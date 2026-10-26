"""Task volume_constrain: the P-V cycle of the imaged chamber. For every tracked frame k the cavity volume is prescribed,
V_k = V0 x V_img,k / V_img,ref (cut surfaces of mesh/), and the model returns the cavity pressure, the displacement and
the stresses. Activation time of frame k: t_k = mod(k - onset, N) BCL / N, onset = the frame of maximum volume +
ONSET_AFTER_ED_FRAMES.
    FE    step solve (MPI): heartFEM's volume-controlled formulation (fe.VolumeProblem); the frames are solved in time
          order from the reference frame, each starting from the previous solution.
    PINN  steps train, solve: with t_k fixed, the cavity pressure P_k solves V_PINN(P, t_k) = V_k (bisection on [0, P_MAX];
          V is monotonic in P); the displacement is the network output at (P_k, t_k). Stresses: F from the P2 displacement
          at the quadrature points; hydrostatic pressure p recovered as the P1 field that best balances the P2 weak
          momentum residual (least squares, base-normal reaction rows dropped); sigma = F S F^T / J - p I with S the
          passive + active second Piola-Kirchhoff stress; cell averages (as the FE's DG0 output) of sigma, F, sigma_ff
          and p; cardiac-frame components from the cell-average F and sigma exactly as the FE.
Output in OUTPUT_DIR: pv_results.npz, summary.csv, vtu/cycle.pvd, run_info.json, params_used.py; step plot (plots.py):
pv_loop.png/.pdf and pv_cycle.mp4.
"""
import json
import time

import numpy as np

from . import model


def fe_cycle(cfg):
    """FE step solve (MPI)."""
    from . import fe
    t_start = time.time()
    mesh_h5 = cfg.output / "_mesh.h5"  # temporary: the serial mesh, read back distributed over the MPI ranks
    g, R = fe.setup(cfg, mesh_h5)
    N = cfg.N_FRAMES
    t_a = model.activation(cfg, np.arange(N))
    ref, ratio = g["reference_frame"], g["V_ratio"]
    fe.log(f"mesh: {len(g['points'])} nodes, {len(g['cells'])} tets; reference frame t{ref}; "
           f"V/V0 {ratio.min():.4f}..{ratio.max():.4f}")
    dt = cfg.BCL_MS / N
    x = np.empty(N)  # activation time unwrapped along the solve order (reference frame first)
    for i in range(N):
        x[(ref + i) % N] = t_a[ref] + i * dt
    states, info = fe.solve_states(
        cfg, g, R, mesh_h5, [(ref + i) % N for i in range(N)], x, lambda pb, k: pb.V0 * ratio[k], cfg.output, "frame",
        describe=lambda k, V: f"  t{k:02d}  V/V0 {ratio[k]:.4f}  t_a {t_a[k]:5.1f} ms",
        row=lambda k, P, V, T, u_max, its, s: {"frame": k, "t_ms": k * cfg.BCL_MS / N, "t_active_ms": t_a[k],
                                              "V_ratio": ratio[k], "V_nL": V * 1e3, "P_kPa": P,
                                              "P_mmHg": P * model.MMHG_PER_KPA, "T_mean_kPa": T, "u_max_um": u_max,
                                              "newton_its": its, "time_s": round(s, 2)})
    if fe.RANK == 0:
        col = states.write(("frame", "t_ms", "t_active_ms", "V_ratio", "V_nL", "P_kPa", "T_mean_kPa"),
                           fe.material_record(cfg, g, onset=model.onset_frame(cfg)))
        info.update({"frames": len(col["frame"]), "mean_frame_s": round(float(col["time_s"].mean()), 1),
                     "max_frame_s": round(float(col["time_s"].max()), 1), "newton_its_total": int(col["newton_its"].sum())})
        mesh_h5.unlink()
        info.update({"total_s": round(time.time() - t_start, 1), "date": time.strftime("%Y-%m-%d %H:%M")})
        (cfg.output / "run_info.json").write_text(json.dumps(info, indent=1))
        fe.log(f"done in {time.time() - t_start:.0f} s -> {cfg.output}: " + json.dumps(info))


def solve_pressures(net, geo, V_target, t, steps):
    """P (M,) with V_PINN(P, t) = V_target for all frames at once (bisection on [0, P_MAX]); flags of frames whose target
    lies outside the trained pressure range (P clipped to 0 or P_MAX)."""
    import torch
    with torch.no_grad():
        volume = lambda P: geo.cavity_volume(net(P, t))  # noqa: E731
        lo, hi = torch.zeros_like(t), torch.full_like(t, net.config["p_scale"])
        below, above = V_target < volume(lo), V_target > volume(hi)
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            smaller = volume(mid) < V_target
            lo, hi = torch.where(smaller, mid, lo), torch.where(smaller, hi, mid)
        P = 0.5 * (lo + hi)
        P = torch.where(below, torch.zeros_like(P), torch.where(above, torch.full_like(P, net.config["p_scale"]), P))
        return P, below | above


def pinn_cycle(cfg):
    """PINN step solve (GPU if visible)."""
    import torch
    from . import pinn
    t_start = time.time()
    device = pinn.device_of()
    net, geo, p2, mat, g, checkpoint = pinn.load_model(cfg, device)
    N = cfg.N_FRAMES
    t_act = model.activation(cfg, np.arange(N))  # time t of each frame (= FE t_active_ms)
    V0 = geo.reference_volume.item()
    t_net = torch.as_tensor(t_act, dtype=torch.float32, device=device)
    target = torch.as_tensor(g["V_ratio"], dtype=torch.float32, device=device) * V0
    t0 = time.time()
    P, clipped = solve_pressures(net, geo, target, t_net, cfg.BISECTION_STEPS)
    t_pressure = time.time() - t0
    print(f"mesh {len(g['points'])} nodes, {len(g['cells'])} tets; V0 {V0 * 1e3:.5f} nL; {N} pressures in {t_pressure:.2f} s "
          f"({cfg.BISECTION_STEPS} bisection steps); frames outside [0, P_MAX]: {int(clipped.sum())}", flush=True)
    states = model.States(cfg, g, g["volume"], cfg.output, "frame")
    for k in range(N):
        t1 = time.time()
        Pk, tk = float(P[k]), float(t_act[k])
        V, u, f, Tk = pinn.state_fields(net, geo, mat, g, p2, Pk, tk, device)
        row = {"frame": k, "t_ms": k * cfg.BCL_MS / N, "t_active_ms": t_act[k], "V_ratio": g["V_ratio"][k],
               "V_nL": V * 1e3, "P_kPa": Pk, "P_mmHg": Pk * model.MMHG_PER_KPA, "T_mean_kPa": Tk,
               "u_max_um": 1e3 * np.linalg.norm(u, axis=1).max(), "P_clipped": int(clipped[k]),
               "time_s": round(time.time() - t1, 2)}
        states.add(k, row, u, f)
        print(f"  t{k:02d}  V/V0 {g['V_ratio'][k]:.4f}  t {t_act[k]:5.1f} ms  T {Tk:6.3f} kPa  ->  P {Pk:+.4f} kPa  ({row['time_s']:.1f} s)", flush=True)
    col = states.write(("frame", "t_ms", "t_active_ms", "V_ratio", "V_nL", "P_kPa", "T_mean_kPa"),
                       {**vars(mat), "fibers": model.fiber_config(cfg)})
    info = {"device": str(device), "nodes": len(g["points"]), "tetrahedra": len(g["cells"]), "p2_dofs": len(p2["dof_points"]),
            "pressure_solve_s": round(t_pressure, 2), "stress_mean_frame_s": round(float(col["time_s"].mean()), 2),
            "total_s": round(time.time() - t_start, 1), "model": "model.pt",
            "model_train_clock_s": checkpoint.get("train_clock_s"), "date": time.strftime("%Y-%m-%d %H:%M")}
    (cfg.output / "run_info.json").write_text(json.dumps(info, indent=1))
    print("done:", json.dumps(info), flush=True)
