"""Task windkessel: the imaged atrium coupled to a closed-loop circulation (heartFEM's coupling through a P-V-t table).

    pvt       P(V, t) of the atrium on the grid of heartFEM getLVbehavior. Volume grid: heartFEM LVbehaviorRun rule
              (SWEEP_N_LOG), built from the largest and smallest imaged cavity volume (IMAGED_V_MAX_NL, IMAGED_V_MIN_NL).
              Time grid: t = 0, SWEEP_DT_MS, ... (interpolated to heartFEM's grid by `table`).
              FE:   swept like heartFEM getLVbehavior (MPI, SWEEP_JOBS independent jobs side by side: job j takes the
                    volumes i with i mod SWEEP_JOBS = j): for every volume a passive solve (heartFEM solveVolume) and then
                    the activation time (solveTa); the sweep of a volume stops once the activation has ended (wall
                    integral of Ct = 0 at two consecutive times); later times are passive. A volume whose solve fails is
                    logged and left out (the smallest volumes: the wall collapses under high tension).
              PINN: the network gives V(P, t) for any (P, t); for every volume and time the pressure solves V(P, t) = V
                    (bisection on the trained range [P_MIN, P_MAX], all entries at once). An entry outside the trained
                    range is flagged (clipped_below, clipped_above) and filled with the range limit; a volume without any
                    entry inside the range is left out.
              Because Ct depends only on the instantaneous sarcomere length and t (heartFEM trackphase off), P is a
              function of the current (V, t): the table is exact for any volume history, and the circuit needs no phase
              tracking.
    table     heartFEM table files (OUTPUT_DIR/table/): per volume a cubic spline in t on heartFEM's time grid, cut to one
              cycle 0 .. BCL; heartFEM ngspice_py.generateLVtable, executed from the original source data/generateLVtable.py,
              writes the ngspice table2d files the circuit reads.
    circuit   the closed-loop circuit (data/atrium_wk.cir, parameters CIRCUIT) in ngspice, metrics of the last cycle;
              circuit_check: half time step and twice the cycles.
    replay    the coupled atrial volume V(t) goes back through the mechanical model (as in heartFEM) at N_REPLAY states
              per cycle: displacement and stresses of the coupled cycle; the pressure must equal the table pressure the
              circuit used. PINN: range check of the table entries around the coupled cycle.
Units of the circuit: pressure mmHg, volume nL, time s, flow nL/s. Time 0 = atrial activation onset.
"""
import csv
import glob
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import model

CASE = "atrium"
MMHG_PER_KPA = model.MMHG_PER_KPA
DATA = Path(__file__).resolve().parent / "data"


# ==================== 1. P(V, t) ====================
def time_grid(cfg):
    t = np.arange(0.0, cfg.SWEEP_T_END_MS + 1e-9, cfg.SWEEP_DT_MS)
    assert np.any(np.isclose(t, cfg.T0_MS)), "T0_MS must be on the sweep time grid"
    return t


def volume_grid(cfg, edv, esv):
    """heartFEM LVbehaviorRun volume steps (descending)."""
    esv_ = 0.4 * edv if (edv - esv) / edv < 0.6 else esv
    v = esv_ * 0.9 * (edv / esv_ * 1.1 / 0.9) ** (np.linspace(0, 1, num=cfg.SWEEP_N_LOG))[::-1]
    return np.concatenate((v[:1] * 1.3, v[:1] * 1.1, v, v[-1:] * 0.9, v[-1:] * 0.7))


def fe_sweep(cfg, job, n_jobs):
    """FE step pvt, one job (MPI): OUTPUT_DIR/pvt/job<job>of<n_jobs>.npz and .log."""
    from . import fe
    job, n_jobs = int(job), int(n_jobs)
    OUT = cfg.output / "pvt"
    T_GRID = time_grid(cfg)
    name = f"job{job}of{n_jobs}"
    comm, rank = fe.COMM, fe.RANK
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)
    mesh_h5 = OUT / f"_mesh_{name}.h5"
    g, R = fe.setup(cfg, mesh_h5)
    logf = open(OUT / f"{name}.log", "w") if rank == 0 else None

    def log(msg):
        if rank == 0:
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

    t_start = time.time()
    pb = fe.VolumeProblem(cfg, comm, g, R, mesh_h5)
    V0_nl = pb.V0 * 1e3
    edv, esv = cfg.IMAGED_V_MAX_NL, cfg.IMAGED_V_MIN_NL  # imaged cavity volumes (nL)
    layer = model.myo_endo(cfg)
    if layer:  # the imaged volumes are inside the myocardium; the cavity is the lumen (the incompressible layer shifts it)
        edv, esv = edv - pb.endo_volume * 1e3, esv - pb.endo_volume * 1e3
    vgrid = volume_grid(cfg, edv, esv)
    mine = np.arange(len(vgrid))[job::n_jobs]
    mine = mine[np.argsort(np.abs(np.log(vgrid[mine] / V0_nl)))]  # outward from the reference volume V0
    log(f"{name}: {comm.size} ranks, {pb.W.dim()} unknowns, V0 {V0_nl:.5f} nL, imaged volume {esv:.5f}..{edv:.5f} nL; "
        f"{len(mine)} of {len(vgrid)} volumes ({vgrid[mine].min():.4f}..{vgrid[mine].max():.4f} nL), "
        f"t 0..{T_GRID[-1]:g} ms step {cfg.SWEEP_DT_MS:g}; tr = {cfg.TR_M_MS_PER_UM:g} ls {cfg.TR_B_MS:+g}"
        + (f"; endocardial layer {pb.endo_volume * 1e3:.5f} nL" if layer else ""))
    nV, nT = len(vgrid), len(T_GRID)
    P = np.full((nV, nT), np.nan)
    Ctm = np.full((nV, nT), np.nan)
    its = np.zeros((nV, nT), int)
    secs = np.zeros((nV, nT))
    t_stop = np.full(nV, np.nan)
    done = np.zeros(nV, bool)

    def save():
        if rank == 0:
            np.savez(OUT / f"{name}.npz", V_grid_nL=vgrid, t_grid_ms=T_GRID, P_kPa=P, Ct_mean=Ctm, newton_its=its,
                     seconds=secs, t_stop_ms=t_stop, done=done, assigned=np.isin(np.arange(nV), mine), V0_nL=V0_nl,
                     EDV_nL=edv, ESV_nL=esv,
                     info=json.dumps({"job": name, "ranks": comm.size, "dt_ms": cfg.SWEEP_DT_MS,
                                      "tr_m": cfg.TR_M_MS_PER_UM, "tr_b": cfg.TR_B_MS, "T_MAX": cfg.T_MAX, "t0": cfg.T0_MS}))

    # passive continuation, separately towards larger and towards smaller volumes: (solution, (V, t = 0))
    chain = {s: (pb.w.vector().copy(), (pb.V0, 0.0)) for s in (+1, -1)}
    for i in mine:
        side = +1 if vgrid[i] >= V0_nl else -1
        vec, state = chain[side]
        pb.set_vector(vec)
        target = (vgrid[i] * 1e-3, 0.0)
        t0 = time.time()
        ok, n_it, rr = pb.march(state, target, h=0.25)
        if not ok:
            log(f"  V[{i}] {vgrid[i]:.4f} nL: passive solve FAILED (|R|/R_ref {rr / pb.R_ref:.1e}); volume skipped")
            continue
        chain[side] = (pb.w.vector().copy(), target)
        current, zero_count = target, 0
        for j, t in enumerate(T_GRID):
            if j > 0:
                t0 = time.time()
                tgt = (target[0], float(t))
                ok, n_it, rr = pb.march(current, tgt)
                if not ok:
                    log(f"  V[{i}] {vgrid[i]:.4f} nL: t {t:g} ms FAILED (|R|/R_ref {rr / pb.R_ref:.1e}); rest skipped")
                    break
                current = tgt
            P[i, j] = fe.assemble(pb.P * fe.dx(pb.mesh)) / pb.wall_volume
            if layer:  # the stop rule follows the activation of the myocardium
                Ctm[i, j] = pb.myo_mean(pb.Ct)
            else:
                Ctm[i, j] = fe.assemble(pb.Ct * fe.dx(pb.mesh)) / pb.wall_volume
            its[i, j], secs[i, j] = n_it, time.time() - t0
            zero_count = zero_count + 1 if (t > cfg.T0_MS and Ctm[i, j] < 1e-12) else 0
            if zero_count >= 2:  # heartFEM: stop once the pressure no longer changes; the rest is passive
                t_stop[i] = t
                P[i, j + 1:], Ctm[i, j + 1:] = P[i, j], 0.0
                break
        done[i] = np.isfinite(P[i]).all()
        log(f"  V[{i:2d}] {vgrid[i]:.4f} nL: P(t=0) {P[i, 0]:+.4f}  max P {np.nanmax(P[i]):+.4f} kPa  "
            f"stop {t_stop[i]:g} ms  Newton {its[i].sum()}  {secs[i].sum():.0f} s  (total {time.time() - t_start:.0f} s)")
        save()
    save()
    if rank == 0:
        mesh_h5.unlink()
    log(f"{name} done in {time.time() - t_start:.0f} s: {done[mine].sum()} / {len(mine)} volumes")


class CavityMap:
    """Cavity volume V(P, t) of the trained network and its inverse P(V, t).

    The network output is linear in its last hidden features: u = sum_k phi_k(P, t) U_k. The nodal fields U_k are
    evaluated once, and only the endocardial nodes are kept, because the cavity volume depends on them alone. This is
    the same function as geo.cavity_volume(net(P, t)) (checked at construction), without computing the displacement of
    the whole wall."""
    CHUNK = 512  # states per evaluation of V(P, t)

    def __init__(self, net, geo, bisection_steps):
        import copy
        import torch
        self.torch, self.steps = torch, bisection_steps
        self.net, self.p_min, self.p_max = net, float(net.config["p_min"]), float(net.config["p_scale"])
        device = geo.dof_points.device
        with torch.no_grad():
            phi = torch.eye(net.whiten_A.shape[0], dtype=torch.float32, device=device)
            n = len(phi)
            u = torch.nn.functional.linear(phi, net.out.weight).reshape(n, net.n_dofs, 3)
            for i, layer in enumerate(net.coarse_out):
                w = torch.nn.functional.linear(phi, layer.weight).reshape(n, -1, 3).permute(1, 0, 2).reshape(-1, 3 * n)
                u = u + net.config["level_scale"] * torch.sparse.mm(net.interpolations[i].to(device), w).reshape(
                    net.n_dofs, n, 3).permute(1, 0, 2)
            normal = net.base_normal
            u = net.config["u_scale"] * (u - (net.base_mask[:, None] * (u @ normal)[..., None]) * normal)
            nodes, local = torch.unique(geo.endo6, return_inverse=True)
            self.basis = u[:, nodes].reshape(n, -1)  # (features, 3 x endocardial DOFs)
            self.sub = copy.copy(geo)  # geometry restricted to the endocardial DOFs, for Geometry.cavity_volume
            self.sub.dof_points, self.sub.endo6 = geo.dof_points[nodes], local
            self.reference_volume = float(self.volume(torch.zeros(1, device=device), torch.zeros(1, device=device)))
            # check against the full network
            P = torch.tensor([0.02, 0.3, 1.0, 0.1], device=device)
            t = torch.tensor([0.0, 60.0, 150.0, 400.0], device=device)
            full = geo.cavity_volume(net(P, t))
            assert torch.allclose(self.volume(P, t), full, rtol=2e-6), (self.volume(P, t), full)

    def volume(self, P, t):
        """Cavity volume (mm^3 = uL), (M,), at loads P (kPa), t (ms)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for c in torch.split(torch.arange(len(P), device=P.device), self.CHUNK):
                phi = (self.net.hidden_difference(P[c], t[c]) @ self.net.whiten_A.T).to(P.dtype)
                out.append(self.sub.cavity_volume((phi @ self.basis).reshape(len(c), -1, 3)))
        return torch.cat(out)

    def solve_pressures(self, V_target, t):
        """P (M,) with V(P, t) = V_target (bisection on the trained range [P_MIN, P_MAX]; V increases with P), and the
        flags of the states whose target lies below / above the trained range (P = the range limit)."""
        torch = self.torch
        floor = torch.full_like(V_target, self.p_min)
        lo, hi = floor.clone(), torch.full_like(V_target, self.p_max)
        below, above = V_target < self.volume(lo, t), V_target > self.volume(hi, t)
        for _ in range(self.steps):
            mid = 0.5 * (lo + hi)
            smaller = self.volume(mid, t) < V_target
            lo, hi = torch.where(smaller, mid, lo), torch.where(smaller, hi, mid)
        P = 0.5 * (lo + hi)
        return torch.where(below, floor, torch.where(above, torch.full_like(P, self.p_max), P)), below, above


def pinn_table(cfg):
    """PINN step pvt: OUTPUT_DIR/pvt/job0of1.npz and .log (the format of the FE sweep files)."""
    import torch
    from . import pinn
    OUT = cfg.output / "pvt"
    T_GRID = time_grid(cfg)
    t_start = time.time()
    device = pinn.device_of()
    OUT.mkdir(parents=True, exist_ok=True)
    net, geo, p2, mat, g, checkpoint = pinn.load_model(cfg, device)
    cav = CavityMap(net, geo, cfg.BISECTION_STEPS)
    t_load = time.time() - t_start
    V0_nl = cav.reference_volume * 1e3
    edv, esv = cfg.IMAGED_V_MAX_NL, cfg.IMAGED_V_MIN_NL  # imaged cavity volumes (nL)
    vgrid = volume_grid(cfg, edv, esv)
    nV, nT = len(vgrid), len(T_GRID)
    VV, TT = np.meshgrid(vgrid * 1e-3, T_GRID, indexing="ij")
    f32 = lambda a: torch.as_tensor(a.ravel(), dtype=torch.float32, device=device)  # noqa: E731
    t0 = time.time()
    P, below, above = cav.solve_pressures(f32(VV), f32(TT))
    t_solve = time.time() - t0
    P = P.cpu().numpy().reshape(nV, nT)
    below, above = below.cpu().numpy().reshape(nV, nT), above.cpu().numpy().reshape(nV, nT)
    clipped = below | above
    done = ~clipped.all(1)  # a volume without any entry inside the trained range is left out
    info = {"device": str(device), "model": "model.pt", "model_train_clock_s": checkpoint.get("train_clock_s"),
            "P_range_kPa": [cav.p_min, cav.p_max], "bisection_steps": cfg.BISECTION_STEPS, "states": int(nV * nT),
            "load_s": round(t_load, 2), "solve_s": round(t_solve, 2), "dt_ms": cfg.SWEEP_DT_MS,
            "clipped_below": int(below.sum()), "clipped_above": int(above.sum()), "volumes_kept": int(done.sum()),
            "tr_m": cfg.TR_M_MS_PER_UM, "tr_b": cfg.TR_B_MS, "T_MAX": cfg.T_MAX, "t0": cfg.T0_MS,
            "date": time.strftime("%Y-%m-%d %H:%M")}
    np.savez(OUT / "job0of1.npz", V_grid_nL=vgrid, t_grid_ms=T_GRID, P_kPa=P, done=done, assigned=np.ones(nV, bool),
             clipped_below=below, clipped_above=above, V0_nL=V0_nl, EDV_nL=edv, ESV_nL=esv, info=json.dumps(info))
    lines = [f"PINN table: {nV} volumes {vgrid.min():.4f}..{vgrid.max():.4f} nL x {nT} times 0..{T_GRID[-1]:g} ms step "
             f"{cfg.SWEEP_DT_MS:g} = {nV * nT} states; V0 {V0_nl:.5f} nL, imaged volume {esv:.5f}..{edv:.5f} nL",
             f"device {device}; model loaded in {t_load:.1f} s; {nV * nT} pressures in {t_solve:.2f} s "
             f"({cfg.BISECTION_STEPS} bisection steps on [{cav.p_min:g}, {cav.p_max:g}] kPa)",
             f"outside the trained range: {int(below.sum())} states below P_MIN, {int(above.sum())} above P_MAX "
             f"({100 * clipped.mean():.1f} % of the table); volumes kept: {int(done.sum())} / {nV}"]
    for i in range(nV):
        lines.append(f"  V[{i:2d}] {vgrid[i]:.4f} nL: P(t=0) {P[i, 0]:+.4f}  max P {P[i].max():+.4f}  min P {P[i].min():+.4f} kPa  "
                     f"clipped below / above {int(below[i].sum())} / {int(above[i].sum())} of {nT}")
    (OUT / "job0of1.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:3]), flush=True)


# ==================== 2. heartFEM table files ====================
def heartfem_time_space(bcl, t0):
    """heartFEM getLVbehavior timeSpace (ms), 0 .. 2 BCL."""
    ts, t = [], 0.0
    while t < 2 * bcl:
        ts.append(t)
        if 0.0 <= t < 4.0:
            t += 0.50
        elif t0 - 0.5 <= t < t0 + 0.5:
            t += 3
        else:
            t += 1.0
    return np.array(ts)


def load_sweep(cfg):
    """P(V, t) of all files of OUTPUT_DIR/pvt/ (kPa -> mmHg); volumes that did not finish (FE) or have no entry inside the
    trained pressure range (PINN) are left out."""
    SWEEP = cfg.output / "pvt"
    files = sorted(glob.glob(str(SWEEP / "job*.npz")))
    assert files, f"no sweep results in {SWEEP}"
    d0 = np.load(files[0])
    Vg, tg = d0["V_grid_nL"], d0["t_grid_ms"]
    P = np.full((len(Vg), len(tg)), np.nan)
    assigned = np.zeros(len(Vg), bool)
    for f in files:
        d = np.load(f)
        assert np.allclose(d["V_grid_nL"], Vg) and np.allclose(d["t_grid_ms"], tg), f"{f}: different grid"
        m = d["done"]
        P[m] = d["P_kPa"][m]
        assigned |= d["assigned"]
    assert assigned.all(), f"volumes not covered by any job: {np.where(~assigned)[0]} (run all n_jobs jobs)"
    ok = np.isfinite(P).all(1)
    print(f"sweep: {len(files)} jobs, {ok.sum()} / {len(Vg)} volumes complete ({Vg[ok].min():.4f}..{Vg[ok].max():.4f} nL); "
          f"not converged: {', '.join(f'{v:.3f}' for v in Vg[~ok]) or 'none'}")
    return Vg[ok], tg, P[ok] * MMHG_PER_KPA


def run_generate_lv_table(casename, period, timetopeak):
    """Execute heartFEM's generateLVtable from the original source. Two lines are dropped (`import vtk`,
    `from heartFEM_G import ngspice_py`; neither is used by the function). One guard is added: a volume without
    active pressure (fibers below l0) has no half-relaxation point, which crashes the original when it writes the
    relaxation-time table; that entry is set to 0.1 s. The relaxation-time table is not read by the circuit."""
    src = (DATA / "generateLVtable.py").read_text()
    src = "\n".join(l for l in src.splitlines() if l.strip() not in ("import vtk", "from heartFEM_G import ngspice_py"))
    line = "            tryTimeInd=np.argwhere(temp_datatable<temp_maxpress/2.)[0,0]"
    assert line in src
    src = src.replace(line, "            if not np.any(temp_datatable<temp_maxpress/2.):\n"
                      "                trdata.append(0.1)\n                continue\n" + line)
    ns = {"__name__": "generateLVtable_original"}
    exec(compile(src, "heartfem/generateLVtable.py", "exec"), ns)
    ns["generateLVtable"](casename, period, timetopeak=timetopeak, loading_casename=casename, scale_T0_LV=1.0)


def read_table2d(path):
    """heartFEM / ngspice table2d file: n_x, n_y, x values, y values, then the n_y x n_x table (rows y, columns x)."""
    lines = [l for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("*")]
    nx, ny = int(lines[0]), int(lines[1])
    x = np.array(lines[2].split(), float)
    y = np.array(lines[3].split(), float)
    z = np.array([l.split() for l in lines[4:4 + ny]], float)
    assert z.shape == (ny, nx)
    return x, y, z


def read_total_table(table_dir):
    """V (nL, ascending), t (ms), total pressure P[t, V] (mmHg) = active table + passive base, as the circuit sees it."""
    V, t, act = read_table2d(Path(table_dir) / f"{CASE}_lvcirtable.txt")
    _, Vb, base = read_table2d(Path(table_dir) / f"{CASE}_lvcirtablebase.txt")
    assert np.allclose(Vb, V)
    return V, t * 1e3, act + base[:, 0][None, :]


def build_table(cfg):
    """Step table -> OUTPUT_DIR/table/:
        atrium_Press_volumeSpace.txt   volumes (nL, descending, as heartFEM)
        atrium_Press_timeSpace.txt     activation times (ms) of heartFEM's grid (0.5 ms below 4 ms, 3 ms around t0, else
                                       1 ms), cut to one cycle 0 .. BCL (the circuit reads only the cyclic time)
        atrium_Press_VolTime_base.txt  passive pressure = min(P at the first time, P at the last time)
        atrium_Press_VolTime.txt       active pressure  = P - passive (exactly 0 once the activation has ended)
        atrium_lvcirtable.txt, atrium_lvcirtablebase.txt, atrium_lvcirtabletr.txt   (heartFEM generateLVtable) ngspice
                                       table2d files (volume nL, time s, pressure mmHg); the circuit reads the first two
        table_check.npz                leave-one-out error of the spline in t, and the assembled table"""
    OUT = cfg.output / "table"
    OUT.mkdir(parents=True, exist_ok=True)
    Vg, tg, P = load_sweep(cfg)  # Vg descending (heartFEM order), P mmHg [V, t]
    from scipy.interpolate import CubicSpline
    loo = []
    for j in range(1, len(tg) - 1):
        keep = np.r_[0:j, j + 1:len(tg)]
        loo.append(np.abs(CubicSpline(tg[keep], P[:, keep], axis=1)(tg[j]) - P[:, j]).max())
    spl = CubicSpline(tg, P, axis=1)
    ts = heartfem_time_space(cfg.BCL_MS, cfg.T0_MS)
    ts = ts[ts <= cfg.BCL_MS]
    if ts[-1] < cfg.BCL_MS:
        ts = np.r_[ts, cfg.BCL_MS]  # the cyclic time reaches BCL
    Pvt = np.where(ts[None, :] <= tg[-1], spl(np.minimum(ts, tg[-1])), P[:, -1:])
    base = np.minimum(Pvt[:, -1], Pvt[:, 0])
    act = Pvt - base[:, None]
    act[np.abs(act) < 1e-9] = 0.0
    cn = str(OUT / CASE)
    np.savetxt(cn + "_Press_volumeSpace.txt", Vg)
    np.savetxt(cn + "_Press_timeSpace.txt", ts)
    np.savetxt(cn + "_Press_VolTime.txt", act, header="rowVolm: (nL) colTime: (ms); active pressure mmHg")
    np.savetxt(cn + "_Press_VolTime_base.txt", base)
    run_generate_lv_table(cn, cfg.BCL_MS, cfg.T0_MS)
    V, t_ms, Ptot = read_total_table(OUT)
    last = t_ms[np.abs(Ptot - Ptot[-1][None, :]).max(1) > 1e-9].max()
    print(f"table: {len(V)} volumes {V.min():.4f}..{V.max():.4f} nL x {len(t_ms)} times 0..{t_ms.max():g} ms; "
          f"P {Ptot.min():.2f}..{Ptot.max():.2f} mmHg; activation over after {last:g} ms")
    print("spline leave-one-out in t (step %g ms): max %.3e mmHg" % (tg[1] - tg[0], max(loo)))
    np.savez(OUT / "table_check.npz", loo=np.array(loo), V_nL=V, t_ms=t_ms, P_mmHg=Ptot)

    print("wrote", OUT)


# ==================== 3. circuit ====================
SUFFIX = {4: "T", 3: "g", 2: "meg", 1: "k", 0: "", -1: "m", -2: "u", -3: "n", -4: "p", -5: "f"}
# columns of the ngspice output (wrdata line of the template)
COLS = ["t", "P_la", "V_la", "P_lv", "V_lv", "P_art", "P_ven", "Q_la_in", "Q_lv_in", "Q_av", "Q_out", "Q_venla",
        "la_phasetime", "lv_elast", "lv_act"]
Q_OPEN = 1e-3  # nL/s: a valve counts as open above this flow (closed-valve leakage is ~1e-6)


def spice(v):
    """Number in SPICE notation (9 significant digits)."""
    if v == 0:
        return "0"
    i = max(-5, min(4, int(np.floor(np.log10(abs(v)) / 3.0))))
    return "{:.9g}{}".format(v / 1000.0 ** i, SUFFIX[i])


def simulate(cfg, params=None, workdir=None, case="atrium"):
    """Fill the template, run ngspice, return (signals of the kept cycles, parameters). Time restarts at 0 at the
    beginning of the kept cycles; signals: COLS (la = atrium, lv = ventricle; pressures mmHg, volumes nL, flows nL/s)."""
    p = dict(cfg.CIRCUIT)
    p.update(params or {})
    workdir = Path(workdir or cfg.output / "circuit")
    table_dir = cfg.output / "table"
    workdir.mkdir(parents=True, exist_ok=True)
    ngspice = shutil.which(cfg.NGSPICE)
    if ngspice is None:
        raise RuntimeError(f"ngspice executable '{cfg.NGSPICE}' not found (params.NGSPICE)")
    bcl = p["cycle"]
    fill = {k: spice(v) for k, v in p.items() if k not in ("n_cycles", "keep_cycles")}
    fill.update({"stopTime": spice(bcl * p["n_cycles"]), "startTime": spice(bcl * (p["n_cycles"] - p["keep_cycles"])),
                 "outfile": "out", "lautablefile": "la_table.txt", "lautablebasefile": "la_tablebase.txt"})
    # ngspice lower-cases model file paths: copy the tables next to the netlist (heartFEM also copies them)
    for src, dst in (("_lvcirtable.txt", "la_table.txt"), ("_lvcirtablebase.txt", "la_tablebase.txt")):
        (workdir / dst).write_text((Path(table_dir) / f"{case}{src}").read_text())
    text = (DATA / "atrium_wk.cir").read_text()
    for k, v in fill.items():
        text = text.replace(f"<<{k}>>", v)
    assert "<<" not in text, [l for l in text.splitlines() if "<<" in l]
    (workdir / "run.cir").write_text(text)
    (workdir / "params.json").write_text(json.dumps(p, indent=1))
    r = subprocess.run([ngspice, "-b", "-o", "run.log", "run.cir"], cwd=workdir, capture_output=True, text=True)
    out = workdir / "out"
    logtxt = (workdir / "run.log").read_text() if (workdir / "run.log").exists() else ""
    if "cannot open" in logtxt:
        raise RuntimeError(f"ngspice could not open a file in {workdir}")
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"ngspice failed in {workdir}: {logtxt[-2000:]}")
    a = np.loadtxt(out, skiprows=1)
    out.unlink()
    d = {c: a[:, i] for i, c in enumerate(COLS)}
    d["t"] = d["t"] - bcl * (p["n_cycles"] - p["keep_cycles"])
    np.savez_compressed(workdir / "result.npz", **d)
    return d, p


def last_cycle(d, bcl, k=1):
    """Cycle k from the end (1 = last), resampled to 1 ms; t in s from the atrial activation onset."""
    t_end = d["t"][-1]
    t = np.linspace(0, bcl, int(round(bcl * 1000)) + 1)
    t0 = t_end - k * bcl
    return {"t": t, **{c: np.interp(t0 + t, d["t"], d[c]) for c in d if c != "t"}}


def _edges(mask):
    """Circular rising / falling edges of a boolean series (sample indices where it switches on / off)."""
    prev = np.roll(mask, 1)
    return np.where(mask & ~prev)[0], np.where(~mask & prev)[0]


def valve_events(c):
    """Ventricular timing from the valve events, not from volume extrema: with ideal valves the volume is exactly
    flat while both valves are closed, so the position of its maximum or minimum is arbitrary there.
    End-diastole = AV valve closure before ejection; end-systole = outflow valve closure. Times in ms from the atrial
    onset (circular in the cycle)."""
    n = len(c["t"]) - 1  # the last sample repeats the first
    dt = (c["t"][1] - c["t"][0]) * 1e3
    av, out, act = c["Q_av"][:n] > Q_OPEN, c["Q_out"][:n] > Q_OPEN, c["lv_act"][:n] > 1e-3
    e = {}
    av_on, av_off = _edges(av)
    out_on, out_off = _edges(out)
    act_on, _ = _edges(act)
    if len(out_on) != 1 or len(out_off) != 1 or len(av_off) == 0 or len(act_on) != 1:
        e["events_ok"] = 0.0
        return e
    ej0, ej1, on = out_on[0], out_off[0], act_on[0]
    back = lambda i, j: (i - j) % n  # samples from j forward to i  # noqa: E731
    av_close = min(av_off, key=lambda k: back(ej0, k))  # last AV closure before ejection
    av_open = min(av_on, key=lambda k: back(k, ej1))  # first AV opening after ejection
    e["events_ok"] = 1.0
    e["lv_t_ED_ms"], e["lv_t_ES_ms"] = av_close * dt, ej1 * dt
    e["lv_t_contraction_ms"], e["lv_t_ejection_start_ms"], e["lv_t_AV_open_ms"] = on * dt, ej0 * dt, av_open * dt
    wait = back(on, av_close)
    wait = wait if wait <= n // 2 else 0  # contraction starts before the AV closure: no wait
    e["lv_wait_ms"] = wait * dt  # ventricle relaxed with both valves closed before it contracts
    e["lv_ivc_ms"] = back(ej0, (av_close + wait) % n) * dt  # isovolumic contraction
    e["lv_ejection_ms"] = back(ej1, ej0) * dt
    e["lv_ivr_ms"] = back(av_open, ej1) * dt  # isovolumic relaxation
    e["lv_filling_ms"] = av.sum() * dt
    e["lv_EDV"], e["lv_ESV"] = c["V_lv"][av_close], c["V_lv"][ej1]
    e["lv_EDP"], e["lv_ESP"] = c["P_lv"][av_close], c["P_lv"][ej1]
    e["art_P_at_ejection_start"] = c["P_art"][ej0]
    return e


def metrics(cfg, d, p):
    """Summary of the last cycle (pressures mmHg, volumes nL, flows nL/s, times ms)."""
    bcl = p["cycle"]
    c, c2 = last_cycle(d, bcl, 1), last_cycle(d, bcl, 2)
    m = {}
    for ch in ("la", "lv"):
        V, P = c[f"V_{ch}"], c[f"P_{ch}"]
        m[f"{ch}_Vmax"], m[f"{ch}_Vmin"] = V.max(), V.min()
        m[f"{ch}_Pmax"], m[f"{ch}_Pmin"] = P.max(), P.min()
        m[f"{ch}_EF"] = (V.max() - V.min()) / V.max()
    # times of the atrial extremes (the atrium has no closed-valve plateau; ventricular timing: valve_events)
    m["la_t_Vmin_ms"], m["la_t_Pmax_ms"] = c["t"][c["V_la"].argmin()] * 1e3, c["t"][c["P_la"].argmax()] * 1e3
    m["la_t_Pmin_ms"], m["lv_t_Pmax_ms"] = c["t"][c["P_la"].argmin()] * 1e3, c["t"][c["P_lv"].argmax()] * 1e3
    m["art_Pmax"], m["art_Pmin"], m["art_Pmean"] = c["P_art"].max(), c["P_art"].min(), c["P_art"].mean()
    m["ven_Pmean"] = c["P_ven"].mean()
    dt = c["t"][1] - c["t"][0]
    q_in = c["Q_venla"]
    m["la_inflow_fwd"] = np.clip(q_in, 0, None).sum() * dt  # nL per cycle, veins -> atrium
    m["la_inflow_back"] = -np.clip(q_in, None, 0).sum() * dt  # atrium -> veins (no inlet valve)
    m["av_fwd"] = np.clip(c["Q_av"], 0, None).sum() * dt
    m["av_back"] = -np.clip(c["Q_av"], None, 0).sum() * dt
    m["out_fwd"] = np.clip(c["Q_out"], 0, None).sum() * dt
    m["cardiac_output_nL_s"] = m["out_fwd"] / bcl
    m["periodic_err"] = max(np.abs(c["V_la"] - c2["V_la"]).max(), np.abs(c["V_lv"] - c2["V_lv"]).max())
    m.update(valve_events(c))
    # atrial contraction window: share of the ventricular filling, a-wave
    early = c["t"] < cfg.ATRIAL_CONTRACTION_END_MS / 1000.0
    qav = np.clip(c["Q_av"], 0, None)
    m["A_fraction"] = qav[early].sum() / max(qav.sum(), 1e-12)
    m["la_a_wave"] = c["P_la"][early].max() - c["P_la"][0]
    return {k: float(v) for k, v in m.items()}


def circuit(cfg):
    """Step circuit: OUTPUT_DIR/circuit/ (run.cir, run.log, params.json, result.npz, metrics.json)."""
    d, p = simulate(cfg)
    m = metrics(cfg, d, p)
    (cfg.output / "circuit" / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


def circuit_check(cfg):
    """Step circuit_check: numerical checks (half time step, twice the cycles) -> OUTPUT_DIR/circuit/numerics_check.json."""
    OUT = cfg.output / "circuit"
    base = json.loads((OUT / "metrics.json").read_text())
    res = {}
    for name, over in (("half_step", {"step": cfg.CIRCUIT["step"] / 2}),
                       ("double_cycles", {"n_cycles": 2 * cfg.CIRCUIT["n_cycles"]})):
        d, p = simulate(cfg, over, OUT / f"check_{name}")
        m = metrics(cfg, d, p)
        (OUT / f"check_{name}" / "result.npz").unlink()
        ks = [k for k in base if k in m and not k.endswith("_ms")]
        res[name] = {"max_abs_change": max(abs(m[k] - base[k]) for k in ks),
                     "max_timing_change_ms": max(abs(m[k] - base[k]) for k in base if k in m and k.endswith("_ms"))}
        print(name, json.dumps(res[name]))
    (OUT / "numerics_check.json").write_text(json.dumps(res, indent=1))


# ==================== 4. replay ====================
def replay_targets(cfg):
    """States t_j = (j + 0.5) BCL / N_REPLAY from the atrial activation onset and the circuit's atrial volume and
    pressure at these times (last cycle of OUTPUT_DIR/circuit/result.npz)."""
    d = dict(np.load(cfg.output / "circuit" / "result.npz"))
    bcl = cfg.BCL_MS / 1000.0
    assert abs(cfg.CIRCUIT["cycle"] - bcl) < 1e-12
    t_ms = (np.arange(cfg.N_REPLAY) + 0.5) * cfg.BCL_MS / cfg.N_REPLAY
    tc = d["t"][-1] - bcl + t_ms / 1000.0
    return d, t_ms, np.interp(tc, d["t"], d["V_la"]), np.interp(tc, d["t"], d["P_la"])


def fe_replay(cfg):
    """FE step replay (MPI) -> OUTPUT_DIR/replay/: summary.csv (one row per state), pv_results.npz (u, stresses, Ct per
    state), vtu/cycle.pvd, replay_vs_circuit.csv (FE pressure next to the circuit pressure), run_info.json.
    The loop starts at the last state (late in the cycle, where the wall is passive) and then runs through the cycle
    in time order, each state starting from the previous solution."""
    from . import fe
    t_start = time.time()
    OUT = cfg.output / "replay"
    _, t_ms, V_c, P_c = replay_targets(cfg)
    mesh_h5 = OUT / "_mesh.h5"
    g, R = fe.setup(cfg, mesh_h5)
    n = len(t_ms)
    order = [(n - 1 + i) % n for i in range(n)]
    x = np.asarray(t_ms, float) + cfg.BCL_MS  # activation time unwrapped along the solve order (the forms see mod BCL)
    x[n - 1] -= cfg.BCL_MS
    states, info = fe.solve_states(
        cfg, g, R, mesh_h5, order, x, lambda pb, k: float(V_c[k]) * 1e-3, OUT, "state",
        describe=lambda k, V: f"  {k:02d}  t {t_ms[k]:6.1f} ms  V {V * 1e3:.4f} nL",
        row=lambda k, P, V, T, u_max, its, s: {"state": k, "t_ms": t_ms[k], "V_nL": V * 1e3, "P_kPa": P,
                                              "P_mmHg": P * MMHG_PER_KPA, "T_mean_kPa": T, "u_max_um": u_max,
                                              "newton_its": its, "time_s": round(s, 2)})
    if fe.RANK == 0:
        col = states.write(("state", "t_ms", "V_nL", "P_kPa", "T_mean_kPa"), fe.material_record(cfg, g))
        info.update({"states": len(col["state"]), "mean_state_s": round(float(col["time_s"].mean()), 1),
                     "max_state_s": round(float(col["time_s"].max()), 1), "newton_its_total": int(col["newton_its"].sum())})
        mesh_h5.unlink()
        rows = list(csv.DictReader(open(OUT / "summary.csv")))
        with open(OUT / "replay_vs_circuit.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["state", "t_ms", "V_circuit_nL", "V_fe_nL", "P_circuit_mmHg", "P_fe_mmHg"])
            for r_ in rows:
                k = int(r_["state"])
                w.writerow([k, f"{t_ms[k]:.3f}", f"{V_c[k]:.6f}", r_["V_nL"], f"{P_c[k]:.5f}", r_["P_mmHg"]])
        dp = np.array([float(r_["P_mmHg"]) for r_ in rows]) - P_c[[int(r_["state"]) for r_ in rows]]
        info.update({"max_abs_dP_mmHg": float(np.abs(dp).max()), "rms_dP_mmHg": float(np.sqrt((dp ** 2).mean())),
                     "total_s": round(time.time() - t_start, 1), "date": time.strftime("%Y-%m-%d %H:%M")})
        (OUT / "run_info.json").write_text(json.dumps(info, indent=1))
        print(f"replay: FE P - circuit P: max |dP| {np.abs(dp).max():.4f} mmHg, rms {np.sqrt((dp ** 2).mean()):.4f}; "
              f"done in {time.time() - t_start:.0f} s -> {OUT}", flush=True)


def range_check(cfg, d, bcl):
    """PINN: number of 1 ms samples of the last cycle whose table neighbourhood (what the circuit's third-order table
    interpolation can touch: the two grid volumes that bracket the atrial volume plus one on each side, and the same in
    time on the sweep grid) holds an entry outside the trained pressure range, and the number of samples."""
    sweep = np.load(cfg.output / "pvt" / "job0of1.npz")
    keep = sweep["done"]
    order = np.argsort(sweep["V_grid_nL"][keep])
    Vg, tg = sweep["V_grid_nL"][keep][order], sweep["t_grid_ms"]
    clipped = (sweep["clipped_below"] | sweep["clipped_above"])[keep][order]
    c = last_cycle(d, bcl)
    iv = np.clip(np.searchsorted(Vg, c["V_la"][:-1]), 1, len(Vg) - 1)  # V between Vg[iv - 1] and Vg[iv]
    it = np.minimum((c["t"][:-1] * 1e3 / (tg[1] - tg[0])).astype(int), len(tg) - 1)  # later times: the last (passive) column
    bad = 0
    for i, j in zip(iv, it):
        bad += bool(clipped[max(i - 2, 0):i + 2, max(j - 1, 0):j + 3].any())
    return bad, len(iv)


def pinn_replay(cfg):
    """PINN step replay -> OUTPUT_DIR/replay/ (the files of the FE replay). For each state the pressure solves
    V(P, t_j) = V_j (bisection on the trained pressure range), the displacement is the network output at (P_j, t_j), and
    the stresses follow as in volume_constrain."""
    import torch
    from . import pinn
    t_start = time.time()
    OUT = cfg.output / "replay"
    d, t_ms, V_c, P_c = replay_targets(cfg)
    bcl = cfg.BCL_MS / 1000.0
    device = pinn.device_of()
    net, geo, p2, mat, g, checkpoint = pinn.load_model(cfg, device)
    cav = CavityMap(net, geo, cfg.BISECTION_STEPS)
    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    t0 = time.time()
    P, below, above = cav.solve_pressures(f32(V_c * 1e-3), f32(t_ms))
    t_pressure = time.time() - t0
    clipped = (below | above).cpu().numpy()
    n_bad, n_samples = range_check(cfg, d, bcl)
    print(f"mesh {len(g['points'])} nodes, {len(g['cells'])} tets; V0 {cav.reference_volume * 1e3:.5f} nL; {cfg.N_REPLAY} "
          f"pressures in {t_pressure:.2f} s; states outside the trained pressure range: {int(clipped.sum())}", flush=True)
    states = model.States(cfg, g, g["volume"], OUT, "state")
    for k in range(cfg.N_REPLAY):
        t1 = time.time()
        Pk, tk = float(P[k]), float(t_ms[k])
        V, u, f, Tk = pinn.state_fields(net, geo, mat, g, p2, Pk, tk, device)
        states.add(k, {"state": k, "t_ms": t_ms[k], "V_nL": V * 1e3, "P_kPa": Pk, "P_mmHg": Pk * MMHG_PER_KPA, "T_mean_kPa": Tk,
                       "u_max_um": 1e3 * np.linalg.norm(u, axis=1).max(), "P_clipped": int(clipped[k]),
                       "time_s": round(time.time() - t1, 2)}, u, f)
        print(f"  {k:02d}  t {t_ms[k]:6.1f} ms  V {V * 1e3:.4f} nL  mean T {Tk:6.3f} kPa  ->  P {Pk:+.4f} kPa  "
              f"({states.states[k][0]['time_s']:.1f} s)", flush=True)
    col = states.write(("state", "t_ms", "V_nL", "P_kPa", "T_mean_kPa"), {**vars(mat), "fibers": model.fiber_config(cfg)})
    rows = [states.states[k][0] for k in sorted(states.states)]
    with open(OUT / "replay_vs_circuit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["state", "t_ms", "V_circuit_nL", "V_pinn_nL", "P_circuit_mmHg", "P_pinn_mmHg"])
        for r in rows:
            k = r["state"]
            w.writerow([k, f"{t_ms[k]:.3f}", f"{V_c[k]:.6f}", f"{r['V_nL']:.6g}", f"{P_c[k]:.5f}", f"{r['P_mmHg']:.6g}"])
    dp = col["P_mmHg"] - P_c
    info = {"device": str(device), "nodes": len(g["points"]), "tetrahedra": len(g["cells"]), "p2_dofs": len(p2["dof_points"]),
            "states": cfg.N_REPLAY, "states_outside_trained_range": int(clipped.sum()),
            "cycle_samples_near_clipped_table_entries": n_bad, "cycle_samples": n_samples,
            "pressure_solve_s": round(t_pressure, 2), "mean_state_s": round(float(col["time_s"].mean()), 2),
            "max_abs_dP_mmHg": float(np.abs(dp).max()), "rms_dP_mmHg": float(np.sqrt((dp ** 2).mean())),
            "max_abs_dV_nL": float(np.abs(col["V_nL"] - V_c).max()),
            "total_s": round(time.time() - t_start, 1), "model": "model.pt",
            "model_train_clock_s": checkpoint.get("train_clock_s"), "date": time.strftime("%Y-%m-%d %H:%M")}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=1))
    print(f"replay: PINN P - circuit P: max |dP| {np.abs(dp).max():.4f} mmHg, rms {np.sqrt((dp ** 2).mean()):.4f}; "
          f"states outside the trained range: {int(clipped.sum())}; cycle samples near clipped table entries: {n_bad} of "
          f"{n_samples}; done in {time.time() - t_start:.0f} s -> {OUT}", flush=True)
