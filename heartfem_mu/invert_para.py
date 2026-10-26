"""Task invert_para: from the tracked wall of two frames of one cycle (ED: passive, maximum volume; PS: contracting) and
the cavity pressure GIVEN at both frames, find the passive scale C and the peak active tension T_MAX. The fiber
architecture is predefined (FIBER_ANGLE_DEG, or two families, TWO_FIBERS); every other constant is fixed.

Misfit (scale-free: lengths are fractions of L = V0^(1/3), V0 the cavity volume of the model at u = 0), per frame:
- rigid motion of the tracking is removed: the tracked frame is aligned to the tracked reference frame by the best
  rigid fit over all corresponding points (the model has no rigid motion);
- surface distance, endo and epi separately: symmetric point-to-plane distance (simulated vertices to the tracked
  surface and tracked points to the simulated surface: nearest point, then the distance along its normal),
  area-weighted rms over the points farther than BASE_EXCLUDE_REL x L from the base plane (reference position);
- cavity volume: image volume V_k = V0 x V_img,k / V_img,ref;
J = sum over ED, PS of ((V - V_img) / (SIGMA_V V_img))^2 + (d_endo / sigma_s)^2 + W_EPI (d_epi / sigma_s)^2,
sigma_s = SIGMA_S_REL x L.
Search (step invert):
  geometry start  no guess: the shape depends only on P/C and T_MAX/C, so C0 makes the passive wall reach the imaged
        ED volume at P_ED (secant on ln C), and T0 makes the wall at (P_PS, t_PS) reach the imaged PS volume with C0
        (secant on ln T_MAX);
  for every start (C0, T0) x START_FACTORS:
  A     ED frame only (passive: T_MAX has no effect), free C;
  B     PS frame only, free T_MAX (C from A).
Each stage is a Nelder-Mead search on ln C or ln T_MAX of J. Every start is then scored by J over both frames; the
lowest gives the result. FE: fe.PressureProblem, MPI (rank 0 drives the search, every solve is collective). PINN: the
parametric network (pinn.ParamPINN); the result must lie inside its training domain (checked and reported).
Output in OUTPUT_DIR: result.json, ED.vtu, PS.vtu; step evaluate: evaluation.json, comparison.png and the distance to
the tracking in ED.vtu, PS.vtu.
"""
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from . import model

DICE_VOXEL = 1.0 / 87.8  # voxel edge of the Dice coefficient as a fraction of L (1 um for this atrium)


def frames(cfg):
    return (cfg.ED_FRAME, cfg.PS_FRAME)


def frame_names(cfg):
    return {cfg.ED_FRAME: "ED", cfg.PS_FRAME: "PS"}


def ct(cfg, k):
    """Activation level Ct of frame k: 0 at ED (passive); at PS 1/2 (1 - cos(pi t / T0_MS)), t = PS_ACTIVATION_MS, on the
    rise, where Ct does not depend on the sarcomere length (the same everywhere)."""
    if k == cfg.ED_FRAME:
        return 0.0
    if not 0.0 <= cfg.PS_ACTIVATION_MS < cfg.T0_MS:
        raise ValueError("PS_ACTIVATION_MS must lie on the rise of Ct: 0 <= PS_ACTIVATION_MS < T0_MS")
    return float(0.5 * (1 - np.cos(np.pi * cfg.PS_ACTIVATION_MS / cfg.T0_MS)))


# ==================== 1. image misfit (numpy + scipy) ====================
def read_stl(path):
    import meshio
    m = meshio.read(path)
    return m.points.astype(np.float64), m.cells_dict["triangle"].astype(np.int64)


def vertex_normals(x, tri):
    """Area-weighted vertex normals and vertex areas (one third of the adjacent triangle areas)."""
    cr = np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]])
    n, a = np.zeros_like(x), np.zeros(len(x))
    for j in range(3):
        np.add.at(n, tri[:, j], cr)
        np.add.at(a, tri[:, j], np.linalg.norm(cr, axis=1) / 6)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30), a


def kabsch(a, b):
    """R, t with R a + t ~ b (least squares over corresponding rows)."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def mesh_surfaces(p, cells):
    """Boundary triangles of the tet mesh, oriented out of the wall: endo, epi, base, and the base normal (into the wall)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    faces = np.concatenate((cells[:, [0, 1, 2]], cells[:, [0, 1, 3]], cells[:, [0, 2, 3]], cells[:, [1, 2, 3]]))
    opposite = np.concatenate((cells[:, 3], cells[:, 2], cells[:, 1], cells[:, 0]))
    _, first, count = np.unique(np.sort(faces, 1), axis=0, return_index=True, return_counts=True)
    bnd, opp = faces[first[count == 1]], opposite[first[count == 1]]
    nrm = np.cross(p[bnd[:, 1]] - p[bnd[:, 0]], p[bnd[:, 2]] - p[bnd[:, 0]])
    inward = np.sum(nrm * (p[opp] - p[bnd[:, 0]]), 1) > 0
    bnd[inward] = bnd[inward][:, [0, 2, 1]]
    nrm[inward] *= -1
    unit = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    _, group, size = np.unique(np.round(unit, 3), axis=0, return_inverse=True, return_counts=True)
    seed = group.ravel() == np.argmax(size)
    n_out = unit[seed].mean(0)
    n_out /= np.linalg.norm(n_out)
    offset = np.median(p[bnd[seed]] @ n_out)
    on_base = (unit @ n_out > 0.9) & (np.abs(p[bnd] @ n_out - offset).max(1) < 1e-5 * np.ptp(p, 0).max())
    side = bnd[~on_base]
    edges = np.concatenate((side[:, [0, 1]], side[:, [1, 2]], side[:, [2, 0]]))
    _, comp = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(p),) * 2),
                                   directed=False)
    parts = [side[comp[side[:, 0]] == c] for c in np.unique(comp[side[:, 0]])]
    area = [np.linalg.norm(np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]), axis=1).sum() for f in parts]
    endo, epi = (parts[0], parts[1]) if area[0] < area[1] else (parts[1], parts[0])
    return {"endo": endo, "epi": epi, "base": bnd[on_base], "base_normal": -n_out}


class Tracking:
    """Tracked surfaces (mesh/surface_cut, rigid motion removed), the simulated surfaces of the mesh mesh/t0.vtu (moved by
    the vertex displacement, mesh units), and the misfit."""

    def __init__(self, cfg):
        import meshio
        self.cfg = cfg
        MESH = cfg.mesh
        corr = np.load(MESH / "surface_cut" / "correspondence.npz")
        self.frames = [int(f) for f in corr["frame_ids"]]
        assert int(corr["reference_frame"]) == cfg.REF_FRAME, "REF_FRAME differs from the saved landmarks"
        lab = corr["surface_labels"]
        self.plane_origin = corr["plane_origin"]
        x0, tri = read_stl(MESH / "surface_cut" / f"t{cfg.REF_FRAME}.stl")
        self.tri = {"endo": tri[lab == 1], "epi": tri[lab == 2]}
        self.track, self.rigid, vols = {}, {}, {}
        for k in self.frames:
            x, t2 = read_stl(MESH / "surface_cut" / f"t{k}.stl")
            assert np.array_equal(t2, tri)
            v = x[self.tri["endo"]] - self.plane_origin
            vols[k] = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6
            R, t = kabsch(x, x0)
            self.rigid[k] = (R, t)
            self.track[k] = x @ R.T + t
        self.V_ratio = {k: vols[k] / vols[cfg.REF_FRAME] for k in self.frames}
        vtu = meshio.read(MESH / "t0.vtu")
        self.X = vtu.points.astype(np.float64)
        self.cells = vtu.cells_dict["tetra"].astype(np.int64)
        s = mesh_surfaces(self.X, self.cells)
        self.sim_tri = {"endo": s["endo"], "epi": s["epi"]}
        self.base_tri, self.base_normal = s["base"], s["base_normal"]
        self.L = float(self.mesh_cavity_volume(np.zeros_like(self.X)) ** (1 / 3))
        n_out = -self.base_normal
        level = float((self.X @ n_out).max())
        exclude = cfg.BASE_EXCLUDE_REL * self.L
        self.sim_keep, self.trk_keep = {}, {}
        for side in ("endo", "epi"):
            vs = np.unique(self.sim_tri[side])
            self.sim_keep[side] = vs[level - self.X[vs] @ n_out >= exclude]
            vt = np.unique(self.tri[side])
            self.trk_keep[side] = vt[level - x0[vt] @ n_out >= exclude]

    def mesh_cavity_volume(self, u):
        """Cavity volume (mesh units^3) of the moved P1 endocardium closed by cones to the base-plane origin."""
        v = (self.X + u)[self.sim_tri["endo"]] - self.plane_origin
        return -np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6

    def surface_errors(self, k, u):
        """{endo, epi: (rms, signed mean, max)} in mesh units for frame k and the vertex displacement u (n, 3)."""
        x, y = self.X + u, self.track[k]
        out = {}
        for side in ("endo", "epi"):
            ts, tt = self.sim_tri[side], self.tri[side]
            ns, as_ = vertex_normals(x, ts)
            nt, at = vertex_normals(y, tt)
            vs, vt = self.sim_keep[side], self.trk_keep[side]
            ids_t, ids_s = np.unique(tt), np.unique(ts)
            j = ids_t[cKDTree(y[ids_t]).query(x[vs])[1]]
            d1 = np.einsum("ij,ij->i", x[vs] - y[j], nt[j])
            i = ids_s[cKDTree(x[ids_s]).query(y[vt])[1]]
            d2 = -np.einsum("ij,ij->i", y[vt] - x[i], ns[i])
            w1, w2 = as_[vs], at[vt]
            rms = np.sqrt(0.5 * ((w1 * d1 ** 2).sum() / w1.sum() + (w2 * d2 ** 2).sum() / w2.sum()))
            mean = 0.5 * ((w1 * d1).sum() / w1.sum() + (w2 * d2).sum() / w2.sum())
            out[side] = (float(rms), float(mean), float(max(np.abs(d1).max(), np.abs(d2).max())))
        return out

    def vertex_distance(self, k, u):
        """Signed distance (mesh units) of every simulated surface vertex (endo and epi) to the tracked surface of the
        same side at frame k, along the tracked normal: > 0 where the model lies farther from the cavity than the
        tracking; NaN for interior vertices."""
        x, y = self.X + u, self.track[k]
        out = np.full(len(x), np.nan)
        for side in ("endo", "epi"):
            nt, _ = vertex_normals(y, self.tri[side])
            vs, ids_t = np.unique(self.sim_tri[side]), np.unique(self.tri[side])
            j = ids_t[cKDTree(y[ids_t]).query(x[vs])[1]]
            out[vs] = np.einsum("ij,ij->i", x[vs] - y[j], nt[j])
        return out

    def image_volume(self, k, V0):
        return V0 * self.V_ratio[k]


def objective(cfg, trk, frames, u_list, V_list, V0):
    """Misfit J (params: SIGMA_V, SIGMA_S_REL, W_EPI) and its parts; u in mesh units, V and V0 in the same unit."""
    sigma_s = cfg.SIGMA_S_REL * trk.L
    J, parts = 0.0, []
    for k, u, V in zip(frames, u_list, V_list):
        e = trk.surface_errors(k, u)
        Vi = trk.image_volume(k, V0)
        dv = (V - Vi) / Vi
        j = (dv / cfg.SIGMA_V) ** 2 + (e["endo"][0] / sigma_s) ** 2 + cfg.W_EPI * (e["epi"][0] / sigma_s) ** 2
        J += j
        parts.append({"frame": k, "J": j, "dV_rel": dv, "V": V, "V_image": Vi, "endo": e["endo"], "epi": e["epi"]})
    return J, parts


# ==================== 2. result meshes OUTPUT_DIR/ED.vtu, PS.vtu ====================
# Each file is the wall mesh (linear tetrahedra, the vertices of mesh/t0.vtu) deformed by the inverted model at that
# frame, in the coordinates of the input file of the frame: the model has no rigid motion, so it is placed with the rigid
# fit that aligns the tracked frame to the reference frame, inverted. Both files open directly over the input surfaces.
#     point data  displacement_um            model displacement from the reference state (same coordinates)
#                 distance_to_tracking_um    (evaluate) signed distance of each surface vertex to the tracked surface,
#                                            > 0 where the model lies farther from the cavity; NaN inside the wall
#     cell data   fiber_1 (, fiber_2)        unit fiber direction of each family
def write_frame(cfg, trk, k, u_um, depth):
    """OUTPUT_DIR/<ED|PS>.vtu for frame k and the vertex displacement u_um (n, 3) in the reference coordinates; depth: the
    transmural depth per cell for the fiber directions."""
    import meshio
    vtu = meshio.read(cfg.mesh / "t0.vtu")
    R, t = trk.rigid[k]  # tracked frame k -> reference: x R^T + t; back: (y - t) R
    cell = {}
    for i, (a, _) in enumerate(model.fiber_angles(cfg, depth), 1):
        a = a[:, None]
        f = np.cos(a) * vtu.cell_data_dict["eC"]["tetra"] + np.sin(a) * vtu.cell_data_dict["eL"]["tetra"]
        cell[f"fiber_{i}"] = [(f / np.linalg.norm(f, axis=1, keepdims=True)) @ R]
    m = meshio.Mesh((trk.X + u_um - t) @ R, [("tetra", trk.cells)], point_data={"displacement_um": u_um @ R}, cell_data=cell)
    cfg.output.mkdir(parents=True, exist_ok=True)
    m.write(cfg.output / f"{frame_names(cfg)[k]}.vtu")


def read_frame(cfg, trk, k):
    """(mesh, vertex displacement in the reference coordinates) of OUTPUT_DIR/<ED|PS>.vtu."""
    import meshio
    m = meshio.read(cfg.output / f"{frame_names(cfg)[k]}.vtu")
    R, _ = trk.rigid[k]
    return m, m.point_data["displacement_um"] @ R.T


# ==================== 3. inversion ====================
class Inversion:
    """Search on (C, T_MAX); a subclass provides solve(theta, frames) -> (vertex displacements um, cavity volumes nL)."""
    SECANT_STEPS = 10

    def misfit(self, theta, frames):
        us, Vs = self.solve(theta, frames)
        self.count += 1
        if any(np.isnan(Vs)):
            return np.inf, []
        return objective(self.cfg, self.trk, frames, us, Vs, self.V0)

    def geometry_start(self):
        cfg = self.cfg
        V_ed, V_ps = self.trk.image_volume(cfg.ED_FRAME, self.V0), self.trk.image_volume(cfg.PS_FRAME, self.V0)

        def secant(fun, x0, x1, target, n=self.SECANT_STEPS, tol=5e-4):
            y0, y1 = fun(x0), fun(x1)
            for _ in range(n):
                if abs(y1 - target) <= tol * abs(target) or not np.isfinite(y1) or y1 == y0:
                    break
                x2 = x1 + (target - y1) * (x1 - x0) / (y1 - y0)
                x0, y0, x1, y1 = x1, y1, x2, fun(x2)
            return x1
        C0 = float(np.exp(secant(lambda z: self.solve((np.exp(z), 1.0), [cfg.ED_FRAME])[1][0],
                                 np.log(0.05), np.log(0.1), V_ed)))
        T0 = float(np.exp(secant(lambda z: self.solve((C0, np.exp(z)), [cfg.PS_FRAME])[1][0],
                                 np.log(20 * C0), np.log(60 * C0), V_ps)))
        return [C0, T0]

    def nelder_mead(self, theta, frame, i):
        """Search on ln theta[i] alone with the frame `frame`."""
        from scipy.optimize import minimize
        cfg = self.cfg
        theta = list(theta)

        def with_z(z):
            th = list(theta)
            th[i] = float(np.exp(z[0]))
            return th
        z0 = np.log(theta[i])
        r = minimize(lambda z: self.misfit(with_z(z), [frame])[0], [z0], method="Nelder-Mead",
                     options={"initial_simplex": np.array([[z0], [z0 + cfg.STEP]]), "xatol": cfg.XATOL,
                              "fatol": cfg.FATOL, "maxfev": cfg.MAXFEV})
        return with_z(r.x), {"J_frame": float(r.fun), "evaluations": int(r.nfev), "converged": bool(r.success)}

    def run_start(self, geo, factors):
        cfg = self.cfg
        t0 = time.time()
        theta = [geo[0] * factors[0], geo[1] * factors[1]]
        res = {"start": {"C": theta[0], "T_MAX": theta[1]}}
        theta, res["stage_A"] = self.nelder_mead(theta, cfg.ED_FRAME, 0)
        theta, res["stage_B"] = self.nelder_mead(theta, cfg.PS_FRAME, 1)
        res["J"] = float(self.misfit(theta, list(frames(cfg)))[0])
        res["C"], res["T_MAX"] = theta
        res["converged"] = res["stage_A"]["converged"] and res["stage_B"]["converged"]
        res["seconds"] = round(time.time() - t0, 1)
        print(f"start {factors}: C {theta[0]:.5f} T_MAX {theta[1]:.4f} J {res['J']:.5g} converged {res['converged']} "
              f"({res['seconds']:.1f} s)", flush=True)
        return res

    def search(self):
        """Geometry start, every start of START_FACTORS, the best start; the final state at the result."""
        cfg = self.cfg
        geo = self.geometry_start()
        print(f"geometry start: C0 {geo[0]:.5f} T0 {geo[1]:.4f}", flush=True)
        starts = [dict(factors=list(f), **self.run_start(geo, f)) for f in cfg.START_FACTORS]
        best = min(starts, key=lambda s: s["J"])
        theta = [best["C"], best["T_MAX"]]
        us, Vs = self.solve(theta, frames(cfg))  # final state at the result
        return geo, starts, theta, us, Vs

    def result(self, geo, starts, theta, us, Vs, spring, extra, solver, t0):
        """result.json content (and the frames' result meshes are written by the caller)."""
        cfg = self.cfg
        J, parts = objective(cfg, self.trk, list(frames(cfg)), us, Vs, self.V0)
        ct_ps = ct(cfg, cfg.PS_FRAME)
        res = {
            "C_kPa": theta[0], "T_MAX_kPa": theta[1], "active_tension_at_PS_kPa": theta[1] * ct_ps,
            "fibers_angle_deg_share": model.fiber_shares(cfg), **spring, "J": J,
            "all_starts_converged": all(s["converged"] for s in starts),
            "spread_over_starts": {"C": [min(s["C"] for s in starts), max(s["C"] for s in starts)],
                                   "T_MAX": [min(s["T_MAX"] for s in starts), max(s["T_MAX"] for s in starts)]},
            "geometry_start": {"C": geo[0], "T_MAX": geo[1]}, "starts": starts,
            "frames": {name: {"frame": k, "Ct": ct(cfg, k), "P_kPa": self.P[k],
                              "cavity_volume_nL": p["V"], "image_volume_nL": p["V_image"], "J": p["J"]}
                       for name, k, p in zip(("ED", "PS"), frames(cfg), parts)},
            "reference_cavity_volume_nL": self.V0, **extra, "solver": solver, "evaluations": self.count,
            "seconds": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d %H:%M")}
        return res, J


class FEInversion(Inversion):
    """Rank 0 drives the search; every solve is collective (all ranks receive the same command)."""

    def __init__(self, cfg):
        from . import fe
        self.cfg, self.fe = cfg, fe
        self.P = {cfg.ED_FRAME: cfg.P_ED_KPA, cfg.PS_FRAME: cfg.P_PS_KPA}
        self.fs = fe.FrameSolver(cfg, frames(cfg), self.P, {k: ct(cfg, k) for k in frames(cfg)}, cfg.cache)
        self.P = self.fs.P
        self.V0 = self.fs.pb.V0 * 1e3  # nL
        if fe.RANK == 0:
            self.trk = Tracking(cfg)
        self.count = 0

    def solve(self, theta, frames):
        if self.fe.RANK == 0:
            self.fe.COMM.bcast(("solve", (list(theta), list(frames))), root=0)
        res = [self.fs.solve(k, theta) for k in frames]  # (u um, V nL, Newton its, seconds) per frame
        return [r[0] for r in res], [r[1] for r in res]

    def worker(self):
        while True:
            cmd, payload = self.fe.COMM.bcast(None, root=0)
            if cmd == "stop":
                return
            self.solve(*payload)


class PINNInversion(Inversion):
    SECANT_STEPS = 12

    def __init__(self, cfg):
        from . import pinn
        self.cfg, self.pinn = cfg, pinn
        self.device = pinn.device_of()
        self.net, g, _, self.geo = pinn.load_param_model(cfg, self.device)
        self.nv = len(g["points"])
        self.depth = g["depth"]  # transmural depth per cell (fiber directions of the output files)
        self.V0 = float(self.geo.reference_volume) * 1e3  # nL
        self.trk = Tracking(cfg)
        self.P = {cfg.ED_FRAME: cfg.P_ED_KPA, cfg.PS_FRAME: cfg.P_PS_KPA}
        self.ct = {k: ct(cfg, k) for k in frames(cfg)}
        self.count = 0

    def solve(self, theta, frames):
        res = [self.pinn.predict(self.net, self.geo, theta, self.P[k], self.ct[k], self.nv) for k in frames]
        return [r[0] for r in res], [r[1] for r in res]


def fe_inversion(cfg):
    """FE step invert (MPI): OUTPUT_DIR/result.json, ED.vtu, PS.vtu."""
    t0 = time.time()
    cfg.cache.mkdir(parents=True, exist_ok=True)
    inv = FEInversion(cfg)
    fe = inv.fe
    fam = model.fiber_shares(cfg)
    fe.log(f"{fe.COMM.size} ranks, {inv.fs.pb.W.dim()} unknowns, V0 {inv.V0:.5f} nL; ED t{cfg.ED_FRAME} (P {cfg.P_ED_KPA} kPa), "
           f"PS t{cfg.PS_FRAME} (P {cfg.P_PS_KPA} kPa); fibers ((endo, epi) angle deg, share) {fam}")
    if fe.RANK != 0:
        inv.worker()  # follows every collective solve of rank 0 until "stop"
        return
    geo, starts, theta, us, Vs = inv.search()
    fe.COMM.bcast(("stop", None), root=0)
    result, J = inv.result(geo, starts, theta, us, Vs, {"K_spring_kPa_per_mm": cfg.K_SPRING}, {},
                           f"FE, {fe.COMM.size} MPI ranks", t0)
    for k, u in zip(frames(cfg), us):
        write_frame(cfg, inv.trk, k, u, inv.fs.g["depth"])
    (cfg.output / "result.json").write_text(json.dumps(result, indent=1, default=float))
    inv.fs.mesh_h5.unlink(missing_ok=True)
    ct_ps = ct(cfg, cfg.PS_FRAME)
    fe.log(f"result: C {theta[0]:.5f} kPa, T_MAX {theta[1]:.4f} kPa (T_MAX Ct at PS {theta[1] * ct_ps:.3f} kPa), J {J:.4g}; "
           f"{inv.count} FE evaluations, {result['seconds']:.0f} s -> {cfg.output}")


def pinn_inversion(cfg):
    """PINN step invert: OUTPUT_DIR/result.json, ED.vtu, PS.vtu."""
    t0 = time.time()
    inv = PINNInversion(cfg)
    fam = model.fiber_shares(cfg)
    print(f"device {inv.device}; V0 {inv.V0:.5f} nL; ED t{cfg.ED_FRAME} (P {cfg.P_ED_KPA} kPa), PS t{cfg.PS_FRAME} "
          f"(P {cfg.P_PS_KPA} kPa); fibers ((endo, epi) angle deg, share) {fam}", flush=True)
    geo, starts, theta, us, Vs = inv.search()
    ct_ps = inv.ct[cfg.PS_FRAME]
    inputs = {"p_ED": cfg.P_ED_KPA / theta[0], "p_PS": cfg.P_PS_KPA / theta[0], "a_PS": theta[1] * ct_ps / theta[0]}
    inside = max(inputs["p_ED"], inputs["p_PS"]) <= cfg.P_MAX_N and inputs["a_PS"] <= cfg.A_MAX_N
    result, J = inv.result(geo, starts, theta, us, Vs, {"K_spring_over_C_per_mm": cfg.K_SPRING_OVER_C},
                           {"network_inputs_at_result": inputs, "training_domain": {"p_max": cfg.P_MAX_N, "a_max": cfg.A_MAX_N},
                            "inside_training_domain": bool(inside)}, f"PINN ({inv.device})", t0)
    for k, u in zip(frames(cfg), us):
        write_frame(cfg, inv.trk, k, u, inv.depth)
    cfg.output.mkdir(parents=True, exist_ok=True)
    (cfg.output / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"result: C {theta[0]:.5f} kPa, T_MAX {theta[1]:.4f} kPa (T_MAX Ct at PS {theta[1] * ct_ps:.3f} kPa), J {J:.4g}; "
          f"inside the training domain: {inside}; {inv.count} evaluations, {result['seconds']:.0f} s -> {cfg.output}", flush=True)
    if not inside:
        print("WARNING: the result lies outside the training domain of the network: widen P_MAX_N / A_MAX_N and retrain",
              flush=True)


# ==================== 4. evaluation ====================
def polydata(x, tri):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    pd, pts, ca = vtk.vtkPolyData(), vtk.vtkPoints(), vtk.vtkCellArray()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(x, dtype=np.float64), deep=True))
    pd.SetPoints(pts)
    ca.SetCells(len(tri), numpy_to_vtkIdTypeArray(np.c_[np.full(len(tri), 3), tri].astype(np.int64).ravel(), deep=True))
    pd.SetPolys(ca)
    return pd


def inside(x, tri, q):
    """Mask of the points q inside the closed triangle surface (x, tri)."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    pts, qp = vtk.vtkPoints(), vtk.vtkPolyData()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(q, dtype=np.float64), deep=True))
    qp.SetPoints(pts)
    sel = vtk.vtkSelectEnclosedPoints()
    sel.SetInputData(qp)
    sel.SetSurfaceData(polydata(x, tri))
    sel.SetTolerance(1e-6)
    sel.Update()
    return vtk_to_numpy(sel.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)


def close_by_fan(x, tri):
    """Close an open surface with one boundary loop by a fan from the loop centroid (orientation kept)."""
    d = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    u, c = np.unique(np.sort(d, 1), axis=0, return_counts=True)
    rim = {tuple(r) for r in u[c == 1]}
    d = d[np.array([tuple(r) in rim for r in np.sort(d, 1)])]
    centre = len(x)
    return np.vstack((x, x[np.unique(d)].mean(0))), np.vstack((tri, np.c_[d[:, 1], d[:, 0], np.full(len(d), centre)]))


def closed_shells(path, R, t):
    """(endo, epi) closed surfaces of a raw input frame (welded), rigidly moved by (R, t)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    x, tri = read_stl(path)
    xu, inv = np.unique(np.round(x, 6), axis=0, return_inverse=True)
    tri = inv.ravel()[tri]
    x = xu @ R.T + t
    e = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(x),) * 2), directed=False)
    parts = [tri[comp[tri[:, 0]] == c] for c in np.unique(comp[tri[:, 0]])]
    assert len(parts) == 2, "expected two closed surfaces"
    vol = [abs(np.einsum("ij,ij->i", x[p[:, 0]], np.cross(x[p[:, 1]], x[p[:, 2]])).sum()) for p in parts]
    endo, epi = (parts[0], parts[1]) if vol[0] < vol[1] else (parts[1], parts[0])
    return (x, endo), (x, epi)


def dice(cfg, trk, k, u):
    """Dice similarity coefficients on voxels of DICE_VOXEL x L: wall (simulated: inside the deformed mesh boundary;
    tracked: inside the epicardial and outside the endocardial closed input surface, on the wall side of the plane fitted
    to the tracked cut rim) and cavity (simulated: deformed endocardium closed by a fan over its base rim; tracked: inside
    the endocardial surface on the wall side of the same plane); both rigidly aligned as the misfit."""
    (xe, te), (xp, tp) = closed_shells(cfg.input / f"t{k}.stl", *trk.rigid[k])
    y = trk.track[k]
    tri = np.vstack((trk.tri["endo"], trk.tri["epi"]))
    e = np.sort(np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), 1)
    ue, ce = np.unique(e, axis=0, return_counts=True)
    rim = y[np.unique(ue[ce == 1])]
    c0 = rim.mean(0)
    nrm = np.linalg.svd(rim - c0)[2][-1]
    nrm *= np.sign(nrm @ trk.base_normal)
    x = trk.X + u
    h = DICE_VOXEL * trk.L
    lo = np.minimum(xp.min(0), x.min(0)) - 2 * h
    hi = np.maximum(xp.max(0), x.max(0)) + 2 * h
    q = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i], h) + h / 2 for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    below = (q - c0) @ nrm >= 0
    in_endo = inside(xe, te, q)
    T_wall, T_cav = inside(xp, tp, q) & ~in_endo & below, in_endo & below
    S_wall = inside(x, np.vstack((trk.sim_tri["endo"], trk.sim_tri["epi"], trk.base_tri)), q)
    S_cav = inside(*close_by_fan(x, trk.sim_tri["endo"]), q)
    d = lambda A, B: float(2 * (A & B).sum() / (A.sum() + B.sum()))  # noqa: E731
    return {"wall_DSC": d(S_wall, T_wall), "cavity_DSC": d(S_cav, T_cav),
            "wall_volume_ratio_model_to_tracking": float(S_wall.sum() / T_wall.sum())}


def evaluate(cfg):
    """Step evaluate: comparison of the inverted model with the tracking at ED and PS -> OUTPUT_DIR/evaluation.json
    (surface distances, cavity volume against the image volume, Dice coefficients), comparison.png, and the distance to
    the tracking in ED.vtu, PS.vtu."""
    from . import plots
    OUT = cfg.output
    trk = Tracking(cfg)
    result = json.loads((OUT / "result.json").read_text())
    frames_ = {name: result["frames"][name]["frame"] for name in ("ED", "PS")}
    meshes, us = {}, {}
    for k in frames_.values():
        meshes[k], us[k] = read_frame(cfg, trk, k)
    V0 = result["reference_cavity_volume_nL"]
    ks = list(frames_.values())
    J, parts = objective(cfg, trk, ks, [us[k] for k in ks], [result["frames"][n]["cavity_volume_nL"] for n in frames_], V0)
    ev = {"C_kPa": result["C_kPa"], "T_MAX_kPa": result["T_MAX_kPa"], "J": J, "L_um": trk.L,
          "sigma_s_um": cfg.SIGMA_S_REL * trk.L, "frames": {}}
    lines = [f"C {result['C_kPa']:.5f} kPa, T_MAX {result['T_MAX_kPa']:.4f} kPa; misfit J {J:.4g}"]
    for name, p in zip(frames_, parts):
        k = p["frame"]
        dsc = dice(cfg, trk, k, us[k])
        ev["frames"][name] = {"frame": k, "endo_rms_um": p["endo"][0], "endo_mean_um": p["endo"][1], "endo_max_um": p["endo"][2],
                              "epi_rms_um": p["epi"][0], "epi_mean_um": p["epi"][1], "epi_max_um": p["epi"][2],
                              "cavity_volume_nL": p["V"], "image_volume_nL": p["V_image"], "volume_error_rel": p["dV_rel"], **dsc}
        lines.append(f"{name} t{k}: endo {p['endo'][0]:.2f} um (mean {p['endo'][1]:+.2f}, max {p['endo'][2]:.1f}), "
                     f"epi {p['epi'][0]:.2f} um (mean {p['epi'][1]:+.2f}, max {p['epi'][2]:.1f}); cavity volume {p['V']:.5f} "
                     f"vs image {p['V_image']:.5f} nL ({100 * p['dV_rel']:+.2f} %); DSC wall {dsc['wall_DSC']:.3f}, "
                     f"cavity {dsc['cavity_DSC']:.3f}")
        m = meshes[k]
        m.point_data["distance_to_tracking_um"] = trk.vertex_distance(k, us[k])
        m.write(OUT / f"{name}.vtu")
    plots.invert_comparison(cfg, trk, us, result, ev)
    (OUT / "evaluation.json").write_text(json.dumps(ev, indent=1, default=float))
    print("\n".join(lines))
