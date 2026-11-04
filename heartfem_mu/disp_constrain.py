"""Displacement constraint of the volume_constrain FE (DISP_CONSTRAIN = True): the wall is pulled toward the tracked wall.

The FE (fe.VolumeProblem) adds the traction w (u_img - u) on the endocardium and epicardium, i.e. the energy
w/2 int |u - u_img|^2 dS, w = DISP_WEIGHT (kPa/mm); everything else is the volume_constrain FE. The traction is a force
that the real heart does not have ("image force"): w balances closeness to the tracking against it, and it is reported.

Steps (CONDA_ENV, numpy + scipy; compare also vtk, matplotlib):
  targets  u_img of every endocardial / epicardial node of mesh/t0.vtu (the reference frame): the node is attached to the
           closest point of the same surface (endo / epi) of the tracked reference frame (input t<ref>.stl), as
           barycentric coordinates in one triangle; all tracked frames share the triangle order, so the point is followed
           through the cycle. Rigid motion of the tracking is removed per frame (least-squares rotation + translation of
           all tracked vertices onto the reference frame; the model has no rigid motion). u_img(ref) = 0.
           -> OUTPUT_DIR/disp_constrain/targets.npz
  compare  every frame, simulated wall (mesh + u) vs tracked wall (rigidly aligned as above), on the cells / points at
           least BASE_EXCLUDE_UM below the base plane of the reference frame:
           wall Dice on voxels of DICE_VOXEL_UM (tracked wall: inside the epicardium and outside the endocardium of the
           closed input surfaces; simulated: inside the closed wall boundary), surface distance (simulated endo / epi
           nodes to the tracked surface of the same side: median, 95 %), misfit to the targets (area-weighted rms over
           the endo + epi nodes: full vector and normal component), image force w |u_img - u| (area-weighted rms).
           -> OUTPUT_DIR/disp_constrain/compare.csv, compare.json, disp_constrain.png
"""
import json
import struct

import numpy as np
from scipy.spatial import cKDTree

from . import model

DICE_VOXEL_UM = 1.0     # voxel edge of the wall Dice (mesh units = um)
NEAR_UM = 12.0          # Dice voxels farther than this from both walls are outside both (skipped for speed)


# ==================== tracked surfaces ====================
def read_stl_soup(path):
    """Triangle vertices of a binary STL in file order, (n_triangles x 3, 3); the order is the tracking's correspondence."""
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    if len(b) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL")
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(np.float64)


def tracked(cfg):
    """Welded vertices of every tracked frame (frames, V, 3) in mesh units, the triangles, the per-triangle surface (0 endo,
    1 epi) and the rigid alignment of every frame onto the reference frame (applied)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ref = model.reference_frame(cfg)
    soup = read_stl_soup(cfg.input / f"t{ref}.stl")
    _, idx, inv = np.unique(np.round(soup, 6), axis=0, return_index=True, return_inverse=True)
    tri = inv.ravel().reshape(-1, 3)
    X = np.empty((cfg.N_FRAMES, len(idx), 3))
    for k in range(cfg.N_FRAMES):
        s = read_stl_soup(cfg.input / f"t{k}.stl")
        if len(s) != len(soup):
            raise ValueError(f"t{k}.stl: the tracked frames must share the triangles of t{ref}.stl")
        X[k] = s[idx]
    e = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]]))
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(idx),) * 2), directed=False)
    if comp.max() != 1:
        raise ValueError("the tracked surface must consist of two closed surfaces (endocardium, epicardium)")
    x = X[ref][tri]
    vol = [abs(np.einsum("ij,ij->i", x[comp[tri[:, 0]] == c, 0],
                         np.cross(x[comp[tri[:, 0]] == c, 1], x[comp[tri[:, 0]] == c, 2])).sum()) for c in (0, 1)]
    side = (comp[tri[:, 0]] != int(np.argmin(vol))).astype(int)  # the smaller volume is the endocardium
    rigid = []
    for k in range(cfg.N_FRAMES):
        Rm, t = rigid_to(X[k], X[ref])
        rigid.append((np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))),
                      np.linalg.norm(X[k].mean(0) - X[ref].mean(0))))  # rotation (deg), centroid shift (mesh units)
        X[k] = X[k] @ Rm.T + t
    return X, tri, side, np.array(rigid)


def rigid_to(Xk, X0):
    """Rotation Rm and translation t with Rm Xk + t ~ X0 (least squares over corresponding vertices)."""
    a, b = Xk.mean(0), X0.mean(0)
    U, _, Vt = np.linalg.svd((Xk - a).T @ (X0 - b))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    return Rm, b - Rm @ a


def closest_on_triangles(P, A, B, C):
    """Barycentric coordinates of the closest point to P on the triangles ABC, and the distance (Ericson 5.1.5)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp, cp = P - B, P - C
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        bary = np.stack([1 - vb / den - vc / den, vb / den, vc / den], -1)
        v_ab, w_ac, w_bc = d1 / (d1 - d3), d2 / (d2 - d6), (d4 - d3) / ((d4 - d3) + (d5 - d6))
    for mask, val in (((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), np.stack([0 * w_bc, 1 - w_bc, w_bc], -1)),
                      ((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - w_ac, 0 * w_ac, w_ac], -1)),
                      ((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - v_ab, v_ab, 0 * v_ab], -1)),
                      ((d6 >= 0) & (d5 <= d6), np.array([0.0, 0, 1])),
                      ((d3 >= 0) & (d4 <= d3), np.array([0.0, 1, 0])),
                      ((d1 <= 0) & (d2 <= 0), np.array([1.0, 0, 0]))):
        bary = np.where(mask[..., None], val, bary)
    Q = bary[..., :1] * A + bary[..., 1:2] * B + bary[..., 2:] * C
    return bary, np.linalg.norm(P - Q, axis=-1)


# ==================== step targets ====================
def targets(cfg):
    """Step targets: u_img of every endo / epi node -> OUTPUT_DIR/disp_constrain/targets.npz."""
    w = model.read_wall(cfg)
    pts = w["points"] / cfg.MM_PER_UNIT
    X, tri, side, rigid = tracked(cfg)
    ref = model.reference_frame(cfg)
    u_img = np.zeros((cfg.N_FRAMES, len(pts), 3))
    data = np.zeros(len(pts), bool)
    lines = []
    for s, name in ((0, "endo"), (1, "epi")):
        nodes = np.unique(w[name])
        sel = np.where(side == s)[0]
        cand = sel[cKDTree(X[ref][tri[sel]].mean(1)).query(pts[nodes], 24)[1]]
        T = X[ref][tri[cand]]
        bary, d = closest_on_triangles(pts[nodes][:, None], T[..., 0, :], T[..., 1, :], T[..., 2, :])
        j = np.argmin(d, 1)
        r = np.arange(len(nodes))
        t_, b_ = tri[cand[r, j]], bary[r, j]
        x0 = np.einsum("ni,nij->nj", b_, X[ref][t_])
        for k in range(cfg.N_FRAMES):
            u_img[k, nodes] = (np.einsum("ni,nij->nj", b_, X[k][t_]) - x0) * cfg.MM_PER_UNIT
        data[nodes] = True
        lines.append(f"{name}: {len(nodes)} nodes, distance to the tracked t{ref} surface median {np.median(d[r, j]):.2f}, "
                     f"max {d[r, j].max():.2f} um")
    out = cfg.output / "disp_constrain"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "targets.npz", u_img=u_img, data_nodes=data, rigid_rotation_deg=rigid[:, 0],
                        rigid_shift_um=rigid[:, 1], reference_frame=ref)
    mag = np.linalg.norm(u_img[:, data], axis=2) / cfg.MM_PER_UNIT
    print("\n".join(lines) + f"\nimage displacement: median over the nodes up to {np.median(mag, 1).max():.2f} um; tracking "
          f"rigid motion removed: up to {rigid[:, 0].max():.1f} deg", flush=True)


def load_targets(cfg):
    path = cfg.output / "disp_constrain" / "targets.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the step targets first")
    return np.load(path)["u_img"]


# ==================== step compare ====================
def polydata(x, tri):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    pd, pts, ca = vtk.vtkPolyData(), vtk.vtkPoints(), vtk.vtkCellArray()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(x, dtype=np.float64), deep=True))
    pd.SetPoints(pts)
    ca.SetCells(len(tri), numpy_to_vtkIdTypeArray(np.c_[np.full(len(tri), 3), tri].astype(np.int64).ravel(), deep=True))
    pd.SetPolys(ca)
    return pd


def signed_distance(x, tri, q):
    """Signed distance of the points q to the closed, outward-oriented surface (x, tri): negative inside."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    f = vtk.vtkImplicitPolyDataDistance()
    f.SetInput(polydata(x, tri))
    pts, out = vtk.vtkPoints(), vtk.vtkDoubleArray()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(q, dtype=np.float64), deep=True))
    f.FunctionValue(pts.GetData(), out)
    return vtk_to_numpy(out)


def outward(x, tri):
    v = x[tri]
    return tri if np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() > 0 else tri[:, ::-1]


class Compare:
    def __init__(self, cfg):
        self.cfg = cfg
        w = model.read_wall(cfg)
        self.w, self.pts = w, w["points"] / cfg.MM_PER_UNIT
        self.X, tri, side, _ = tracked(cfg)
        ref = model.reference_frame(cfg)
        self.endo_f, self.epi_f = outward(self.X[ref], tri[side == 0]), outward(self.X[ref], tri[side == 1])
        self.n_out = w["n_out"]
        self.level = float(np.median(self.pts[w["base_nodes"]] @ self.n_out))
        self.nodes = {s: np.unique(w[s]) for s in ("endo", "epi")}

    def below(self, x):
        return self.level - x @ self.n_out

    def frame(self, k, u_mm):
        cfg, x, Xk = self.cfg, self.pts + u_mm / self.cfg.MM_PER_UNIT, self.X[k]
        lo, hi = np.minimum(x.min(0), Xk.min(0)) - 2, np.maximum(x.max(0), Xk.max(0)) + 2
        q = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i], DICE_VOXEL_UM) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        q = q[self.below(q) >= cfg.BASE_EXCLUDE_UM]
        q = q[(cKDTree(x).query(q, distance_upper_bound=NEAR_UM)[0] < np.inf) |
              (cKDTree(Xk).query(q, distance_upper_bound=NEAR_UM)[0] < np.inf)]
        sim = signed_distance(x, self.w["bnd"], q) < 0
        trk = (signed_distance(Xk, self.epi_f, q) < 0) & (signed_distance(Xk, self.endo_f, q) > 0)
        d = []
        for s, f in (("endo", self.endo_f), ("epi", self.epi_f)):
            n = self.nodes[s][self.below(self.pts[self.nodes[s]]) >= cfg.BASE_EXCLUDE_UM]
            d.append(model.surface_distance(Xk, f, x[n]))
        d = np.concatenate(d)
        return {"frame": int(k), "wall_dice": float(2 * (sim & trk).sum() / (sim.sum() + trk.sum())),
                "distance_median_um": float(np.median(d)), "distance_p95_um": float(np.percentile(d, 95))}


_C = None


def _frame(args):
    global _C
    if _C is None:
        _C = Compare(args[0])
    return _C.frame(args[1], args[2])


def compare(cfg):
    """Step compare: simulation vs tracking, every frame -> OUTPUT_DIR/disp_constrain/compare.csv, compare.json, .png."""
    import csv
    import multiprocessing as mp
    r = np.load(cfg.output / "pv_results.npz")
    u, frames = r["u"].astype(np.float64), [int(k) for k in r["frame"]]
    with mp.Pool(max(1, int(cfg.MPI_RANKS))) as pool:
        rows = pool.map(_frame, [(cfg, k, u[i]) for i, k in enumerate(frames)])
    # misfit to the targets and the image force, area-weighted over the endo + epi nodes
    w = model.read_wall(cfg)
    pts = w["points"]
    area, nrm = np.zeros(len(pts)), np.zeros((len(pts), 3))
    for f in (w["endo"], w["epi"]):
        cr = np.cross(pts[f[:, 1]] - pts[f[:, 0]], pts[f[:, 2]] - pts[f[:, 0]])
        for j in range(3):
            np.add.at(area, f[:, j], np.linalg.norm(cr, axis=1) / 6)
            np.add.at(nrm, f[:, j], cr)
    data = area > 0
    a = area[data] / area[data].sum()
    n = nrm[data] / np.linalg.norm(nrm[data], axis=1, keepdims=True)
    weight = float(cfg.DISP_WEIGHT)
    U = load_targets(cfg)
    for i, row in enumerate(rows):
        k = row["frame"]
        d = U[k][data] - u[i][data]  # mm
        row.update(t_ms=float(r["t_ms"][i]), P_kPa=float(r["P_kPa"][i]),
                   misfit_rms_um=float(np.sqrt((a * (d ** 2).sum(1)).sum())) / cfg.MM_PER_UNIT,
                   misfit_normal_rms_um=float(np.sqrt((a * np.einsum("nj,nj->n", d, n) ** 2).sum())) / cfg.MM_PER_UNIT,
                   image_force_rms_kPa=weight * float(np.sqrt((a * (d ** 2).sum(1)).sum())))
    out = cfg.output / "disp_constrain"
    keys = ("frame", "t_ms", "P_kPa", "wall_dice", "distance_median_um", "distance_p95_um", "misfit_rms_um",
            "misfit_normal_rms_um", "image_force_rms_kPa")
    with open(out / "compare.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(keys)
        wr.writerows([[f"{row[c]:.6g}" if isinstance(row[c], float) else row[c] for c in keys] for row in rows])
    col = {c: np.array([row[c] for row in rows]) for c in keys}
    summary = {"weight_kPa_per_mm": weight, "frames": len(rows),
               "wall_dice_mean": float(col["wall_dice"].mean()), "wall_dice_min": float(col["wall_dice"].min()),
               "wall_dice_min_frame": int(col["frame"][np.argmin(col["wall_dice"])]),
               "distance_median_um_mean": float(col["distance_median_um"].mean()),
               "misfit_rms_um": float(np.sqrt((col["misfit_rms_um"] ** 2).mean())),
               "misfit_normal_rms_um": float(np.sqrt((col["misfit_normal_rms_um"] ** 2).mean())),
               "image_force_rms_kPa": float(np.sqrt((col["image_force_rms_kPa"] ** 2).mean())),
               "P_max_kPa": float(col["P_kPa"].max())}
    summary["image_force_over_P_max"] = summary["image_force_rms_kPa"] / summary["P_max_kPa"]
    (out / "compare.json").write_text(json.dumps(summary, indent=1))
    from . import plots
    plots.disp_constrain_figure(cfg, col)
    print(json.dumps(summary, indent=1), flush=True)
