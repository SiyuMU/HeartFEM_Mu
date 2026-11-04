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
           -> OUTPUT_DIR/mesh/disp_targets.npz (an input of the step solve, next to the other preprocess products)
  compare  the movie OUTPUT_DIR/disp_compare.mp4 (plots.disp_compare_movie): every frame, the simulated wall on the
           long-axis section of pv_cycle.mp4, coloured by sigma_ff, over the tracked wall (rigidly aligned as above,
           blurred grey band, endo / epi outlines), with the wall Dice of the frame: voxels of DICE_VOXEL_UM at least
           BASE_EXCLUDE_UM below the base plane, tracked wall = inside the epicardium and outside the endocardium of the
           closed input surfaces, simulated wall = inside the closed boundary of the deformed mesh.
"""
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
    np.savez_compressed(cfg.mesh / "disp_targets.npz", u_img=u_img, data_nodes=data, rigid_rotation_deg=rigid[:, 0],
                        rigid_shift_um=rigid[:, 1], reference_frame=ref)
    mag = np.linalg.norm(u_img[:, data], axis=2) / cfg.MM_PER_UNIT
    print("\n".join(lines) + f"\nimage displacement: median over the nodes up to {np.median(mag, 1).max():.2f} um; tracking "
          f"rigid motion removed: up to {rigid[:, 0].max():.1f} deg", flush=True)


def load_targets(cfg):
    path = cfg.mesh / "disp_targets.npz"
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


class Tracking:
    """The tracked surfaces of every frame (closed endocardium and epicardium, rigidly aligned as the targets) and the
    wall Dice of a simulated frame."""

    def __init__(self, cfg):
        self.cfg = cfg
        w = model.read_wall(cfg)
        self.bnd, self.pts = w["bnd"], w["points"] / cfg.MM_PER_UNIT
        self.X, tri, side, _ = tracked(cfg)
        ref = model.reference_frame(cfg)
        self.endo_f, self.epi_f = outward(self.X[ref], tri[side == 0]), outward(self.X[ref], tri[side == 1])
        self.n_out = w["n_out"]
        self.level = float(np.median(self.pts[w["base_nodes"]] @ self.n_out))

    def wall(self, k, q):
        """Mask of the points q inside the tracked wall of frame k."""
        Xk = self.X[k]
        return (signed_distance(Xk, self.epi_f, q) < 0) & (signed_distance(Xk, self.endo_f, q) > 0)

    def dice(self, k, u_mm):
        x, Xk = self.pts + u_mm / self.cfg.MM_PER_UNIT, self.X[k]
        lo, hi = np.minimum(x.min(0), Xk.min(0)) - 2, np.maximum(x.max(0), Xk.max(0)) + 2
        q = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i], DICE_VOXEL_UM) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        q = q[self.level - q @ self.n_out >= self.cfg.BASE_EXCLUDE_UM]
        q = q[(cKDTree(x).query(q, distance_upper_bound=NEAR_UM)[0] < np.inf) |
              (cKDTree(Xk).query(q, distance_upper_bound=NEAR_UM)[0] < np.inf)]
        sim, trk = signed_distance(x, self.bnd, q) < 0, self.wall(k, q)
        return float(2 * (sim & trk).sum() / (sim.sum() + trk.sum()))


_T = None


def _dice(args):
    global _T
    if _T is None:
        _T = Tracking(args[0])
    return _T.dice(args[1], args[2])


def compare(cfg):
    """Step compare: wall Dice of every frame and the movie OUTPUT_DIR/disp_compare.mp4."""
    import multiprocessing as mp
    from . import plots
    r = np.load(cfg.output / "pv_results.npz")
    u = r["u"].astype(np.float64)
    with mp.Pool(max(1, int(cfg.MPI_RANKS))) as pool:
        dice = pool.map(_dice, [(cfg, int(k), u[i]) for i, k in enumerate(r["frame"])])
    print(f"wall Dice, simulation vs tracking: mean {np.mean(dice):.3f}, lowest {np.min(dice):.3f} "
          f"(t{int(r['frame'][int(np.argmin(dice))])})", flush=True)
    plots.disp_compare_movie(cfg, Tracking(cfg), np.array(dice))
