"""Displacement constraint of the volume_constrain FE (DISP_CONSTRAIN = True): the wall is pulled toward the tracked wall.

The FE (fe.VolumeProblem) adds the traction w (u_img - u) on the endocardium and epicardium, i.e. the energy
w/2 int |u - u_img|^2 dS, w = DISP_WEIGHT (kPa/mm); everything else is the volume_constrain FE. The traction is a force
that the real heart does not have ("image force"): w balances closeness to the tracking against it, and it is reported.

Tracked wall = the cut surfaces of the preprocess (mesh/surface_cut/t*.stl: endocardium and epicardium cut at the base
plane of the reference frame, followed through the cycle by the tracking; all frames share the triangles), with the base
flattened: (1) rigid alignment of every frame onto the reference frame (least-squares rotation + translation of all
cut-surface vertices; the model has no rigid motion); (2) the rim (open edge) moved back into the base plane along its
normal, since the model holds its base in the plane (u . n = 0): every vertex gets the normal correction of the nearest
rim vertex of its surface, weighted by phi(d) = (1 + cos(pi d / L)) / 2 for d < L = FLAT_DEPTH_UM (d: depth below the
base plane in the reference frame), 0 below; in-plane positions unchanged. The targets AND the prescribed cavity
volumes come from these surfaces, and the Dice compares with them directly (bases aligned by construction).

Steps (CONDA_ENV, numpy + scipy + meshio; compare also vtk, matplotlib):
  targets  u_img of every endocardial / epicardial node of mesh/t0.vtu (the reference frame): the node is attached to the
           closest point of the same surface of the flattened reference frame, as barycentric coordinates in one
           triangle, and followed through the flattened frames; u_img(ref) = 0. Cavity volume of every flattened frame
           (endocardium closed by cones to the base-plane origin, as model.frame_volumes).
           -> OUTPUT_DIR/mesh/disp_targets.npz (an input of the step solve, next to the other preprocess products)
  compare  the wall Dice of every frame (voxels of DICE_VOXEL_UM below the base plane; tracked wall = inside the
           epicardium and outside the endocardium of the flattened surfaces, each closed by a fan over its rim;
           simulated wall = inside the closed boundary of the deformed mesh) and the movie OUTPUT_DIR/disp_compare.mp4
           (plots.disp_compare_movie: three long-axis sections and one short-axis section).
"""
import numpy as np
from scipy.spatial import cKDTree

from . import model

DICE_VOXEL_UM = 1.0     # voxel edge of the wall Dice (mesh units = um)
NEAR_UM = 12.0          # Dice voxels farther than this from both walls are outside both (skipped for speed)
FLAT_DEPTH_UM = 20.0    # L: depth below the base plane over which the base flattening fades out (mesh units = um)


# ==================== tracked surfaces ====================
def flattened(cfg):
    """The cut surfaces of every frame, rigidly aligned onto the reference frame and base-flattened: vertices (frames, V, 3)
    in mesh units, triangles, per-triangle surface label (1 endo, 2 epi), and per frame the removed rigid rotation (deg),
    the centroid shift and the largest rim height above the base plane before the flattening (mesh units)."""
    import meshio
    w = model.read_wall(cfg)
    n_out = w["n_out"]
    level = float(np.median(w["points"][w["base_nodes"]] @ n_out)) / cfg.MM_PER_UNIT  # base plane: x . n_out = level
    lab = np.load(cfg.mesh / "surface_cut" / "correspondence.npz")["surface_labels"]
    ref = model.reference_frame(cfg)
    X, tri = np.empty((cfg.N_FRAMES,) + meshio.read(cfg.mesh / "surface_cut" / f"t{ref}.stl").points.shape), None
    for k in range(cfg.N_FRAMES):
        m = meshio.read(cfg.mesh / "surface_cut" / f"t{k}.stl")
        X[k] = m.points
        t = m.cells_dict["triangle"].astype(np.int64)
        if tri is None:
            tri = t
        elif not np.array_equal(t, tri):
            raise ValueError(f"surface_cut/t{k}.stl: the cut frames must share the triangles")
    X0 = X[ref].copy()
    d = np.clip(level - X0 @ n_out, 0, None)
    phi = np.where(d < FLAT_DEPTH_UM, 0.5 * (1 + np.cos(np.pi * d / FLAT_DEPTH_UM)), 0.0)
    nearest = {}  # surface -> (its vertices, the nearest rim vertex of each)
    for s in (1, 2):
        f = tri[lab == s]
        e, c = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), 1), axis=0, return_counts=True)
        rim, verts = np.unique(e[c == 1]), np.unique(f)
        nearest[s] = (verts, rim[cKDTree(X0[rim]).query(X0[verts])[1]])
    info = []
    for k in range(cfg.N_FRAMES):
        Rm, t = rigid_to(X[k], X0)
        shift = np.linalg.norm(X[k].mean(0) - X0.mean(0))
        X[k] = X[k] @ Rm.T + t
        h = X[k] @ n_out - level  # height above the base plane
        rim_h = 0.0
        for verts, near in nearest.values():
            rim_h = max(rim_h, np.abs(h[near]).max())
            X[k][verts] -= (h[near] * phi[verts])[:, None] * n_out
        info.append((np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))), shift, rim_h))
    return X, tri, lab, np.array(info)


def cavity_volume(x, tri, lab, origin):
    """Cavity volume of one frame: the endocardial triangles closed by cones to the base-plane origin (as
    model.frame_volumes), mesh units^3."""
    v = x[tri[lab == 1]] - origin
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6


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
    """Step targets: u_img of every endo / epi node and the cavity volume of every frame, from the flattened tracked
    surfaces -> OUTPUT_DIR/mesh/disp_targets.npz."""
    w = model.read_wall(cfg)
    pts = w["points"] / cfg.MM_PER_UNIT
    X, tri, lab, info = flattened(cfg)
    ref = model.reference_frame(cfg)
    u_img = np.zeros((cfg.N_FRAMES, len(pts), 3))
    data = np.zeros(len(pts), bool)
    lines = []
    for s, name in ((1, "endo"), (2, "epi")):
        nodes = np.unique(w[name])
        sel = np.where(lab == s)[0]
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
    origin = np.load(cfg.mesh / "surface_cut" / "correspondence.npz")["plane_origin"]
    vol = np.array([cavity_volume(X[k], tri, lab, origin) for k in range(cfg.N_FRAMES)])
    ratio = vol / vol[ref]
    np.savez_compressed(cfg.mesh / "disp_targets.npz", u_img=u_img, data_nodes=data, V_ratio=ratio,
                        rigid_rotation_deg=info[:, 0], rigid_shift_um=info[:, 1], rim_height_um=info[:, 2],
                        reference_frame=ref)
    mag = np.linalg.norm(u_img[:, data], axis=2) / cfg.MM_PER_UNIT
    cut = model.frame_volumes(cfg)
    print("\n".join(lines) + f"\nimage displacement: median over the nodes up to {np.median(mag, 1).max():.2f} um; tracking "
          f"rigid motion removed: up to {info[:, 0].max():.1f} deg; rim height above the base plane before the flattening: "
          f"up to {info[:, 2].max():.1f} um; cavity volume / reference frame {ratio.min():.4f}..{ratio.max():.4f} "
          f"(not flattened {(cut / cut[ref]).min():.4f}..{(cut / cut[ref]).max():.4f})", flush=True)


def load_targets(cfg):
    """Image displacement u_img (frames, nodes, 3; mm) and cavity volume / reference frame (frames) of the step targets."""
    path = cfg.mesh / "disp_targets.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the step targets first")
    d = np.load(path)
    if "V_ratio" not in d:
        raise ValueError(f"{path}: written by an older version (no base flattening): run the step targets again")
    return d["u_img"], d["V_ratio"]


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


def fan_closed(tri, n_vertices):
    """Triangles of an open surface with one boundary loop closed by a fan to an added vertex (index n_vertices), and the
    loop vertices (the added vertex is their centroid)."""
    d = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    e, c = np.unique(np.sort(d, 1), axis=0, return_counts=True)
    rim = {tuple(r) for r in e[c == 1]}
    d = d[np.array([tuple(r) in rim for r in np.sort(d, 1)])]
    return np.vstack((tri, np.c_[d[:, 1], d[:, 0], np.full(len(d), n_vertices)])), np.unique(d)


class Tracking:
    """The flattened tracked surfaces of every frame (endocardium and epicardium, each closed by a fan over its rim in the
    base plane) and the wall Dice of a simulated frame."""

    def __init__(self, cfg):
        self.cfg = cfg
        w = model.read_wall(cfg)
        self.bnd, self.pts, self.n_out, self.base_nodes = w["bnd"], w["points"] / cfg.MM_PER_UNIT, w["n_out"], w["base_nodes"]
        self.X, tri, lab, _ = flattened(cfg)
        ref = model.reference_frame(cfg)
        self.closed = {}  # surface -> (outward closed triangles, rim vertices)
        for s, name in ((1, "endo"), (2, "epi")):
            f, rim = fan_closed(tri[lab == s], self.X.shape[1])
            self.closed[name] = (outward(np.vstack((self.X[ref], self.X[ref][rim].mean(0))), f), rim)
        self.level = float(np.median(self.pts[self.base_nodes] @ self.n_out))

    def surface(self, k, name):
        """Closed tracked surface `name` (endo, epi) of frame k: vertices and outward triangles."""
        f, rim = self.closed[name]
        return np.vstack((self.X[k], self.X[k][rim].mean(0))), f

    def wall(self, k, q):
        """Mask of the points q inside the tracked wall of frame k."""
        return (signed_distance(*self.surface(k, "epi"), q) < 0) & (signed_distance(*self.surface(k, "endo"), q) > 0)

    def dice(self, k, u_mm):
        x, Xk = self.pts + u_mm / self.cfg.MM_PER_UNIT, self.X[k]
        lo, hi = np.minimum(x.min(0), Xk.min(0)) - 2, np.maximum(x.max(0), Xk.max(0)) + 2
        q = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i], DICE_VOXEL_UM) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        q = q[self.level - q @ self.n_out >= 0]  # below the base plane
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
