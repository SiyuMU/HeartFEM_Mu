"""Model inputs and outputs shared by the FE and the PINN solver (numpy, scipy and meshio only: imported in both
environments).

- the wall mesh of the reference frame (mesh/t0.vtu from preprocess.py) and its boundary: base plane, endocardium,
  epicardium;
- the fiber families: angle from eC (circumferential) toward eL (longitudinal) as an (endo, epi) pair, linear in the
  transmural depth d_endo / (d_endo + d_epi); one family at FIBER_ANGLE_DEG, or two (TWO_FIBERS) with shares
  FIBER_WEIGHTS;
- the cavity volume of every tracked frame and the activation time of a frame (volume_constrain);
- the per-state result files (vtu/, summary.csv, pv_results.npz) and the stress in the current cardiac frame.
"""
import csv
import shutil

import numpy as np

BASE_ID, ENDO_ID, EPI_ID = 1, 2, 3  # facet markers of the FE mesh
COMPS = ("CC", "LL", "RR", "CL", "CR", "LR")  # stress components in the cardiac frame
MMHG_PER_KPA = 7.50062


def myo_endo(cfg):
    """True for the wall of myocardium + endocardium (WALL = "myo_endo"; absent or "myo": myocardium only)."""
    return getattr(cfg, "WALL", "myo") == "myo_endo"


# ==================== wall mesh ====================
def read_wall(cfg, name="t0.vtu"):
    """t0.vtu in mm with its local frame (eC, eL, eR per cell) and boundary: outward boundary faces `bnd`, the base
    (faces `on_base`, nodes, outward normal `n_out`), endocardium and epicardium (the two other connected surfaces; the
    smaller one is the endocardium). name: another mesh file of the mesh folder (WALL = "myo_endo": t0_endo.vtu)."""
    import meshio
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    vtu = meshio.read(cfg.mesh / name)
    points = vtu.points.astype(np.float64) * cfg.MM_PER_UNIT
    cells = vtu.cells_dict["tetra"].astype(np.int64)
    lrc = {k: vtu.cell_data_dict[k]["tetra"].astype(np.float64) for k in ("eC", "eL", "eR")}

    # Boundary faces (one tetrahedron), oriented out of the wall.
    faces = np.concatenate((cells[:, [0, 1, 2]], cells[:, [0, 1, 3]], cells[:, [0, 2, 3]], cells[:, [1, 2, 3]]))
    opposite = np.concatenate((cells[:, 3], cells[:, 2], cells[:, 1], cells[:, 0]))
    _, first, count = np.unique(np.sort(faces, 1), axis=0, return_index=True, return_counts=True)
    bnd, opp = faces[first[count == 1]], opposite[first[count == 1]]
    nrm = np.cross(points[bnd[:, 1]] - points[bnd[:, 0]], points[bnd[:, 2]] - points[bnd[:, 0]])
    inward = np.sum(nrm * (points[opp] - points[bnd[:, 0]]), 1) > 0
    bnd[inward] = bnd[inward][:, [0, 2, 1]]
    nrm[inward] *= -1
    unit = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    # Base: the largest group of faces sharing one outward normal, then all faces on that plane.
    _, group, size = np.unique(np.round(unit, 3), axis=0, return_inverse=True, return_counts=True)
    seed = group.ravel() == np.argmax(size)
    n_out = unit[seed].mean(0)
    n_out /= np.linalg.norm(n_out)
    offset = np.median(points[bnd[seed]] @ n_out)
    scale = np.ptp(points, 0).max()
    on_base = (unit @ n_out > 0.9) & (np.abs(points[bnd] @ n_out - offset).max(1) < 1e-5 * scale)
    # The rest splits into two connected surfaces: endocardium (smaller area) and epicardium.
    side = bnd[~on_base]
    edges = np.concatenate((side[:, [0, 1]], side[:, [1, 2]], side[:, [2, 0]]))
    _, comp = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points),) * 2),
                                   directed=False)
    parts = [side[comp[side[:, 0]] == c] for c in np.unique(comp[side[:, 0]])]
    if len(parts) != 2:
        raise ValueError(f"expected endocardium and epicardium, found {len(parts)} surfaces")
    area = [np.linalg.norm(np.cross(points[f[:, 1]] - points[f[:, 0]], points[f[:, 2]] - points[f[:, 0]]), axis=1).sum()
            for f in parts]
    endo, epi = (parts[0], parts[1]) if area[0] < area[1] else (parts[1], parts[0])
    return {"points": points, "cells": cells, **lrc, "bnd": bnd, "on_base": on_base, "n_out": n_out,
            "base_nodes": np.unique(bnd[on_base]), "endo": endo, "epi": epi}


def surface_distance(points, faces, queries, k=16):
    """Unsigned distance from the query points to a triangulated surface (exact point-triangle distance to the k
    triangles with the nearest centroids)."""
    from scipy.spatial import cKDTree
    tri = points[faces]
    idx = cKDTree(tri.mean(1)).query(queries, k)[1]
    A, B, C = tri[idx, 0], tri[idx, 1], tri[idx, 2]
    P = queries[:, None, :]
    nrm = np.cross(B - A, C - A)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    h = np.einsum("qkj,qkj->qk", P - A, nrm)
    Q = P - h[..., None] * nrm  # projection onto the triangle plane
    inside = np.all([np.einsum("qkj,qkj->qk", np.cross(V - U, Q - U), nrm) >= 0 for U, V in ((A, B), (B, C), (C, A))], 0)

    def segment(U, V):
        t = np.clip(np.einsum("qkj,qkj->qk", P - U, V - U) / np.einsum("qkj,qkj->qk", V - U, V - U), 0, 1)
        return np.linalg.norm(P - U - t[..., None] * (V - U), axis=2)
    d = np.where(inside, np.abs(h), np.minimum(np.minimum(segment(A, B), segment(B, C)), segment(C, A)))
    return d.min(1)


def transmural_depth(points, cells, endo, epi):
    """Per cell: d_endo / (d_endo + d_epi) at the centroid (0 at the endocardium, 1 at the epicardium)."""
    centroids = points[cells].mean(1)
    d = surface_distance(points, endo, centroids)
    return d / (d + surface_distance(points, epi, centroids))


# ==================== fibers ====================
def fiber_families(cfg):
    """(endo, epi) angles in degrees and shares of the fiber families: one family at FIBER_ANGLE_DEG unless TWO_FIBERS;
    two families at FIBER_ANGLES_DEG with shares FIBER_WEIGHTS."""
    def pair(a):
        if len(a) != 2:
            raise ValueError(f"a fiber angle is an (endo, epi) pair in degrees, got {a}")
        return float(a[0]), float(a[1])
    if not cfg.TWO_FIBERS:
        return [pair(cfg.FIBER_ANGLE_DEG)], np.ones(1)
    angles, weights = [pair(a) for a in cfg.FIBER_ANGLES_DEG], np.asarray(cfg.FIBER_WEIGHTS, float)
    if len(angles) != 2 or len(weights) != 2 or abs(weights.sum() - 1) > 1e-12 or (weights < 0).any():
        raise ValueError("TWO_FIBERS needs two (endo, epi) FIBER_ANGLES_DEG and two nonnegative FIBER_WEIGHTS that sum to 1")
    return angles, weights


def fiber_shares(cfg):
    """[((endo, epi) angle deg, share)] of the fiber families (the form stored by invert_para)."""
    angles, weights = fiber_families(cfg)
    return [(a, float(w)) for a, w in zip(angles, weights)]


def fiber_angles(cfg, depth):
    """[(angle in radians per cell, share)] of the fiber families: each (endo, epi) pair linear in the depth."""
    return [(np.radians((1 - depth) * endo + depth * epi), w) for (endo, epi), w in fiber_shares(cfg)]


def fiber_fields(cfg, w, depth):
    """Per cell and family: fiber f_i = eC rotated toward eL by its angle, sheet eR, normal f_i x eR; and the shares."""
    angles, weights = fiber_families(cfg)
    s0 = w["eR"]
    fibers, normals = [], []
    for endo_deg, epi_deg in angles:
        alpha = np.radians((1 - depth) * endo_deg + depth * epi_deg)[:, None]
        f = np.cos(alpha) * w["eC"] + np.sin(alpha) * w["eL"]
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        fibers.append(f)
        normals.append(np.cross(f, s0))
    return {"f0": fibers[0], "s0": s0, "n0": normals[0], "fibers": np.array(fibers), "normals": np.array(normals),
            "weights": weights}


def fiber_config(cfg):
    """The fiber settings stored with a trained network (checked against params.py when it is loaded)."""
    angles, weights = fiber_families(cfg)
    return {"angles_deg": [list(map(float, a)) for a in angles], "weights": weights.tolist()}


# ==================== tracked frames (volume_constrain) ====================
def frame_volumes(cfg):
    """Cavity volume of every tracked frame: endocardial cut surface (mesh/surface_cut) closed by cones to the base-plane
    origin, mesh units^3."""
    import meshio
    corr = np.load(cfg.mesh / "surface_cut" / "correspondence.npz")
    endo_tri = corr["surface_labels"] == 1
    vols = []
    for k in range(cfg.N_FRAMES):
        stl = meshio.read(cfg.mesh / "surface_cut" / f"t{k}.stl")
        x = stl.points[stl.cells_dict["triangle"]].astype(np.float64)[endo_tri] - corr["plane_origin"]
        vols.append(np.einsum("ij,ij->i", x[:, 0], np.cross(x[:, 1], x[:, 2])).sum() / 6)
    return np.array(vols)


def reference_frame(cfg):
    """Number of the tracked frame of the mesh (stress-free reference), from mesh/parameter.txt."""
    return int((cfg.mesh / "parameter.txt").read_text().split("=")[1])


def onset_frame(cfg):
    """Activation onset as a frame number: the tracked frame of maximum cavity volume + ONSET_AFTER_ED_FRAMES (43.5 here)."""
    return float(np.argmax(frame_volumes(cfg))) + cfg.ONSET_AFTER_ED_FRAMES


def activation(cfg, frames, onset=None):
    """Activation time t_a (ms since the onset of contraction) of each frame, t_a = mod(k - onset, N) BCL / N."""
    onset = onset_frame(cfg) if onset is None else onset
    return np.mod(np.asarray(frames, float) - onset, cfg.N_FRAMES) * cfg.BCL_MS / cfg.N_FRAMES


# ==================== results of a sequence of states ====================
def cardiac_components(Fc, sig, eC, eR):
    """Cauchy stress in the current cardiac frame: r = F^-T eR (wall normal), c = F eC made normal to r, l = r x c."""
    r = np.linalg.solve(np.transpose(Fc, (0, 2, 1)), eR[:, :, None])[:, :, 0]
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    c = np.einsum("cij,cj->ci", Fc, eC)
    c -= np.einsum("ci,ci->c", c, r)[:, None] * r
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    e = {"C": c, "L": np.cross(r, c), "R": r}
    return {f"sigma_{cc}": np.einsum("ci,cij,cj->c", e[cc[0]], sig, e[cc[1]]) for cc in COMPS}


class States:
    """Displacement and stresses of a sequence of states (the frames of volume_constrain, the replay states of windkessel)
    -> out/: vtu/<name>NN.vtu (one per state), vtu/cycle.pvd, summary.csv, pv_results.npz, params_used.py.
    Wall means skip the cells closer than BASE_EXCLUDE_UM to the base (boundary-condition stress)."""

    def __init__(self, cfg, g, cell_volume, out, name):
        self.cfg, self.g, self.vol, self.out, self.name = cfg, g, cell_volume, out, name
        pts = g["points"]
        self.far = (pts[g["cells"]].mean(1) - g["base_origin"]) @ g["base_normal"] / cfg.MM_PER_UNIT >= cfg.BASE_EXCLUDE_UM
        self.region = g.get("region")  # WALL = "myo_endo": 1 myocardium, 2 endocardial layer
        self.states = {}
        (out / "vtu").mkdir(parents=True, exist_ok=True)

    def add(self, k, row, u, f):
        """State k: summary row (completed by the wall means), vertex displacement u (mm) and the cell fields f (sigma, F,
        sigma_ff, p, Ct; with two fiber families also sigma_ff2, Ct_f1, Ct_f2)."""
        import meshio
        cfg, g, far, vol = self.cfg, self.g, self.far, self.vol
        loc = cardiac_components(f["F"], f["sigma"], g["eC"], g["eR"])
        extra = {key: f[key] for key in ("sigma_ff2", "Ct_f1", "Ct_f2") if key in f}  # two fiber families only
        data = {"u": u, "sigma": f["sigma"], "sigma_ff": f["sigma_ff"], "p": f["p"], "Ct": f["Ct"], **loc, **extra}
        if self.region is None:
            for key in ("sigma_ff", "sigma_CC", "sigma_LL", "sigma_RR") + (("sigma_ff2",) if extra else ()):
                row[f"{key}_wall_mean_kPa"] = float((vol * data[key])[far].sum() / vol[far].sum())
        else:  # wall means of the myocardium, and the means of the layer
            myo, lay = far & (self.region == 1), far & (self.region == 2)
            for key in ("sigma_ff", "sigma_CC", "sigma_LL", "sigma_RR") + (("sigma_ff2",) if extra else ()):
                row[f"{key}_wall_mean_kPa"] = float((vol * data[key])[myo].sum() / vol[myo].sum())
            for key in ("sigma_ff", "sigma_CC", "sigma_LL", "sigma_RR"):
                row[f"{key}_endo_mean_kPa"] = float((vol * data[key])[lay].sum() / vol[lay].sum())
        self.states[k] = (row, {key: val.astype(np.float32) for key, val in data.items()})
        f32 = np.float32
        meshio.Mesh(g["points"] / cfg.MM_PER_UNIT, [("tetra", g["cells"])], point_data={"u_um": (u / cfg.MM_PER_UNIT).astype(f32)},
                    cell_data={"sigma_ff": [data["sigma_ff"].astype(f32)], "Ct": [data["Ct"].astype(f32)],
                               **{key: [val.astype(f32)] for key, val in extra.items()},
                               **{f"sigma_{cc}": [loc[f"sigma_{cc}"].astype(f32)] for cc in COMPS},
                               "sigma": [f["sigma"].reshape(-1, 9).astype(f32)], "p": [f["p"].astype(f32)],
                               "far_from_base": [far.astype(np.int32)],
                               **({} if self.region is None else {"region": [self.region.astype(np.int32)]})}
                    ).write(self.out / "vtu" / f"{self.name}{k:02d}.vtu")

    def write(self, columns, material):
        """vtu/cycle.pvd, summary.csv, pv_results.npz (the summary columns `columns`, the fields of every state, mesh,
        local frame and `material`, a dict stored as JSON), params_used.py; returns the summary columns."""
        import json
        g, out = self.g, self.out
        ks = sorted(self.states)
        rows = [self.states[k][0] for k in ks]
        with open(out / "vtu" / "cycle.pvd", "w") as fh:
            fh.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
            fh.writelines(f'<DataSet timestep="{r_["t_ms"]:.4f}" file="{self.name}{r_[self.name]:02d}.vtu"/>\n' for r_ in rows)
            fh.write("</Collection>\n</VTKFile>\n")
        with open(out / "summary.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r_.items()} for r_ in rows)
        col = {key: np.array([r_[key] for r_ in rows]) for key in rows[0]}
        fields = {key: np.stack([self.states[k][1][key] for k in ks]) for key in self.states[ks[0]][1]}
        two = {"f0_2": g["fibers"][1]} if len(g["fibers"]) == 2 else {}
        if self.region is not None:
            two["region"] = self.region
        np.savez_compressed(out / "pv_results.npz", **{k: col[k] for k in columns},
                            **fields, points=g["points"], cells=g["cells"], f0=g["f0"], **two, eC=g["eC"], eL=g["eL"], eR=g["eR"],
                            far_from_base=self.far, cell_volume=self.vol, material=np.array(json.dumps(material)))
        shutil.copy(self.cfg.params_file, out / "params_used.py")
        return col
