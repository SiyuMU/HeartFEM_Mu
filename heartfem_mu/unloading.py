"""Task unloading: from the end-diastolic (ED) wall, its cavity pressure P_ED_KPA and the cavity volume V0_NL of the
stress-free (unloaded) state, find the passive scale C and the unloaded geometry X: inflating X to P_ED with C gives the
ED wall, and the cavity of X has the volume V0_NL. At ED the wall is passive (Ct = 0); the fiber architecture is
predefined (FIBER_ANGLE_DEG, or two families, TWO_FIBERS) on the unloaded geometry; every other constant is fixed. The
input is one surface of the ED wall (two closed shells cut at the base plane by the step preprocess, or one closed wall
already cut there) with its base plane and apex (INPUT_DIR/base_plane.json or the browser app); the step preprocess
meshes it (CACHE_DIR/mesh), and the unloaded geometry has the nodes and tetrahedra of that ED mesh.

Method (FE; unloading_fe.py): inverse elastostatics on the fixed ED mesh x. Unknowns: the forward displacement u (P2),
the pressure p (P1), the five rigid-body multipliers of fe.PressureProblem and the inverse displacement U (P1); the
unloaded geometry is X = x - U with straight cells. Equations: the residual of fe.PressureProblem on the reference X,
pulled back to the ED mesh (dV_X = det A dv, Grad_X = grad_x A^-1, N dA_X = cof(A) n da, A = I - grad U), and U = u at
the vertices (x_v = X_v + u(X_v)): the discrete forward problem on X, solved for X as one Newton system (MUMPS). The
local frame (eC, eL, eR) of X (the preprocess functions on X) and the base origin (mean of the base nodes of X) are
lagged, and C is found in the same loop. Iteration k: frame and origin of X_k, inverse solve at C_k -> X_(k+1), then a secant
step on ln C toward ln V0(X) = ln V0_NL (slope SECANT_SLOPE0 at the start, steps at most MAX_STEP_LN_C). Stop: X changes
by less than TOL_X_UM and V0 is met to V0_TOL_REL or the next step in ln C is below TOL_LN_C.
Check (the result is accepted only if it passes): the forward problem fe.PressureProblem on X, loaded from zero to
P_ED in CHECK_LOAD_STEPS equal steps without a warm start, must reproduce the ED wall (max |x_ED - X - u| < 1e-3 um).
The equilibrium is not unique in general: below a critical V0 the inverse solution exists, but loading the unloaded
wall from zero ends in another state; the check reports this (result.json "reaches_ed": false).
Output in OUTPUT_DIR: result.json (C, V0, the check, d ln C / d ln V0, the iterations) and unloaded.vtu (the unloaded
wall: tetrahedra with the local frame, depth and fibers per cell, the displacement to the ED wall per node).

This module needs numpy and scipy only (the frame helper also the preprocess modules of CONDA_ENV); the FE solver is
unloading_fe.py (CONDA_ENV_FENICS).
"""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

ENDO, EPI, BASE = 1, 2, 4  # boundary labels of the local-frame functions of preprocess (boundary_masks, solve_lrc)
CHECK_TOL_UM = 1e-3        # forward check: the unloaded wall loaded to P_ED reproduces the ED wall to this distance


def fe_unload(cfg):
    """Step unload (FE, MPI)."""
    from . import unloading_fe
    unloading_fe.unload(cfg)


# ==================== 1. wall topology (numpy) ====================
class Topology:
    """Fixed topology of the wall mesh (model.read_wall): cells, boundary triangles with the preprocess labels, the FE
    facets (as fe.geometry), base nodes and plane, and the apex as a material point (epicardial triangle,
    barycentric)."""

    def __init__(self, w):
        from .model import BASE_ID, ENDO_ID, EPI_ID
        self.cells, self.triangles = w["cells"], w["bnd"]
        self.endo, self.epi, self.base_nodes, self.n_out = w["endo"], w["epi"], w["base_nodes"], w["n_out"]
        endo = set(map(tuple, np.sort(w["endo"], 1)))
        keys = map(tuple, np.sort(self.triangles, 1))
        self.labels = np.array([BASE if b else ENDO if k in endo else EPI for k, b in zip(keys, w["on_base"])], np.uint8)
        labels = {tuple(f): ENDO_ID for f in np.sort(w["endo"], 1)}  # same order as fe.geometry
        labels.update({tuple(f): EPI_ID for f in np.sort(w["epi"], 1)})
        labels.update({tuple(f): BASE_ID for f in np.sort(w["bnd"][w["on_base"]], 1)})
        self.facets, self.facet_labels = np.array(list(labels.keys())), np.array(list(labels.values()))
        self.sign0 = np.sign(signed_volumes(w["points"], self.cells))
        self.apex_tri = self.apex_bary = None

    def set_apex(self, apex, points):
        """Attach the apex (a point on the epicardium of `points`) to its epicardial triangle; returns the distance."""
        i, b, d = closest_triangle(points[self.epi], apex)
        self.apex_tri, self.apex_bary = self.epi[i], b
        return d

    def apex(self, points):
        return self.apex_bary @ points[self.apex_tri]


def signed_volumes(points, cells):
    e = points[cells[:, 1:]] - points[cells[:, :1]]
    return np.linalg.det(e) / 6


def wall_volume_nL(points_um, cells):
    return float(np.abs(signed_volumes(points_um, cells)).sum() * 1e-6)


def cavity_volume_nL(points_um, topo):
    """Cavity volume closed by the base plane (the endocardial faces point into the cavity), nL."""
    x = points_um[topo.endo] - points_um[topo.base_nodes].mean(0)
    V = -np.einsum("ij,ij->i", x[:, 0], np.cross(x[:, 1], x[:, 2])).sum() / 6 * 1e-6
    if V <= 0:
        raise ValueError("cavity volume not positive: check the endocardium of the mesh")
    return float(V)


def closest_triangle(tri_points, q, k=32):
    """(index, barycentric coordinates, distance) of the point of the triangles (m, 3, 3) closest to q."""
    from scipy.spatial import cKDTree
    idx = cKDTree(tri_points.mean(1)).query(q, min(k, len(tri_points)))[1]
    A, B, C = (tri_points[idx, j] for j in range(3))
    n = np.cross(B - A, C - A)
    area2 = np.einsum("ij,ij->i", n, n)
    Q = q - (np.einsum("ij,ij->i", q - A, n) / area2)[:, None] * n
    b = np.stack([np.einsum("ij,ij->i", np.cross(C - B, Q - B), n), np.einsum("ij,ij->i", np.cross(A - C, Q - C), n),
                  np.einsum("ij,ij->i", np.cross(B - A, Q - A), n)], 1) / area2[:, None]
    bc = np.clip(b, 0, None)
    bc /= bc.sum(1, keepdims=True)
    P = bc[:, :1] * A + bc[:, 1:2] * B + bc[:, 2:] * C
    d = np.linalg.norm(P - q, axis=1)
    j = int(np.argmin(d))
    return int(idx[j]), bc[j], float(d[j])


# ==================== 2. local frame of a geometry (helper process in CONDA_ENV) ====================
_NEIGHBORHOODS = {}  # sha1 of the cells -> per cell its face neighbors within preprocess.LRC_SEARCH_LAYERS layers


def neighborhoods(pp, cells, adjacency):
    """The candidate donors of preprocess.correct_lrc_frames (its breadth-first search, which depends on the topology
    only), computed once per topology."""
    key = hashlib.sha1(cells.tobytes()).hexdigest()
    if key not in _NEIGHBORHOODS:
        out = []
        for cell_id in range(len(cells)):
            visited = {int(cell_id)}
            frontier = {int(cell_id)}
            for _ in range(pp.LRC_SEARCH_LAYERS):
                nearby = set()
                for neighbor_id in frontier:
                    nearby.update(adjacency.indices[adjacency.indptr[neighbor_id]:adjacency.indptr[neighbor_id + 1]])
                frontier = nearby - visited
                visited.update(frontier)
                if not frontier:
                    break
            out.append(np.array(sorted(visited - {cell_id}), dtype=int))
        _NEIGHBORHOODS[key] = out
    return _NEIGHBORHOODS[key]


def correct_lrc_frames_cached(points, cells, directions, valid, quality, graph=None):
    """preprocess.correct_lrc_frames with its neighbor search cached per topology; every other statement unchanged, so
    the frames are the same (the topology is fixed while the geometry moves)."""
    from . import preprocess as pp
    if pp.LRC_SEARCH_LAYERS <= 0:
        return 0
    if not 0.5 < pp.LRC_MAJORITY_FRACTION <= 1:
        raise ValueError("LRC_MAJORITY_FRACTION must be greater than 0.5 and at most 1.")
    reliable = valid & (quality >= pp.LRC_MIN_QUALITY)
    centers = points[cells].mean(axis=1)
    adjacency = pp.tetra_graph(points, cells) if graph is None else graph
    agreement = np.cos(np.deg2rad(pp.LRC_AGREEMENT_ANGLE))
    threshold = np.cos(np.deg2rad(pp.LRC_OUTLIER_ANGLE))
    replacements = []
    near = neighborhoods(pp, cells, adjacency)
    eL = directions["eL"]
    for cell_id in range(len(cells)):
        donors = near[cell_id]
        donors = donors[reliable[donors]]
        if len(donors) < 3:
            continue
        vectors = eL[donors]
        support = vectors @ vectors.T >= agreement
        group = support[np.argmax(support.sum(axis=1))].copy()
        required = max(3, int(np.ceil(pp.LRC_MAJORITY_FRACTION * len(donors))))
        while np.sum(group) >= required:
            mean = vectors[group].mean(axis=0)
            mean /= np.linalg.norm(mean)
            refined = group & (vectors @ mean >= agreement)
            if np.array_equal(refined, group):
                break
            group = refined
        if np.sum(group) < required:
            continue
        if valid[cell_id] and eL[cell_id] @ mean >= threshold:
            continue
        donors = donors[group]
        donor = donors[np.argmin(np.linalg.norm(centers[donors] - centers[cell_id], axis=1))]
        replacements.append((cell_id, donor))
    if replacements:
        targets, donors = np.asarray(replacements).T
        directions["eL"][targets] = directions["eL"][donors]
        valid[targets] = True
    return len(replacements)


def local_frame(points, cells, triangles, labels, apex):
    """{eC, eL, eR} per cell and the repair diagnostics of a geometry (points um, labels per boundary triangle), by the
    preprocess functions in the order of preprocess.export_reference."""
    from . import preprocess as pp
    masks = np.zeros(len(points), dtype=np.uint8)
    for label in (ENDO, EPI, BASE):
        masks[np.unique(triangles[labels == label])] |= label
    if any(not np.any(masks & label) for label in (ENDO, EPI, BASE)) or np.any((masks & 3) == 3):
        raise ValueError("Could not distinguish endocardium, epicardium, and base.")
    cell_data, longitudinal_reference = pp.solve_lrc(points, cells, masks, apex, pp.APEX_RADIUS)
    normals = pp.wall_normals(points, cells, triangles, labels)
    volumes = np.abs(np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]])) / 6
    edge_pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    edge_lengths_squared = np.sum((points[cells[:, edge_pairs[:, 0]]]
                                   - points[cells[:, edge_pairs[:, 1]]]) ** 2, axis=2).sum(axis=1)
    quality = 12 * (3 * volumes) ** (2 / 3) / edge_lengths_squared
    diagnostics = pp.repair_lrc(points, cells, cell_data, normals, quality, longitudinal_reference)
    diagnostics["min_quality"] = float(quality.min())
    return cell_data, diagnostics


def serve_frames():
    """Helper process (CONDA_ENV) of the FE step: reads lines "<request.npz>\\t<response.npz>" on stdin and answers
    "ok <json diagnostics>" or "error <message>". correct_lrc_frames is replaced by its cached form in this process."""
    from . import preprocess as pp
    pp.correct_lrc_frames = correct_lrc_frames_cached
    for line in sys.stdin:
        try:
            src, dst = line.rstrip("\n").split("\t")
            d = np.load(src)
            cell_data, diagnostics = local_frame(d["points"], d["cells"], d["triangles"], d["labels"], d["apex"])
            np.savez(dst, **cell_data)
            print("ok " + json.dumps(diagnostics), flush=True)
        except Exception:
            print("error " + traceback.format_exc().replace("\n", " | "), flush=True)


class FrameClient:
    """Rank 0 of the FE step: the helper serve_frames in CONDA_ENV (the FEniCS environment has no trimesh / vtk),
    started in a clean environment (no MPI variables of the FE step)."""

    def __init__(self, cfg, work):
        from .run import ROOT, conda_base
        self.work = Path(work)
        env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "LANG") if k in os.environ}
        env.update(PATH="/usr/bin:/bin", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
        script = (f"source {shlex.quote(str(Path(conda_base()) / 'etc/profile.d/conda.sh'))} && conda activate "
                  f"{shlex.quote(cfg.CONDA_ENV)} && exec python -u -c "
                  f"{shlex.quote('from heartfem_mu.unloading import serve_frames; serve_frames()')}")
        self.proc = subprocess.Popen(["/bin/bash", "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     env=env, text=True)
        self.calls = 0

    def __call__(self, points, topo):
        req, resp = self.work / "frame_request.npz", self.work / "frame_response.npz"
        np.savez(req, points=points, cells=topo.cells, triangles=topo.triangles, labels=topo.labels,
                 apex=topo.apex(points))
        self.proc.stdin.write(f"{req}\t{resp}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line.startswith("ok "):
            raise RuntimeError(f"local frame helper: {line or 'no answer'}")
        d = np.load(resp)
        self.calls += 1
        return {k: d[k].copy() for k in ("eC", "eL", "eR")}, json.loads(line[3:])

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# ==================== 3. results ====================
def write_mesh(path, points_um, cells, cell_data, point_data):
    import meshio
    meshio.write(path, meshio.Mesh(np.asarray(points_um, float), [("tetra", np.asarray(cells, np.int64))],
                                   cell_data={k: [np.asarray(v, float)] for k, v in cell_data.items()},
                                   point_data={k: np.asarray(v, float) for k, v in point_data.items()}))


def read_result(cfg):
    return json.loads((cfg.output / "result.json").read_text())
