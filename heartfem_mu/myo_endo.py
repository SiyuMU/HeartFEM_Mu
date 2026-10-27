"""Wall of myocardium + endocardium (WALL = "myo_endo"; windkessel FE only). numpy, scipy and meshio (preprocess_layers:
the preprocessing environment, trimesh, vtk, gmsh).

The two-region mesh mesh/t0_endo.vtu (cell data eC, eL, eR, region: 1 myocardium, 2 endocardium; the myocardial nodes
first, then the nodes of the layer alone; the myocardial cells first) comes from one of two steps (ENDO_GROW):
- grow_layer (ENDO_GROW = True; step endo_layer after preprocess): the myocardial mesh mesh/t0.vtu unchanged plus a layer
  of thickness ENDO_THICKNESS_UM grown from its inner surface into the cavity -> mesh/t0_endo.vtu and
  mesh/endo_layer_info.json. Every inner-surface node moves along its normal (area-weighted face
  normals; on the base ring projected into the base plane, so the layer's base stays on the plane) in ENDO_SUBLAYERS
  steps; every surface triangle sweeps a prism per step, split into three tetrahedra with the diagonal of every side quad
  from the lower-numbered surface node at the bottom to the other node at the top, so neighboring prisms share their
  side triangles: a conforming mesh, the two regions share the interface nodes (perfect bonding). A layer cell takes the
  local frame of the myocardial tetrahedron it grows from.
- preprocess_layers (ENDO_GROW = False; step preprocess): the input surface has both layers, three closed nested surfaces
  (lumen, myocardial endocardium, epicardium; ordered by enclosed volume). As preprocess.export_reference for the two
  wall regions: both cut at the base plane, the cut walls (lumen + inner base annulus + myocardial endocardium;
  myocardial endocardium + outer base annulus + epicardium) remeshed and filled by gmsh as two volumes that share the
  remeshed myocardial endocardium (a conforming mesh, perfect bonding) -> mesh/t0_endo.vtu, mesh/t0.vtu (the myocardium
  alone), parameter.txt, endo_layer_info.json, preprocess_info.json. The local frame is that of the myocardium alone
  (preprocess's Laplace problems and repairs on the myocardial region); a layer cell takes the frame of the myocardial
  tetrahedron of the interface face nearest to it.
- fe_geometry: the FE input of the two-region wall (fe.geometry with WALL = "myo_endo"): boundary of t0_endo.vtu (the
  endocardial boundary is now the lumen), the region of every cell, the myo/endo interface, the fibers per region
  (myocardium: depth from the interface to the epicardium with the myocardial fiber settings; layer: depth from the
  lumen to the interface with the fiber settings in ENDO_MATERIAL), base-frame origin of the myocardial base nodes.
"""
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import model

MYO, ENDO = 1, 2
MATERIAL_KEYS = ("C", "C_FF", "C_SS", "C_NN", "C_NS", "C_FS", "C_FN", "T_MAX")


def endo_settings(cfg):
    """The layer's material and fiber settings as a settings object (fibers read by model.fiber_families)."""
    m = cfg.ENDO_MATERIAL
    return SimpleNamespace(TWO_FIBERS=cfg.TWO_FIBERS, **m)


def _tet_volume(points, tets):
    p = points[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0


def _quality(points, tets):
    """12 (3 V)^(2/3) / sum of squared edge lengths (1 = regular tetrahedron)."""
    p = points[tets]
    l2 = sum(np.sum((p[:, i] - p[:, j]) ** 2, 1) for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    return 12 * (3 * np.abs(_tet_volume(points, tets))) ** (2 / 3) / l2


def _enclosed_volume(points, faces, n_out, offset):
    """Volume enclosed by a surface open at the base and the base plane (faces oriented out of the wall)."""
    x = points - offset * n_out
    return -np.einsum("ij,ij->i", x[faces[:, 0]], np.cross(x[faces[:, 1]], x[faces[:, 2]])).sum() / 6.0


def grow_layer(cfg):
    """Step endo_layer: mesh/t0.vtu -> mesh/t0_endo.vtu and mesh/endo_layer_info.json (lengths in mesh units)."""
    import meshio
    vtu = meshio.read(cfg.mesh / "t0.vtu")
    points = vtu.points.astype(np.float64)  # mesh units (um), unchanged in the output
    cells = vtu.cells_dict["tetra"].astype(np.int64)
    lrc = {k: vtu.cell_data_dict[k]["tetra"].astype(np.float64) for k in ("eC", "eL", "eR")}
    w = model.read_wall(cfg)  # boundary of the myocardial mesh (topology and normals do not depend on the units)
    faces, n_out = w["endo"], w["n_out"]  # inner surface, oriented out of the wall (into the cavity)
    offset = float(np.median(points[w["base_nodes"]] @ n_out))
    all_faces = np.sort(np.concatenate([cells[:, q] for q in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])]), 1)
    owner_of = {tuple(f): i % len(cells) for i, f in enumerate(all_faces)}  # boundary faces have one owner
    own = np.array([owner_of[tuple(f)] for f in np.sort(faces, 1)])  # the myocardial cell each layer prism grows from
    thickness, sublayers = float(cfg.ENDO_THICKNESS_UM), int(cfg.ENDO_SUBLAYERS)
    if thickness <= 0 or sublayers < 1:
        raise ValueError("ENDO_THICKNESS_UM must be > 0 and ENDO_SUBLAYERS >= 1")

    nodes = np.unique(faces)
    local = np.full(len(points), -1)
    local[nodes] = np.arange(len(nodes))
    fn = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])  # 2 area n
    normal = np.zeros((len(nodes), 3))
    for k in range(3):
        np.add.at(normal, local[faces[:, k]], fn)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    scale = np.ptp(points, 0).max()
    ring = np.abs(points[nodes] @ n_out - offset) < 1e-5 * scale  # inner-surface nodes on the base plane
    normal[ring] -= (normal[ring] @ n_out)[:, None] * n_out
    normal[ring] /= np.linalg.norm(normal[ring], axis=1, keepdims=True)
    ids, new = [nodes], []
    for k in range(1, sublayers + 1):
        x = points[nodes] + (k / sublayers) * thickness * normal
        x[ring] -= ((x[ring] @ n_out) - offset)[:, None] * n_out  # exactly on the base plane
        new.append(x)
        ids.append(len(points) + (k - 1) * len(nodes) + np.arange(len(nodes)))
    pts = np.concatenate([points] + new)
    loc = local[np.sort(faces, axis=1)]  # bottom (v0 v1 v2) by surface node number
    tets, parent = [], []
    for k in range(sublayers):
        v0, v1, v2 = (ids[k][loc[:, i]] for i in range(3))
        t0, t1, t2 = (ids[k + 1][loc[:, i]] for i in range(3))
        for tet in (np.c_[v0, v1, v2, t2], np.c_[v0, v1, t1, t2], np.c_[v0, t0, t1, t2]):
            tets.append(tet)
            parent.append(own)
    tets, parent = np.concatenate(tets), np.concatenate(parent)
    vol = _tet_volume(pts, tets)
    tets[vol < 0] = tets[vol < 0][:, [0, 1, 3, 2]]
    vol = np.abs(vol)
    all_cells = np.concatenate((cells, tets))
    region = np.r_[np.full(len(cells), MYO), np.full(len(tets), ENDO)].astype(np.int32)
    meshio.Mesh(pts, [("tetra", all_cells)], cell_data={**{k: [np.concatenate((v, v[parent]))] for k, v in lrc.items()},
                                                         "region": [region]}).write(cfg.mesh / "t0_endo.vtu")

    # checks: positive tetrahedra, the old inner surface interior, layer volume = difference of the enclosed volumes
    w2 = model.read_wall(cfg, "t0_endo.vtu")
    v_myo = _enclosed_volume(points, faces, n_out, offset)
    v_lumen = _enclosed_volume(pts, w2["endo"], n_out, offset)
    still = len({tuple(f) for f in np.sort(faces, 1)} & {tuple(f) for f in np.sort(w2["bnd"], 1)})
    nl = cfg.MM_PER_UNIT ** 3 * 1e3  # mesh units^3 -> nL
    info = {"thickness": thickness, "sublayers": sublayers, "nodes_myo": len(points), "nodes": len(pts),
            "tets_myo": len(cells), "tets_layer": len(tets), "inner_surface_nodes": len(nodes),
            "base_ring_nodes": int(ring.sum()), "layer_volume_nL": float(vol.sum() * nl),
            "cavity_myo_inner_nL": float(v_myo * nl), "cavity_lumen_nL": float(v_lumen * nl),
            "volume_identity_rel_err": float(abs(v_myo - v_lumen - vol.sum()) / vol.sum()),
            "min_tet_volume_over_median": float(vol.min() / np.median(vol)),
            "layer_quality_min": float(_quality(pts, tets).min()), "layer_quality_median": float(np.median(_quality(pts, tets)))}
    if info["min_tet_volume_over_median"] < 1e-3 or still or info["volume_identity_rel_err"] > 1e-8:
        raise ValueError(f"invalid endocardial layer (old inner faces on the boundary: {still}): {info}")
    (cfg.mesh / "endo_layer_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)
    return info


# ==================== the input surface has both layers (ENDO_GROW = False) ====================
LUMEN, INTERFACE, EPI, INNER_BASE, OUTER_BASE = 1, 2, 3, 4, 5  # patches of the cut surfaces of both layers


def _cut_layers(shells, origin, normal, keep_sign):
    """The two cut walls (preprocess.cut_wall: lumen + myocardial endocardium; myocardial endocardium + epicardium) as
    one surface: points, triangles, patch of every triangle; the clipped surfaces of the myocardial wall (vtk); the
    volumes of the two cut walls."""
    from . import preprocess
    endo_wall, _ = preprocess.cut_wall(shells[:2], origin, normal, keep_sign)
    myo_wall, myo_surfaces = preprocess.cut_wall(shells[1:], origin, normal, keep_sign)
    points, inverse = np.unique(np.concatenate((endo_wall.vertices, myo_wall.vertices)), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    fe_, fm = inverse[:len(endo_wall.vertices)][endo_wall.faces], inverse[len(endo_wall.vertices):][myo_wall.faces]
    shared = {tuple(f) for f in np.sort(fe_, 1)} & {tuple(f) for f in np.sort(fm, 1)}  # the myocardial endocardium
    scale = np.linalg.norm(np.ptp(points, 0))
    flat = lambda f: np.abs((points[f] - origin) @ normal).max(1) < scale * 1e-6  # noqa: E731
    in_e = np.array([tuple(f) in shared for f in np.sort(fe_, 1)])
    in_m = np.array([tuple(f) in shared for f in np.sort(fm, 1)])
    faces = np.concatenate((fe_, fm[~in_m]))
    patch = np.concatenate((np.where(in_e, INTERFACE, np.where(flat(fe_), INNER_BASE, LUMEN)),
                            np.where(flat(fm[~in_m]), OUTER_BASE, EPI)))
    if in_e.sum() != in_m.sum() or in_e.sum() != len(shared) or len(np.unique(patch)) != 5:
        raise ValueError("the two cut walls do not share the myocardial endocardium; check that the three surfaces are "
                         "nested and that the base plane cuts each of them in one loop")
    return points, faces, patch, myo_surfaces, {"myocardium": abs(myo_wall.volume), "endocardium": abs(endo_wall.volume)}


def _mesh_layers(points, faces, patch, directory, mesh_size, gmsh):
    """gmsh (the executable and the mesh options of preprocess.tetrahedralize; the Python API gives a different mesh in
    every run): every patch of the cut surfaces one discrete surface, the patches joined at their shared rims (the base
    loops), reparametrized and remeshed, the two volumes filled. Returns points, tetrahedra of the myocardium and of the
    layer."""
    import subprocess
    import meshio
    import trimesh
    keys = (LUMEN, INTERFACE, EPI, INNER_BASE, OUTER_BASE)
    for k in keys:
        trimesh.Trimesh(points, faces[patch == k], process=False).export(directory / f"patch{k}.stl")
    # One surface per patch (surface k = patch k, the merge order), not ClassifySurfaces: it would join a patch to its
    # neighbor where they meet at a small angle (a wall meeting the base plane obliquely).
    (directory / "layers.geo").write_text("".join(f'Merge "patch{k}.stl";\n' for k in keys) + f'''Coherence Mesh;
CreateTopology;
CreateGeometry;
Surface Loop(1) = {{{LUMEN}, {INNER_BASE}, {INTERFACE}}};
Volume(1) = {{1}};
Surface Loop(2) = {{{INTERFACE}, {OUTER_BASE}, {EPI}}};
Volume(2) = {{2}};
''' + "".join(f"Physical Surface({k}) = {{{k}}};\n" for k in keys) + f'''Physical Volume({MYO}) = {{2}};
Physical Volume({ENDO}) = {{1}};
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.OptimizeNetgen = 1;
Mesh.MeshSizeMin = {mesh_size};
Mesh.MeshSizeMax = {mesh_size};
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.MshFileVersion = 2.2;
''')
    result = subprocess.run([gmsh, "layers.geo", "-3", "-o", "layers.msh", "-v", "2"], cwd=directory,
                            capture_output=True, text=True, timeout=300)
    if result.returncode or not (directory / "layers.msh").exists():
        raise ValueError("Gmsh failed on the two layers. Check the surfaces or the mesh size. "
                         + (result.stderr + result.stdout)[-1200:])
    mesh = meshio.read(directory / "layers.msh")
    tags = {c: mesh.cell_data_dict["gmsh:physical"][c] for c in ("tetra", "triangle")}
    tets, triangles = mesh.get_cells_type("tetra"), mesh.get_cells_type("triangle")
    myo, endo = tets[tags["tetra"] == MYO], tets[tags["tetra"] == ENDO]
    if not len(myo) or not len(endo) or len(myo) + len(endo) != len(tets):
        raise ValueError("Gmsh did not fill both layers with tetrahedra.")
    # The surface k of the mesh must lie on the patch k of the input (the merge order): of all patches, its triangles
    # are closest to patch k (median distance of their centroids).
    for k in keys:
        c = mesh.points[triangles[tags["triangle"] == k]].mean(1)
        d = [np.median(model.surface_distance(points, faces[patch == j], c)) for j in keys]
        if not len(c) or int(np.argmin(d)) != keys.index(k):
            raise ValueError(f"Gmsh surface {k} is not the input patch {k}.")
    xyz = mesh.points
    # The myocardial nodes first (in the order of their first use), then the nodes of the layer alone.
    first = lambda a: a[np.sort(np.unique(a, return_index=True)[1])]  # noqa: E731  distinct, in the order of first use
    order = np.concatenate((first(myo.ravel()), first(endo.ravel()[~np.isin(endo.ravel(), myo)])))
    new = np.full(len(xyz), -1, dtype=np.int64)
    new[order] = np.arange(len(order))
    return xyz[order], new[myo], new[endo]


def preprocess_layers(cfg):
    """Step preprocess with ENDO_GROW = False (see the module docstring); lengths in mesh units."""
    import meshio
    import trimesh
    from scipy.spatial import cKDTree
    from . import preprocess
    start = time.time()
    if shutil.which(preprocess.GMSH) is None:
        raise FileNotFoundError("Gmsh is not installed or GMSH is not on PATH.")
    if not cfg.USE_SAVED_LANDMARKS:
        raise ValueError("ENDO_GROW = False needs USE_SAVED_LANDMARKS = True: the base plane, apex and reference frame in "
                         "input/base_plane.json (the landmark app places them on a surface of one layer)")
    saved = json.loads((cfg.input / "base_plane.json").read_text())
    reference_stl, frame = preprocess.reference_surface(cfg.input)
    if Path(saved["reference_stl"]).name != reference_stl.name:
        raise ValueError(f"The saved landmarks belong to {saved['reference_stl']}, the input surface is {reference_stl.name}.")
    origin, normal = preprocess.plane_vectors(saved)
    apex = np.asarray(saved["apex"], dtype=float)
    surface = trimesh.load_mesh(reference_stl, process=True)
    shells = sorted(surface.split(only_watertight=False), key=lambda shell: abs(shell.volume))
    if len(shells) != 3 or not all(shell.is_watertight for shell in shells):
        raise ValueError(f"ENDO_GROW = False: {reference_stl.name} must hold three closed surfaces (lumen, myocardial "
                         f"endocardium, epicardium), found {len(shells)}")
    scale = np.linalg.norm(surface.extents)
    if apex.shape != (3,) or not np.all(np.isfinite(apex)):
        raise ValueError("The saved apex must be a finite point on the outer surface.")
    if trimesh.proximity.closest_point_naive(shells[2], apex[None])[1][0] > scale * 1e-5:
        raise ValueError("The saved apex must lie on the outer surface (epicardium).")
    signed_apex = float(np.dot(apex - origin, normal))
    if abs(signed_apex) < preprocess.APEX_RADIUS:
        raise ValueError("Move the base plane farther from the apex patch.")
    keep_sign = 1 if signed_apex > 0 else -1
    print(f"Saved landmarks: reference {reference_stl.name} (three surfaces); meshing both layers at size "
          f"{cfg.MESH_SIZE}...", flush=True)
    sp, sf, patch, myo_surfaces, cut_volume = _cut_layers(shells, origin, normal, keep_sign)
    cfg.mesh.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cfg.mesh, prefix=".mesh-") as temporary:
        points, myo, endo = _mesh_layers(sp, sf, patch, Path(temporary), float(cfg.MESH_SIZE), preprocess.GMSH)
    distance = (points - origin) @ normal  # the remeshed flat base is within round-off of the plane: put it on the plane
    on_plane = np.abs(distance) < scale * 1e-5
    points[on_plane] -= distance[on_plane, None] * normal
    v_myo, v_endo = np.abs(_tet_volume(points, myo)), np.abs(_tet_volume(points, endo))
    for name, v in (("myocardium", v_myo.sum()), ("endocardium", v_endo.sum())):
        if abs(v - cut_volume[name]) > 0.05 * cut_volume[name]:
            raise ValueError(f"Tetrahedral volume of the {name} does not match its cut wall volume.")

    # Local frame of the myocardium alone (preprocess.export_reference on the myocardial region).
    n_myo = int(myo.max()) + 1
    pm = points[:n_myo]
    faces = np.concatenate((myo[:, [0, 1, 2]], myo[:, [0, 1, 3]], myo[:, [0, 2, 3]], myo[:, [1, 2, 3]]))
    _, first, count = np.unique(np.sort(faces, 1), axis=0, return_index=True, return_counts=True)
    triangles = faces[first[count == 1]]
    masks, labels = preprocess.boundary_masks(pm, triangles, myo_surfaces, origin, normal, scale)
    frames, longitudinal_reference = preprocess.solve_lrc(pm, myo, masks, apex, preprocess.APEX_RADIUS)
    normals = preprocess.wall_normals(pm, myo, triangles, labels)
    quality = _quality(pm, myo)
    diagnostics = preprocess.repair_lrc(pm, myo, frames, normals, quality, longitudinal_reference)
    inner = triangles[labels == 1]  # the myocardial endocardium = the interface
    near = cKDTree(pm[inner].mean(1)).query(points[endo].mean(1))[1]
    owner = {tuple(f): i % len(myo) for i, f in enumerate(np.sort(faces, 1))}  # boundary faces have one owner
    parent = np.array([owner[tuple(f)] for f in np.sort(inner, 1)])[near]

    cells = np.concatenate((myo, endo))
    region = np.r_[np.full(len(myo), MYO), np.full(len(endo), ENDO)].astype(np.int32)
    lrc = {k: frames[k] for k in ("eC", "eL", "eR")}
    meshio.Mesh(pm, [("tetra", myo)], cell_data={k: [v] for k, v in lrc.items()}).write(cfg.mesh / "t0.vtu")
    meshio.Mesh(points, [("tetra", cells)], cell_data={**{k: [np.concatenate((v, v[parent]))] for k, v in lrc.items()},
                                                       "region": [region]}).write(cfg.mesh / "t0_endo.vtu")
    (cfg.mesh / "parameter.txt").write_text(f"unloaded_time_point = {frame}\n")

    # checks: the inner surface of the myocardium interior, layer volume = difference of the enclosed volumes
    w = model.read_wall(cfg, "t0.vtu")
    w2 = model.read_wall(cfg, "t0_endo.vtu")
    n_out = w2["n_out"]
    offset = float(np.median(points[w2["base_nodes"]] @ n_out))
    v_in = _enclosed_volume(points, w["endo"], n_out, offset)
    v_lumen = _enclosed_volume(points, w2["endo"], n_out, offset)
    still = len({tuple(f) for f in np.sort(w["endo"], 1)} & {tuple(f) for f in np.sort(w2["bnd"], 1)})
    thickness = model.surface_distance(points, w["endo"], points[np.unique(w2["endo"])])
    nl = cfg.MM_PER_UNIT ** 3 * 1e3  # mesh units^3 -> nL
    info = {"layer": "input surfaces (ENDO_GROW = False)", "thickness_min": float(thickness.min()),
            "thickness_median": float(np.median(thickness)), "thickness_max": float(thickness.max()),
            "nodes_myo": n_myo, "nodes": len(points), "tets_myo": len(myo), "tets_layer": len(endo),
            "lumen_surface_nodes": int(len(np.unique(w2["endo"]))), "layer_volume_nL": float(v_endo.sum() * nl),
            "cavity_myo_inner_nL": float(v_in * nl), "cavity_lumen_nL": float(v_lumen * nl),
            "volume_identity_rel_err": float(abs(v_in - v_lumen - v_endo.sum()) / v_endo.sum()),
            "min_tet_volume_over_median": float(v_endo.min() / np.median(v_endo)),
            "layer_quality_min": float(_quality(points, endo).min()),
            "layer_quality_median": float(np.median(_quality(points, endo)))}
    if info["min_tet_volume_over_median"] < 1e-3 or still or info["volume_identity_rel_err"] > 1e-8:
        raise ValueError(f"invalid two-layer mesh (myocardial endocardium on the boundary: {still}): {info}")
    (cfg.mesh / "endo_layer_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)
    result = {"tetrahedra": len(cells), "tetrahedra_myo": len(myo), "tetrahedra_layer": len(endo), "invalid_lrc": 0,
              "low_quality_tetrahedra": int(np.sum(_quality(points, cells) < preprocess.LRC_MIN_QUALITY)), **diagnostics}
    print(f"Saved {cfg.mesh / 't0_endo.vtu'} and {cfg.mesh / 't0.vtu'}: {result}", flush=True)
    info = {"mesh_size": cfg.MESH_SIZE, **result, "time_s": round(time.time() - start, 1)}
    (cfg.mesh / "preprocess_info.json").write_text(json.dumps(info, indent=1))
    print(f"preprocess done in {info['time_s']} s", flush=True)


def fe_geometry(cfg):
    """fe.geometry for WALL = "myo_endo": the keys of fe.geometry plus region, cell_weights (shares per family and cell),
    interface_faces."""
    import meshio
    w = model.read_wall(cfg, "t0_endo.vtu")
    points, cells = w["points"], w["cells"]
    region = meshio.read(cfg.mesh / "t0_endo.vtu").cell_data_dict["region"]["tetra"].astype(np.int64)
    myo = region == MYO
    labels = {tuple(f): model.ENDO_ID for f in np.sort(w["endo"], 1)}
    labels.update({tuple(f): model.EPI_ID for f in np.sort(w["epi"], 1)})
    labels.update({tuple(f): model.BASE_ID for f in np.sort(w["bnd"][w["on_base"]], 1)})
    base_nodes = w["base_nodes"][np.isin(w["base_nodes"], cells[myo])]  # myocardial base: the frame of the myo wall
    g = {"points": points, "cells": cells, "facets": np.array(list(labels.keys())),
         "facet_labels": np.array(list(labels.values())), "base_origin": points[base_nodes].mean(0),
         "base_normal": -w["n_out"], **{k: w[k] for k in ("eC", "eL", "eR")}}
    # interface = faces of the myocardial cells that are interior to the whole mesh
    mf = np.sort(np.concatenate([cells[myo][:, q] for q in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])]), 1)
    mf_u, mf_n = np.unique(mf, axis=0, return_counts=True)
    whole = {tuple(f) for f in np.sort(w["bnd"], 1)}
    interface = np.array([f for f in mf_u[mf_n == 1] if tuple(f) not in whole])
    centroids = points[cells].mean(1)
    d_int = model.surface_distance(points, interface, centroids)
    depth = np.empty(len(cells))
    depth[myo] = d_int[myo] / (d_int[myo] + model.surface_distance(points, w["epi"], centroids[myo]))
    d_lum = model.surface_distance(points, w["endo"], centroids[~myo])
    depth[~myo] = d_lum / (d_lum + d_int[~myo])
    f_myo = model.fiber_fields(cfg, w, depth)
    f_endo = model.fiber_fields(endo_settings(cfg), w, depth)
    pick = lambda a, b: np.where(myo[:, None], a, b) if a.ndim == 2 else np.where(myo, a, b)  # noqa: E731
    g.update({"f0": pick(f_myo["f0"], f_endo["f0"]), "s0": f_myo["s0"], "n0": pick(f_myo["n0"], f_endo["n0"]),
              "fibers": np.array([pick(a, b) for a, b in zip(f_myo["fibers"], f_endo["fibers"])]),
              "normals": np.array([pick(a, b) for a, b in zip(f_myo["normals"], f_endo["normals"])]),
              "weights": f_myo["weights"],
              "cell_weights": np.array([np.where(myo, a, b) for a, b in zip(f_myo["weights"], f_endo["weights"])]),
              "region": region, "interface_faces": interface})
    return g
