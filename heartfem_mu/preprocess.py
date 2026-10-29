"""Preprocessing (env CONDA_ENV: numpy, scipy, trimesh, meshio, vtk, gmsh): the raw tracked surfaces -> the
tetrahedral mesh of the wall of the reference frame and its local frame, and the corresponding cut surfaces of every
tracked frame.

Step `preprocess` of every example. Reads INPUT_DIR/t*.stl (closed endocardial and epicardial surfaces of each frame,
one triangle connectivity) and INPUT_DIR/base_plane.json (base plane, apex, reference frame) and writes mesh/:
    t0.vtu, parameter.txt   tetrahedral wall mesh of the reference frame (cut at the base plane) with the cell vectors
                            eC, eL, eR (circumferential, longitudinal, radial) and its frame number
    surface_cut/            (volume_constrain, invert_para) every input frame cut at the base plane with the reference
                            cut topology: t<k>.stl and correspondence.npz (endo / epi labels, provenance)
    preprocess_info.json    mesh statistics
windkessel and unloading read one surface file (the reference frame; unloading: the end-diastolic wall) and write no
surface_cut; unloading also takes a wall already cut at the base plane (one closed surface: endocardium, epicardium and
the flat base in the plane), meshed as it is. mesh/ is OUTPUT_DIR/mesh (invert_para, unloading: CACHE_DIR/mesh).
With USE_SAVED_LANDMARKS (default) the base plane, apex and reference frame come from INPUT_DIR/base_plane.json and no
browser is needed; otherwise the browser app of landmarks.py places them (two-shell surfaces only).
The tetrahedra depend on the number of threads of gmsh (params.MESH_THREADS, set by run.py): it meshes the surface
patches in parallel.
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import meshio
import numpy as np
import trimesh
import vtk
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

# ==================== options (the main settings are in the example's params.py) ====================
SURFACE_ONLY = False  # True: only re-export surface_cut from the saved landmarks.
REMESH_SURFACE = True  # Remesh the wall surface (no slivers); False keeps the input STL triangles.
APEX_RADIUS = 8.0  # Epicardial apex-patch radius, in STL coordinate units.
LRC_SEARCH_LAYERS = 3  # Face-neighbor search depth; 0 disables outlier correction.
LRC_OUTLIER_ANGLE = 45.0  # Replace eL only above this angle (degrees).
LRC_AGREEMENT_ANGLE = 30.0  # Neighbor eL directions must agree within this angle.
LRC_MAJORITY_FRACTION = 0.6  # At least this fraction of eligible neighbors must agree.
LRC_MIN_QUALITY = 0.05  # Reconstruct directions in flatter tetrahedra from connected neighbors.
LRC_REPAIR_PASSES = 3  # Recheck after projection/filling; stop sooner when unchanged.
RADIAL_MAX_TILT = 45.0  # Larger radial adjustments use a neighboring radial reference.
GMSH = "gmsh"  # Executable on PATH, or the full path to your Gmsh executable.


# ==================== 1. the reference surface ====================
def reference_surface(data_dir):
    """windkessel, unloading: the one STL file of the input folder and its frame number (t<k>.stl -> k, any other name
    -> 0)."""
    files = sorted(Path(data_dir).glob("*.stl"))
    if len(files) != 1:
        raise ValueError(f"Expected one STL file (the reference frame) in {data_dir}, found {len(files)}.")
    stem = files[0].stem
    return files[0], int(stem[1:]) if stem[:1] == "t" and stem[1:].isdigit() else 0


# ==================== 2. validate and cut the myocardial wall ====================
def plane_vectors(data):
    """Validate coordinates and return a unit normal without changing STL units."""
    origin = np.asarray(data["origin"], dtype=float)
    normal = np.asarray(data["normal"], dtype=float)
    if origin.shape != (3,) or normal.shape != (3,):
        raise ValueError("Origin and normal must each contain three numbers.")
    length = np.linalg.norm(normal)
    if not np.all(np.isfinite([origin, normal])) or not np.isfinite(length) or length == 0:
        raise ValueError("Plane coordinates must be finite and the normal nonzero.")
    return origin, normal / length


def load_surface_points(path):
    """Weld exact STL duplicates in first-corner order, without reordering triangles."""
    triangles = np.asarray(trimesh.load_mesh(path, process=False).triangles)
    corners = triangles.reshape(-1, 3)
    if not len(corners) or not np.all(np.isfinite(corners)):
        raise ValueError(f"{Path(path).name}: empty or nonfinite surface.")
    _, first, inverse = np.unique(corners, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    mapping = np.empty(len(order), dtype=np.int64)
    mapping[order] = np.arange(len(order))
    first = first[order]
    return corners[first], mapping[inverse].reshape(-1, 3), first


def cut_surface_mapping(points, faces, origin, normal, keep_sign):
    """Cut once; represent every retained/new point by a source edge and fixed fraction."""
    signed = ((points - origin) @ normal) * keep_sign
    tolerance = np.linalg.norm(np.ptp(points, axis=0)) * 1e-10
    signed[np.abs(signed) <= tolerance] = 0
    kept = np.flatnonzero(signed >= 0)
    original_ids = {int(point_id): index for index, point_id in enumerate(kept)}
    sources = [(int(point_id), int(point_id)) for point_id in kept]
    fractions = [0.0] * len(kept)
    crossings = {}
    cut_faces, parents = [], []
    for face_id, face in enumerate(faces):
        polygon = []
        for start, end in zip(face, np.roll(face, -1)):
            if signed[start] >= 0:
                polygon.append(original_ids[int(start)])
            if signed[start] * signed[end] < 0:
                edge = tuple(sorted((int(start), int(end))))
                if edge not in crossings:
                    crossings[edge] = len(sources)
                    sources.append(edge)
                    fractions.append(float(signed[edge[0]] / (signed[edge[0]] - signed[edge[1]])))
                polygon.append(crossings[edge])
        # Preserve source winding, including the two triangles of a clipped quad.
        for corner in range(1, len(polygon) - 1):
            cut_faces.append((polygon[0], polygon[corner], polygon[corner + 1]))
            parents.append(face_id)
    if not cut_faces:
        raise ValueError("The plane removes the whole surface.")
    cut_faces = np.asarray(cut_faces, dtype=np.int64)
    used, compact = np.unique(cut_faces, return_inverse=True)
    return (np.asarray(sources, dtype=np.int64)[used], np.asarray(fractions)[used],
            compact.reshape(-1, 3), np.asarray(parents, dtype=np.int64))


def export_surface_series(data_dir, output_dir, plane_file=None, landmarks=None):
    """Export corresponding open endo/epi surfaces; do not change tetrahedra or volumes."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    saved = json.loads(Path(plane_file).read_text()) if landmarks is None else landmarks
    frame = int(saved["reference_frame"])
    if Path(saved["reference_stl"]).name != f"t{frame}.stl":
        raise ValueError("The saved reference filename and frame do not agree.")
    origin, normal = plane_vectors(saved)
    apex = np.asarray(saved["apex"], dtype=float)
    if apex.shape != (3,) or not np.all(np.isfinite(apex)):
        raise ValueError("The saved plane must include a finite apex point.")
    signed_apex = float((apex - origin) @ normal)
    if abs(signed_apex) <= 1e-10:
        raise ValueError("The saved apex lies on the cut plane.")
    keep_sign = 1 if signed_apex > 0 else -1
    files = sorted((path for path in data_dir.glob("t*.stl") if path.stem[1:].isdigit()),
                   key=lambda path: int(path.stem[1:]))
    if len(files) < 2:
        raise ValueError("At least two STL frames are required.")
    points, faces, first_corners = load_surface_points(data_dir / f"t{frame}.stl")

    # Identify the two original closed shells; the smaller enclosed volume is endocardium.
    reference = trimesh.Trimesh(points, faces, process=False)
    components = trimesh.graph.connected_components(reference.face_adjacency, nodes=np.arange(len(faces)))
    if len(components) != 2 or not reference.is_watertight or not reference.is_winding_consistent:
        raise ValueError("Expected two closed, consistently wound source surfaces.")
    components = sorted(components, key=lambda ids: abs(trimesh.Trimesh(points, faces[ids], process=False).volume))
    source_labels = np.zeros(len(faces), dtype=np.uint8)
    for label, face_ids in enumerate(components, 1):
        source_labels[face_ids] = label
    sources, fractions, cut_faces, parents = cut_surface_mapping(points, faces, origin, normal, keep_sign)
    labels = source_labels[parents]
    reference_cut = points[sources[:, 0]] * (1 - fractions[:, None]) + points[sources[:, 1]] * fractions[:, None]

    # Require one open cut loop per wall surface. No artificial cap is added.
    for label in (1, 2):
        shell = trimesh.Trimesh(reference_cut, cut_faces[labels == label], process=False)
        edges, counts = np.unique(np.sort(shell.edges, axis=1), axis=0, return_counts=True)
        boundary = edges[counts == 1]
        if not len(boundary) or np.any(counts > 2):
            raise ValueError("The plane must make an open, manifold cut through both surfaces.")
        nodes, inverse = np.unique(boundary, return_inverse=True)
        boundary = inverse.reshape(-1, 2)
        graph = coo_matrix((np.ones(2 * len(boundary)), (
            boundary.ravel(), boundary[:, ::-1].ravel(),
        )), shape=(len(nodes), len(nodes))).tocsr()
        if np.any(np.bincount(boundary.ravel()) != 2) or connected_components(graph)[0] != 1:
            raise ValueError("Choose a plane with one closed cut loop on each surface.")

    if output_dir.resolve() == data_dir.resolve():
        raise ValueError("The cut output directory must differ from the source directory.")
    if output_dir.exists() and any(path.name != "correspondence.npz" and not (
            path.is_file() and path.suffix == ".stl" and path.stem.startswith("t") and path.stem[1:].isdigit()
    ) for path in output_dir.iterdir()):
        raise ValueError("The surface_cut directory contains unrelated files; keep them elsewhere.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".surface-cut-") as temporary:
        staged = Path(temporary) / "new"
        staged.mkdir()
        for path in files:
            current, current_faces, _ = load_surface_points(path)
            if current.shape != points.shape or not np.array_equal(current_faces, faces):
                raise ValueError(f"{path.name}: ordered point/triangle correspondence does not match the reference.")
            # Keep the reference edge fractions and topology; do not fit and re-cut a new plane.
            mapped = current[sources[:, 0]] * (1 - fractions[:, None]) + current[sources[:, 1]] * fractions[:, None]
            stored = mapped.astype(np.float32)
            triangles = stored[cut_faces]
            areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
            if np.any(areas == 0) or len(np.unique(stored, axis=0)) != len(stored):
                raise ValueError(f"{path.name}: a cut element or point collapses at STL precision.")
            trimesh.Trimesh(stored, cut_faces, process=False).export(staged / path.name)

        # STL has no persistent IDs. Keep exact provenance and shared connectivity separately.
        np.savez_compressed(staged / "correspondence.npz", frame_ids=[int(path.stem[1:]) for path in files],
                            reference_frame=frame, source_vertex_corners=first_corners, source_edges=sources,
                            edge_fraction=fractions, faces=cut_faces, source_face_ids=parents,
                            surface_labels=labels, reference_points=reference_cut,
                            plane_origin=origin, plane_normal=normal, keep_sign=keep_sign)
        # Validate every frame before replacing an older export; retain it on failure.
        previous = Path(temporary) / "previous"
        if output_dir.exists():
            output_dir.rename(previous)
        try:
            staged.rename(output_dir)
        except OSError:
            if previous.exists():
                previous.rename(output_dir)
            raise
    return {"frames": len(files), "source_points": len(points), "cut_points": len(reference_cut),
            "cut_triangles": len(cut_faces), "reference_frame": frame}


def polydata(mesh):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(
        np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).astype(np.int64).ravel(), deep=True))
    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)
    return surface


def cut_wall(shells, origin, normal, keep_sign):
    """Join the cut inner and outer surfaces with an annulus, not a cavity disk."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal * keep_sign)
    surfaces = []
    boundaries = vtk.vtkAppendPolyData()
    for shell in shells:
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(polydata(shell))
        clip.SetClipFunction(plane)
        # Clipping leaves quads along the cut; split them so every face is a triangle.
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(clip.GetOutputPort())
        triangulate.Update()
        surface = triangulate.GetOutput()
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(surface)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        if not surface.GetNumberOfCells() or not edges.GetOutput().GetNumberOfCells():
            raise ValueError("The base plane must cross both inner and outer surfaces.")
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputConnection(edges.GetOutputPort())
        connectivity.SetExtractionModeToAllRegions()
        connectivity.Update()
        if connectivity.GetNumberOfExtractedRegions() != 1:
            raise ValueError("Choose a plane with one closed cut loop on each surface.")
        surfaces.append(surface)
        boundaries.AddInputData(edges.GetOutput())

    boundaries.Update()
    cap = vtk.vtkContourTriangulator()
    cap.SetInputConnection(boundaries.GetOutputPort())
    cap.Update()
    append = vtk.vtkAppendPolyData()
    for surface in surfaces:
        append.AddInputData(surface)
    append.AddInputData(cap.GetOutput())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(append.GetOutputPort())
    clean.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    closed = normals.GetOutput()
    faces = vtk_to_numpy(closed.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    wall = trimesh.Trimesh(vtk_to_numpy(closed.GetPoints().GetData()), faces, process=True)
    if not wall.is_watertight or len(wall.split()) != 1:
        raise ValueError("The cut wall is not a single watertight solid; adjust the plane.")
    return wall, surfaces


def precut_surfaces(wall, origin, normal, scale):
    """A wall already cut at the base plane (one closed surface): its inner and outer side surfaces (as cut_wall
    returns them; the inner one has the smaller area), i.e. every face that is not in the plane."""
    on_base = np.abs((wall.vertices[wall.faces] - origin) @ normal).max(axis=1) < scale * 1e-5
    if not on_base.any():
        raise ValueError("One closed surface is read as a wall already cut at the base plane: its base must lie in "
                         "the plane.")
    parts = trimesh.Trimesh(wall.vertices, wall.faces[~on_base], process=False).split(only_watertight=False)
    if len(parts) != 2:
        raise ValueError(f"The cut wall must have an inner and an outer surface besides its base, found {len(parts)}.")
    return [polydata(part) for part in sorted(parts, key=lambda part: part.area)]


# ==================== 3. generate tetrahedra with Gmsh ====================
def tetrahedralize(wall, directory, mesh_size, gmsh, remesh=REMESH_SURFACE):
    wall.export(directory / "wall.stl")
    geometry = directory / "wall.geo"
    if remesh:
        # The cut edges and base annulus leave tiny, thin input triangles, and tetrahedra attached to
        # them are slivers whatever the mesh size. Split the surface at the sharp base rims into
        # endocardium, epicardium and base, reparametrize, remesh at a uniform size and optimize the
        # tetrahedra. New surface nodes lie on the input triangles.
        surface = f'''Merge "wall.stl";
ClassifySurfaces{{40 * Pi / 180, 1, 0, Pi}};
CreateGeometry;
Surface Loop(1) = Surface{{:}};
Volume(1) = {{1}};
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.OptimizeNetgen = 1;
'''
    else:
        surface = '''Merge "wall.stl";
surfaces[] = Surface "*";
Surface Loop(1) = {surfaces[]};
Volume(1) = {1};
'''
    geometry.write_text(surface + f'''Mesh.MeshSizeMin = {mesh_size};
Mesh.MeshSizeMax = {mesh_size};
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.MshFileVersion = 2.2;
''')
    result = subprocess.run(
        [gmsh, "wall.geo", "-3", "-o", "wall.msh", "-v", "2"],
        cwd=directory, capture_output=True, text=True, timeout=300,
    )
    if result.returncode or not (directory / "wall.msh").exists():
        raise ValueError("Gmsh failed. Adjust the cut or mesh size. " + (result.stderr + result.stdout)[-1200:])
    mesh = meshio.read(directory / "wall.msh")
    cells = mesh.get_cells_type("tetra")
    triangles = mesh.get_cells_type("triangle")
    if not len(cells) or not len(triangles):
        raise ValueError("Gmsh did not produce tetrahedra and boundary triangles.")
    # Remove unused surface/geometry nodes before assembling the volume problem.
    used = np.unique(cells)
    mapping = np.full(len(mesh.points), -1, dtype=int)
    mapping[used] = np.arange(len(used))
    triangles = mapping[triangles]
    if np.any(triangles < 0):
        raise ValueError("Gmsh produced a boundary disconnected from the volume.")
    return mesh.points[used], mapping[cells], triangles


# ==================== 4. label boundaries and solve LRC ====================
def boundary_masks(points, triangles, surfaces, origin, normal, scale):
    """Label boundary nodes: 1=endo, 2=epi, 4=base; intersections retain both bits."""
    centers = points[triangles].mean(axis=1)
    on_base = np.max(np.abs((points[triangles] - origin) @ normal), axis=1) < scale * 1e-6
    locators = []
    for surface in surfaces:
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(surface)
        locator.BuildLocator()
        locators.append(locator)
    labels = np.full(len(triangles), 4, dtype=np.uint8)
    for index in np.flatnonzero(~on_base):
        distances = []
        for locator in locators:
            closest = [0.0, 0.0, 0.0]
            cell_id, sub_id, distance = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
            locator.FindClosestPoint(centers[index], closest, cell_id, sub_id, distance)
            distances.append(float(distance))
        labels[index] = 1 if distances[0] < distances[1] else 2
    masks = np.zeros(len(points), dtype=np.uint8)
    for label in (1, 2, 4):
        masks[np.unique(triangles[labels == label])] |= label
    if any(not np.any(masks & label) for label in (1, 2, 4)) or np.any((masks & 3) == 3):
        raise ValueError("Could not distinguish endocardium, epicardium, and base.")
    return masks, labels


def wall_normals(points, cells, triangles, labels):
    """Area-weighted wall normals per boundary tetrahedron, oriented endo -> epi."""
    face_nodes = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces = cells[:, face_nodes].reshape(-1, 3)
    _, inverse = np.unique(np.sort(np.vstack((faces, triangles)), axis=1),
                           axis=0, return_inverse=True)
    owners = np.full(inverse.max() + 1, -1, dtype=int)
    owners[inverse[:len(faces)]] = np.repeat(np.arange(len(cells)), 4)
    boundary_owners = owners[inverse[len(faces):]]
    if np.any(boundary_owners < 0):
        raise ValueError("A boundary triangle has no tetrahedral owner.")

    # The artificial base cap is not a myocardial wall and must not constrain eL.
    wall = labels != 4
    boundary_owners = boundary_owners[wall]
    vertices = points[triangles[wall]]
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    outward = vertices.mean(axis=1) - points[cells[boundary_owners]].mean(axis=1)
    normals[np.sum(normals * outward, axis=1) < 0] *= -1
    normals[labels[wall] == 1] *= -1
    combined = np.zeros((len(cells), 3))
    np.add.at(combined, boundary_owners, normals)
    lengths = np.linalg.norm(combined, axis=1)
    if np.any(lengths[np.unique(boundary_owners)] <= 1e-15):
        raise ValueError("Wall normals cancel; inspect the boundary tetrahedra.")
    return combined / np.maximum(lengths[:, None], 1e-15)


def project_directions(vectors, normals):
    """Remove the normal component; zero normals leave interior vectors unconstrained."""
    projected = vectors - np.sum(vectors * normals, axis=-1)[..., None] * normals
    lengths = np.linalg.norm(projected, axis=-1)
    projected /= np.maximum(lengths[..., None], 1e-15)
    projected[lengths <= 1e-8] = 0
    return projected, lengths


def solve_lrc(points, cells, masks, apex, apex_radius):
    """Solve two P1 Laplace problems and recover stable gradient references."""
    scale = np.linalg.norm(np.ptp(points, axis=0))
    coordinates = (points - points.mean(axis=0)) / scale
    edges = coordinates[cells[:, 1:]] - coordinates[cells[:, :1]]
    determinants = np.linalg.det(edges)
    if np.any(np.abs(determinants) < 1e-15):
        raise ValueError("The tetrahedral mesh contains degenerate elements.")
    gradients = np.empty((len(cells), 4, 3))
    gradients[:, 1:] = np.linalg.inv(edges).transpose(0, 2, 1)
    gradients[:, 0] = -gradients[:, 1:].sum(axis=1)
    volumes = np.abs(determinants) / 6
    stiffness = np.einsum("nig,njg,n->nij", gradients, gradients, volumes)
    matrix = coo_matrix((stiffness.ravel(), (
        np.repeat(cells, 4, axis=1).ravel(), np.tile(cells, (1, 4)).ravel(),
    )), shape=(len(points), len(points))).tocsr()

    def laplace(zero_nodes, one_nodes):
        if not len(zero_nodes) or not len(one_nodes) or np.intersect1d(zero_nodes, one_nodes).size:
            raise ValueError("The Laplace boundary patches are empty or overlap.")
        values = np.zeros(len(points))
        values[one_nodes] = 1
        fixed = np.union1d(zero_nodes, one_nodes)
        free = np.setdiff1d(np.arange(len(points)), fixed)
        values[free] = spsolve(matrix[free][:, free], -(matrix[free][:, fixed] @ values[fixed]))
        if not np.all(np.isfinite(values)):
            raise ValueError("The Laplace solve failed; check mesh connectivity.")
        return values

    endo = np.flatnonzero(masks & 1)
    epi = np.flatnonzero(masks & 2)
    base = np.flatnonzero(masks & 4)
    distances = np.linalg.norm(points[epi] - apex, axis=1)
    apex_nodes = epi[distances <= max(apex_radius, distances.min() * 1.01)]
    masks[apex_nodes] |= 8
    # R: endocardium -> epicardium. L: apex patch -> base, with natural other boundaries.
    phi_r = laplace(endo, epi)
    phi_l = laplace(apex_nodes, base)
    radial = np.einsum("ni,nij->nj", phi_r[cells], gradients)
    longitudinal = np.einsum("ni,nij->nj", phi_l[cells], gradients)
    # Recover gradients through volume-weighted vertex patches BEFORE normalization.
    # Tiny cut tetrahedra must not dominate a direction just because it is normalized.
    weights = np.bincount(cells.ravel(), weights=np.repeat(volumes, 4), minlength=len(points))
    for vectors in (longitudinal, radial):
        recovered = np.zeros_like(points)
        for corner in range(4):
            np.add.at(recovered, cells[:, corner], vectors * volumes[:, None])
        vectors[:] = (recovered / weights[:, None])[cells].mean(axis=1)
    radial_norm = np.linalg.norm(radial, axis=1)
    radial /= np.maximum(radial_norm[:, None], 1e-15)
    radial[radial_norm <= 1e-10] = 0
    # Keep the positive phi_L gradient as the apex -> base sign reference.
    return {"eL": longitudinal.copy(), "eR": radial}, longitudinal


def tetra_graph(points, cells):
    """Face-connected graph weighted by center distance; never jump across the cavity."""
    face_nodes = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces = np.sort(cells[:, face_nodes].reshape(-1, 3), axis=1)
    _, first, inverse, counts = np.unique(
        faces, axis=0, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    shared = np.flatnonzero(counts == 2)
    starts = np.cumsum(counts) - counts
    owners = first[shared] // 4
    partners = order[starts[shared] + 1] // 4
    centers = points[cells].mean(axis=1)
    distances = np.maximum(np.linalg.norm(centers[owners] - centers[partners], axis=1), 1e-15)
    return coo_matrix((np.tile(distances, 2), (
        np.concatenate((owners, partners)), np.concatenate((partners, owners)),
    )), shape=(len(cells), len(cells))).tocsr()


def correct_lrc_frames(points, cells, directions, valid, quality, graph=None):
    """Check only eL across all tetrahedra, keeping the local radial references unchanged."""
    if LRC_SEARCH_LAYERS <= 0:
        return 0
    if not 0.5 < LRC_MAJORITY_FRACTION <= 1:
        raise ValueError("LRC_MAJORITY_FRACTION must be greater than 0.5 and at most 1.")
    reliable = valid & (quality >= LRC_MIN_QUALITY)
    centers = points[cells].mean(axis=1)
    adjacency = tetra_graph(points, cells) if graph is None else graph
    agreement = np.cos(np.deg2rad(LRC_AGREEMENT_ANGLE))
    threshold = np.cos(np.deg2rad(LRC_OUTLIER_ANGLE))
    replacements = []

    for cell_id in range(len(cells)):
        visited = {int(cell_id)}
        frontier = {int(cell_id)}
        for _ in range(LRC_SEARCH_LAYERS):
            nearby = set()
            for neighbor_id in frontier:
                nearby.update(adjacency.indices[
                    adjacency.indptr[neighbor_id]:adjacency.indptr[neighbor_id + 1]])
            frontier = nearby - visited
            visited.update(frontier)
            if not frontier:
                break
        # Exclude the target itself, even when its original frame is valid.
        donors = np.array(sorted(visited - {cell_id}), dtype=int)
        donors = donors[reliable[donors]]
        if len(donors) < 3:
            continue

        # Seed the dominant eL group with the neighbor having the most angular support.
        vectors = directions["eL"][donors]
        support = vectors @ vectors.T >= agreement
        group = support[np.argmax(support.sum(axis=1))].copy()
        required = max(3, int(np.ceil(LRC_MAJORITY_FRACTION * len(donors))))
        # Remove fringe directions until the whole group agrees with its own mean.
        while np.sum(group) >= required:
            mean = vectors[group].mean(axis=0)
            mean /= np.linalg.norm(mean)
            refined = group & (vectors @ mean >= agreement)
            if np.array_equal(refined, group):
                break
            group = refined
        if np.sum(group) < required:
            continue
        if valid[cell_id] and directions["eL"][cell_id] @ mean >= threshold:
            continue
        # Copy only from the agreeing majority, never from a closer rejected outlier.
        donors = donors[group]
        donor = donors[np.argmin(np.linalg.norm(centers[donors] - centers[cell_id], axis=1))]
        replacements.append((cell_id, donor))

    # Apply simultaneously from original frames, so corrections never propagate in a pass.
    if replacements:
        targets, donors = np.asarray(replacements).T
        directions["eL"][targets] = directions["eL"][donors]
        valid[targets] = True
    return len(replacements)


def fill_invalid_directions(points, cells, vectors, valid, normals, graph=None):
    """Harmonically fill invalid patches along shared faces, then apply target constraints."""
    donors = np.flatnonzero(valid)
    if not len(donors):
        raise ValueError("No valid directions are available for filling.")
    missing = np.flatnonzero(~valid)
    count = len(missing)
    if not count:
        return 0
    graph = tetra_graph(points, cells) if graph is None else graph
    distances = dijkstra(graph, indices=donors, min_only=True)
    if not np.all(np.isfinite(distances[missing])):
        raise ValueError("An invalid mesh component has no connected valid directions.")
    adjacency = graph.copy()
    adjacency.data[:] = 1
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    interpolated = spsolve(laplacian[missing][:, missing], -(laplacian[missing][:, donors] @ vectors[donors]))
    candidates, lengths = project_directions(interpolated, normals[missing])
    # Cancellation at a singularity: use the nearest usable CONNECTED donor, not a global KD-tree.
    for row in np.flatnonzero(lengths <= 1e-8):
        cell_id = missing[row]
        distances = dijkstra(graph, indices=cell_id)
        nearby = donors[np.argsort(distances[donors])]
        projected, sizes = project_directions(vectors[nearby], normals[cell_id])
        usable = (sizes > 1e-8) & np.isfinite(distances[nearby])
        if not np.any(usable):
            raise ValueError("No connected neighbor has a usable direction in the target plane.")
        candidates[row] = projected[np.flatnonzero(usable)[0]]
    if not np.all(np.isfinite(candidates)):
        raise ValueError("Direction interpolation failed; inspect mesh connectivity.")
    vectors[missing] = candidates
    valid[missing] = True
    return count


def finalize_lrc(points, cells, directions, normals, graph=None):
    """Keep repaired eL primary, adjust the local radial reference, then rebuild eC."""
    if not 0 < RADIAL_MAX_TILT < 90:
        raise ValueError("RADIAL_MAX_TILT must be between 0 and 90 degrees.")
    # Reproject copied eL at its new location, not at the donor's wall tangent plane.
    longitudinal, lengths = project_directions(directions["eL"], normals)
    valid = lengths > 1e-8
    filled = fill_invalid_directions(points, cells, longitudinal, valid, normals, graph)

    reference = directions["eR"].copy()
    on_wall = np.linalg.norm(normals, axis=1) > 0.5
    reference[on_wall] = normals[on_wall]
    radial, lengths = project_directions(reference, longitudinal)
    reliable = lengths >= np.cos(np.deg2rad(RADIAL_MAX_TILT))
    fallback = fill_invalid_directions(points, cells, radial, reliable, longitudinal, graph)
    # Preserve the local endo -> epi sign where a nonzero reference is available.
    radial[np.sum(radial * reference, axis=1) < 0] *= -1
    directions.update(eL=longitudinal, eR=radial, eC=np.cross(longitudinal, radial))
    return filled, fallback


def repair_lrc(points, cells, directions, normals, quality, longitudinal_reference):
    """Repair weak patches, anchor eL signs, and recheck the FINAL projected directions."""
    if not 0 <= LRC_MIN_QUALITY < 1 or LRC_REPAIR_PASSES < 0:
        raise ValueError("Use 0 <= LRC_MIN_QUALITY < 1 and nonnegative LRC_REPAIR_PASSES.")
    graph = tetra_graph(points, cells)
    radial_reference = directions["eR"].copy()
    constraints = radial_reference.copy()
    on_wall = np.linalg.norm(normals, axis=1) > 0.5
    constraints[on_wall] = normals[on_wall]
    directions["eL"], _ = project_directions(directions["eL"], constraints)
    poor = quality < LRC_MIN_QUALITY
    directions["eL"][poor] = 0
    radial_reference[poor] = 0
    reference_lengths = np.linalg.norm(longitudinal_reference, axis=1)
    reference = longitudinal_reference / np.maximum(reference_lengths[:, None], 1e-15)
    corrected = np.zeros(len(cells), dtype=bool)
    reoriented = np.zeros(len(cells), dtype=bool)
    filled_total = 0
    radial_fallback = 0

    # Initial filling, then bounded correction passes. Reuse the original radial reference.
    for pass_id in range(LRC_REPAIR_PASSES + 1):
        if pass_id:
            previous = directions["eL"].copy()
            changes = correct_lrc_frames(points, cells, directions, np.ones(len(cells), dtype=bool), quality, graph)
            corrected |= np.any(previous != directions["eL"], axis=1)
        directions["eR"] = radial_reference.copy()
        filled, radial_fallback = finalize_lrc(points, cells, directions, normals, graph)
        filled_total += filled
        reverse = (reference_lengths > 1e-8) & (np.sum(directions["eL"] * reference, axis=1) < -1e-8)
        directions["eL"][reverse] *= -1
        directions["eC"] = np.cross(directions["eL"], directions["eR"])
        reoriented |= reverse
        if pass_id and not changes and not filled and not np.any(reverse):
            break

    # Report unresolved local jumps rather than equating orthogonality with correctness.
    adjacency = graph.copy()
    adjacency.data[:] = 1
    mean = adjacency @ directions["eL"]
    lengths = np.linalg.norm(mean, axis=1)
    mean /= np.maximum(lengths[:, None], 1e-15)
    cosine = np.sum(directions["eL"] * mean, axis=1)
    for values in directions.values():
        if not np.all(np.isfinite(values)) or not np.allclose(np.linalg.norm(values, axis=1), 1):
            raise ValueError("Invalid final direction frame; inspect the mesh and landmarks.")
    if not np.allclose(np.sum(directions["eL"] * directions["eR"], axis=1), 0, atol=1e-8):
        raise ValueError("Final longitudinal and radial directions are not orthogonal.")
    return {
        "corrected_lrc_frames": int(corrected.sum()), "filled_lrc": filled_total,
        "radial_fallback": radial_fallback, "reoriented_lrc_frames": int(reoriented.sum()),
        "remaining_lrc_outliers": int(np.sum((lengths > 1e-8) & (cosine < np.cos(np.deg2rad(LRC_OUTLIER_ANGLE))))),
        "opposed_neighbor_frames": int(np.sum((lengths > 1e-8) & (cosine < 0))),
    }


# ==================== 5. export the confirmed reference ====================
def export_reference(reference_stl, frame, origin, normal, apex, output_dir,
                     mesh_size, apex_radius, gmsh):
    """Return UI diagnostics only after both requested output files are ready."""
    if not np.isfinite(mesh_size) or mesh_size <= 0 or not np.isfinite(apex_radius) or apex_radius <= 0:
        raise ValueError("Mesh size and apex radius must be positive and finite.")
    mesh = trimesh.load_mesh(reference_stl, process=True)
    shells = sorted(mesh.split(only_watertight=False), key=lambda shell: abs(shell.volume))
    precut = len(shells) == 1 and shells[0].is_watertight  # a wall already cut at the base plane (unloading)
    if not precut and (len(shells) != 2 or not all(shell.is_watertight for shell in shells)):
        raise ValueError("Two closed inner/outer surfaces (or one closed wall cut at the base plane) are required.")
    apex = np.asarray(apex, dtype=float)
    scale = np.linalg.norm(mesh.extents)
    if apex.shape != (3,) or not np.all(np.isfinite(apex)):
        raise ValueError("Select a finite apex point on the outer surface.")
    _, distance, _ = trimesh.proximity.closest_point_naive(shells[-1], apex[None])
    if distance[0] > scale * 1e-5:
        raise ValueError("Select the apex on the outer surface.")
    signed_apex = float(np.dot(apex - origin, normal))
    if abs(signed_apex) < apex_radius:
        raise ValueError("Move the base plane farther from the apex patch.")
    keep_sign = 1 if signed_apex > 0 else -1
    if precut:
        wall, surfaces = shells[0], precut_surfaces(shells[0], origin, normal, scale)
    else:
        wall, surfaces = cut_wall(shells, origin, normal, keep_sign)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".mesh-") as temporary:
        directory = Path(temporary)
        points, cells, triangles = tetrahedralize(wall, directory, mesh_size, gmsh)
        # The remeshed flat base is within round-off of the cut plane; put it exactly on the plane.
        distance = (points - origin) @ normal
        on_plane = np.abs(distance) < scale * 1e-5
        points[on_plane] -= distance[on_plane, None] * normal
        volumes = np.abs(np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]])) / 6
        if abs(volumes.sum() - abs(wall.volume)) > 0.05 * abs(wall.volume):
            raise ValueError("Tetrahedral volume does not match the cut wall volume.")
        masks, labels = boundary_masks(points, triangles, surfaces, origin, normal, scale)
        cell_data, longitudinal_reference = solve_lrc(points, cells, masks, apex, apex_radius)
        normals = wall_normals(points, cells, triangles, labels)
        edge_pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        edge_lengths_squared = np.sum((points[cells[:, edge_pairs[:, 0]]]
                                       - points[cells[:, edge_pairs[:, 1]]]) ** 2, axis=2).sum(axis=1)
        quality = 12 * (3 * volumes) ** (2 / 3) / edge_lengths_squared
        diagnostics = repair_lrc(points, cells, cell_data, normals, quality, longitudinal_reference)
        # Build the final VTU directly: no intermediate VTU write/read round trip.
        grid = vtk.vtkUnstructuredGrid()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True))
        grid.SetPoints(vtk_points)
        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(cells), numpy_to_vtkIdTypeArray(
            np.column_stack((np.full(len(cells), 4), cells)).astype(np.int64).ravel(), deep=True))
        grid.SetCells(vtk.VTK_TETRA, vtk_cells)
        # Export only the three cell vectors. Diagnostics remain internal.
        for name, values in cell_data.items():
            array = numpy_to_vtk(values, deep=True)
            array.SetName(name)
            grid.GetCellData().AddArray(array)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(directory / "t0.vtu"))
        writer.SetInputData(grid)
        if writer.Write() != 1:
            raise OSError("Could not write the VTU file.")
        (directory / "parameter.txt").write_text(f"unloaded_time_point = {frame}\n")
        (directory / "t0.vtu").replace(output_dir / "t0.vtu")
        (directory / "parameter.txt").replace(output_dir / "parameter.txt")
    return {
        "tetrahedra": len(cells), "invalid_lrc": 0,
        "low_quality_tetrahedra": int(np.sum(quality < LRC_MIN_QUALITY)), **diagnostics,
    }


# ==================== 6. run preprocessing ====================
def main(cfg):
    start = time.time()
    data_dir, out, plane_file = cfg.input, cfg.mesh, cfg.input / "base_plane.json"
    series = cfg.TASK not in ("windkessel", "unloading")  # every frame -> surface_cut (volume_constrain, invert_para)
    if series and SURFACE_ONLY:
        if not plane_file.is_file():
            raise FileNotFoundError("No saved base plane. Set SURFACE_ONLY = False to select it in the app first.")
        print("Checking correspondence and exporting cut surfaces...", flush=True)
        surfaces = export_surface_series(data_dir, out / "surface_cut", plane_file)
        print(f"Saved {surfaces['frames']} corresponding frames to {out / 'surface_cut'}", flush=True)
        # The mesh is unchanged, so preprocess_info.json (mesh statistics) is kept as is.
        print(f"surface export done in {time.time() - start:.1f} s", flush=True)
        return
    # Fail early with a clear message rather than after the user finishes annotation.
    if shutil.which(GMSH) is None:
        raise FileNotFoundError("Gmsh is not installed or GMSH is not on PATH.")
    if not cfg.USE_SAVED_LANDMARKS:
        from . import landmarks
        landmarks.serve(cfg, out, series)
        return
    # Same exports as the app's "Confirm apex", from the saved landmarks (no browser).
    saved = json.loads(plane_file.read_text())
    if series:
        reference_stl, frame = data_dir / saved["reference_stl"], int(saved["reference_frame"])
    else:
        reference_stl, frame = reference_surface(data_dir)
        if Path(saved["reference_stl"]).name != reference_stl.name:
            raise ValueError(f"The saved landmarks belong to {saved['reference_stl']}, the input surface is {reference_stl.name}.")
    origin, normal = plane_vectors(saved)
    print(f"Saved landmarks: reference {reference_stl.name}; meshing at size {cfg.MESH_SIZE}...", flush=True)
    result = export_reference(reference_stl, frame, origin, normal, np.asarray(saved["apex"], dtype=float), out,
                              cfg.MESH_SIZE, APEX_RADIUS, GMSH)
    print(f"Saved {out / 't0.vtu'}: {result}", flush=True)
    info = {"mesh_size": cfg.MESH_SIZE, **result}
    if series:
        surfaces = export_surface_series(data_dir, out / "surface_cut", plane_file)
        print(f"Saved {surfaces['frames']} corresponding frames to {out / 'surface_cut'}", flush=True)
        info["frames"] = surfaces["frames"]
    info["time_s"] = round(time.time() - start, 1)
    (out / "preprocess_info.json").write_text(json.dumps(info, indent=1))
    print(f"preprocess done in {info['time_s']} s", flush=True)
