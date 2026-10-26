"""Browser app that places the base plane and the apex on the reference frame (preprocess.py with
USE_SAVED_LANDMARKS = False): http://127.0.0.1:8765, English, offline. On "Confirm apex" it writes the same exports as
preprocess.py with saved landmarks and overwrites INPUT_DIR/base_plane.json (hand-placed landmarks are not recomputable).
The HTML and JavaScript are embedded here; only the third-party rendering libraries live in data/assets/.
"""

import json
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import trimesh

from . import preprocess as pp

r = 1 / 3  # reference-frame heuristic: the frame whose cavity volume is r of the way from the minimum to the maximum
REFERENCE_FRAME = None  # None: select by volume; an integer: use that original frame.
INITIAL_NORMAL = (0.0, 0.0, 1.0)
HOST = "127.0.0.1"
PORT = 8765
ASSET_DIR = Path(__file__).resolve().parent / "data" / "assets"


# ==================== 1. select the reference frame ====================
def select_reference_frame(data_dir, r=r):
    """Return the approximate unloaded frame ID without writing any files."""
    if not 0 <= r <= 1:
        raise ValueError("r must be between 0 and 1.")

    # Sort by numeric frame ID and measure the cavity inside the smaller shell.
    files = sorted(
        (file for file in Path(data_dir).glob("t*.stl") if file.stem[1:].isdigit()),
        key=lambda file: int(file.stem[1:]),
    )
    if len(files) < 2:
        raise ValueError("At least two STL frames are required.")

    volumes = []
    for file in files:
        shells = trimesh.load_mesh(file, process=True).split(only_watertight=False)
        if len(shells) != 2 or not all(
            shell.is_watertight and shell.is_winding_consistent for shell in shells
        ):
            raise ValueError(f"{file.name}: expected two closed, consistently wound shells.")
        volumes.append(min(abs(shell.volume) for shell in shells))
    volumes = np.asarray(volumes)
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0) or np.ptp(volumes) == 0:
        raise ValueError("Volumes must be positive, finite, and nonconstant.")

    # Follow the cycle from the minimum to the next maximum, including wraparound.
    order = (np.arange(len(files)) + np.argmin(volumes)) % len(files)
    branch = order[:np.argmax(volumes[order]) + 1]

    # Match volume, not frame index. This is a reference heuristic, not zero stress.
    target_volume = volumes.min() + r * np.ptp(volumes)
    selected_index = branch[np.argmin(np.abs(volumes[branch] - target_volume))]
    return int(files[selected_index].stem[1:])


# ==================== 2. browser UI ====================
# The HTML and JavaScript are embedded here. Only the third-party rendering
# libraries live in assets/. Plane edits stay in the browser until confirmation.
PAGE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Base Plane</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px system-ui, sans-serif; color: #293749; background: #f5f7fa; }
  header { position: fixed; left: 0; right: 0; padding: 16px 24px; display: flex;
    align-items: center; gap: 12px; z-index: 1; pointer-events: none; }
  header { top: 0; } .spacer { flex: 1; }
  #status { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    max-width: 90vw; padding: 10px 16px; background: white; border-radius: 7px; }
  #status:empty { display: none; }
  button { pointer-events: auto; border: 0; border-radius: 7px; background: #246bce;
    color: white; padding: 10px 24px; cursor: pointer; }
  button:hover { background: #1c59ad; } button:disabled { opacity: .5; cursor: wait; }
  #back { background: white; color: #293749; }
  #viewport { width: 100vw; height: 100dvh; touch-action: none; }
  canvas { display: block; }
</style>
<header><strong id="stage">1 · Plane</strong><span id="reference"></span><span class="spacer"></span>
  <button id="back" hidden>Back</button><button id="save" disabled>Confirm plane</button></header>
<main id="viewport" aria-label="Transparent heart wall, base plane, and apex"></main>
<div id="status" role="status">Loading...</div>
<script type="importmap">{"imports": {"three": "/three.module.min.js"}}</script>
<script type="module">
const status = document.querySelector('#status');
const confirmButton = document.querySelector('#save');
const backButton = document.querySelector('#back');
try {
  const THREE = await import('three');
  const { OrbitControls } = await import('/OrbitControls.js');
  const response = await fetch('/api/scene');
  if (!response.ok) throw new Error('Could not load the reference frame.');
  const data = await response.json();
  const viewport = document.querySelector('#viewport');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.localClippingEnabled = true;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  viewport.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#f5f7fa');
  scene.add(new THREE.HemisphereLight(0xffffff, 0x566782, 2));
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(1, 2, 3);
  scene.add(light);

  // Draw the inner surface first, then the transparent outer wall.
  // Separate geometries also ensure apex picking never selects the endocardium.
  const keptClip = new THREE.Plane(), removedClip = new THREE.Plane();
  const surfaces = data.surfaces.map((surface, index) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(surface.vertices, 3));
    geometry.setIndex(surface.faces);
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    const material = new THREE.MeshStandardMaterial({
      color: index === 0 ? 0xdb947c : 0x80b3ce, roughness: 0.65,
      transparent: true, opacity: index === 0 ? 0.64 : 0.2,
      depthWrite: false, side: THREE.DoubleSide, clippingPlanes: [keptClip]
    });
    const kept = new THREE.Mesh(geometry, material);
    const removed = new THREE.Mesh(geometry, material.clone());
    removed.material.color.set(0x92969c);
    removed.material.opacity = index === 0 ? 0.3 : 0.12;
    removed.material.clippingPlanes = [removedClip];
    kept.renderOrder = removed.renderOrder = index + 1;
    scene.add(kept, removed);
    return kept;
  });
  const outer = surfaces[1];
  const radius = outer.geometry.boundingSphere.radius;
  const center = new THREE.Vector3().fromArray(data.initial_origin);
  const camera = new THREE.PerspectiveCamera(40, 1, radius / 1000, radius * 100);
  camera.position.copy(center).add(new THREE.Vector3(1.5, 1, 3.8).multiplyScalar(radius));
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.copy(center);
  orbit.enableDamping = true;
  orbit.minDistance = radius * 0.3;
  orbit.maxDistance = radius * 15;
  orbit.update();

  const plane = new THREE.Mesh(new THREE.PlaneGeometry(radius * 2.5, radius * 2.5),
    new THREE.MeshBasicMaterial({ color: 0x20b6b6, transparent: true, opacity: 0.1,
      side: THREE.DoubleSide, depthWrite: false }));
  plane.add(new THREE.LineSegments(new THREE.EdgesGeometry(plane.geometry),
    new THREE.LineBasicMaterial({ color: 0x117b88 })));
  scene.add(plane);
  const normalAxis = new THREE.Vector3(0, 0, 1);
  const normal = new THREE.Vector3();
  const arrow = new THREE.ArrowHelper(normalAxis, center, radius * 1.05, 0x117b88);
  arrow.traverse(object => {
    if (object.material) { object.material.depthTest = false; object.material.transparent = true; }
    object.renderOrder = 4;
  });
  scene.add(arrow);
  function handle(color, size) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius * size, 20, 16),
      new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }));
    mesh.renderOrder = 5;
    scene.add(mesh);
    return mesh;
  }
  const tiltHandle = handle(0x117b88, 0.04);
  const marker = handle(0xf09a26, 0.028);
  let apex = data.apex ? new THREE.Vector3().fromArray(data.apex) : null;
  let stage = 'plane';

  function updatePlane() {
    normal.copy(normalAxis).applyQuaternion(plane.quaternion).normalize();
    arrow.position.copy(plane.position);
    arrow.setDirection(normal);
    tiltHandle.position.copy(plane.position).addScaledVector(normal, radius * 1.05);
    const keepSign = apex ? Math.sign(apex.clone().sub(plane.position).dot(normal)) || -1 : -1;
    keptClip.setFromNormalAndCoplanarPoint(normal.clone().multiplyScalar(keepSign), plane.position);
    removedClip.copy(keptClip).negate();
    marker.visible = stage === 'apex' && Boolean(apex);
    if (apex) marker.position.copy(apex);
  }
  plane.position.fromArray(data.origin);
  plane.quaternion.setFromUnitVectors(normalAxis, new THREE.Vector3().fromArray(data.normal).normalize());
  updatePlane();
  document.querySelector('#reference').textContent = `t${data.frame}`;
  status.textContent = data.status === 'Saved plane could not be loaded' ? data.status : '';
  confirmButton.disabled = false;

  // Confirm the plane first. Its handles disappear and cannot be edited in the apex step.
  function setStage(value) {
    stage = value;
    const editingPlane = stage === 'plane';
    document.querySelector('#stage').textContent = editingPlane ? '1 · Plane' : '2 · Apex';
    confirmButton.textContent = editingPlane ? 'Confirm plane' : 'Confirm apex';
    backButton.hidden = editingPlane;
    arrow.visible = tiltHandle.visible = editingPlane;
    plane.material.opacity = editingPlane ? 0.1 : 0.04;
    updatePlane();
    status.textContent = '';
    status.title = '';
  }
  backButton.onclick = () => { if (!busy && !drag) setStage('plane'); };
  let drag = null, clickStart = null, busy = false;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const canvas = renderer.domElement;
  function cast(event) {
    const bounds = canvas.getBoundingClientRect();
    pointer.set(2 * (event.clientX - bounds.left) / bounds.width - 1,
      1 - 2 * (event.clientY - bounds.top) / bounds.height);
    raycaster.setFromCamera(pointer, camera);
  }
  canvas.addEventListener('pointerdown', event => {
    if (busy || event.button !== 0 || drag) return;
    cast(event);
    clickStart = stage === 'apex' ? { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false } : null;
    const handles = stage === 'plane' ? [tiltHandle] : (apex ? [marker] : []);
    const hit = raycaster.intersectObjects(handles, false)[0];
    let action = hit ? (hit.object === marker ? 'apex' : 'tilt') : null;
    if (!action && stage === 'plane') {
      const planeHit = raycaster.intersectObject(plane, false)[0];
      const wallHit = raycaster.intersectObject(outer, false)[0];
      if (planeHit && (!wallHit || planeHit.distance < wallHit.distance)) action = 'move';
    }
    if (!action) return;
    event.stopImmediatePropagation();
    canvas.setPointerCapture(event.pointerId);
    orbit.enabled = false;
    canvas.style.cursor = 'grabbing';
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, action };
  }, true);
  canvas.addEventListener('pointermove', event => {
    if (clickStart && Math.hypot(event.clientX - clickStart.x, event.clientY - clickStart.y) > 5) {
      clickStart.moved = true;
    }
    if (!drag || drag.id !== event.pointerId) return;
    event.stopImmediatePropagation();
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
    if (drag.action === 'apex') {
      cast(event);
      const hit = raycaster.intersectObject(outer, false)[0];
      if (hit) apex = hit.point.clone();
    } else if (drag.action === 'tilt') {
      plane.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(up, dx * 0.008));
      plane.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(right, dy * 0.008));
    } else {
      const horizontal = normal.dot(right), vertical = -normal.dot(up);
      const projection = Math.hypot(horizontal, vertical);
      const pixels = projection > 0.15 ? (dx * horizontal + dy * vertical) / projection : -dy;
      const scale = 2 * camera.position.distanceTo(plane.position)
        * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / canvas.clientHeight;
      plane.position.addScaledVector(normal, pixels * scale);
    }
    drag.x = event.clientX;
    drag.y = event.clientY;
    updatePlane();
    status.textContent = '';
  }, true);
  function endDrag(event) {
    if (!busy && !drag && stage === 'apex' && event.type === 'pointerup'
        && clickStart?.id === event.pointerId && !clickStart.moved) {
      cast(event);
      const hit = raycaster.intersectObject(outer, false)[0];
      if (hit) { apex = hit.point.clone(); updatePlane(); status.textContent = ''; }
    }
    clickStart = null;
    if (!drag || drag.id !== event.pointerId) return;
    event.stopImmediatePropagation();
    drag = null;
    orbit.enabled = true;
    canvas.style.cursor = '';
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  }
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, endDrag, true);

  // LRC is still calculated and saved in the VTU; no extra visualization controls are needed here.
  confirmButton.onclick = async () => {
    if (busy || drag) return;
    if (stage === 'plane') { setStage('apex'); return; }
    if (!apex) { status.textContent = 'Select an apex on the outer wall'; return; }
    if (!confirm('Generate t0.vtu, parameter.txt and surface_cut? Previous outputs will be replaced.')) return;
    busy = true;
    confirmButton.disabled = true;
    backButton.disabled = true;
    status.textContent = 'Generating reference mesh and cut surfaces...';
    try {
      const response = await fetch('/api/plane', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Plane-Token': data.token },
        body: JSON.stringify({ origin: plane.position.toArray(), normal: normal.toArray(), apex: apex.toArray() })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Save failed.');
      status.textContent = result.remaining_lrc_outliers ? 'Saved · Review directions' : result.low_quality_tetrahedra ? 'Saved · Review mesh quality' : 'Saved · t0.vtu + surface_cut';
      status.title = `${result.tetrahedra} cells; ${result.surface_frames ?? 0} cut frames; ${result.remaining_lrc_outliers ?? 0} local eL jumps; ${result.low_quality_tetrahedra ?? 0} low-quality cells`;
    } catch (error) {
      status.textContent = error.message;
    } finally {
      busy = false;
      confirmButton.disabled = false;
      backButton.disabled = false;
    }
  };
  function resize() {
    camera.aspect = viewport.clientWidth / viewport.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewport.clientWidth, viewport.clientHeight);
  }
  window.addEventListener('resize', resize);
  resize();
  renderer.setAnimationLoop(() => { orbit.update(); renderer.render(scene, camera); });
} catch (error) {
  status.textContent = `Viewer unavailable: ${error.message}`;
}
</script>
</html>
"""


def create_server(cfg, reference_stl, frame, output_dir, series, port=PORT):
    """Serve one reference mesh locally; write only after a confirmed save request. series: also export the cut
    surfaces of every tracked frame (volume_constrain, invert_para)."""
    reference_stl = Path(reference_stl).resolve()
    plane_file = cfg.input / "base_plane.json"
    page = PAGE if series else single_frame_page()
    mesh = trimesh.load_mesh(reference_stl, process=True)
    shells = sorted(mesh.split(only_watertight=False), key=lambda shell: abs(shell.volume))
    if len(shells) != 2:
        raise ValueError("The viewer requires separate inner and outer surfaces.")
    origin, normal = pp.plane_vectors({"origin": mesh.bounds.mean(axis=0), "normal": INITIAL_NORMAL})
    scene = {
        "frame": int(frame),
        "surfaces": [{"vertices": shell.vertices.ravel().tolist(),
                      "faces": shell.faces.ravel().tolist()} for shell in shells],
        "initial_origin": origin.tolist(),
        "initial_normal": normal.tolist(),
        "origin": origin.tolist(),
        "normal": normal.tolist(),
        "apex": None,
        "status": "Not saved",
        "token": secrets.token_urlsafe(32),
    }

    # Only restore a saved plane belonging to this reference and coordinate system.
    if plane_file.exists():
        try:
            saved = json.loads(plane_file.read_text())
            if saved["reference_frame"] == frame and Path(saved["reference_stl"]).name == reference_stl.name:
                origin, normal = pp.plane_vectors(saved)
                scene.update(origin=origin.tolist(), normal=normal.tolist(),
                             apex=saved.get("apex"), status="Saved landmarks")
        except (OSError, ValueError, KeyError, TypeError):
            scene["status"] = "Saved plane could not be loaded"

    # Serve only these files, never arbitrary paths or the whole data directory.
    assets = {}
    for name in ("three.module.min.js", "three.core.min.js", "OrbitControls.js"):
        assets["/" + name] = (ASSET_DIR / name, "text/javascript")
    for path, _ in assets.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing viewer asset: {path}")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def parse_request(self):
            if not super().parse_request():
                return False
            if urlsplit("//" + self.headers.get("Host", "")).hostname not in ("localhost", "127.0.0.1", "::1"):
                self.send_error(403, "Use the localhost or forwarded localhost URL.")
                return False
            return True

        def respond(self, content, content_type="application/json", status=200):
            payload = content if isinstance(content, bytes) else json.dumps(content, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self.respond(page.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/scene":
                self.respond(scene)
            elif self.path in assets:
                path, content_type = assets[self.path]
                self.respond(path.read_bytes(), content_type)
            else:
                self.respond({"error": "Not found."}, status=404)

        def do_POST(self):
            if self.path != "/api/plane":
                self.respond({"error": "Not found."}, status=404)
                return
            # A per-session token prevents other websites from submitting plane changes.
            if not secrets.compare_digest(self.headers.get("X-Plane-Token", ""), scene["token"]):
                self.respond({"error": "Reload the page before saving."}, status=403)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size.")
                data = json.loads(self.rfile.read(size))
                origin, normal = pp.plane_vectors(data)
                apex = np.asarray(data["apex"], dtype=float)
                if apex.shape != (3,) or not np.all(np.isfinite(apex)):
                    raise ValueError("Select an apex on the outer surface first.")
                result = pp.export_reference(reference_stl, frame, origin, normal, apex,
                                             output_dir, cfg.MESH_SIZE, pp.APEX_RADIUS, pp.GMSH)
                saved = {
                    "reference_frame": int(frame),
                    "reference_stl": reference_stl.name,
                    "origin": origin.tolist(),
                    "normal": normal.tolist(),
                    "offset": float(np.dot(normal, origin)),
                    "equation": "dot(normal, point) = offset",
                    "coordinate_system": "original_STL",
                    "length_unit": "STL_coordinate_unit",
                    "apex": apex.tolist(),
                    "keep_sign": 1 if np.dot(apex - origin, normal) > 0 else -1,
                }
                if series:  # run the surface step last, using this confirmation rather than an older saved plane
                    surfaces = pp.export_surface_series(cfg.input, Path(output_dir) / "surface_cut", landmarks=saved)
                # Atomically replace the previous plane; source meshes remain untouched.
                payload = json.dumps(saved, indent=2, allow_nan=False) + "\n"
                plane_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = plane_file.with_suffix(".json.tmp")
                temporary.write_text(payload)
                temporary.replace(plane_file)
                scene.update(origin=saved["origin"], normal=saved["normal"],
                             apex=saved["apex"], status="Saved landmarks")
                if series:
                    self.respond({"status": "Exported reference and cut surfaces", "surface_frames": surfaces["frames"], **result})
                else:
                    self.respond({"status": "Exported reference", **result})
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.respond({"error": str(error)}, status=400)
            except Exception as error:
                self.respond({"error": f"Export failed: {error}"}, status=500)

    return HTTPServer((HOST, port), Handler)


def single_frame_page():
    """The page for one input frame (windkessel): no cut surfaces."""
    page = PAGE
    for old, new in (("Generate t0.vtu, parameter.txt and surface_cut?", "Generate t0.vtu and parameter.txt?"),
                     ("'Generating reference mesh and cut surfaces...'", "'Generating reference mesh...'"),
                     ("'Saved · t0.vtu + surface_cut'", "'Saved · t0.vtu'"),
                     ("${result.surface_frames ?? 0} cut frames; ", "")):
        assert old in page
        page = page.replace(old, new)
    return page


def serve(cfg, out, series):
    """Run the app until Ctrl+C."""
    if series:
        frame = REFERENCE_FRAME
        if frame is None:
            print("Selecting reference frame...", flush=True)
            frame = select_reference_frame(cfg.input, r)
        reference_stl = cfg.input / f"t{frame}.stl"
    else:
        reference_stl, frame = pp.reference_surface(cfg.input)
    with create_server(cfg, reference_stl, frame, out, series) as server:
        print(f"Reference {Path(reference_stl).name} | Open http://{HOST}:{PORT}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
