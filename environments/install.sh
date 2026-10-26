#!/usr/bin/env bash
# Install and check the two conda environments of HeartFEM_Mu (Linux x86-64, conda or miniforge).
#
#   bash environments/install.sh                       # create pinn_stress and heartfem_fe, then check them
#   bash environments/install.sh --check               # only check existing environments
#   bash environments/install.sh --pinn NAME --fe NAME # other environment names (then set CONDA_ENV and
#                                                      # CONDA_ENV_FENICS in the examples' params.py)
#
# pinn_stress.yml (CONDA_ENV): Python 3.11, numpy, scipy, torch 2.5.1 with CUDA 12.1, matplotlib, trimesh, meshio, vtk, gmsh;
#   preprocessing, the PINN and the figures. The movies use the ffmpeg binary of the pip package imageio-ffmpeg, linked
#   here as <env>/bin/ffmpeg (the binary that wrote the example movies).
# heartfem_fe.yml (CONDA_ENV_FENICS): Python 3.8, legacy FEniCS 2019.1 (dolfin) with MPICH and PETSc/MUMPS, ngspice 41;
#   the FE steps and the circuit.
# Both files pin the exact versions of the environments that produced the example outputs. An environment that already
# exists is not changed, only checked. The checks: imports and versions, GPU visible to torch, gmsh, ffmpeg, a small
# dolfin problem solved with MUMPS on 2 MPI ranks (also tests the compilers of the form JIT), ngspice.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINN=pinn_stress
FE=heartfem_fe
CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --pinn) PINN="$2"; shift 2;;
    --fe) FE="$2"; shift 2;;
    --check) CHECK_ONLY=1; shift;;
    -h|--help) sed -n '2,17p' "$0"; exit 0;;
    *) echo "unknown option: $1 (see --help)"; exit 2;;
  esac
done

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ]; then
  echo "conda not found: install Miniforge (https://github.com/conda-forge/miniforge) or Miniconda first"; exit 1
fi
BASE="$("$CONDA_BIN" info --base)"
# shellcheck disable=SC1091
source "$BASE/etc/profile.d/conda.sh"

exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

create() {  # create <name> <yml>
  if exists "$1"; then
    echo "== $1: exists, not changed (checked below)"
  elif [ "$CHECK_ONLY" = 1 ]; then
    echo "== $1: does not exist (run without --check to create it)"; return 1
  else
    echo "== $1: creating from $2 (several GB to download the first time)"
    conda env create -n "$1" -f "$2"
  fi
}

ok=1
create "$PINN" "$HERE/pinn_stress.yml" || ok=0
create "$FE" "$HERE/heartfem_fe.yml" || ok=0
[ "$ok" = 1 ] || exit 1

echo "== checking $PINN"
conda activate "$PINN"
if [ ! -e "$CONDA_PREFIX/bin/ffmpeg" ]; then  # the movies: ffmpeg of imageio-ffmpeg
  ln -s "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$CONDA_PREFIX/bin/ffmpeg"
  echo "   linked $CONDA_PREFIX/bin/ffmpeg"
fi
python - <<'PY'
import gmsh, matplotlib, meshio, numpy, scipy, torch, trimesh, vtk
print(f"   python ok: numpy {numpy.__version__}, scipy {scipy.__version__}, torch {torch.__version__}, "
      f"vtk {vtk.vtkVersion.GetVTKVersion()}, gmsh {gmsh.__version__}, trimesh {trimesh.__version__}")
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"   GPUs visible to torch: {n}" + ("" if n else " (the PINN then runs on the CPU: GPU = None in params.py)"))
PY
echo "   $(gmsh --version 2>&1 | head -1 | sed 's/^/gmsh /')"
echo "   $(ffmpeg -version 2>&1 | head -1)"
conda deactivate

echo "== checking $FE"
conda activate "$FE"
python - <<'PY'
import dolfin, meshio, mpi4py, scipy
from petsc4py import PETSc
mumps = PETSc.Sys.hasExternalPackage("mumps")
print(f"   python ok: dolfin {dolfin.__version__}, PETSc {'.'.join(map(str, PETSc.Sys.getVersion()))} (MUMPS: {mumps}), "
      f"mpi4py {mpi4py.__version__}, scipy {scipy.__version__}")
if not mumps:
    raise SystemExit("PETSc without MUMPS: the FE steps need it")
PY
TEST=$(mktemp -d)
cat > "$TEST/poisson.py" <<'PY'
import dolfin as d
mesh = d.UnitCubeMesh(d.MPI.comm_world, 6, 6, 6)
V = d.FunctionSpace(mesh, "P", 2)
u, v, w = d.TrialFunction(V), d.TestFunction(V), d.Function(V)
d.solve(d.inner(d.grad(u), d.grad(v)) * d.dx == d.Constant(1.0) * v * d.dx, w, d.DirichletBC(V, 0.0, "on_boundary"),
        solver_parameters={"linear_solver": "mumps"})
peak = w.vector().max()
if d.MPI.rank(mesh.mpi_comm()) == 0:
    print(f"   dolfin form JIT + MUMPS on {d.MPI.size(mesh.mpi_comm())} MPI ranks ok (max u {peak:.4f}, expected 0.0562)")
PY
(cd "$TEST" && OMP_NUM_THREADS=1 mpirun -n 2 python poisson.py 2>&1 | grep -v "^Calling FFC\|Solving linear variational problem\|^$")
rm -rf "$TEST"
echo "   $(ngspice --version 2>&1 | grep -m1 -i 'ngspice-')"
conda deactivate

echo "== done. params.py of every example: CONDA_ENV = \"$PINN\", CONDA_ENV_FENICS = \"$FE\""
