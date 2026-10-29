"""All settings of this example (read by the library heartfem_mu through main.py). Paths are relative to this folder.

Unloading: from the end-diastolic (ED) wall, its cavity pressure P_ED_KPA and the cavity volume V0_NL of the
stress-free (unloaded) state, find the passive scale C and the unloaded geometry: inflating the unloaded wall to P_ED
with C gives the ED wall. The wall is passive at ED; the fiber architecture is PREDEFINED on the unloaded geometry (one
family at FIBER_ANGLE_DEG, or two families, TWO_FIBERS). Mechanics: heartFEM laws (Lee lab, lcleeHeart), as the
invert_para model of the same atrium (../../invert_para/FE_two_fibers); every other constant is fixed below. The method
(inverse elastostatics on the ED mesh, the loop over frame and C, the check) is described in heartfem_mu/unloading.py.

The input of this example is the surface of an ED wall made with the same model: the mesh of the tracked frame t12
(the stress-free reference of ../../invert_para) inflated to P_ED with two fiber families.
"""

# ==================== run settings (main.py) ====================
TASK = "unloading"               # "volume_constrain", "windkessel", "invert_para" or "unloading" (C and the stress-free
                                 # geometry from the ED wall, P_ED and the unloaded cavity volume)
SOLVER = "FE"                    # "FE" (legacy FEniCS, MPI on the CPU); unloading has no PINN solver
CPUS = None                      # CPU cores of the run (Linux taskset list, e.g. "0-7" or "0,2,4,6"); None: any core
MPI_RANKS = 8                    # MPI ranks of the FE step (one core each)
CONDA_ENV = "pinn_stress"        # conda env of the Python steps (numpy, scipy, matplotlib, trimesh, meshio, vtk, gmsh); the
                                 # FE step also runs the local-frame functions of preprocess in it (a helper process)
CONDA_ENV_FENICS = "heartfem_fe" # conda env of the FE step (legacy FEniCS 2019.1, mpi4py, petsc4py)

# ---- folders
INPUT_DIR = "input"              # raw data: one STL file (the ED wall, here ed.stl) and base_plane.json (its landmarks)
CACHE_DIR = "cache"              # preprocess products: CACHE_DIR/mesh (the ED mesh t0.vtu)
OUTPUT_DIR = "output"            # results: result.json and unloaded.vtu (the unloaded wall, tetrahedra)

# ---- preprocess (more options at the top of heartfem_mu/preprocess.py)
MESH_SIZE = 5.0                  # tetrahedral mesh size, um
USE_SAVED_LANDMARKS = True       # base plane and apex (on the ED surface) from input/base_plane.json; False: browser app
MESH_THREADS = 1                 # threads of the gmsh mesher (step preprocess): the tetrahedra depend on the thread
                                 # count (gmsh meshes the surface patches in parallel; deterministic for a fixed count)
MM_PER_UNIT = 1e-3               # STL / mesh coordinates are um; the model works in mm (stresses in kPa)

# ---- given values
P_ED_KPA = 0.1813                # cavity pressure at ED, kPa: 1.36 mmHg, 3 dpf ventricular end-diastolic pressure
                                 # (Salehin et al. 2021); at end-diastole the AV canal is open, atrial = ventricular pressure
V0_NL = 0.6771817993716098       # cavity volume of the unloaded wall, nL (closed by the base plane): here the cavity of the
                                 # t12 mesh of ../../invert_para. A 1 % change of V0_NL changes C by about 7 % (result.json:
                                 # dlnC_dlnV0); below a critical V0 no unloaded wall loads back to the ED wall (check)

# ---- fixed material constants (heartFEM laws; zebrafish values as the invert_para model)
C_FF, C_SS, C_NN, C_NS, C_FS, C_FN = 29.9, 13.3, 13.3, 26.6, 53.2, 53.2   # Fung exponents (C itself is the result)
CA0, CA0MAX, B, L0, LR = 4.35, 4.35, 4.75, 1.58, 2.0   # active law of heartFEM: inactive at ED (Ct = 0); it only sets
                                 # the residual scale of the forward problem (Newton tolerance), as in invert_para
K_SPRING = 0.0                   # epicardial spring of heartFEM, kPa/mm; unloading supports 0 only (no spring: at 48 hpf
                                 # the heart beats in a fluid-filled pericardial cavity). Then the shapes depend on P/C only
# base: u . n = 0, no rigid motion

# ---- predefined fiber architecture on the unloaded geometry (not inferred); angle from eC toward eL as an (endo, epi)
# pair, linear in the transmural depth d_endo / (d_endo + d_epi); the local frame is recomputed on the unloaded geometry
TWO_FIBERS = True                # two families at FIBER_ANGLES_DEG (False: one family, set FIBER_ANGLE_DEG instead)
FIBER_ANGLES_DEG = ((0.0, 0.0), (90.0, 90.0))  # (endo, epi) of family 1 and of family 2; 0 = eC, 90 = eL
FIBER_WEIGHTS = (0.5, 0.5)       # shares (sum 1): W = sum_i w_i C (e^Q_i - 1)

# ---- search (one loop over the local frame, the base origin and C; heartfem_mu/unloading.py)
C_START_KPA = 0.4                # C of the first iteration, kPa (the unloaded wall of the first iteration starts at the ED
                                 # wall)
SECANT_SLOPE0 = 0.2              # d ln V0 / d ln C for the first C step (later: the secant of the last two iterations)
MAX_STEP_LN_C = 0.9              # largest step of ln C per iteration (a factor 2.5)
TOL_X_UM = 1e-4                  # stop when the unloaded wall changes by less than this (max over the nodes), um ...
V0_TOL_REL = 1e-8                # ... and its cavity volume meets V0_NL to this (relative) ...
TOL_LN_C = 1e-6                  # ... or the next step of ln C is below this
MAX_ITERATIONS = 100
CHECK_LOAD_STEPS = 10            # check: the unloaded wall is loaded from zero to P_ED in this many equal steps
