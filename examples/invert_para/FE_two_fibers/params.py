"""All settings of this example (read by the library heartfem_mu through main.py). Paths are relative to this folder.

Inverse problem: from the tracked wall of two frames of one cycle (ED: passive, maximum volume; PS: contracting) and
the cavity pressure GIVEN at both frames, find the passive scale C and the peak active tension T_MAX. The fiber
architecture is PREDEFINED (one family at FIBER_ANGLE_DEG, or two families, TWO_FIBERS). Mechanics: heartFEM laws (Lee
lab, lcleeHeart), as the volume-driven model of the same atrium (../../volume_constrain/FE_two_fibers); every other constant is
fixed below.
"""

# ==================== run settings (main.py) ====================
TASK = "invert_para"             # "volume_constrain" (P-V cycle of the tracked volumes), "windkessel" (the chamber in a
                                 # closed-loop circulation) or "invert_para" (C and T_MAX from two tracked frames)
SOLVER = "FE"                    # "FE" (legacy FEniCS, MPI on the CPU) or "PINN" (physics-informed network, GPU)
CPUS = None                      # CPU cores of the run (Linux taskset list, e.g. "0-7" or "0,2,4,6"); None: any core
MPI_RANKS = 8                    # MPI ranks of the FE inversion (one core each)
CONDA_ENV = "pinn_stress"        # conda env of the Python steps (numpy, scipy, matplotlib, trimesh, meshio, vtk, gmsh)
CONDA_ENV_FENICS = "heartfem_fe" # conda env of the FE steps (legacy FEniCS 2019.1, mpi4py, petsc4py, ngspice)

# ---- folders
INPUT_DIR = "input"              # raw data: t12.stl (reference), t43.stl (ED), t51.stl (PS), base_plane.json
CACHE_DIR = "cache"              # intermediate files: CACHE_DIR/mesh/ (preprocess products)
OUTPUT_DIR = "output"            # results: result.json, evaluation.json, ED.vtu, PS.vtu, comparison.png

# ---- preprocess (more options at the top of heartfem_mu/preprocess.py)
MESH_SIZE = 5.0                  # tetrahedral mesh size, um
USE_SAVED_LANDMARKS = True       # base plane, apex and reference frame from input/base_plane.json; False: browser app
MESH_THREADS = 1                 # threads of the gmsh mesher (step preprocess): the tetrahedra depend on the thread
                                 # count (gmsh meshes the surface patches in parallel; deterministic for a fixed count).
                                 # 1 as the original FE package (its pipeline ran with OMP_NUM_THREADS=1); the PINN
                                 # example uses 36, a slightly different mesh

# ---- frames (numbers of the input files t<k>.stl) and their activation
REF_FRAME = 12                   # stress-free reference configuration (the mesh); must match input/base_plane.json
ED_FRAME = 43                    # end-diastole: maximum volume, passive (Ct = 0)
PS_FRAME = 51                    # peak systolic pressure, contracting
PS_ACTIVATION_MS = 80.0          # time of the PS frame after the activation onset, ms; must be < T0_MS (on the rise of
                                 # Ct, where Ct = 1/2 (1 - cos(pi t / T0_MS)) is the same everywhere: 0.75 here).
                                 # How 80 ms was obtained: 52 frames per cycle of 550 ms (10.58 ms per frame); onset at
                                 # the maximum-volume frame 43 + 0.5 (usually + 0; a small offset as the tracking is not
                                 # dense in time); PS frame 51: (51 - 43.5) x 550 / 52 = 79.3 ms, rounded to 80 ms
MM_PER_UNIT = 1e-3               # STL / mesh coordinates are um; the model works in mm (stresses in kPa)

# ---- given cavity pressures, kPa (no measurement of the atrial pressure exists at 48 hpf; Hu et al. 2000)
P_ED_KPA = 0.1813                # 1.36 mmHg: 3 dpf ventricular end-diastolic pressure (Salehin et al. 2021); at
                                 # end-diastole the AV canal is open, atrial = ventricular pressure
P_PS_KPA = 0.2738                # 2.05 mmHg = 7.52 mmHg (3 dpf ventricular peak, Salehin et al. 2021) x 0.68 / 2.49 (adult
                                 # zebrafish atrial / ventricular peak pressure, Hu et al. 2001 / 2000)

# ---- fixed material constants (heartFEM laws; zebrafish values as the volume-driven model)
C_FF, C_SS, C_NN, C_NS, C_FS, C_FN = 29.9, 13.3, 13.3, 26.6, 53.2, 53.2   # Fung exponents (C itself is inferred)
CA0, CA0MAX, B, L0, LR = 4.35, 4.35, 4.75, 1.58, 2.0   # active law; LR: sarcomere length (um) in the reference state
T0_MS = 120.0                    # time to peak tension (the relaxation law is not needed: ED passive, PS on the rise)
K_SPRING = 0.0                   # epicardial spring of heartFEM, kPa/mm: traction -K (u . N) N on the epicardium (N its
                                 # reference normal). 0 = no spring (default: at 48 hpf the heart beats in a fluid-filled
                                 # pericardial cavity and has no epicardium yet; the previous model version used 0.09)
# base: u . n = 0, no rigid motion

# ---- predefined fiber architecture (not inferred); angle from eC toward eL as an (endo, epi) pair, linear in the
# transmural depth d_endo / (d_endo + d_epi) (as the volume_constrain and windkessel examples)
TWO_FIBERS = True                # two families at FIBER_ANGLES_DEG (False: one family, set FIBER_ANGLE_DEG instead)
FIBER_ANGLES_DEG = ((0.0, 0.0), (90.0, 90.0))  # (endo, epi) of family 1 and of family 2; 0 = eC, 90 = eL
FIBER_WEIGHTS = (0.5, 0.5)       # shares (sum 1): W = sum_i w_i C (e^Q_i - 1), S_act = sum_i w_i S_act,i (own ls_i)

# ---- misfit (scale-free: the lengths are fractions of L = V0^(1/3), the cube root of the reference cavity volume)
SIGMA_V = 0.01                   # cavity-volume tolerance (relative)
SIGMA_S_REL = 0.02               # surface-distance tolerance sigma_s = SIGMA_S_REL x L (1.76 um for this atrium)
W_EPI = 0.2                      # weight of the epicardial surface relative to the endocardial one
BASE_EXCLUDE_REL = 0.1139        # surface points closer than BASE_EXCLUDE_REL x L to the base plane are not compared
                                 # (10 um here: boundary condition and cut)
# J = sum over ED, PS of ((V - V_img) / (SIGMA_V V_img))^2 + (d_endo / sigma_s)^2 + W_EPI (d_epi / sigma_s)^2

# ---- search
START_FACTORS = ((1.0, 1.0), (0.5, 2.0), (2.0, 0.5))   # starts: (C, T_MAX) of the geometry start x these factors;
                                 # ((1.0, 1.0),) runs one start only (about 3x faster)
XATOL, FATOL, MAXFEV = 2e-4, 1e-4, 250   # Nelder-Mead stop (on ln C or ln T_MAX) and evaluation cap per stage
STEP = 0.5                       # initial simplex step (ln units)

