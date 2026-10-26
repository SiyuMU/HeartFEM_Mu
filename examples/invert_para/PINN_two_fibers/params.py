"""All settings of this example (read by the library heartfem_mu through main.py). Paths are relative to this folder.
Solver: a parametric physics-informed network (PINN) instead of FE (../FE_two_fibers: the same inverse problem by FE).

Inverse problem: from the tracked wall of two frames of one cycle (ED: passive, maximum volume; PS: contracting) and
the cavity pressure GIVEN at both frames, find the passive scale C and the peak active tension T_MAX. The fiber
architecture is PREDEFINED (one family at FIBER_ANGLE_DEG, or two families, TWO_FIBERS). Mechanics: heartFEM laws (Lee
lab, lcleeHeart), as the volume-driven model of the same atrium (../../volume_constrain/FE_two_fibers); every other constant is
fixed below.
"""

# ==================== run settings (main.py) ====================
TASK = "invert_para"             # "volume_constrain" (P-V cycle of the tracked volumes), "windkessel" (the chamber in a
                                 # closed-loop circulation) or "invert_para" (C and T_MAX from two tracked frames)
SOLVER = "PINN"                  # "FE" (legacy FEniCS, MPI on the CPU) or "PINN" (physics-informed network, GPU)
CPUS = None                      # CPU cores of the run (Linux taskset list, e.g. "0-7" or "0,2,4,6"); None: any core
GPU = 3                          # GPU of the training and the solves (nvidia-smi index); None: the CPU (training ~10x slower)
CONDA_ENV = "pinn_stress"        # conda env of all steps (numpy, scipy, torch with CUDA, matplotlib, trimesh, meshio, vtk, gmsh)

# ---- folders
INPUT_DIR = "input"              # raw data: t12.stl (reference), t43.stl (ED), t51.stl (PS), base_plane.json
CACHE_DIR = "cache"              # intermediate files: CACHE_DIR/mesh/ (preprocess products), CACHE_DIR/model/ (network)
OUTPUT_DIR = "output"            # results: result.json, evaluation.json, ED.vtu, PS.vtu, comparison.png

# ---- preprocess (more options at the top of heartfem_mu/preprocess.py)
MESH_SIZE = 5.0                  # tetrahedral mesh size, um
USE_SAVED_LANDMARKS = True       # base plane, apex and reference frame from input/base_plane.json; False: browser app
MESH_THREADS = 36                # threads of the gmsh mesher (step preprocess): the tetrahedra depend on the thread
                                 # count (gmsh meshes the surface patches in parallel; deterministic for a fixed count;
                                 # a node with fewer cores gives another mesh); 36 = all cores of the node of these runs

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
K_SPRING_OVER_C = 0.0            # epicardial spring of heartFEM: traction -K (u . N) N on the epicardium (N its reference
                                 # normal), K = K_SPRING_OVER_C x C (1/mm). 0 = no spring (default: at 48 hpf the heart
                                 # beats in a fluid-filled pericardial cavity and has no epicardium yet). Relative to C
                                 # (unlike K_SPRING of the FE example) because the network works in units of C with two inputs:
                                 # a fixed K would make K / C change with the unknown C. A new value needs a new training
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


# ==================== PINN ====================
# The network maps (p, a) to the P2 nodal displacement of the wall (fibers fixed above), with p = P / C and
# a = T_MAX Ct(t) / C: with C, T_MAX and P scaled together the displacement does not change, so these two inputs cover
# every parameter set (Ct is uniform on the rise and 0 when passive - for every fiber family - so one tension level a
# serves both frames).
# ---- physics discretization (heartfem_mu/pinn.py), in units of C
KAPPA_OVER_C = 10000.0           # incompressibility penalty on the nodal (P1-projected) volume change, / C
RIGID_OVER_C = 10000.0           # no rigid translation in the base plane and no rotation (penalty), / C
QUAD_ORDER = 2                   # tetrahedron rule: 8-point degree-3 conical rule
# ---- domain of the inputs (the result must lie inside: step invert reports it)
P_MAX_N = 10.0                   # p = P / C in [0, P_MAX_N]
A_MAX_N = 110.0                  # a = T_MAX Ct / C in [0, A_MAX_N]
# ---- network
WIDTH, DEPTH = 128, 5            # tanh MLP; WIDTH = number of learned displacement modes of the linear nodal output
LEVELS_UM = (8.0, 20.0, 50.0)    # multilevel output: coarse displacements on voxel clusters of these sizes (preconditioner)
U_SCALE_MM = 0.01                # output scale (10 um)
SEED = 0
# ---- training (energy minimization, no FE or image data)
ADAM_STEPS = 1000                # warm-up: random loads, lr ADAM_LR -> ADAM_LR_END, load ramp and kappa ramp
ADAM_LR, ADAM_LR_END = 1e-5, 2e-6
ADAM_LOADS, ADAM_CHUNK = 8, 4
LOAD_RAMP_STEPS, KAPPA_RAMP_STEPS, KAPPA_START_OVER_C = 200, 400, 100.0
GRID_STAGES = ((5, 5, 1), (8, 8, 42), (14, 14, 101))  # L-BFGS load grids (n_p, n_a, first outer step):
                                 # 24, 63, 195 loads (without the unloaded state p = a = 0)
GRID_P0_N, GRID_A0_N = 0.2, 5.0  # log-spacing scales of p and a (dense at small loads)
WEIGHT_POWER = 0.5               # per-load objective Pi_i / |Pi_i(stage start)|^WEIGHT_POWER
LBFGS_HISTORY, LBFGS_INNER, LBFGS_CHUNK = 100, 20, 11
WHITEN, WHITEN_FROM_OUTER = 1e-5, 101  # whitened output parametrization on the finest grid
STOP_WINDOW, STOP_TOL = 10, 1e-4 # stop on the finest grid when the objective has flattened
TRAIN_HOURS = 6.0                # safety cap on the optimization time only
