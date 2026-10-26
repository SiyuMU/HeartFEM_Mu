"""All settings of this example (read by the library heartfem_mu through main.py). Paths are relative to this folder.

The mechanics (mesh, fibers, material, activation, boundary conditions) are identical to the FE P-V reference
(../FE_two_fibers/params.py); only the solver differs: a coordinate-free physics-informed network instead of FE.
"""

# ==================== run settings (main.py) ====================
TASK = "volume_constrain"        # "volume_constrain" (P-V cycle of the tracked volumes), "windkessel" (the chamber in a
                                 # closed-loop circulation) or "invert_para" (C and T_MAX from two tracked frames)
SOLVER = "PINN"                  # "FE" (legacy FEniCS, MPI on the CPU) or "PINN" (physics-informed network, GPU)
CPUS = None                      # CPU cores of the run (Linux taskset list, e.g. "0-7" or "0,2,4,6"); None: any core
GPU = 1                          # GPU of the training and the solves (nvidia-smi index); None: the CPU (training ~10x slower)
CONDA_ENV = "pinn_stress"        # conda env of all steps (numpy, scipy, torch with CUDA, matplotlib, trimesh, meshio, vtk, gmsh)

# ---- folders
INPUT_DIR = "input"              # raw data: t0.stl ... t51.stl (closed endo + epi surfaces of each frame, um) + base_plane.json
OUTPUT_DIR = "output"            # results; OUTPUT_DIR/mesh/ holds the preprocess products (t0.vtu, parameter.txt, surface_cut/)

# ---- preprocess (more options at the top of heartfem_mu/preprocess.py)
MESH_SIZE = 5.0                  # tetrahedral mesh size, um
USE_SAVED_LANDMARKS = True       # base plane, apex and reference frame from input/base_plane.json; False: browser app
MESH_THREADS = 36                # threads of the gmsh mesher (step preprocess): the tetrahedra depend on the thread
                                 # count (gmsh meshes the surface patches in parallel; deterministic for a fixed count;
                                 # a node with fewer cores gives another mesh); 36 = all cores of the node of these runs

# ---- frames and units
N_FRAMES = 52                    # frames uniformly covering one cardiac cycle
MM_PER_UNIT = 1e-3               # STL / mesh coordinates are um; the model works in mm (stresses in kPa)

# ---- fibers: angle from eC (circumferential, 0 deg) toward eL (longitudinal, 90 deg) as an (endo, epi) pair, linear in
# the transmural depth d_endo / (d_endo + d_epi); sheet = eR (radial), normal = fiber x sheet for every family
TWO_FIBERS = True                # two families at FIBER_ANGLES_DEG (False: one family, set FIBER_ANGLE_DEG instead)
FIBER_ANGLES_DEG = ((0.0, 0.0), (90.0, 90.0))  # (endo, epi) of family 1 and of family 2; 0 = eC, 90 = eL
FIBER_WEIGHTS = (0.5, 0.5)       # shares (sum 1): W = sum_i w_i C (e^Q_i - 1), S_act = sum_i w_i S_act,i (own ls_i)

# ---- passive myocardium (heartFEM Fung law), kPa
C, C_FF, C_SS, C_NN, C_NS, C_FS, C_FN = 0.1, 29.9, 13.3, 13.3, 26.6, 53.2, 53.2
K_SPRING = 0.0                   # epicardial spring of heartFEM, kPa/mm: traction -K (u . N) N on the epicardium (N its
                                 # reference normal). 0 = no spring (default: at 48 hpf the heart beats in a fluid-filled
                                 # pericardial cavity and has no epicardium yet; the previous model version used 0.09); > 0 needs a new
                                 # training

# ---- active tension (heartFEM simpleActiveMaterial): S_act = T_MAX Ct(t, ls) Ca0^2 / (Ca0^2 + ECa(ls)^2) along the fiber
CA0, CA0MAX, B, L0, LR = 4.35, 4.35, 4.75, 1.58, 2.0   # LR: sarcomere length (um) in the reference state
T_MAX = 20.0                     # kPa
BCL_MS = 550.0                   # cycle length (Cairelli et al. 2024, zebrafish 3 dpf)
T0_MS = 120.0                    # time to peak tension
TR_M_MS_PER_UM = 73.0            # relaxation time tr = m ls + b, a field (ls = LR |F f0|)
TR_B_MS = -33.0
ONSET_AFTER_ED_FRAMES = 0.5      # activation onset in frames after the maximum-volume frame (found from the tracking: t43
                                 # here). Usually 0; a small offset if the tracking is not dense in time (here 0.5)

# ---- PINN: physics discretization (heartfem_mu/pinn.py)
KAPPA = 1000.0                   # incompressibility penalty on the nodal (P1-projected) volume change, kPa
RIGID_PENALTY = 1000.0           # kPa; no rigid translation in the base plane and no rotation (FE: Lagrange multipliers)
QUAD_ORDER = 2                   # tetrahedron rule: 2 = 8-point degree-3 conical rule (3 = 27 points, 1 = 4 points)

# ---- PINN: network (P, t) -> P2 nodal displacements
P_MAX = 1.6                      # trained cavity-pressure range [0, P_MAX], kPa
WIDTH, DEPTH = 128, 5            # tanh MLP; WIDTH = number of learned displacement modes of the linear nodal output
LEVELS_UM = (8.0, 20.0, 50.0)    # multilevel output: coarse displacements on voxel clusters of these sizes (preconditioner)
U_SCALE_MM = 0.01                # output scale (10 um)
SEED = 0
# network input: cavity pressure P and activation time t in [0, BCL_MS) (frame k: t = mod(k - onset, N) BCL / N,
# as the FE); inside the model the features (P / P_MAX, Ct(t, LR lambda_k)) from heartFEM's formula
FEATURE_STRETCHES = (0.8, 1.0, 1.2, 1.4, 1.6)  # fiber stretches lambda_k of the Ct features (all 0 after t = 320.6 ms)

# ---- PINN: training (energy minimization, no FE data)
ADAM_STEPS = 1000                # warm-up: random loads, lr ADAM_LR -> ADAM_LR_END, load ramp and kappa ramp
ADAM_LR, ADAM_LR_END = 1e-5, 2e-6
ADAM_LOADS, ADAM_CHUNK = 8, 4    # random (P, t) per step, loads per forward/backward pass
LOAD_RAMP_STEPS, KAPPA_RAMP_STEPS, KAPPA_START = 200, 400, 10.0
GRID_STAGES = ((4, 1), (6, 42), (8, 6, 101))  # L-BFGS load grids (n_P[, n_T], first outer step): n_P pressures x the
                                 # times of n_T tension levels -> 35, 83, 111 loads; coarse grids first (cheap iterations)
GRID_P0, GRID_T0 = 0.02, 0.5     # log-spacing scales of the pressures and tension levels, kPa (dense at small loads)
GRID_RELAX_FACTOR = 4 / 3        # times per stage: the rise times of the n_T tension levels (exact) + round(4/3 n_T)
GRID_RELAX_END_MS = 320.0        # relaxation times uniform on (t0, 320] ms (passive times need no points, see FEATURE_STRETCHES)
WEIGHT_POWER = 0.5               # per-load objective Pi_i / |Pi_i(stage start)|^WEIGHT_POWER
LBFGS_HISTORY, LBFGS_INNER, LBFGS_CHUNK = 100, 20, 11
WHITEN = 1e-5                    # whitened output parametrization on the finest grid: u = W A (h - h0) with
                                 # A = (G / max eig + WHITEN I)^(-1/2), G the Gram matrix of the last hidden features over
                                 # the L-BFGS loads; W is transformed when A is set, so the displacement field is unchanged
WHITEN_FROM_OUTER = 101          # A is set at the first grid stage that starts at or after this outer step (= the finest
                                 # grid); the earlier stages use A = I (whitening barely trained features slows training)
STOP_WINDOW, STOP_TOL = 10, 1e-4  # stop on the finest grid when the relative decrease of the objective over the last
                                 # STOP_WINDOW outer steps is below STOP_TOL (about 30 outer steps after the whitening)
TRAIN_HOURS = 2.0                # optimization-time cap; training normally stops before by the rule above (~48 min alone on
                                 # an RTX 3090 Ti); 5 outer steps without progress also stop it on the finest grid and move
                                 # on to the next grid on a coarser one

# ---- volume-constrained solve (step solve)
BISECTION_STEPS = 40             # per frame: V(P, t_k) = V_k for P in [0, P_MAX]

# ---- movie / figures
BASE_EXCLUDE_UM = 10.0           # wall means in summary.csv skip cells this close to the base (boundary-condition stress)
FFMPEG = "ffmpeg"                # ffmpeg executable for output/pv_cycle.mp4 (skipped if not found)
MOVIE_FPS = 6
