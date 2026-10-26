"""All settings of this example (read by the library heartfem_mu through main.py). Paths are relative to this folder.

Mechanics: heartFEM laws (Lee lab, lcleeHeart) with zebrafish values from the ventricle literature (Cairelli et al.
2024, Salehin et al. 2021); identical to the Windkessel FE example (../FE_two_fibers/params.py).
Solver of the atrium: a coordinate-free physics-informed network (PINN) instead of FE.
Circuit: identical to the Windkessel FE package (structure of heartFEM / Cairelli's 3 dpf circuit with the chamber roles
swapped; the values in CIRCUIT were calibrated once for the FE package and are fixed, see README.md).
"""

# ==================== run settings (main.py) ====================
TASK = "windkessel"              # "volume_constrain" (P-V cycle of the tracked volumes), "windkessel" (the chamber in a
                                 # closed-loop circulation) or "invert_para" (C and T_MAX from two tracked frames)
SOLVER = "PINN"                  # "FE" (legacy FEniCS, MPI on the CPU) or "PINN" (physics-informed network, GPU)
CPUS = None                      # CPU cores of the run (Linux taskset list, e.g. "0-7" or "0,2,4,6"); None: any core
GPU = 1                          # GPU of the training and the solves (nvidia-smi index); None: the CPU (training ~10x slower)
CONDA_ENV = "pinn_stress"        # conda env of all steps (numpy, scipy, torch with CUDA, matplotlib, trimesh, meshio, vtk, gmsh)
CONDA_ENV_FENICS = "heartfem_fe" # only its ngspice is used, if CONDA_ENV has none

# ---- folders
INPUT_DIR = "input"              # raw data: t12.stl (closed endo + epi surface of the reference frame, um) + base_plane.json
OUTPUT_DIR = "output"            # results; OUTPUT_DIR/mesh/ holds the preprocess products (t0.vtu, parameter.txt)

# ---- preprocess (more options at the top of heartfem_mu/preprocess.py)
MESH_SIZE = 5.0                  # tetrahedral mesh size, um
USE_SAVED_LANDMARKS = True       # base plane and apex from input/base_plane.json; False: browser app
MESH_THREADS = 36                # threads of the gmsh mesher (step preprocess): the tetrahedra depend on the thread
                                 # count (gmsh meshes the surface patches in parallel; deterministic for a fixed count;
                                 # a node with fewer cores gives another mesh); 36 = all cores of the node of these runs

# ---- units
MM_PER_UNIT = 1e-3               # STL / mesh coordinates are um; the model works in mm (stresses in kPa)

# ---- fibers: angle from eC (circumferential, 0 deg) toward eL (longitudinal, 90 deg) as an (endo, epi) pair, linear in
# the transmural depth d_endo / (d_endo + d_epi); sheet = eR (radial), normal = fiber x sheet for every family
TWO_FIBERS = True                # False: one family at FIBER_ANGLE_DEG
FIBER_ANGLE_DEG = (0.0, 0.0)     # one family (TWO_FIBERS = False): (endo, epi); 0 = circumferential (eC), 90 = longitudinal (eL)
FIBER_ANGLES_DEG = ((0.0, 0.0), (90.0, 90.0))  # two families (TWO_FIBERS = True): (endo, epi) of family 1 and of family 2
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

# ---- PINN: physics discretization (heartfem_mu/pinn.py)
KAPPA = 1000.0                   # incompressibility penalty on the nodal (P1-projected) volume change, kPa
RIGID_PENALTY = 1000.0           # kPa; no rigid translation in the base plane and no rotation (FE: Lagrange multipliers)
QUAD_ORDER = 2                   # tetrahedron rule: 2 = 8-point degree-3 conical rule (3 = 27 points, 1 = 4 points)

# ---- PINN: network (P, t) -> P2 nodal displacements
P_MIN, P_MAX = -0.05, 1.6        # trained cavity-pressure range [P_MIN, P_MAX], kPa (P_MIN < 0: the relaxing atrium of
                                 # the coupled cycle is smaller than its stress-free volume, its pressure is negative)
WIDTH, DEPTH = 128, 5            # tanh MLP; WIDTH = number of learned displacement modes of the linear nodal output
LEVELS_UM = (8.0, 20.0, 50.0)    # multilevel output: coarse displacements on voxel clusters of these sizes (preconditioner)
U_SCALE_MM = 0.01                # output scale (10 um)
SEED = 0
# network input: cavity pressure P and activation time t in [0, BCL_MS) (ms from the atrial activation onset, the time of
# the table and of the circuit); inside the model the features (P / P_MAX, Ct(t, LR lambda_k)) from heartFEM's formula
FEATURE_STRETCHES = (0.8, 1.0, 1.2, 1.4, 1.6)  # fiber stretches lambda_k of the Ct features (all 0 after t = 320.6 ms)

# ---- PINN: training (energy minimization, no FE data)
ADAM_STEPS = 1000                # warm-up: random loads, lr ADAM_LR -> ADAM_LR_END, load ramp and kappa ramp
ADAM_LR, ADAM_LR_END = 1e-5, 2e-6
ADAM_LOADS, ADAM_CHUNK = 8, 4    # random (P, t) per step, loads per forward/backward pass
LOAD_RAMP_STEPS, KAPPA_RAMP_STEPS, KAPPA_START = 200, 400, 10.0
GRID_STAGES = ((4, 1), (6, 42), (8, 6, 101))  # L-BFGS load grids (n_P[, n_T], first outer step): n_P pressures in
                                 # [0, P_MAX] (+ the negative ones) x the times of n_T tension levels; coarse grids first
                                 # (cheap iterations)
GRID_NEGATIVE = (3, 4, 6)        # pressures in [P_MIN, 0) of the three grids -> 62, 139, 195 loads
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
                                 # STOP_WINDOW outer steps is below STOP_TOL; 5 outer steps without progress also stop it
                                 # on the finest grid and move on to the next grid on a coarser one
TRAIN_HOURS = 36.0               # safety cap on the optimization time only: training ends by the rule above (see the
                                 # measured times in README.md)

# ---- volume-constrained solve (steps pvt, replay)
BISECTION_STEPS = 40             # V(P, t) = V_target for P in the trained pressure range

# ---- summaries
BASE_EXCLUDE_UM = 10.0           # wall means skip cells this close to the base (boundary-condition stress)
FFMPEG = "ffmpeg"                # ffmpeg executable for output/pv_cycle.mp4 (skipped if not found)
MOVIE_FPS = 6

# ==================== Windkessel coupling ====================
# ---- step pvt: PINN cavity pressure P(V, t) on the grid of heartFEM getLVbehavior
# Volume grid (heartFEM LVbehaviorRun rule): SWEEP_N_LOG log-spaced volumes from 0.9 ESV* to 1.1 EDV, plus 1.1x and
# 1.3x the largest and 0.9x and 0.7x the smallest. EDV, ESV = largest and smallest imaged cavity volume;
# ESV* = 0.4 EDV if (EDV - ESV) / EDV < 0.6, else ESV.
IMAGED_V_MAX_NL = 0.949115       # largest and smallest cavity volume of the imaged cycle, nL (cavity closed at the base
IMAGED_V_MIN_NL = 0.495989       # plane; frames t43 and t5 of the 52-frame recording = 1.4016 and 0.7324 x the volume of
                                 # the reference frame t12). They only set the volume range of the table; the coupled
                                 # volume is not prescribed
SWEEP_N_LOG = 50
SWEEP_DT_MS = 5.0                # activation-time step (T0_MS must be on the grid); 2.5 ms changes P by 0.001 mmHg
SWEEP_T_END_MS = 400.0           # last time of the grid; later times are passive (the tension is off after 321 ms at the latest)

# ---- step circuit: closed-loop circuit (heartfem_mu/data/atrium_wk.cir) in ngspice
NGSPICE = "ngspice"              # executable on PATH (env with ngspice >= 41, XSPICE code models), or its full path
# Units: pressure mmHg, volume nL, time s, flow nL/s; R mmHg s/nL, C nL/mmHg, E mmHg/nL. Time 0 = atrial activation onset.
CIRCUIT = {
    # resistances
    "R_in": 1.2152473599804687,      # veins -> atrium (no valve)
    "R_AV": 0.5300946572638963,      # atrium -> ventricle, in series with the AV valve
    "R_out": 0.6965130560716649,     # ventricle -> arteries, in series with the outflow valve
    "R_sys": 5.5,                    # arteries -> veins
    # compliances
    "C_art": 0.1677407894897985,     # arterial pool (bulbus arteriosus)
    "C_ven": 3.59477124,             # venous pool (Cairelli's 3 dpf value, not calibrated)
    # ventricle: P = E(t) (V - V0), E = E_min + (E_max - E_min) e(tau); e rises over t_rise, falls over t_fall
    "E_min": 2.9999999999999996,
    "E_max": 50.45378491522287,
    "V0": 0.11790631708778505,
    "t_on": 0.14487027653746604,     # start of the ventricular activation
    "t_rise": 0.2327449717555155,
    "t_fall": 0.024304947199609375,
    # atrium: gain of the active table (heartFEM Windkessel_scale_T0_LV); 1 = the model as it is
    "atrium_scale": 1.0,
    # valves (ngspice sidiode) and inertances
    "valve_roff": 1e6, "valve_ron": 1e-9, "valve_rrev": 10.0, "valve_vfwd": 0.0,
    "L_in": 0.0, "L_AV": 0.0, "L_out": 0.0, "L_sys": 0.0,
    # initial state (P_ven0 sets the total blood volume; the others only shape the transient)
    "P_ven0": 1.5328457229811743, "P_art0": 3.0, "V_atrium0": 0.8, "V_ventricle0": 0.5,
    # time base
    "cycle": BCL_MS / 1000.0, "step": 1e-5, "n_cycles": 30, "keep_cycles": 2,
}
ATRIAL_CONTRACTION_END_MS = 220.0  # window for "ventricular filling during atrial contraction" and the a-wave

# ---- step replay: the coupled atrial volume V(t) goes back through the PINN
N_REPLAY = 52                    # states per cycle, at t = (j + 0.5) BCL / N_REPLAY
