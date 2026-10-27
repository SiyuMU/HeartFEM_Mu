"""All settings of this example (read by the library heartfem_mu through main.py). Paths are relative to this folder.

FE model: heartFEM laws (Lee lab, lcleeHeart) with zebrafish values from the ventricle literature (Cairelli et al.
2024, Salehin et al. 2021); T_MAX, LR, T0_MS and TR_B_MS were tuned to the imaged chamber (atrium).
Circuit: structure of heartFEM / Cairelli's 3 dpf circuit with the chamber roles swapped; the values in CIRCUIT were
calibrated to measured zebrafish hemodynamics (see README.md).
Wall: myocardium + endocardium (WALL = "myo_endo"): a 2 um endocardial layer grown from the myocardial inner surface
into the cavity, solved together with the myocardium; the cavity is the lumen inside the layer.
"""

# ==================== run settings (main.py) ====================
TASK = "windkessel"              # "volume_constrain" (P-V cycle of the tracked volumes), "windkessel" (the chamber in a
                                 # closed-loop circulation) or "invert_para" (C and T_MAX from two tracked frames)
SOLVER = "FE"                    # "FE" (legacy FEniCS, MPI on the CPU) or "PINN" (physics-informed network, GPU)
CPUS = None                      # CPU cores of the run (Linux taskset list, e.g. "0-7" or "0,2,4,6"); None: any core
MPI_RANKS = 8                    # MPI ranks of the FE replay (one core each)
SWEEP_JOBS, SWEEP_RANKS = 18, 1  # step pvt: parallel sweep jobs x MPI ranks each. The partition of the volumes
                                 # sets the continuation path of each volume (round-off level); this example's
                                 # output was made with 18 x 1
CONDA_ENV = "pinn_stress"        # conda env of the Python steps (numpy, scipy, matplotlib, trimesh, meshio, vtk, gmsh)
CONDA_ENV_FENICS = "heartfem_fe" # conda env of the FE steps (legacy FEniCS 2019.1, mpi4py, petsc4py, ngspice)

# ---- folders
INPUT_DIR = "input"              # raw data: t12.stl (closed endo + epi surface of the reference frame, um) + base_plane.json
OUTPUT_DIR = "output"            # results; OUTPUT_DIR/mesh/ holds the preprocess products (t0.vtu, parameter.txt) and the
                                 # mesh with the endocardial layer (t0_endo.vtu, endo_layer_info.json; step endo_layer)

# ---- preprocess (more options at the top of heartfem_mu/preprocess.py)
MESH_SIZE = 5.0                  # tetrahedral mesh size, um
USE_SAVED_LANDMARKS = True       # base plane and apex from input/base_plane.json; False: browser app
                                 # (places them by hand and overwrites input/base_plane.json; not recomputable)
MESH_THREADS = 1                 # threads of the gmsh mesher (step preprocess): the tetrahedra depend on the thread
                                 # count (gmsh meshes the surface patches in parallel; deterministic for a fixed count).
                                 # 1 as the original FE package (its pipeline ran with OMP_NUM_THREADS=1); the PINN
                                 # example uses 36, a slightly different mesh

# ---- units
MM_PER_UNIT = 1e-3               # STL / mesh coordinates are um; the FE works in mm (stresses in kPa)

# ---- fibers: angle from eC (circumferential, 0 deg) toward eL (longitudinal, 90 deg) as an (endo, epi) pair, linear in
# the transmural depth d_endo / (d_endo + d_epi); sheet = eR (radial), normal = fiber x sheet for every family
TWO_FIBERS = True                # two families at FIBER_ANGLES_DEG (False: one family, set FIBER_ANGLE_DEG instead)
FIBER_ANGLES_DEG = ((0.0, 0.0), (90.0, 90.0))  # (endo, epi) of family 1 and of family 2; 0 = eC, 90 = eL
FIBER_WEIGHTS = (0.5, 0.5)       # shares (sum 1): W = sum_i w_i C (e^Q_i - 1), S_act = sum_i w_i S_act,i (own ls_i)

# ---- passive myocardium (heartFEM Fung law), kPa
C, C_FF, C_SS, C_NN, C_NS, C_FS, C_FN = 0.1, 29.9, 13.3, 13.3, 26.6, 53.2, 53.2
K_SPRING = 0.0                   # epicardial spring of heartFEM, kPa/mm: traction -K (u . N) N on the epicardium (N its
                                 # reference normal). 0 = no spring (default: at 48 hpf the heart beats in a fluid-filled
                                 # pericardial cavity and has no epicardium yet; the previous model version used 0.09)

# ---- wall: "myo" (myocardium only; also when WALL is absent) or "myo_endo" (myocardium + endocardial layer; windkessel
# FE only). The layer shares the interface nodes with the myocardium. It has the laws of the myocardium (Fung passive
# energy, heartFEM active fiber stress, the fiber families) with the values in ENDO_MATERIAL; the activation timing
# (T0_MS, TR_*, CA0, CA0MAX, B, L0, LR) is shared. The cavity is the lumen (inside the layer).
WALL = "myo_endo"
ENDO_GROW = True                 # True: the layer is grown from the myocardial inner surface into the cavity (step
                                 # endo_layer; the input is the myocardium). False: the input surface has both layers
                                 # (three closed surfaces: lumen, myocardial endocardium, epicardium), meshed together
                                 # by the step preprocess; ENDO_THICKNESS_UM and ENDO_SUBLAYERS are then not read
ENDO_THICKNESS_UM = 2.0          # layer thickness along the surface normal, um
ENDO_SUBLAYERS = 1               # element layers through the thickness
ENDO_PRESSURE_JUMP = True        # one hydrostatic pressure field per region: p can jump at the interface, as it does
                                 # where the two incompressible materials differ (False: one continuous p)
ENDO_MATERIAL = {                # the layer: passive values and fibers of the myocardium, no active tension
    "C": 0.1, "C_FF": 29.9, "C_SS": 13.3, "C_NN": 13.3, "C_NS": 26.6, "C_FS": 53.2, "C_FN": 53.2,
    "T_MAX": 0.0,                # kPa; 0: the endocardium does not contract
    "FIBER_ANGLES_DEG": ((0.0, 0.0), (90.0, 90.0)),  # (lumen side, myocardial side) of family 1 and of family 2
    "FIBER_WEIGHTS": (0.5, 0.5),
}

# ---- active tension (heartFEM simpleActiveMaterial): S_act = T_MAX Ct(t, ls) Ca0^2 / (Ca0^2 + ECa(ls)^2) along the fiber
CA0, CA0MAX, B, L0, LR = 4.35, 4.35, 4.75, 1.58, 2.0   # LR: sarcomere length (um) in the reference state
T_MAX = 20.0                     # kPa
BCL_MS = 550.0                   # cycle length: Cairelli et al. 2024 3 dpf value (the imaged heart is 48 hpf, see README)
T0_MS = 120.0                    # time to peak tension
TR_M_MS_PER_UM = 73.0            # relaxation time tr = m ls + b (a field; ls = LR |F f0|): m as in Cairelli's heartFEM runs
TR_B_MS = -33.0                  # b tuned on the volume-driven P-V loop of this atrium

# ---- summaries
BASE_EXCLUDE_UM = 10.0           # wall means skip cells this close to the base (boundary-condition stress)
FFMPEG = "ffmpeg"                # ffmpeg executable for output/pv_cycle.mp4 (skipped if not found)
MOVIE_FPS = 6

# ==================== Windkessel coupling ====================
# ---- step pvt: FE cavity pressure P(V, t), swept like heartFEM getLVbehavior
# Volume grid (heartFEM LVbehaviorRun rule): SWEEP_N_LOG log-spaced volumes from 0.9 ESV* to 1.1 EDV, plus 1.1x and
# 1.3x the largest and 0.9x and 0.7x the smallest. EDV, ESV = largest and smallest imaged cavity volume;
# ESV* = 0.4 EDV if (EDV - ESV) / EDV < 0.6, else ESV.
IMAGED_V_MAX_NL = 0.949115       # largest and smallest cavity volume of the imaged cycle, nL (cavity closed at the base
IMAGED_V_MIN_NL = 0.495989       # plane; frames t43 and t5 of the 52-frame recording = 1.4016 and 0.7324 x the volume of
                                 # the reference frame t12). They only set the volume range of the table; the coupled
                                 # volume is not prescribed. They are inside the myocardium: with the layer the
                                 # table's grid is built from them minus the (constant) layer volume
SWEEP_N_LOG = 50
SWEEP_DT_MS = 5.0                # activation-time step (T0_MS must be on the grid); 2.5 ms changes P by 0.001 mmHg
SWEEP_T_END_MS = 400.0           # upper bound; every volume stops earlier, two steps after the tension is off

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
    # atrium: gain of the active FE table (heartFEM Windkessel_scale_T0_LV); 1 = the FE model as it is
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

# ---- step replay: the coupled atrial volume V(t) goes back through the FE model
N_REPLAY = 52                    # states per cycle, at t = (j + 0.5) BCL / N_REPLAY
