"""All settings of this example (read by the library heartfem_mu through main.py). Paths are relative to this folder.

Material and activation are heartFEM laws (Lee lab, lcleeHeart) with zebrafish values from the ventricle literature
(Cairelli et al. 2024, Salehin et al. 2021); T_MAX, LR, T0_MS, ONSET_AFTER_ED_FRAMES and TR_B_MS were tuned to the imaged
chamber (LA) as documented in README.md.
"""

# ==================== run settings (main.py) ====================
TASK = "volume_constrain"        # "volume_constrain" (P-V cycle of the tracked volumes), "windkessel" (the chamber in a
                                 # closed-loop circulation) or "invert_para" (C and T_MAX from two tracked frames)
SOLVER = "FE"                    # "FE" (legacy FEniCS, MPI on the CPU) or "PINN" (physics-informed network, GPU)
CPUS = None                      # CPU cores of the run (Linux taskset list, e.g. "0-7" or "0,2,4,6"); None: any core
MPI_RANKS = 8                    # MPI ranks of the FE solve (one core each)
CONDA_ENV = "pinn_stress"        # conda env of the Python steps (numpy, scipy, matplotlib, trimesh, meshio, vtk, gmsh)
CONDA_ENV_FENICS = "heartfem_fe" # conda env of the FE steps (legacy FEniCS 2019.1, mpi4py, petsc4py, ngspice)

# ---- folders
INPUT_DIR = "input"              # raw data: t0.stl ... t51.stl (closed endo + epi surfaces of each frame, um) + base_plane.json
OUTPUT_DIR = "output"            # results; OUTPUT_DIR/mesh/ holds the preprocess products (t0.vtu, parameter.txt, surface_cut/)

# ---- preprocess (more options at the top of heartfem_mu/preprocess.py)
MESH_SIZE = 5.0                  # tetrahedral mesh size, um
USE_SAVED_LANDMARKS = True        # base plane, apex and reference frame from input/base_plane.json; False: browser app
                                 # (places them by hand and overwrites input/base_plane.json; not recomputable)
MESH_THREADS = 36                # threads of the gmsh mesher (step preprocess): the tetrahedra depend on the thread
                                 # count (gmsh meshes the surface patches in parallel; deterministic for a fixed count;
                                 # a node with fewer cores gives another mesh); 36 = all cores of the node of these runs

# ---- frames and units
N_FRAMES = 52                    # frames uniformly covering one cardiac cycle
MM_PER_UNIT = 1e-3               # STL / mesh coordinates are um; the FE works in mm (stresses in kPa)

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
                                 # pericardial cavity and has no epicardium yet; the previous model version used 0.09)

# ---- active tension (heartFEM simpleActiveMaterial): S_act = T_MAX Ct(t, ls) Ca0^2 / (Ca0^2 + ECa(ls)^2) along the fiber
CA0, CA0MAX, B, L0, LR = 4.35, 4.35, 4.75, 1.58, 2.0   # LR: sarcomere length (um) in the reference state
T_MAX = 20.0                     # kPa
BCL_MS = 550.0                   # cycle length: Cairelli et al. 2024 3 dpf value (the imaged heart is 48 hpf, see README)
T0_MS = 120.0                    # time to peak tension
TR_M_MS_PER_UM = 73.0            # relaxation time tr = m ls + b (a field; ls = LR |F f0|): m as in Cairelli's heartFEM runs
TR_B_MS = -33.0                  # b tuned so that the P-V loop matches the constant-tr (100 ms) reference
ONSET_AFTER_ED_FRAMES = 0.5      # activation onset in frames after the maximum-volume frame (found from the tracking: t43
                                 # here). Usually 0; a small offset if the tracking is not dense in time (here 0.5)

# ---- movie (step plot)
FFMPEG = "ffmpeg"                # ffmpeg executable for output/pv_cycle.mp4 (skipped if not found)
MOVIE_FPS = 6

# ---- summaries
BASE_EXCLUDE_UM = 10.0           # wall means in summary.csv skip cells this close to the base (boundary-condition stress)
