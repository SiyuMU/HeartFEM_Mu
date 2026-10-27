"""Run an example: read its params.py, check it, and run the pipeline of its mode (TASK, SOLVER) step by step.

    python main.py                  # every step
    python main.py <step> ...       # only these steps
    python main.py --from <step>    # this step and the following ones
    python main.py --list           # the steps of this example

Every step runs in its own process, in the conda environment it needs:
CONDA_ENV (numpy, scipy, matplotlib, trimesh, meshio, vtk, gmsh, torch) for preprocessing, the PINN and the figures,
CONDA_ENV_FENICS (legacy FEniCS 2019, mpi4py, ngspice) for the FE steps (MPI). Compute resources (params.py):
    CPUS          cores the steps may use (Linux taskset list, e.g. "0-7"; None: any). The preprocess step ignores it: the
                  tetrahedral mesh depends on the number of threads gmsh can use (MESH_THREADS).
    MPI_RANKS     FE: MPI ranks of the solve (one thread each); windkessel FE sweep: SWEEP_JOBS jobs x SWEEP_RANKS ranks.
    GPU           PINN: the GPU (nvidia-smi index; None: the CPU).
This file needs only the Python standard library (it runs in any Python 3 environment).
"""
import importlib
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]  # HeartFEM_Mu/
TASKS = ("volume_constrain", "windkessel", "invert_para")
SOLVERS = ("FE", "PINN")


# ==================== 1. pipelines ====================
@dataclass(frozen=True)
class Step:
    name: str
    function: str           # module.function of heartfem_mu, called with (cfg, *args)
    about: str
    fenics: bool = False    # runs in CONDA_ENV_FENICS (else CONDA_ENV)
    mpi: str = ""           # "ranks": mpirun -n MPI_RANKS; "jobs": SWEEP_JOBS parallel jobs of SWEEP_RANKS ranks
    gpu: bool = False       # uses the GPU (PINN)
    args: tuple = ()


PREPROCESS = Step("preprocess", "preprocess.main", "raw surfaces -> tetrahedral mesh, local frame, cut surfaces")
ENDO_LAYER = Step("endo_layer", "myo_endo.grow_layer", "endocardial layer grown on the mesh (WALL = myo_endo)")
PREPROCESS_LAYERS = Step("preprocess", "myo_endo.preprocess_layers",
                         "raw surfaces of both layers -> two-region tetrahedral mesh, local frame (WALL = myo_endo)")
PIPELINES = {
    ("volume_constrain", "FE"): (
        PREPROCESS,
        Step("solve", "volume_constrain.fe_cycle", "volume-driven FE cycle of the tracked frames", fenics=True, mpi="ranks"),
        Step("plot", "plots.pv_figures", "pv_loop.png/.pdf, pv_cycle.mp4"),
    ),
    ("volume_constrain", "PINN"): (
        PREPROCESS,
        Step("train", "pinn.train_load_network", "train the (P, t) network (energy minimization)", gpu=True),
        Step("solve", "volume_constrain.pinn_cycle", "volume-constrained cycle with the network", gpu=True),
        Step("plot", "plots.pv_figures", "pv_loop.png/.pdf, pv_cycle.mp4"),
    ),
    ("windkessel", "FE"): (
        PREPROCESS,
        Step("pvt", "windkessel.fe_sweep", "FE P(V, t) sweep", fenics=True, mpi="jobs"),
        Step("table", "windkessel.build_table", "heartFEM P-V-t table files", fenics=True),
        Step("figures_table", "plots.windkessel_figures", "pvt_table.png", args=("table",)),
        Step("circuit", "windkessel.circuit", "closed-loop circuit in ngspice", fenics=True),
        Step("circuit_check", "windkessel.circuit_check", "half time step, twice the cycles", fenics=True),
        Step("figures_cycle", "plots.windkessel_figures", "P-V loops, flows, metrics", args=("cycle",)),
        Step("replay", "windkessel.fe_replay", "coupled volume back through the FE model", fenics=True, mpi="ranks"),
        Step("figures_fields", "plots.windkessel_figures", "fields summary, pv_cycle.mp4", args=("fields",)),
    ),
    ("windkessel", "PINN"): (
        PREPROCESS,
        Step("train", "pinn.train_load_network", "train the (P, t) network (energy minimization)", gpu=True),
        Step("pvt", "windkessel.pinn_table", "P(V, t) table from the network", gpu=True),
        Step("table", "windkessel.build_table", "heartFEM P-V-t table files"),
        Step("figures_table", "plots.windkessel_figures", "pvt_table.png", args=("table",)),
        Step("circuit", "windkessel.circuit", "closed-loop circuit in ngspice"),
        Step("circuit_check", "windkessel.circuit_check", "half time step, twice the cycles"),
        Step("figures_cycle", "plots.windkessel_figures", "P-V loops, flows, metrics", args=("cycle",)),
        Step("replay", "windkessel.pinn_replay", "coupled volume back through the network", gpu=True),
        Step("figures_fields", "plots.windkessel_figures", "fields summary, pv_cycle.mp4", args=("fields",)),
    ),
    ("invert_para", "FE"): (
        PREPROCESS,
        Step("invert", "invert_para.fe_inversion", "search C, T_MAX with the FE model", fenics=True, mpi="ranks"),
        Step("evaluate", "invert_para.evaluate", "comparison with the tracking"),
    ),
    ("invert_para", "PINN"): (
        PREPROCESS,
        Step("train", "pinn.train_param_network", "train the (p, a) network (energy minimization)", gpu=True),
        Step("invert", "invert_para.pinn_inversion", "search C, T_MAX with the network", gpu=True),
        Step("evaluate", "invert_para.evaluate", "comparison with the tracking"),
    ),
}


# ==================== 2. settings ====================
def _names(text):
    return tuple(text.split())


_FIBERS = _names("TWO_FIBERS")  # + the fiber settings of its case, FIBER_SETTINGS
_LAWS = _names("C_FF C_SS C_NN C_NS C_FS C_FN CA0 CA0MAX B L0 LR T0_MS")
_FORWARD = _names("INPUT_DIR OUTPUT_DIR MESH_SIZE MESH_THREADS USE_SAVED_LANDMARKS MM_PER_UNIT C K_SPRING T_MAX BCL_MS "
                  "TR_M_MS_PER_UM TR_B_MS BASE_EXCLUDE_UM FFMPEG MOVIE_FPS") + _FIBERS + _LAWS
_VOLUME = _names("N_FRAMES ONSET_AFTER_ED_FRAMES")
_WINDKESSEL = _names("IMAGED_V_MAX_NL IMAGED_V_MIN_NL SWEEP_N_LOG SWEEP_DT_MS SWEEP_T_END_MS NGSPICE CIRCUIT "
                     "ATRIAL_CONTRACTION_END_MS N_REPLAY")
_TRAINING = _names("QUAD_ORDER WIDTH DEPTH LEVELS_UM U_SCALE_MM SEED ADAM_STEPS ADAM_LR ADAM_LR_END ADAM_LOADS ADAM_CHUNK "
                   "LOAD_RAMP_STEPS KAPPA_RAMP_STEPS GRID_STAGES WEIGHT_POWER LBFGS_HISTORY LBFGS_INNER LBFGS_CHUNK "
                   "WHITEN WHITEN_FROM_OUTER STOP_WINDOW STOP_TOL TRAIN_HOURS")
_LOAD_NETWORK = _TRAINING + _names("KAPPA RIGID_PENALTY P_MAX FEATURE_STRETCHES KAPPA_START GRID_P0 GRID_T0 "
                                   "GRID_RELAX_FACTOR GRID_RELAX_END_MS BISECTION_STEPS")
_INVERSE = _names("INPUT_DIR CACHE_DIR OUTPUT_DIR MESH_SIZE MESH_THREADS USE_SAVED_LANDMARKS REF_FRAME ED_FRAME PS_FRAME "
                  "PS_ACTIVATION_MS MM_PER_UNIT P_ED_KPA P_PS_KPA SIGMA_V SIGMA_S_REL W_EPI BASE_EXCLUDE_REL START_FACTORS "
                  "XATOL FATOL MAXFEV STEP") + _FIBERS + _LAWS
_PARAM_NETWORK = _TRAINING + _names("K_SPRING_OVER_C KAPPA_OVER_C RIGID_OVER_C P_MAX_N A_MAX_N KAPPA_START_OVER_C "
                                    "GRID_P0_N GRID_A0_N")
_RUN_FE = _names("TASK SOLVER CPUS MPI_RANKS CONDA_ENV CONDA_ENV_FENICS")
_RUN_PINN = _names("TASK SOLVER CPUS GPU CONDA_ENV")
PARAMETERS = {  # every setting of an example's params.py, per mode (+ FIBER_SETTINGS of its fiber case)
    ("volume_constrain", "FE"): _RUN_FE + _FORWARD + _VOLUME,
    ("volume_constrain", "PINN"): _RUN_PINN + _FORWARD + _VOLUME + _LOAD_NETWORK,
    ("windkessel", "FE"): _RUN_FE + _names("SWEEP_JOBS SWEEP_RANKS") + _FORWARD + _WINDKESSEL,
    ("windkessel", "PINN"): _RUN_PINN + _names("CONDA_ENV_FENICS P_MIN GRID_NEGATIVE") + _FORWARD + _WINDKESSEL + _LOAD_NETWORK,
    ("invert_para", "FE"): _RUN_FE + _INVERSE + _names("K_SPRING"),
    ("invert_para", "PINN"): _RUN_PINN + _INVERSE + _PARAM_NETWORK,
}
FIBER_SETTINGS = {  # the fiber settings that TWO_FIBERS selects; params.py needs only those of its case
    False: _names("FIBER_ANGLE_DEG"),
    True: _names("FIBER_ANGLES_DEG FIBER_WEIGHTS"),
}
# Wall (optional setting WALL): "myo" (myocardium only; also when WALL is absent) or "myo_endo" (myocardium + endocardial
# layer, windkessel FE only), which needs these settings in addition; ENDO_MATERIAL holds the layer's values of the
# keys below and its fiber settings of the TWO_FIBERS case. ENDO_GROW selects where the layer comes from: True, grown
# from the myocardial mesh (step endo_layer; needs GROW_SETTINGS); False, the input surface has both layers (step
# preprocess meshes them; GROW_SETTINGS not read).
WALLS = ("myo", "myo_endo")
WALL_MODES = {("windkessel", "FE")}  # the modes that accept WALL = "myo_endo"
WALL_SETTINGS = _names("ENDO_GROW ENDO_PRESSURE_JUMP ENDO_MATERIAL")
GROW_SETTINGS = _names("ENDO_THICKNESS_UM ENDO_SUBLAYERS")
ENDO_MATERIAL_KEYS = _names("C C_FF C_SS C_NN C_NS C_FS C_FN T_MAX")


def load(example):
    """Settings of an example (its params.py) as attributes, with the folders: root, params_file, input, output, cache
    (invert_para), mesh (the preprocess products: OUTPUT_DIR/mesh, invert_para CACHE_DIR/mesh)."""
    root = Path(example).resolve()
    params_file = root / "params.py"
    if not params_file.is_file():
        raise FileNotFoundError(f"no params.py in {root}")
    namespace = {"__file__": str(params_file), "__name__": "params"}
    exec(compile(params_file.read_text(), str(params_file), "exec"), namespace)  # no __pycache__ in the example
    values = {k: v for k, v in namespace.items() if k.isupper()}
    task, solver = values.get("TASK"), values.get("SOLVER")
    if task not in TASKS or solver not in SOLVERS:
        raise ValueError(f"{params_file}: TASK must be one of {TASKS} and SOLVER one of {SOLVERS}, got {task!r}, {solver!r}")
    two = values.get("TWO_FIBERS", False)
    if not isinstance(two, bool):
        raise ValueError(f"{params_file}: TWO_FIBERS must be True or False, got {two!r}")
    expected = set(PARAMETERS[task, solver]) | set(FIBER_SETTINGS[two])
    wall = values.get("WALL", "myo")
    if wall not in WALLS:
        raise ValueError(f"{params_file}: WALL must be one of {WALLS}, got {wall!r}")
    not_read = set(FIBER_SETTINGS[not two])  # the other case's fiber settings: not read
    if wall == "myo_endo":
        if (task, solver) not in WALL_MODES:
            raise ValueError(f"{params_file}: WALL = 'myo_endo' (myocardium + endocardium) is available for TASK = "
                             f"'windkessel', SOLVER = 'FE' only, got TASK = {task!r}, SOLVER = {solver!r}")
        grow = values.get("ENDO_GROW")
        if not isinstance(grow, bool):
            raise ValueError(f"{params_file}: ENDO_GROW must be True (layer grown from the myocardial mesh) or False "
                             f"(the input surface has both layers), got {grow!r}")
        expected |= set(WALL_SETTINGS) | (set(GROW_SETTINGS) if grow else set())
        not_read |= set() if grow else set(GROW_SETTINGS)
    missing = sorted(expected - set(values))
    unknown = sorted(set(values) - expected - not_read - {"WALL"})
    if missing or unknown:
        raise ValueError(f"{params_file}: settings do not fit TASK = {task!r}, SOLVER = {solver!r}"
                         + (f"; missing: {', '.join(missing)}" if missing else "")
                         + (f"; unknown: {', '.join(unknown)}" if unknown else "")
                         + " (start from the params.py of the example of this mode)")
    if wall == "myo_endo":
        keys = set(ENDO_MATERIAL_KEYS) | set(FIBER_SETTINGS[two])
        m = values["ENDO_MATERIAL"]
        if not isinstance(m, dict) or set(m) != keys:
            raise ValueError(f"{params_file}: ENDO_MATERIAL must be a dict with the keys {', '.join(sorted(keys))}"
                             + (f"; missing: {', '.join(sorted(keys - set(m)))}; unknown: {', '.join(sorted(set(m) - keys))}"
                                if isinstance(m, dict) else ""))
    cfg = SimpleNamespace(**values)
    cfg.root, cfg.params_file = root, params_file
    cfg.input, cfg.output = root / cfg.INPUT_DIR, root / cfg.OUTPUT_DIR
    if task == "invert_para":
        cfg.cache = root / cfg.CACHE_DIR
    cfg.mesh = (cfg.cache if task == "invert_para" else cfg.output) / "mesh"
    return cfg


def check_input(cfg):
    """The input files the mode needs (before the first step)."""
    need = [cfg.input / "base_plane.json"]
    if cfg.TASK == "volume_constrain":
        need += [cfg.input / f"t{k}.stl" for k in range(cfg.N_FRAMES)]
    elif cfg.TASK == "invert_para":
        need += [cfg.input / f"t{k}.stl" for k in (cfg.REF_FRAME, cfg.ED_FRAME, cfg.PS_FRAME)]
    elif len(list(cfg.input.glob("*.stl"))) != 1:
        raise ValueError(f"windkessel needs exactly one STL file (the reference frame) in {cfg.input}")
    missing = [str(p.relative_to(cfg.root)) for p in need if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"input files missing for TASK = {cfg.TASK!r}: {', '.join(missing)}")


def cpu_list(cpus):
    """CPUS as a taskset list ("0-7", "0,2,4"), or None."""
    if cpus is None:
        return None
    if isinstance(cpus, int):
        raise ValueError("CPUS is a list of cores, e.g. \"0-7\" or (0, 1, 2, 3), not a count")
    text = ",".join(str(int(c)) for c in cpus) if isinstance(cpus, (list, tuple, range)) else str(cpus).replace(" ", "")
    count = 0
    for part in text.split(","):
        a, _, b = part.partition("-")
        count += int(b or a) - int(a) + 1
    return text, count


# ==================== 3. running the steps ====================
def conda_base():
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if exe is None:
        raise RuntimeError("conda not found (CONDA_EXE / PATH): needed to activate CONDA_ENV and CONDA_ENV_FENICS")
    return subprocess.run([exe, "info", "--base"], capture_output=True, text=True, check=True).stdout.strip()


def step_process(cfg, step, base, args=()):
    """(bash command, environment) of one step."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # no __pycache__ in the library and example folders
    cmd = ["python", "-u", "-m", "heartfem_mu", str(cfg.root), step.name, *map(str, args)]
    if step.mpi:
        ranks = cfg.MPI_RANKS if step.mpi == "ranks" else cfg.SWEEP_RANKS
        cmd = ["mpirun", "-n", str(ranks)] + cmd
    cpus = cpu_list(cfg.CPUS)
    if step.name == PREPROCESS.name:  # gmsh meshes the surface patches in parallel: the mesh depends on the thread count
        env["OMP_NUM_THREADS"] = str(cfg.MESH_THREADS)
    else:
        if cpus:
            cmd = ["taskset", "-c", cpus[0]] + cmd
        if cfg.SOLVER == "FE":  # one thread per process (per MPI rank)
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if cfg.SOLVER == "PINN":
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = "" if cfg.GPU is None else str(cfg.GPU)
    name = cfg.CONDA_ENV_FENICS if step.fenics else cfg.CONDA_ENV
    script = f'set -e; source "{base}/etc/profile.d/conda.sh"; conda activate {shlex.quote(name)}; '
    if cfg.TASK == "windkessel" and not step.fenics:  # ngspice of CONDA_ENV_FENICS if CONDA_ENV has none
        script += (f'command -v {shlex.quote(cfg.NGSPICE)} > /dev/null || '
                   f'export PATH="$PATH:{base}/envs/{cfg.CONDA_ENV_FENICS}/bin"; ')
    return script + "exec " + shlex.join(cmd), env


def pipeline(cfg):
    """The steps of the example's mode; WALL = "myo_endo" adds the step endo_layer after preprocess (ENDO_GROW) or meshes
    both layers in the step preprocess (ENDO_GROW False)."""
    steps = PIPELINES[cfg.TASK, cfg.SOLVER]
    if getattr(cfg, "WALL", "myo") == "myo_endo":
        steps = steps[:1] + (ENDO_LAYER,) + steps[1:] if cfg.ENDO_GROW else (PREPROCESS_LAYERS,) + steps[1:]
    return steps


def run_steps(cfg, steps):
    base = conda_base()
    print(f"== {cfg.root.name}: TASK {cfg.TASK}, SOLVER {cfg.SOLVER}, fibers: {'two families' if cfg.TWO_FIBERS else 'one family'}"
          + ("; wall: myocardium + endocardial layer" + (" (grown)" if cfg.ENDO_GROW else " (from the input surface)")
             if getattr(cfg, "WALL", "myo") == "myo_endo" else "")
          + f"; steps {', '.join(s.name for s in steps)}", flush=True)
    unused = [k for k in FIBER_SETTINGS[not cfg.TWO_FIBERS] if hasattr(cfg, k)]
    if unused:
        print(f"   ({', '.join(unused)} in params.py not used: TWO_FIBERS = {cfg.TWO_FIBERS})", flush=True)
    unused = [k for k in GROW_SETTINGS if getattr(cfg, "WALL", "myo") == "myo_endo" and not cfg.ENDO_GROW and hasattr(cfg, k)]
    if unused:
        print(f"   ({', '.join(unused)} in params.py not used: ENDO_GROW = False)", flush=True)
    t0 = time.time()
    for i, step in enumerate(steps, 1):
        print(f"== step {i}/{len(steps)}: {step.name} ({step.about}) {time.strftime('%F %T')}", flush=True)
        if step.mpi == "jobs":  # independent sweep jobs side by side, each with its own log
            (cfg.output / "pvt").mkdir(parents=True, exist_ok=True)
            procs = []
            for j in range(cfg.SWEEP_JOBS):
                script, env = step_process(cfg, step, base, (j, cfg.SWEEP_JOBS))
                log = open(cfg.output / "pvt" / f"job{j}of{cfg.SWEEP_JOBS}.stdout", "w")
                procs.append((subprocess.Popen(["bash", "-c", script], env=env, cwd=cfg.root, stdout=log,
                                               stderr=subprocess.STDOUT), log))
            codes = [p.wait() for p, _ in procs]
            for _, log in procs:
                log.close()
            failed = [j for j, c in enumerate(codes) if c]
            if failed:
                raise SystemExit(f"step {step.name} failed: jobs {failed} (logs output/pvt/job*.stdout)")
        else:
            script, env = step_process(cfg, step, base)
            if subprocess.run(["bash", "-c", script], env=env, cwd=cfg.root).returncode:
                raise SystemExit(f"step {step.name} failed")
    print(f"== done in {(time.time() - t0) / 60:.1f} min {time.strftime('%F %T')}", flush=True)


def run(main_file, argv=()):
    """Entry of main.py: the example is the folder of main_file."""
    cfg = load(Path(main_file).resolve().parent)
    steps = pipeline(cfg)
    names = [s.name for s in steps]
    argv = list(argv)
    if argv[:1] in (["--list"], ["-l"]):
        for s in steps:
            print(f"{s.name:15s} {s.about}{'  [FE, MPI]' if s.fenics else ''}{'  [GPU]' if s.gpu else ''}")
        return
    if argv[:1] == ["--from"]:
        if len(argv) != 2 or argv[1] not in names:
            raise SystemExit(f"usage: python main.py --from <step>, steps: {' '.join(names)}")
        chosen = steps[names.index(argv[1]):]
    elif argv:
        unknown = [a for a in argv if a not in names]
        if unknown:
            raise SystemExit(f"unknown steps {unknown}; steps of this example: {' '.join(names)}")
        chosen = [s for s in steps if s.name in argv]
    else:
        chosen = steps
    if chosen[0].name == PREPROCESS.name:
        check_input(cfg)
    if cfg.SOLVER == "FE" or any(s.fenics for s in chosen):
        cpus = cpu_list(cfg.CPUS)
        ranks = max(cfg.MPI_RANKS, cfg.SWEEP_JOBS * cfg.SWEEP_RANKS if cfg.TASK == "windkessel" else 0)
        if cpus and cpus[1] < ranks:
            print(f"warning: CPUS has {cpus[1]} cores for {ranks} MPI ranks", flush=True)
    run_steps(cfg, chosen)


def run_step(example, name, args):
    """One step in this process (python -m heartfem_mu <example> <step> [args]; started by run_steps)."""
    cfg = load(example)
    step = {s.name: s for s in pipeline(cfg)}[name]
    module, function = step.function.rsplit(".", 1)
    getattr(importlib.import_module(f"heartfem_mu.{module}"), function)(cfg, *step.args, *args)
