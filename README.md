# HeartFEM_Mu

**Author:** Siyu Mu · **License:** MIT (`LICENSE`)

Mechanics of a beating heart-chamber wall reconstructed from 4D images, solved either by finite elements (FE, legacy
FEniCS) or by a physics-informed neural network (PINN), with the material and activation laws of heartFEM (Lee lab,
`lcleeHeart`). The data of the examples are the tracked wall of a 48 hpf zebrafish atrium (52 frames of one cycle).

Three tasks, each with an FE and a PINN solver and with one or two fiber families:

| task | question | input | main results |
|---|---|---|---|
| `volume_constrain` | P-V cycle of the tracked chamber: the cavity volume of every frame is prescribed | the 52 tracked frames | cavity pressure, displacement and stresses of every frame; P-V loop |
| `windkessel` | the chamber in a closed-loop circulation (heartFEM's coupling: a P-V-t table of the chamber in an ngspice circuit) | the reference frame | coupled cycle (pressures, volumes, flows, valve timing), and its displacement and stresses |
| `invert_para` | the passive stiffness C and the peak active tension T_MAX from two tracked frames (end-diastole, peak systole) with given cavity pressures | three frames (reference, ED, PS) | C, T_MAX, misfit, surface distances and Dice coefficients against the tracking |

The wall is the myocardium. The windkessel FE mode can also solve a wall of myocardium + endocardium (`WALL =
"myo_endo"`): a thin endocardial layer, grown from the myocardial inner surface into the cavity or given with the input
surface, solved together with the myocardium (example `windkessel/FE_two_fibers_endo`).

The FE solves a nonlinear problem per state (MPI on the CPU). The PINN is trained once per geometry, without FE or
image data, by minimizing the potential energy of the same mechanics; it then returns the equilibrium displacement of
any load in milliseconds (GPU).

## Layout
```
HeartFEM_Mu/
├── README.md              this file
├── LICENSE                MIT License
├── THIRD_PARTY_NOTICES.md the works HeartFEM_Mu contains or derives from (heartFEM, three.js) and their licenses
├── CITATION.cff           how to cite HeartFEM_Mu
├── environments/          the two conda environments (pinn_stress.yml, heartfem_fe.yml) and install.sh
├── heartfem_mu/           the library
│   ├── run.py             entry of main.py: settings check, the pipeline of each mode, environments and resources
│   ├── preprocess.py      raw surfaces -> tetrahedral wall mesh, local frame (eC, eL, eR), cut surfaces of every frame
│   ├── landmarks.py       browser app that places the base plane and the apex (only if USE_SAVED_LANDMARKS = False)
│   ├── model.py           shared by FE and PINN: wall boundary, fiber families, frame volumes, activation time, result files
│   ├── myo_endo.py        wall of myocardium + endocardium (WALL = "myo_endo", windkessel FE): the layer grown on the mesh or meshed from the input, two-region FE input
│   ├── fe.py              FE solver (dolfin): volume-driven problem (volume_constrain, windkessel), pressure-driven problem (invert_para)
│   ├── pinn.py            PINN solver (torch): material laws, P2 discretization, the networks, energies, training
│   ├── volume_constrain.py   task: P-V cycle (FE, PINN)
│   ├── windkessel.py      task: P-V-t table (FE sweep, PINN), heartFEM table files, ngspice circuit, replay
│   ├── invert_para.py     task: image misfit, search for C and T_MAX (FE, PINN), evaluation
│   ├── plots.py           all figures and movies
│   └── data/              circuit template atrium_wk.cir, heartFEM generateLVtable.py (original source), three.js
└── examples/
    ├── volume_constrain/{FE_one_fiber, FE_two_fibers, PINN_one_fiber, PINN_two_fibers}/
    ├── windkessel/{FE_one_fiber, FE_two_fibers, PINN_one_fiber, PINN_two_fibers, FE_two_fibers_endo}/
    └── invert_para/{FE_one_fiber, FE_two_fibers, PINN_one_fiber, PINN_two_fibers}/
        each:  main.py  params.py  input/  output/ (invert_para also cache/)  README.md
```
The library has two layers: data (`preprocess`, `model`, `myo_endo`) and solvers (`fe`, `pinn`) below the three task modules,
`plots` on top. `fe.py` is imported only in the FEniCS environment and `pinn.py` only in the torch environment; the other
modules need numpy and scipy (and matplotlib, vtk for the figures).

## Install
Linux x86-64 with conda (Miniforge or Miniconda). The PINN runs on an NVIDIA GPU with a driver for CUDA 12.1 (driver
>= 525); without a GPU set `GPU = None` in `params.py` (training about ten times slower).
```bash
bash environments/install.sh
```
creates the two conda environments from `environments/*.yml`, which pin the exact versions of the environments that
produced the example outputs, and checks them:
- `pinn_stress` (`CONDA_ENV`, Python 3.11): numpy, scipy, torch 2.5.1 with CUDA 12.1, matplotlib, trimesh, meshio, vtk,
  gmsh; preprocessing, the PINN and all figures. The movies use the ffmpeg binary of the pip package imageio-ffmpeg,
  which the script links as `<env>/bin/ffmpeg`.
- `heartfem_fe` (`CONDA_ENV_FENICS`, Python 3.8): legacy FEniCS 2019.1 (dolfin), mpi4py, petsc4py with MUMPS, MPICH,
  meshio, scipy, ngspice 41 (XSPICE code models); the FE steps and the circuit.

The checks: imports and versions, the GPUs torch sees, gmsh, ffmpeg, a small dolfin problem solved with MUMPS on 2 MPI
ranks (this also tests the compilers of the form JIT), ngspice. An environment that already exists is only checked.
```bash
bash environments/install.sh --check                    # only check the two environments
```
```bash
bash environments/install.sh --pinn myenv --fe myenv_fe # other names: then set CONDA_ENV, CONDA_ENV_FENICS in params.py
```
Without the script: `conda env create -f environments/pinn_stress.yml` and `conda env create -f
environments/heartfem_fe.yml`, then link ffmpeg as the script does (otherwise the movies are skipped). Nothing has to be
installed from this folder: `main.py` puts `HeartFEM_Mu/` on the Python path.

## Run an example
```bash
cd examples/volume_constrain/FE_two_fibers
python main.py                  # every step, from input/ to output/
```
```bash
python main.py --list           # the steps of this example
```
```bash
python main.py plot             # only the named steps (here: the figures again)
```
```bash
python main.py --from solve     # this step and the following ones
```
`main.py` runs in any Python 3; every step then runs in its own process in the conda environment it needs
(`conda activate` of `CONDA_ENV` or `CONDA_ENV_FENICS`). The steps of the six modes:

| mode | steps |
|---|---|
| volume_constrain FE | preprocess, solve (FE, MPI), plot |
| volume_constrain PINN | preprocess, train (GPU), solve (GPU), plot |
| windkessel FE | preprocess, pvt (FE sweep, parallel MPI jobs), table, figures_table, circuit, circuit_check, figures_cycle, replay (FE, MPI), figures_fields |
| windkessel PINN | preprocess, train (GPU), pvt (GPU), table, figures_table, circuit, circuit_check, figures_cycle, replay (GPU), figures_fields |
| invert_para FE | preprocess, invert (FE, MPI), evaluate |
| invert_para PINN | preprocess, train (GPU), invert (GPU), evaluate |

With `WALL = "myo_endo"` (windkessel FE) the step `endo_layer` (the endocardial layer grown on the mesh) follows
`preprocess` (`ENDO_GROW = True`), or `preprocess` meshes both layers of the input surface (`ENDO_GROW = False`).

## Settings
Each example's `params.py` holds every setting of its mode; `main.py` refuses a `params.py` with a missing or unknown
setting for its `TASK` and `SOLVER` (start a new case from the example of the same mode). The first section selects
the mode and the compute resources:

| setting | meaning |
|---|---|
| `TASK` | `"volume_constrain"`, `"windkessel"` or `"invert_para"` |
| `SOLVER` | `"FE"` or `"PINN"` |
| `TWO_FIBERS` | (fiber section) `True`: two fiber families, set by `FIBER_ANGLES_DEG` and `FIBER_WEIGHTS`; `False`: one, set by `FIBER_ANGLE_DEG`. `params.py` holds only the settings of its case: to switch, replace them (`main.py` names the missing ones; settings of the other case are not read) |
| `CPUS` | CPU cores the steps may use, a Linux `taskset` list such as `"0-7"` or `"0,2,4,6"`; `None`: any core |
| `MPI_RANKS` | FE: MPI ranks of the solve (one thread each) |
| `SWEEP_JOBS, SWEEP_RANKS` | windkessel FE: parallel jobs of the P(V, t) sweep x MPI ranks each |
| `GPU` | PINN: the GPU (index of `nvidia-smi`); `None`: the CPU |
| `CONDA_ENV`, `CONDA_ENV_FENICS` | the two conda environments |
| `MESH_THREADS` | (preprocess section) threads of the gmsh mesher; the mesh depends on it (see Reproducibility) |
| `WALL` | (optional, wall section) `"myo"`: the myocardium (also when `WALL` is absent, as in every example but one); `"myo_endo"`: myocardium + endocardial layer, windkessel FE only (every other mode refuses it), with the settings `ENDO_GROW` (`True`: the layer grown from the myocardial mesh, `ENDO_THICKNESS_UM`, `ENDO_SUBLAYERS`; `False`: the input surface has both layers), `ENDO_PRESSURE_JUMP`, `ENDO_MATERIAL` (see `examples/windkessel/FE_two_fibers_endo`) |

All other settings (geometry, fibers, material, activation, circulation, misfit, network, training) and their sources
are described in the example's `README.md`.

## Examples
Each example folder is complete: `input/` (the raw data), `params.py`, `main.py`, the `output/` of its own run from
`input/` with this library, and a `README.md` with the model, the settings and the results.

The git repository holds the small output files (figures, movies, tables, json, logs). The large ones (`mesh/`,
`cache/`, every `.vtu`, `.npz` and `.pt` file; 2.4 GB) are in three archives of the release
[v1.0.0](https://github.com/SiyuMU/HeartFEM_Mu/releases/tag/v1.0.0), one per task, to unpack in `HeartFEM_Mu/`:
```bash
curl -LO https://github.com/SiyuMU/HeartFEM_Mu/releases/download/v1.0.0/heartfem_mu_v1.0.0_outputs_volume_constrain.tar.gz
```
```bash
tar -xzf heartfem_mu_v1.0.0_outputs_volume_constrain.tar.gz
```
(the same for `windkessel` and `invert_para`; `SHA256SUMS` lists the checksums). The example
`windkessel/FE_two_fibers_endo`, added in v1.1.0, has its large files in one archive of the release
[v1.1.0](https://github.com/SiyuMU/HeartFEM_Mu/releases/tag/v1.1.0):
```bash
curl -LO https://github.com/SiyuMU/HeartFEM_Mu/releases/download/v1.1.0/heartfem_mu_v1.1.0_outputs_windkessel_FE_two_fibers_endo.tar.gz
```
```bash
tar -xzf heartfem_mu_v1.1.0_outputs_windkessel_FE_two_fibers_endo.tar.gz
```
(checksum in `SHA256SUMS_v1.1.0` of that release). Without the archives, `python main.py` recomputes every file from
`input/`.

<!-- EXAMPLES TABLE -->
| example | fibers | main result |
|---|---|---|
| `volume_constrain/FE_two_fibers` | two fiber families | P at max volume 0.238 kPa, peak P 1.348 kPa (t51), largest displacement 19.0 um |
| `volume_constrain/FE_one_fiber` | one fiber family | P at max volume 0.260 kPa, peak P 1.250 kPa (t51), largest displacement 23.6 um |
| `volume_constrain/PINN_two_fibers` | two fiber families | P at max volume 0.235 kPa, peak P 1.354 kPa (t51), largest displacement 18.9 um |
| `volume_constrain/PINN_one_fiber` | one fiber family | P at max volume 0.260 kPa, peak P 1.245 kPa (t51), largest displacement 24.0 um |
| `windkessel/FE_two_fibers` | two fiber families | atrial P max / min 4.11 / -0.22 mmHg; ventricle EDV / ESV 0.565 / 0.229 nL; mean arterial P 4.88 mmHg |
| `windkessel/FE_one_fiber` | one fiber family | atrial P max / min 3.84 / -0.23 mmHg; ventricle EDV / ESV 0.569 / 0.230 nL; mean arterial P 4.91 mmHg |
| `windkessel/PINN_two_fibers` | two fiber families | atrial P max / min 4.03 / -0.22 mmHg; ventricle EDV / ESV 0.567 / 0.229 nL; mean arterial P 4.89 mmHg |
| `windkessel/PINN_one_fiber` | one fiber family | atrial P max / min 3.84 / -0.22 mmHg; ventricle EDV / ESV 0.571 / 0.230 nL; mean arterial P 4.92 mmHg |
| `invert_para/FE_two_fibers` | two fiber families | C 0.07704 kPa, T_MAX 4.368 kPa, J 25.72 |
| `invert_para/FE_one_fiber` | one fiber family | C 0.07060 kPa, T_MAX 4.016 kPa, J 33.49 |
| `invert_para/PINN_two_fibers` | two fiber families | C 0.07757 kPa, T_MAX 4.355 kPa, J 25.82 |
| `invert_para/PINN_one_fiber` | one fiber family | C 0.07108 kPa, T_MAX 4.018 kPa, J 33.58 |
<!-- /EXAMPLES TABLE -->

Wall of myocardium + endocardium (`WALL = "myo_endo"`):

| example | wall | main result |
|---|---|---|
| `windkessel/FE_two_fibers_endo` | two fiber families, 2 um endocardial layer | atrial P max / min 3.66 / -0.24 mmHg; ventricle EDV / ESV 0.554 / 0.227 nL; mean arterial P 4.80 mmHg |

## Reproducibility
- **Preprocessing** is deterministic. gmsh meshes the surface patches in parallel, so the tetrahedra depend on the
  number of threads gmsh uses: `MESH_THREADS` (36 in `volume_constrain/*` and in the PINN examples, 1 in
  `windkessel/FE_*` and `invert_para/FE_*`). A node with fewer cores than `MESH_THREADS` gives a different, equally
  valid mesh; for the same reason the preprocess step ignores `CPUS`.
- **FE** runs agree with each other to round-off (relative 1e-15 in pressures and volumes; single float32 roundings in
  the stored stress fields): MPI with the MUMPS direct solver is not bit-reproducible from run to run. The windkessel
  sweep also depends at round-off level on the partition of the volumes over the jobs (`SWEEP_JOBS`).
- **PINN.** The steps after the training (solve, pvt, replay, invert, evaluate) reproduce the results of a given network
  to float32 roundings on the GPU (relative 1e-7; the search path of the inversion is the same). Training on a GPU is
  not bit-reproducible: a new training differs from the network in `output/` by the run-to-run spread given in each
  example's README (Accuracy, Verification).

## References
- heartFEM (Lee lab; W. X. Chan): `lcleeHeart` forms (`forms_shavik`, `simpleActiveMaterial`), `ngspice_py`.
- Cairelli A. G. et al. (2024) Role of tissue biomechanics in the formation and function of myocardial trabeculae in
  zebrafish embryos. J Physiol 602(4):597-617. doi:10.1113/JP285490
- Guccione J. M., Waldman L. K., McCulloch A. D. (1993) Mechanics of active contraction in cardiac muscle: Part II -
  cylindrical models of the systolic left ventricle. J Biomech Eng 115:82-90. doi:10.1115/1.2895474
- Salehin N. et al. (2021) Assessing pressure-volume relationship in developing heart of zebrafish in-vivo. Ann Biomed
  Eng 49:2080-2093.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`LICENSE`). HeartFEM_Mu contains a source file and the circuit of heartFEM
(MIT, Copyright (c) 2019 W. X. Chan) and three.js (MIT) for the landmark app; their notices are in
`THIRD_PARTY_NOTICES.md`.
