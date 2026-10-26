# invert_para · PINN · one fiber family

**Parameter inversion by a parametric PINN: C and T_MAX from two tracked frames.**

From the tracked wall of two frames of one cardiac cycle of a 48 hpf zebrafish atrium, ED (end-diastole: maximum volume,
passive) and PS (peak systolic pressure: contracting), and the cavity pressure given at both frames, this example finds
two parameters of the heartFEM mechanics:

- **C**, the scale of the passive Fung law (kPa);
- **T_MAX**, the peak active fiber tension (kPa).

The fiber architecture is predefined, not inferred: two fiber families at 0° and 90° with equal shares (default), or one
family at a chosen angle. The solver is a parametric physics-informed network (PINN), trained once without any FE or
image data, that returns the equilibrium wall displacement for any (C, T_MAX, P); one inversion then takes seconds. The
example `../FE_one_fiber` solves the same problem with FE (same misfit, same search) and is the reference for this one.

This example has **one fiber family** (`TWO_FIBERS = False`), circumferential (0°). The example `../PINN_two_fibers` is
the same model with two families, circumferential and longitudinal, each with half of the myocardium.

## This example
```
PINN_one_fiber/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  all settings: run settings (`TASK = "invert_para"`, `SOLVER = "PINN"`, resources), then the model
├── input/     raw data: t12.stl (reference), t43.stl (ED), t51.stl (PS), base_plane.json
├── cache/     intermediate files: mesh/ (step preprocess), model/ (step train: the network)
└── output/    results: result.json, evaluation.json, ED.vtu, PS.vtu, comparison.png
```
The code is the library `../../../heartfem_mu/` (`preprocess.py`, `model.py`, `pinn.py`, `invert_para.py`, `plots.py`); the example folder holds the data, the settings and the results. `params.py` holds every setting of this example: the run settings (mode, resources, conda environments) at the top, then the model, the numerics and the output options, each with its meaning.

The git repository holds the small files of `output/` and `cache/` (figures, movies, tables, json, logs). The large ones
(`mesh/`, every `.vtu`, `.npz` and `.pt` file) are in the release archive
`heartfem_mu_v1.0.0_outputs_invert_para.tar.gz` (see Examples in `../../../README.md`); `python main.py` recomputes all
of them from `input/`.

## Run
```bash
python main.py                  # every step, input/ -> output/
```
The two conda environments (`CONDA_ENV`, `CONDA_ENV_FENICS` in `params.py`) are installed once with `bash ../../../environments/install.sh` (see Install in `../../../README.md`).

The steps (`python main.py --list`); each runs in its own process in the conda environment it needs:

| step | does | code (`heartfem_mu/`) | runs in |
|---|---|---|---|
| `preprocess` | raw surfaces -> tetrahedral mesh, local frame, cut surfaces | `preprocess.main` | `CONDA_ENV`, `MESH_THREADS` threads |
| `train` | train the (p, a) network (energy minimization) | `pinn.train_param_network` | `CONDA_ENV`, `GPU` |
| `invert` | search C, T_MAX with the network | `invert_para.pinn_inversion` | `CONDA_ENV`, `GPU` |
| `evaluate` | comparison with the tracking | `invert_para.evaluate` | `CONDA_ENV` |

`python main.py <step> ...` runs only the named steps, `python main.py --from <step>` a step and the following ones.
Resources (`params.py`): `GPU`, the GPU of the training and the solves (the `nvidia-smi` index; `None`: the CPU,
training about ten times slower), and `CPUS`, the cores the steps may use (a `taskset` list; `None`: any core).

## Input (`input/`)
Identical to `../FE_one_fiber`: `t12.stl` (reference frame, stress-free, the mesh), `t43.stl` (ED), `t51.stl` (PS),
`base_plane.json` (saved landmarks: base cut plane and apex). Frames 12, 43 and 51 of the 52 tracked frames of one cycle
of a 4D recording of the heart at 48 hpf, with one triangle connectivity.

## Pipeline
1. **`preprocess`**: as `../FE_one_fiber`: cut surfaces with point-to-point correspondence and the 5 um tetrahedral
   mesh of the reference frame with its local frame (`cache/mesh/`).
2. **`train`** (GPU): trains the network for the fibers of `params.py` by minimizing the potential
   energy over a range of loads (no FE or image data); writes `cache/model/`.
3. **`invert`**: the search of `../FE_one_fiber` with the network as the solver (geometry start; stages A and B from
   each start; lowest J over both frames); checks that the result lies in the training domain; writes
   `output/result.json`, `ED.vtu`, `PS.vtu`.
4. **`evaluate`**: as `../FE_one_fiber`: `output/evaluation.json`, `comparison.png`, distances added to the vtu files.

## Settings (`params.py`)
The folders, frames, `PS_ACTIVATION_MS`, pressures, fixed material constants, fibers, misfit and search settings are
identical to `../FE_one_fiber` and explained in its README (*Settings*), except the epicardial spring: `K_SPRING_OVER_C`
(K / C, 1/mm) here instead of `K_SPRING` (kPa/mm). Specific to the PINN:

| group | setting | meaning |
|---|---|---|
| physics | `KAPPA_OVER_C` | incompressibility penalty on the nodal volume change, in units of C (10000) |
| | `RIGID_OVER_C` | penalty against rigid translation in the base plane and rigid rotation (10000) |
| | `QUAD_ORDER` | quadrature rule of the tetrahedra (2: 8 points) |
| training domain | `P_MAX_N` | upper bound of the input p = P / C (10) |
| | `A_MAX_N` | upper bound of the input a = T_MAX Ct / C (110) |
| network | `WIDTH`, `DEPTH` | tanh MLP, 5 layers of 128 |
| | `LEVELS_UM` | coarse output heads on clusters of 8, 20, 50 um (help the optimization) |
| | `U_SCALE_MM` | output scale of the displacement (0.01 mm) |
| | `SEED` | random seed |
| Adam warm-up | `ADAM_STEPS`, `ADAM_LR`, `ADAM_LR_END`, `ADAM_LOADS`, `ADAM_CHUNK` | 1000 steps, learning rate 1e-5 → 2e-6, 8 random loads per step (in chunks of 4) |
| | `LOAD_RAMP_STEPS`, `KAPPA_RAMP_STEPS`, `KAPPA_START_OVER_C` | loads ramped from 0 over 200 steps; penalty from 100 to KAPPA over 400 steps |
| L-BFGS | `GRID_STAGES` | load grids (n_p, n_a, first outer step): 5 × 5 from step 1, 8 × 8 from 42, 14 × 14 from 101 |
| | `GRID_P0_N`, `GRID_A0_N` | spacing scales of the grids (0.2, 5): values x0 (e^(s ln(1 + x_max / x0)) − 1), dense at small loads |
| | `WEIGHT_POWER` | each load weighted by 1 / \|Pi\|^0.5 of its energy at the start of the grid stage |
| | `LBFGS_HISTORY`, `LBFGS_INNER`, `LBFGS_CHUNK` | L-BFGS memory (100), iterations per outer step (20), loads per chunk (11) |
| | `WHITEN`, `WHITEN_FROM_OUTER` | whitened output parametrization on the finest grid (from outer step 101) |
| stop | `STOP_WINDOW`, `STOP_TOL` | stop when the objective decreased by less than 1e-4 (relative) over the last 10 outer steps |
| | `TRAIN_HOURS` | safety cap on the training time (6 h) |

## Model
The mechanics are those of `../FE_one_fiber` (heartFEM laws; see its README): Fung passive law with C inferred, active
fiber tension with T_MAX inferred, predefined fibers (`TWO_FIBERS = True`: families at (0°, 0°) and (90°, 90°) as (endo,
epi), linear in the transmural depth, shares 0.5 / 0.5; `False`: one family at `FIBER_ANGLE_DEG` = (0°, 0°)), given
cavity pressure, base u · n = 0, no rigid motion, no epicardial spring by default. Optional: `K_SPRING_OVER_C` > 0
restores heartFEM's epicardial spring −K (u · N) N with K = `K_SPRING_OVER_C` × C (1/mm). It is given relative to C,
unlike `K_SPRING` of `../FE_one_fiber` (kPa/mm): the network works in units of C with two inputs (below), and a fixed K
would make K / C change with the unknown C. The network must be trained again for a new value. At ED Ct = 0; at PS
(`PS_ACTIVATION_MS` = 80 ms < t0 = 120 ms) Ct = 0.75, the same everywhere.

**Scaling.** With C, T_MAX and P multiplied by one factor every stress scales and the displacement does not change.
The network is therefore trained once in units of C, with two inputs:
```
p = P / C        in [0, 10]  (P_MAX_N)
a = T_MAX Ct / C in [0, 110] (A_MAX_N)
```
ED is (p = P_ED / C, a = 0), PS is (p = P_PS / C, a = 0.75 T_MAX / C). Ct is folded into a because on the rise it is
uniform over the wall (it does not depend on the sarcomere length); time is not an input. One trained network serves
every (C, T_MAX, P) whose (p, a) lies in this range; the step `invert` checks that the result does.

**Discretization and network** (as the volume-driven PINN, `../../volume_constrain/PINN_one_fiber`): displacement in the FE's
space, quadratic (P2) tetrahedra on the 5 um mesh (35706 nodes), 8-point quadrature; incompressibility as a penalty on
the nodal (P1-projected) volume change, KAPPA / C = 10000; rigid-body penalty / C = 10000. Network: features (p / 10,
a / 110) → 5 × 128 tanh MLP → linear nodal output plus three coarse heads on 8 / 20 / 50 um clusters;
u = 0.01 mm × (NN(p, a) − NN(0, 0)), so u = 0 unloaded; u · n = 0 on the base. The checkpoint stores the fibers.

**Training** (no FE or image data): minimize the total potential energy in units of C,
```
Pi / C = int sum_i w_i [(e^Q_i - 1) + Psi_act(lambda_i; a)] dV + (KAPPA/C)/2 sum m Jbar^2 - p (V - V0) + rigid-body penalty / C
```
over many loads: Adam 1000 steps on random loads (load and penalty ramps; a = 0 for a quarter of them), then
full-batch L-BFGS (history 100) on fixed load grids refined in three stages, 5 × 5 → 8 × 8 → 14 × 14 values of (p, a)
(24, 63, 195 loads, from outer step 1 / 42 / 101), spaced as x0 (e^(s ln(1 + x_max / x0)) − 1) with s uniform in
[0, 1] (x0 = 0.2 for p and 5 for a: dense at small loads), each load weighted by 1 / |Pi|^0.5; whitened output
parametrization on the finest grid; stop when the objective has flattened (relative decrease over 10 outer steps
below 1e-4).

## Inverse problem (step `invert`)
Identical to `../FE_one_fiber` (see its README for the details):
- **Given pressures:** P_ED = 0.1813 kPa (1.36 mmHg, 3 dpf ventricular end-diastolic pressure, Salehin et al. 2021);
  P_PS = 0.2738 kPa (2.05 mmHg = 3 dpf ventricular peak 7.52 mmHg × adult atrial / ventricular peak ratio 0.68 / 2.49,
  Hu et al. 2001, 2000). No atrial measurement exists at 48 hpf.
- **Misfit:** J = Σ over ED, PS of ((V − V_img) / (0.01 V_img))² + (d_endo / σ_s)² + 0.2 (d_epi / σ_s)², σ_s = 0.02 L,
  L = V0^(1/3) (scale-free); tracking rigidly aligned, symmetric point-to-plane surface distances, points within
  0.114 L of the base excluded.
- **Search:** geometry start (C0 from the passive ED volume, T0 from the PS volume, no guess), then for each of the
  starts (C0, T0) × (1, 1), (0.5, 2), (2, 0.5): Nelder-Mead stage A (ED: C) and stage B (PS: T_MAX); stop at a simplex
  size of 2e-4 and a change of J below 1e-4 (at most 250 evaluations). Each start is scored by J over both frames; the
  lowest gives the result.

## Output (`output/`)
Same files as `../FE_one_fiber`:

| file | content |
|---|---|
| `result.json` | C, T_MAX, T_MAX Ct at PS, fibers; misfit J; convergence and spread of the starts; per start the start point, each stage and the result; per frame the activation time, pressure, cavity and image volume; network inputs at the result and whether they lie in the training domain; evaluations, time |
| `evaluation.json` | per frame: endo / epi surface distance (rms, signed mean, max), cavity volume vs image, Dice (wall, cavity) |
| `ED.vtu`, `PS.vtu` | the deformed wall at each frame, in the coordinates of `input/t43.stl` / `input/t51.stl`; point data `displacement_um`, `distance_to_tracking_um`; cell data `fiber_1` (, `fiber_2`) |
| `comparison.png` | long-axis section through the apex at ED and PS, tracking vs model, with C and T_MAX |

`cache/` holds `mesh/` (preprocess products) and `model/` (`model.pt`: weights, input ranges, fibers and K / C;
`train_log.csv`: objective per step; `train_info.json`: time, steps, stop reason).

## Result
This example's run (`output/`), one fiber family 0°, pressures 0.1813 / 0.2738 kPa, with the FE result of
`../FE_one_fiber` (same input and settings):

| | C (kPa) | T_MAX (kPa) | J |
|---|---|---|---|
| geometry start | 0.07014 | 4.0039 | |
| start (1, 1) | 0.07108 | 4.0181 | 33.585 |
| start (0.5, 2) | 0.07108 | 4.0183 | 33.585 |
| start (2, 0.5) | 0.07107 | 4.0180 | 33.585 |
| **result** | **0.07108** | **4.0181** | **33.585** |
| FE (`../FE_one_fiber`) | 0.07060 | 4.0162 | 33.490 |
| PINN − FE | +0.68 % | +0.05 % | |

All stages of all starts stopped by the tolerances. The active tension at PS is T_MAX × Ct = 4.018 × 0.75 = 3.01 kPa.
Network inputs at the result: p_ED 2.55, p_PS 3.85, a_PS 42.4, inside the training domain (p ≤ 10, a ≤ 110). Against the
tracking (`output/evaluation.json`, `comparison.png`):

| | ED (t43) | PS (t51) |
|---|---|---|
| endocardial distance rms (signed mean, max) | 5.74 um (+0.42, 24.6) | 7.35 um (+1.17, 27.4) |
| epicardial distance rms (signed mean, max) | 5.22 um (+1.24, 22.4) | 7.42 um (+3.06, 24.9) |
| cavity volume vs image | 0.94732 vs 0.94912 nL (-0.19 %) | 0.69366 vs 0.69424 nL (-0.08 %) |
| DSC wall / cavity | 0.523 / 0.854 | 0.413 / 0.807 |

The example `../PINN_two_fibers` (two families 0° / 90°): C 0.07757 kPa, T_MAX 4.3545 kPa, J 25.82; ED endo / epi 5.65 /
5.32 um, PS 5.89 / 6.04 um. At PS two families fit the tracking better (endocardial rms 5.9 um with two families, 7.4 um
with one), with the same two unknowns.

## Accuracy against the FE example
Same input, settings and search as `../FE_one_fiber`; the network replaces the FE solves:

| | PINN | FE | PINN − FE |
|---|---|---|---|
| C (kPa) | 0.07108 | 0.07060 | +0.68 % |
| T_MAX (kPa) | 4.0181 | 4.0162 | +0.05 % |
| J | 33.585 | 33.490 | |
| ED endo / epi rms (um) | 5.74 / 5.22 | 5.73 / 5.21 | |
| ED DSC wall / cavity | 0.523 / 0.854 | 0.523 / 0.855 | |
| PS endo / epi rms (um) | 7.35 / 7.42 | 7.34 / 7.41 | |
| PS DSC wall / cavity | 0.413 / 0.807 | 0.413 / 0.807 | |
| evaluations | 166 | 166 | |

## Verification
Tests with the same network, training and search, at an activation time with Ct(PS) = 0.742 (79.3 ms; this scales T_MAX
by 0.75 / 0.742 and changes nothing else):

- **Energy.** With one family at 0° the energy equals that of a single-fiber network (difference 0.0 on four displacement fields and loads); two families at the same angle equal one family (any shares).
- **Forward accuracy against FE** at the same (C, T_MAX): cavity volume within 0.12 %, displacement error rms 0.06 - 0.14 um, maximum 0.3 um (one family at (0.0759, 5.544); two families at (0.077, 4.41) and (0.06, 6.0)); sigma_s is 1.76 um.
- **Recovery of known parameters** from FE solutions used as "tracking" (so including the network error): one family, truths (0.0759, 5.544), (0.05, 5.0), (0.10, 3.0) and the first with surface noise of 0.01 L and 0.02 L: C +0.45 to +0.63 %, T_MAX −0.05 to +0.2 %; two families, (0.077, 4.41), (0.06, 6.0) and the first with noise 0.02 L: C +0.6 to +0.7 %, T_MAX −0.3 to −0.2 %. The three starts always agree.
- **Ablations** (one change each, both fiber cases): stop tolerance 1e-5 instead of 1e-4 (training +20 %) reduced the C bias only slightly (0.7 → 0.5 % with two families); KAPPA / C 1e5 instead of 1e4 did not reduce it overall and doubled the training time. Both were not adopted.
- **Another training.** GPU training is not bit-reproducible. Two independent trainings of this recipe give C within 0.01 % and T_MAX within 0.003 % of each other.

## Limitations
- **Accuracy of the network.** Against FE the network keeps a small positive bias in C (0.5 - 0.7 %; its ED cavity
  volume is about 0.08 % too large at a given C) and T_MAX within 0.3 %. Training longer (stop tolerance 1e-5) or a 10×
  stiffer incompressibility penalty did not remove it (development study).
- **Training domain.** p ≤ 10 and a ≤ 110 cover this heart with a wide margin (result p 2.3 - 3.5, a 42); for
  another heart or other pressures check `inside_training_domain` in `result.json` and widen the range if needed.
- The limitations of the inverse problem itself (assumed pressures, activation timing and fibers; stages A and B fit
  the frames one after the other; the remaining misfit near the base) are those of `../FE_one_fiber`; see its README.

## References
See `../FE_one_fiber/README.md`; in addition, for the network training: Liu D. C., Nocedal J. (1989) On the limited
memory BFGS method for large scale optimization. Math Program 45:503-528.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`); third-party notices (heartFEM, three.js) in
`../../../THIRD_PARTY_NOTICES.md`.