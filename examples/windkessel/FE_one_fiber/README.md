# windkessel · FE · one fiber family

**FE atrium in a closed-loop circulation (Windkessel coupling).**

A finite-element model of the beating zebrafish embryonic atrium (48 hpf), coupled to a lumped-parameter (Windkessel)
circulation. Unlike a volume-driven model, the atrial volume is not prescribed: it follows from the balance between the
atrial pressure (from the FE model) and the rest of the circulation (a circuit solved with ngspice).

The coupling follows heartFEM (Lee lab) and the zebrafish circuit of Cairelli et al. 2024:

1. the FE model is evaluated once on a grid of cavity volumes and activation times, giving a pressure table P(V, t);
2. the circuit reads this table as its atrial element and is run to a periodic steady state;
3. the atrial volume of the coupled cycle is sent back through the FE model, which returns the wall displacement and
   stresses of that cycle and closes the loop.

In Cairelli's circuit the ventricle is the detailed chamber. Here the roles are swapped: the **atrium is the FE
chamber** and the **ventricle is a lumped time-varying elastance**.

This example has **one fiber family** (`TWO_FIBERS = False`), circumferential (0°). The example `../FE_two_fibers` is
the same model with two families, circumferential and longitudinal, each with half of the myocardium. The circuit
parameters are the same in both.

## This example
```
FE_one_fiber/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  all settings: run settings (`TASK = "windkessel"`, `SOLVER = "FE"`, resources), then the model
├── input/     raw data: t12.stl (the reference frame), base_plane.json
└── output/    result figures at the top, the data of every step in sub-folders (mesh, pvt, table, circuit, replay)
```
The code is the library `../../../heartfem_mu/` (`preprocess.py`, `model.py`, `fe.py`, `windkessel.py`, `plots.py`, `data/atrium_wk.cir`, `data/generateLVtable.py`); the example folder holds the data, the settings and the results. `params.py` holds every setting of this example: the run settings (mode, resources, conda environments) at the top, then the model, the numerics and the output options, each with its meaning.

The git repository holds the small files of `output/` (figures, movies, tables, json, logs). The large ones (`mesh/`,
every `.vtu`, `.npz` and `.pt` file) are in the release archive `heartfem_mu_v1.0.0_outputs_windkessel.tar.gz` (see
Examples in `../../../README.md`); `python main.py` recomputes all of them from `input/`.

## Run
```bash
python main.py                  # every step, input/ -> output/
```
The two conda environments (`CONDA_ENV`, `CONDA_ENV_FENICS` in `params.py`) are installed once with `bash ../../../environments/install.sh` (see Install in `../../../README.md`).

The steps (`python main.py --list`); each runs in its own process in the conda environment it needs:

| step | does | code (`heartfem_mu/`) | runs in |
|---|---|---|---|
| `preprocess` | raw surfaces -> tetrahedral mesh, local frame, cut surfaces | `preprocess.main` | `CONDA_ENV`, `MESH_THREADS` threads |
| `pvt` | FE P(V, t) sweep | `windkessel.fe_sweep` | `CONDA_ENV_FENICS`, `SWEEP_JOBS` MPI jobs x `SWEEP_RANKS` ranks |
| `table` | heartFEM P-V-t table files | `windkessel.build_table` | `CONDA_ENV_FENICS` |
| `figures_table` | pvt_table.png | `plots.windkessel_figures` | `CONDA_ENV` |
| `circuit` | closed-loop circuit in ngspice | `windkessel.circuit` | `CONDA_ENV_FENICS` |
| `circuit_check` | half time step, twice the cycles | `windkessel.circuit_check` | `CONDA_ENV_FENICS` |
| `figures_cycle` | P-V loops, flows, metrics | `plots.windkessel_figures` | `CONDA_ENV` |
| `replay` | coupled volume back through the FE model | `windkessel.fe_replay` | `CONDA_ENV_FENICS`, MPI, `MPI_RANKS` ranks |
| `figures_fields` | fields summary, pv_cycle.mp4 | `plots.windkessel_figures` | `CONDA_ENV` |

`python main.py <step> ...` runs only the named steps, `python main.py --from <step>` a step and the following ones.
Resources (`params.py`): `CPUS`, the cores the steps may use (a `taskset` list; `None`: any core), and `MPI_RANKS`
(replay) and `SWEEP_JOBS` x `SWEEP_RANKS` (sweep); the FE steps run one thread per MPI rank.

## Input (`input/`)
| file | content |
|---|---|
| `t12.stl` | Surface of the reference frame, in um: the closed endocardial and epicardial surfaces of a 48 hpf zebrafish atrium. It is frame t12 of the 52 tracked frames of one cycle of a 4D (3D + time) recording of the heart at 48 hpf. |
| `base_plane.json` | Saved landmarks for `t12.stl`: the base cut plane and the apex. Placed by hand once in the landmark browser app (`heartfem_mu/landmarks.py`, `USE_SAVED_LANDMARKS = False`); the default run only reads it. |

The coupling uses the images for two things only:
- the mesh, from the reference frame t12 (the only frame in `input/`);
- the range of cavity volumes that the pressure table must cover. It is built from the largest and smallest cavity
  volume of the imaged cycle, which are two settings in `params.py` (`IMAGED_V_MAX_NL` = 0.9491 nL, frame t43, and
  `IMAGED_V_MIN_NL` = 0.4960 nL, frame t5). The other 51 frames are not needed and are not part of this example.

The imaged volume curve is not prescribed anywhere.

## FE model of the atrium (`heartfem_mu/fe.py`)
The mechanics are heartFEM's (`lcleeHeart`): a Fung-type hyperelastic, incompressible myocardium with heartFEM's
time-varying active fiber stress. Lengths are in mm in the solver, stresses in kPa.

- **Geometry.** The wall is frame t12 cut at the base plane, meshed with linear tetrahedra (5 um). Frame t12 is the
  stress-free reference configuration. Local frame per cell from two Laplace problems: eL (longitudinal), eR (radial),
  eC = eR x eL. Fibers: family i, f_i = cos θ_i eC + sin θ_i eL, sheet s0 = eR, normal n_i = f_i x s0. Default two
  families, `FIBER_ANGLES_DEG` = ((0°, 0°), (90°, 90°)), circumferential and longitudinal, with shares
  w_i = `FIBER_WEIGHTS` = 0.5 / 0.5; `TWO_FIBERS = False`: one family at `FIBER_ANGLE_DEG` = (0°, 0°) (circumferential).
  Every angle is an (endo, epi) pair, linear in the transmural depth d_endo / (d_endo + d_epi) (the defaults are the same at every depth).
- **Passive myocardium** (heartFEM `forms_shavik.PassiveMatSEF`): W = C (e^Q - 1) - p (J - 1), with
  Q = C_ff E_ff² + C_ss E_ss² + C_nn E_nn² + C_ns E_ns² + C_fs E_fs² + C_fn E_fn² in the (f0, s0, n0) frame; p enforces
  J = 1. Two families: W = Σ_i w_i C (e^(Q_i) - 1) - p (J - 1), Q_i in the frame (f_i, s0, n_i).
- **Active stress** (heartFEM `simpleActiveMaterial`, trackphase off):
  S_act = T_MAX · Ct(t, ls) · Ca0² / (Ca0² + ECa(ls)²) · f0 ⊗ f0, with the current sarcomere length ls = LR |F f0|,
  ECa = Ca0max / sqrt(exp(B (ls - l0)) - 1), and

  ```
  Ct = 1/2 (1 - cos(pi t / t0))            0 <= t < t0
  Ct = 1/2 (1 + cos(pi (t - t0) / tr))     t0 <= t < t0 + tr,     tr = m ls + b
  Ct = 0                                    afterwards
  ```
  t is the time since the atrial activation onset. The relaxation time tr depends on the local sarcomere length, so
  Ct is a field: longer fibers relax later. Two families: S_act = Σ_i w_i T_MAX Ct_i g(ls_i) f_i ⊗ f_i, each family
  with its own ls_i = LR |F f_i|, tr_i and Ct_i; T_MAX is the tension of the whole myocardium (shares sum to 1).
- **Cavity.** The cavity pressure P is the Lagrange multiplier of the constraint V(u) = V_target (heartFEM's
  volume-controlled formulation), V(u) = -1/3 ∫_endo (X + u) · cof(F) N dS.
- **Boundary conditions.** The base moves in its plane only (u · n = 0); rigid translation in the base plane and
  rigid rotation are removed by multipliers. There is no epicardial spring: at 48 hpf the heart beats in a
  fluid-filled pericardial cavity and has no epicardium yet (Peralta et al. 2013, 2014; Serluca 2008). Optional:
  `K_SPRING` > 0 restores heartFEM's epicardial spring (traction −K (u · N) N, N the reference epicardial normal;
  default 0).
- **Discretisation.** u in P2, p in P1, P and the rigid-body multipliers as global unknowns; Newton with a direct
  solver (MUMPS) and automatic sub-stepping.

A **state** is a pair (cavity volume V, activation time t). The model is quasi-static and has no history variable, so
its solution, and in particular the cavity pressure, depends only on the current (V, t). This is what allows the table
look-up below.

| setting | value | source |
|---|---|---|
| passive C, C_ff, C_ss = C_nn, C_ns, C_fs = C_fn | 0.1 kPa, 29.9, 13.3, 26.6, 53.2 | zebrafish ventricle literature (Salehin et al. 2021); heartFEM law |
| Ca0 = Ca0max, B, l0 | 4.35, 4.75 /um, 1.58 um | heartFEM defaults |
| LR (sarcomere length of the reference state) | 2.0 um | tuned on this atrium |
| T_MAX | 20 kPa | tuned on this atrium |
| t0 (time to peak tension) | 120 ms | tuned on this atrium |
| tr = m ls + b | m = 73 ms/um, b = -33 ms | m: Cairelli's heartFEM runs; b tuned on this atrium |
| cycle length | 550 ms | Cairelli et al. 2024, 3 dpf (see Limitations) |

"Tuned on this atrium" refers to the volume-driven P-V loop of the same FE model, the example
`../../volume_constrain/FE_one_fiber`.

## Step pvt: pressure table P(V, t)
As heartFEM's `getLVbehavior`: for every volume of the grid, a passive solve at that volume, then the activation time is
advanced at fixed volume and the cavity pressure is recorded.

- **Volume grid** (heartFEM `LVbehaviorRun` rule): 50 log-spaced volumes from 0.9 ESV* to 1.1 EDV, plus 1.1x and
  1.3x the largest and 0.9x and 0.7x the smallest; 54 volumes, 0.24 to 1.36 nL. EDV = 0.949 nL and ESV = 0.496 nL are
  the largest and smallest imaged cavity volume (`params.IMAGED_V_MAX_NL`, `IMAGED_V_MIN_NL`); ESV* = 0.4 EDV because (EDV - ESV) / EDV < 0.6.
- **Time grid:** 0 to 400 ms in 5 ms steps. As in heartFEM, a volume stops two steps after the tension is off (between
  240 and 305 ms, later for larger volumes); from then on the pressure is the passive pressure. A 2.5 ms step changes
  P by 0.001 mmHg.
- **Not converged:** the four volumes 0.239, 0.342, 0.350 and 0.358 nL fail between 75 and 105 ms (the wall collapses
  under high tension) and are left out. The table has 50 volumes, 0.31 to 1.36 nL; the coupled atrium stays between
  0.47 and 0.83 nL.

## Step table: table files
The sweep is interpolated in time (cubic spline) to heartFEM's time grid, cut to one cycle (0 to 550 ms; heartFEM writes
two cycles, the circuit reads the cyclic time only), split into a passive part P_passive(V) and an active part
P_active(V, t) = P - P_passive, and passed to heartFEM's own `generateLVtable` (original source in
`heartfem_mu/data/generateLVtable.py`, executed unchanged except for one guard, see `windkessel.run_generate_lv_table`).
It writes the ngspice `table2d` files: 50 volumes x 553 times, volume in nL, time in s, pressure in mmHg.

## Steps circuit, circuit_check: the circuit (`heartfem_mu/data/atrium_wk.cir`)
Pressure is the voltage (mmHg), flow the current (nL/s), volume the charge (nL).

```
 venous pool --R_in (no valve)--> ATRIUM --R_AV + AV valve--> VENTRICLE --R_out + outflow valve--> arterial pool
   C_ven  ^                     FE table                      elastance                             C_art   |
          +------------------------------------ R_sys (systemic resistance) <------------------------------+
```

| element | law |
|---|---|
| venous and arterial pool | C dP/dt = inflow - outflow |
| R_in, R_sys | Q = ΔP / R |
| R_AV + AV valve, R_out + outflow valve | Q = ΔP / R if ΔP > 0, else 0 (ideal diode in series with the resistance) |
| atrium | P_A = P_passive(V_A) + P_active(V_A, t), from the FE table; dV_A/dt = Q_in - Q_AV |
| ventricle | P_V = E(t) (V_V - V0); dV_V/dt = Q_AV - Q_out |

The inlet has no valve, as in the embryonic heart and in Cairelli's circuit. The atrium therefore has no isovolumic
phase, and part of the blood flows back into the veins while it contracts. Inertances are zero.

**Atrium.** heartFEM's table source (`LV.cir`, source mode 3). Because the FE pressure depends only on the current
(V, t), the look-up is exact for any volume history, and heartFEM's relaxation-phase tracking (mode 4) is not needed.

**Ventricle.** Time-varying elastance (Suga & Sagawa 1974) in the form of the left ventricle of Korakianitis &
Shi 2006:

```
P_V = E(t) (V_V - V0),      E(t) = E_min + (E_max - E_min) e(tau),      tau = (t - t_on) mod T
e = 1/2 (1 - cos(pi tau / t_rise))                    0 <= tau < t_rise
e = 1/2 (1 + cos(pi (tau - t_rise) / t_fall))         t_rise <= tau < t_rise + t_fall
e = 0                                                 otherwise
```
The activation e has the same shape as the FE atrium's Ct, with constant durations. heartFEM's own circuits use a
fixed compliance plus a cos² pressure pulse for lumped chambers; a symmetric pulse cannot give the short isovolumic
relaxation measured in the embryo, which is why rise and fall have separate durations here.

**Solution.** ngspice transient analysis, 10 us step, 30 cycles from the initial state in `params.CIRCUIT`; the last
cycle is analysed. Ventricular timing is taken from the valve events (end-diastole = AV valve closure, end-systole =
outflow valve closure), not from volume extrema, because the volume is exactly flat while both valves are closed.

### Parameters (`params.CIRCUIT`)
Units: R in mmHg s/nL, C in nL/mmHg, E in mmHg/nL, V in nL, t in ms from the atrial activation onset.

| parameter | value | role | origin |
|---|---|---|---|
| R_in | 1.215 | veins -> atrium: atrial refilling, backflow | calibrated |
| R_AV | 0.530 | atrium -> ventricle: atrial emptying, a-wave | calibrated |
| R_out | 0.697 | ventricle -> arteries | calibrated |
| R_sys | 5.50 | arteries -> veins: mean arterial pressure | calibrated |
| C_art | 0.168 | arterial pool: pulse pressure | calibrated |
| C_ven | 3.59 | venous pool | Cairelli's 3 dpf circuit |
| E_min, E_max | 3.00, 50.5 | ventricular stiffness, relaxed and contracted | calibrated |
| V0 | 0.118 | ventricular volume at zero pressure | calibrated |
| t_on, t_rise, t_fall | 145, 233, 24 ms | ventricular activation | calibrated |
| P_ven0 | 1.53 mmHg | initial venous pressure; sets the total blood volume (7.31 nL) | calibrated |
| atrium_scale | 1 | gain of the active FE table: the FE model is used as it is | fixed |
| valves, inertances | ideal, 0 | | Cairelli's circuit |

### Where the calibrated values come from
The 12 calibrated values were found once by an automatic search over these parameters (coordinate pattern search, 960
circuit runs) and are fixed in `params.py`; the search is not part of this example. Its targets (value ± tolerance) were
all for the ventricle and the arteries:

| target | value | source |
|---|---|---|
| ventricular EDV, ESV | 0.57 ± 0.04, 0.23 ± 0.04 nL | Cairelli et al. 2024, 3 dpf ventricle (same size scale as this atrium) |
| ventricular EDP, peak pressure | 1.36 ± 0.5, 7.5 ± 1.0 mmHg | Salehin et al. 2021, 3 dpf, measured |
| mean arterial (bulbus) pressure | 4.9 ± 0.7 mmHg | Salehin et al. 2021 |
| filling, isovolumic contraction, ejection, isovolumic relaxation | 280 ± 40, 50 ± 25, 180 ± 30, 21 ± 20 ms | Salehin et al. 2021 |

Further conditions: the ventricle starts to contract within 20 ms of the AV valve closure, backflow <= 25 % of the AV
flow, >= 60 % of the ventricular filling during the atrial contraction, a-wave >= 1 mmHg, periodic steady state.

The FE table is not changed by the calibration, and no atrial quantity was a target. The atrial P-V loop, the atrial
volume course and the backflow are therefore predictions of the coupled model.

Relations that tie the values to the targets (checks):

| value | relation | check |
|---|---|---|
| E_min | EDP / (EDV - V0) | 1.36 / (0.57 - 0.118) = 3.0 |
| E_max | ≈ ESP / (ESV - V0) | 5.39 / (0.230 - 0.118) = 48 (the valve closes 3 ms after the stiffness peak) |
| t_rise | ≈ isovolumic contraction + ejection | 51 + 180 = 231 ms |
| t_fall | ≈ isovolumic relaxation | 20 ms |
| t_on | AV closure + (cycle - sum of the four measured phases) | 132 + (550 - 531) = 151 ms |
| R_sys | (mean arterial P - venous P) / cardiac output | (4.9 - 1.52) / 0.618 = 5.47 |

## Step replay: FE replay
The atrial volume of the last cycle is sampled at 52 times, t_j = (j + 0.5) · 550 / 52 ms, and each state (V_j, t_j) is
solved with the FE model. This gives the displacement and stresses of the coupled cycle. It also verifies the coupling:
the FE pressure of each state must equal the table pressure the circuit used.

## Output (`output/`)
Figures and the metric table, at the top of `output/`:

| file | content |
|---|---|
| `pv_loops.png` | P-V loops of the atrium (FE) and the ventricle, with the valve events |
| `flows.png` | flows through the inlet, the AV valve, the outflow valve and the systemic resistance over the cycle |
| `pressures_volumes.png` | the four pressures and the two chamber volumes over the cycle |
| `pvt_table.png` | the P-V-t table read by the circuit: three 3-D views, the working range, and a map (white line: P = 0) |
| `pv_cycle.mp4` | the coupled cycle state by state (52 states from the atrial activation onset): the deforming wall on a long-axis section through the apex coloured by the fiber stress, next to the atrial P-V loop (same layout as the movie of the volume-driven examples) |
| `metrics.csv` | metrics of the last cycle next to the measured values |

Data of each step:

| folder | content |
|---|---|
| `mesh/` | `t0.vtu` (tetrahedra, eC / eL / eR per cell), `parameter.txt` |
| `pvt/` | `job*.npz`: P_kPa[volume, time], wall-mean Ct, Newton iterations, stop time, per job; `job*.log` |
| `table/` | heartFEM files `atrium_Press_*.txt`; ngspice tables `atrium_lvcirtable.txt` (active), `atrium_lvcirtablebase.txt` (passive), `atrium_lvcirtabletr.txt`; `table_check.npz` (the assembled table and the interpolation check) |
| `circuit/` | `run.cir` (the filled netlist), `run.log`, `result.npz` (last two cycles: t, P_la, V_la, P_lv, V_lv, P_art, P_ven, flows, ventricular elastance; la = atrium, lv = ventricle), `metrics.json`, `numerics_check.json` |
| `replay/` | `summary.csv` (one row per state), `pv_results.npz` (u, sigma, sigma_ff (family 1), sigma_ff2 (family 2), sigma_CC / LL / RR, p, Ct (Σ_i w_i Ct_i), Ct_f1, Ct_f2 per state, mesh, fibers f0, f0_2), `vtu/cycle.pvd` for ParaView, `replay_vs_circuit.csv`, `fields_summary.csv` / `.txt` |

## Result
Last of 30 cycles (`output/pv_loops.png`, `flows.png`, `pressures_volumes.png`, `metrics.csv`). This example: one fiber
family; the other column is the example `../FE_two_fibers` (same circuit). Measured values: Salehin et al. 2021 (3 dpf);
ventricular volumes: Cairelli et al. 2024.

| | quantity | two families | one family | measured (3 dpf) |
|---|---|---|---|---|
| atrium (FE) | volume max / min | 0.840 / 0.489 nL (ejection fraction 42 %) | 0.830 / 0.473 nL (ejection fraction 43 %) | - |
|  | pressure max / min | 4.11 / -0.22 mmHg | 3.84 / -0.23 mmHg | - |
|  | a-wave | 3.37 mmHg | 3.10 mmHg | - |
| ventricle | EDV / ESV | 0.565 / 0.229 nL (ejection fraction 60 %) | 0.569 / 0.230 nL (ejection fraction 60 %) | 0.57 / 0.23 |
|  | EDP / peak / ESP | 1.34 / 7.45 / 5.35 mmHg | 1.35 / 7.51 / 5.39 mmHg | 1.36 / 7.52 / - |
|  | filling / isovolumic contraction / ejection / isovolumic relaxation | 280 / 51 / 180 / 20 ms | 281 / 51 / 180 / 20 ms | 280 / 50 / 180 / 21 |
|  | filling during the atrial contraction | 83 % | 84 % | mainly atrial |
| circulation | arterial pressure min / mean / max | 4.16 / 4.88 / 5.55 mmHg | 4.19 / 4.91 / 5.59 mmHg | mean 4.90 |
|  | venous pressure | 1.51 mmHg | 1.52 mmHg | - |
|  | stroke volume, cardiac output | 0.336 nL, 0.61 nL/s | 0.340 nL, 0.62 nL/s | - |
|  | backflow atrium -> veins | 0.079 nL (23 % of the AV flow) | 0.080 nL (24 % of the AV flow) | - |

With two families the atrium contracts with a higher pressure (+7 % peak, +9 % a-wave) and ejects slightly less; the
ventricle and the circulation change by at most 1-2 %. The circuit parameters were calibrated with one family and are
unchanged.

The cycle, in time from the atrial activation onset:

| time (ms) | phase | AV valve | outflow valve |
|---|---|---|---|
| 0 - 131 | atrial contraction fills the ventricle (atrial pressure up to 4.11 mmHg at 50 ms); backflow to the veins from 18 to 103 ms | open | closed |
| 131 - 201 | isovolumic contraction; the ventricle stiffens from ~150 ms | closed | closed |
| 201 - 381 | ejection; ventricular pressure peaks at 258 ms | closed | open |
| 381 - 401 | isovolumic relaxation | closed | closed |
| 401 - 550 | passive filling of the ventricle through the relaxed atrium | open | closed |

- The atrial loop runs counterclockwise without isovolumic segments. While it relaxes, the atrial pressure dips
  below zero (-0.22 mmHg at 212 ms) and draws blood from the veins; the atrium refills from 124 ms to the end of the
  cycle.
- For the ventricle, the last and the first row are one filling period (401 -> 131 ms of the next beat).
- The arterial pool is charged during ejection and discharges through R_sys during the rest of the cycle, so the
  flow to the periphery never stops.
- One family (`../FE_one_fiber`): atrial pressure up to 3.84 mmHg at 48 ms, backflow 17 - 108 ms, minimum
  -0.23 mmHg at 213 ms, ventricle end-diastole at 132 ms; the other event times are the same.

FE fields of the coupled cycle (`output/replay/fields_summary.txt`; the sarcomere length is that of family 1):

| quantity | two families | one family |
|---|---|---|
| largest displacement | 16.0 um (cavity radius 49 - 59 um) | 25.8 um (48 - 58 um) |
| sarcomere length, wall mean | 1.83 - 2.08 um (cells, 5 - 95 %: 1.75 - 2.23 um); family-1 shortening 12.1 % | 1.68 - 2.07 um (1.56 - 2.18 um); 18.6 % |
| wall-mean stresses at peak pressure | 50 ms: sigma_ff = sigma_CC = 1.72 kPa, sigma_LL = 0.92 kPa, sigma_RR = -0.27 kPa | 48 ms: 2.35, 0.30, -0.28 kPa |
| in-plane mean stress vs Laplace P r / 2h at peak pressure | 1.32 vs 1.51 kPa | 1.33 vs 1.40 kPa |
| J = det F | volume mean 0.9989 - 1.0004; away from the base, \|J - 1\| > 0.2 in 0.00 % of the wall | 0.9985; 0.19 % |

## Verification
- **Two fiber families.** With `TWO_FIBERS = False` the FE replay of the one-family circuit result gives the one-family replay again (P within 2e-15 kPa, fields bit-identical); the mixture code path is tested in `../../volume_constrain/FE_two_fibers` (Verification).
- **Table vs FE.** In the replay, the FE pressure of the 52 coupled states differs from the table pressure used by the circuit by at most 0.0055 mmHg (rms 0.0018; two families: 0.0068 / 0.0020). The look-up, the interpolation and the circuit's volume are consistent with the FE.
- **Conservation** (one family). Over the last cycle the same blood volume, 0.33980 nL, passes the inlet (net), the AV valve, the outflow valve and R_sys, and equals the ventricular stroke volume. The total blood volume varies by 2e-5 nL over the cycle.
- **Element laws** (one family). |P_V - E (V_V - V0)| <= 1.5e-7 mmHg; valve flows equal ΔP / R within 6e-5 nL/s; reverse valve flow < 1e-5 nL/s.
- **Periodic steady state.** The last two cycles differ by <= 4e-6 nL in the chamber volumes (one family).
- **Reproducibility.** Independent runs of the whole pipeline from `input/` (different job splits of the sweep) agree within 2.4e-4 mmHg in the pressure table, 7e-6 in every circuit metric (event times identical), 2e-6 kPa in the replay pressure, 3e-5 um in the displacement and 2e-6 (relative) in the stresses. The mesh depends on the number of mesher threads (`MESH_THREADS`; this example 1, the PINN example 36): node coordinates then differ by round-off (4e-8 um) and the tetrahedra are numbered differently; compare cell data by cell vertices, not by cell index.
- **Numerics** (step `circuit_check`). Half the time step: metrics change by <= 1.5e-04; twice the cycles: <= 2e-07; the event times do not change (`output/circuit/numerics_check.json`; two families 1.9e-04 / 2e-07).
- **Sweep resolution.** Halving the sweep step to 2.5 ms changes P by 0.001 mmHg (tested at the reference volume). Leaving one time out of the 5 ms grid, i.e. a local 10 ms step, changes the interpolated P by at most 0.022 mmHg.
- **Circuit elements against heartFEM.** The atrial table block, the valve model, the link sub-circuit and the volume integrators are heartFEM's; the only change is the name of the integrator model in ngspice >= 41.

## Limitations
- **Developmental stage.** The pressure and timing targets are measurements at 3 dpf; the imaged heart is 48 hpf, for
  which no pressure measurements exist. The agreement of the ventricular and arterial values with the measurements
  is the result of the calibration, not an independent validation.
- **Atrial pressure level.** The atrial peak pressure is 51 % of the ventricular peak. It is set by the FE active
  tension (T_MAX and the material constants, taken from the ventricle literature), not by the circuit.
- **Ventricle.** A lumped element with a linear passive curve, calibrated, not measured. V0 and E_max are constrained
  jointly rather than one by one; the search started from a single point and uniqueness was not examined.
- **Relaxation of the ventricle.** t_fall = 24 ms is ten times shorter than t_rise; it is forced by the 21 ms
  isovolumic relaxation of Salehin et al.
- **Arterial compliance.** C_art is 14 times the value of Cairelli's circuit.
- **Cycle length.** 550 ms is Cairelli's 3 dpf value; their 2 dpf value is 700 ms.
- **FE model.** Idealised fibers (two families, circumferential and longitudinal, equal shares; not measured); stress-free reference = image frame t12; the base is fixed normal
  to the cut plane; tr(ls) uses the instantaneous sarcomere length. At the base / endocardium corner the boundary
  condition produces local extremes of stress and volume change (J 0.36 - 1.36).
- **Smallest volumes.** Table volumes at high tension below the working range do not converge (wall collapse): with
  two families 11 volumes, 0.239 and 0.342 - 0.439 nL (one family: four, below 0.36 nL). The coupled cycle stays above
  0.489 nL, and the complete volumes 0.449 - 0.55 nL are the same as with one family, so the table is as dense over
  the whole cycle; the replay closes the loop to 0.007 mmHg.

## References
- Cairelli A. G. et al. (2024) Role of tissue biomechanics in the formation and function of myocardial trabeculae in
  zebrafish embryos. J Physiol 602(4):597-617. doi:10.1113/JP285490
- Guccione J. M., Waldman L. K., McCulloch A. D. (1993) Mechanics of active contraction in cardiac muscle: Part II -
  cylindrical models of the systolic left ventricle. J Biomech Eng 115:82-90.
- heartFEM (Lee lab; W. X. Chan): `lcleeHeart` forms (`forms_shavik`, `simpleActiveMaterial`), `ngspice_py`
  (`LV.cir`, `3dpf_new.cir`, `generateLVtable.py`).
- Korakianitis T., Shi Y. (2006) A concentrated parameter model for the human cardiovascular system including heart
  valve dynamics and atrioventricular interaction. Med Eng Phys 28:613-628. (Ventricular elastance and activation
  function; checked here against the reproduction in Scarsoglio S. et al. (2014), arXiv:1409.2814, Eqs. 8-10.)
- Peralta M. et al. (2013) Heartbeat-driven pericardiac fluid forces contribute to epicardium morphogenesis. Curr Biol
  23(18):1726-1735.
- Peralta M., González-Rosa J. M., Marques I. J., Mercader N. (2014) The epicardium in the embryonic and adult
  zebrafish. J Dev Biol 2:101-116.
- Salehin N. et al. (2021) Assessing pressure-volume relationship in developing heart of zebrafish in-vivo. Ann Biomed
  Eng 49:2080-2093.
- Serluca F. C. (2008) Development of the proepicardial organ in the zebrafish. Dev Biol 315:18-27.
- Suga H., Sagawa K. (1974) Instantaneous pressure-volume relationships and their ratio in the excised, supported
  canine left ventricle. Circ Res 35:117-126.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`); third-party notices (heartFEM, three.js) in
`../../../THIRD_PARTY_NOTICES.md`.