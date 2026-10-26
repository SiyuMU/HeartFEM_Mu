# windkessel · PINN · two fiber families

**PINN atrium in a closed-loop circulation (Windkessel coupling).**

The beating zebrafish embryonic atrium (48 hpf), coupled to a lumped-parameter (Windkessel) circulation, with a
physics-informed network (PINN) in the place of the finite-element (FE) model. The atrial volume is not prescribed: it
follows from the balance between the atrial pressure (from the PINN) and the rest of the circulation (a circuit solved
with ngspice).

This is the PINN counterpart of the Windkessel FE example `../FE_two_fibers`. The mechanics, the coupling strategy, the
circuit and its parameters are the same; only the solver of the atrial wall differs:

1. a network is trained once, without any FE data, to return the equilibrium displacement of the wall for any cavity
   pressure P and activation time t;
2. the pressure table P(V, t) that the circuit needs is read off the trained network on a grid of cavity volumes and
   activation times (seconds on a GPU, against hours for the FE sweep of the same table);
3. the circuit reads this table as its atrial element and is run to a periodic steady state;
4. the atrial volume of the coupled cycle is sent back through the network, which returns the wall displacement and
   stresses of that cycle and closes the loop.

As in the FE package, the **atrium is the detailed chamber** and the **ventricle is a lumped time-varying elastance**.

This example has **two fiber families** (`TWO_FIBERS = True`): circumferential (0°) and longitudinal (90°), each with
half of the myocardium (passive and active). The example `../PINN_one_fiber` is the same model with one circumferential
family (`TWO_FIBERS = False`). The circuit parameters are the same in both.

## This example
```
PINN_two_fibers/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  all settings: run settings (`TASK = "windkessel"`, `SOLVER = "PINN"`, resources), then the model
├── input/     raw data: t12.stl (the reference frame), base_plane.json
└── output/    result figures at the top, model.pt and the training log, the data of every step in sub-folders
```
The code is the library `../../../heartfem_mu/` (`preprocess.py`, `model.py`, `pinn.py`, `windkessel.py`, `plots.py`, `data/atrium_wk.cir`, `data/generateLVtable.py`); the example folder holds the data, the settings and the results. `params.py` holds every setting of this example: the run settings (mode, resources, conda environments) at the top, then the model, the numerics and the output options, each with its meaning.

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
| `train` | train the (P, t) network (energy minimization) | `pinn.train_load_network` | `CONDA_ENV`, `GPU` |
| `pvt` | P(V, t) table from the network | `windkessel.pinn_table` | `CONDA_ENV`, `GPU` |
| `table` | heartFEM P-V-t table files | `windkessel.build_table` | `CONDA_ENV` |
| `figures_table` | pvt_table.png | `plots.windkessel_figures` | `CONDA_ENV` |
| `circuit` | closed-loop circuit in ngspice | `windkessel.circuit` | `CONDA_ENV` |
| `circuit_check` | half time step, twice the cycles | `windkessel.circuit_check` | `CONDA_ENV` |
| `figures_cycle` | P-V loops, flows, metrics | `plots.windkessel_figures` | `CONDA_ENV` |
| `replay` | coupled volume back through the network | `windkessel.pinn_replay` | `CONDA_ENV`, `GPU` |
| `figures_fields` | fields summary, pv_cycle.mp4 | `plots.windkessel_figures` | `CONDA_ENV` |

`python main.py <step> ...` runs only the named steps, `python main.py --from <step>` a step and the following ones.
Resources (`params.py`): `GPU`, the GPU of the training and the solves (the `nvidia-smi` index; `None`: the CPU,
training about ten times slower), and `CPUS`, the cores the steps may use (a `taskset` list; `None`: any core).

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

## Mechanics of the atrium
The mechanics are heartFEM's (`lcleeHeart`), exactly as in the FE example: a Fung-type hyperelastic, incompressible
myocardium with heartFEM's time-varying active fiber stress. Lengths are in mm in the solver, stresses in kPa.

- **Geometry.** The wall is frame t12 cut at the base plane, meshed with tetrahedra (5 um). Frame t12 is the
  stress-free reference configuration. Local frame per cell from two Laplace problems: eL (longitudinal), eR (radial),
  eC = eR x eL. Fibers: family i, f_i = cos θ_i eC + sin θ_i eL, sheet s0 = eR, normal n_i = f_i x s0. Default two
  families, `FIBER_ANGLES_DEG` = ((0°, 0°), (90°, 90°)), circumferential and longitudinal, with shares
  w_i = `FIBER_WEIGHTS` = 0.5 / 0.5; `TWO_FIBERS = False`: one family at `FIBER_ANGLE_DEG` = (0°, 0°) (circumferential).
  Every angle is an (endo, epi) pair, linear in the transmural depth d_endo / (d_endo + d_epi) (the defaults are the same at every depth).
- **Passive myocardium** (heartFEM `forms_shavik.PassiveMatSEF`): W = C (e^Q - 1), with
  Q = C_ff E_ff² + C_ss E_ss² + C_nn E_nn² + C_ns E_ns² + C_fs E_fs² + C_fn E_fn² in the (f0, s0, n0) frame, and J = 1.
  Two families: W = Σ_i w_i C (e^(Q_i) - 1), Q_i in the frame (f_i, s0, n_i).
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
  with its own ls_i = LR |F f_i|, tr_i and Ct_i; in the training energy Σ_i w_i Psi_act(lambda_i; t), lambda_i = |F f_i|.
- **Cavity.** Cavity pressure P on the endocardium; cavity volume V(u) of the deformed endocardium, closed at the base.
- **Boundary conditions.** The base moves in its plane only (u · n = 0); no rigid translation in the base plane and no
  rigid rotation. There is no epicardial spring: at 48 hpf the heart beats in a fluid-filled pericardial cavity and has
  no epicardium yet (Peralta et al. 2013, 2014). Optional: `K_SPRING` > 0 restores heartFEM's epicardial spring,
  K/2 ∫_epi (u · N)² dA in the training energy (traction −K (u · N) N; default 0); the network must then be trained
  again.

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

"Tuned on this atrium" refers to the volume-driven P-V loop of the FE model, the example
`../../volume_constrain/FE_two_fibers`.

## Step train: the network
The FE solves a nonlinear problem for each state. Here one network is trained once by minimizing the potential energy of
the same mechanics. Architecture, energy and training recipe are those of the volume-driven PINN example
(`../../volume_constrain/PINN_two_fibers`), unchanged. The one difference is the range of cavity pressures the network
is trained for, see "Pressure range".

**Energy.** The equilibrium displacement u for the loads (P, t) minimizes

```
Pi(u; P, t) = int_wall [ C (e^Q - 1) + Psi_act(lambda; t) ] dV               passive Fung law + active stress
              + incompressibility penalty
              - P (V_cav(u) - V_cav(0))                                     work of the cavity pressure
              + rigid-body penalty,        with u . n = 0 on the base
```
- Psi_act(lambda; t) = T_MAX int_1^lambda Ct(t, LR s) g(LR s) s ds, so that dPsi_act/dE_ff is exactly the active
  stress above (closed form on the rise; 6-point Gauss per smooth piece in relaxation).
- Displacement in the FE's space: **quadratic (P2) tetrahedra** on the 5 um mesh (35706 nodes); 8-point quadrature per
  tetrahedron; F from the P2 shape-function gradients, so the network needs no coordinates.
- Incompressibility: penalty KAPPA/2 sum_a m_a Jbar_a², Jbar_a the lumped P1 projection of J - 1 on vertex a
  (KAPPA = 1000 kPa); the FE enforces J = 1 with a multiplier. Rigid-body motion: penalty (1000 kPa); FE: multipliers.
- Cavity volume: exact for the curved P2 endocardium, closed by cones to the base origin.

**Network.**
```
features(P, t) -> 5 x 128 tanh MLP -> h -> linear layers -> nodal displacement of all 35706 P2 nodes
                                           (+ three coarse linear heads on 8 / 20 / 50 um clusters, interpolated and added)
u(P, t) = 0.01 mm x (U(P, t) - U(0, 0)),   base-normal component removed on the base nodes
```
- Inputs: the cavity pressure P (kPa) and the activation time t (ms from the atrial onset: the time of the table and
  of the circuit). Features: P / P_MAX and heartFEM's Ct(t, LR lambda_k) at five fixed fiber stretches
  lambda_k = 0.8, 1.0, 1.2, 1.4, 1.6; all five are 0 once the wall has relaxed, so every passive time is the same input.
  The features describe t only; they are the same for one or two fiber families.
- The output is linear in the last hidden features; the coarse heads only precondition the optimization.
- u = 0 exactly in the unloaded state, and u · n = 0 exactly on the base.
- 14.7 M weights (almost all in the nodal output layer); the saved model is 59 MB.

**Training** (no FE data). The loss is the energy Pi, summed over many loads.
1. **Adam warm-up**, 1000 steps: 8 random loads per step (P uniform in the pressure range; t on the rise, in the
   relaxation window (120, 320] ms, or passive), a load ramp over 200 steps, a KAPPA ramp from 10 to 1000 over 400
   steps, learning rate 1e-5 -> 2e-6.
2. **Full-batch L-BFGS** (history 100, 20 iterations per outer step, strong-Wolfe line search) on fixed load grids,
   refined in three stages from outer step 1 / 42 / 101:
   - pressures log-spaced and dense at small |P| (scale 0.02 kPa);
   - times: the exact rise times of n_T tension levels log-spaced in [0, T_MAX], plus round(4/3 n_T) times uniform in
     (120, 320] ms; passive times need no points (their features equal those of t = 0);
   - each load weighted by 1 / |Pi|^0.5, with Pi taken at the start of the stage.
3. On the last grid the output layers are re-parametrized with whitened hidden features (`WHITEN`): the displacement
   field is unchanged, the optimization is better conditioned.
4. Training stops on the last grid when the objective has flattened (relative decrease over 10 outer steps below
   1e-4), or after 5 outer steps without progress. `TRAIN_HOURS` is only a safety cap.

Every component of this recipe has a one-change ablation; they are documented with the volume-driven example
(`../../volume_constrain/PINN_two_fibers/README.md`).

**Pressure range.** In the coupled cycle the relaxing atrium is smaller than its stress-free volume
(V0 = 0.677 nL) for about a fifth of the cycle: the wall recoils and the cavity pressure is negative (down to
-0.030 kPa on the cycle, 188 to 312 ms; the table entries that the circuit's interpolation touches go down to about
-0.05 kPa). The volume-driven loop never has P < 0, and the volume-driven network is trained for P in [0, 1.6] kPa
only. Here the loads cover [P_MIN, P_MAX] = [-0.05, 1.6] kPa at every activation time:
- Adam: P uniform in [P_MIN, P_MAX];
- L-BFGS grids: the 4 / 6 / 8 pressures in [0, P_MAX] of the volume-driven example, plus 3 / 4 / 6 pressures
  log-spaced in [P_MIN, 0) with the same scale (`GRID_NEGATIVE`). Finest grid: -0.050, -0.037, -0.026, -0.017, -0.010,
  -0.005, 0, 0.018, 0.050, 0.11, 0.23, 0.44, 0.84, 1.6 kPa x 14 times (0, 13, 23, 36, 57, 120, 145, 170, ..., 320 ms),
  without the unloaded state = 195 loads; 62 and 139 loads on the first two grids.

A model trained for P >= 0 only cannot be used here: extrapolated below zero it puts the atrial pressure minimum at
-0.28 instead of -0.23 mmHg and 10 ms late, and its fields in the suction phase are wrong by 10 % (displacement) to 20 -
50 % (stresses). This is the one change of this example against the volume-driven one, and that test is its ablation.

## Step pvt: pressure table P(V, t)
The network returns the displacement for (P, t), hence the cavity volume V(P, t). At fixed t the volume increases with
P, so the pressure of a state (V, t) is the root of V(P, t) = V: bisection on [P_MIN, P_MAX], 40 halvings, all states of
the table at once. Only the endocardial nodes are evaluated (the volume depends on them alone).

- **Volume grid** (heartFEM `LVbehaviorRun` rule, as the FE example): 50 log-spaced volumes from 0.9 ESV* to 1.1 EDV,
  plus 1.1x and 1.3x the largest and 0.9x and 0.7x the smallest; 54 volumes, 0.24 to 1.36 nL. EDV = 0.949 nL and
  ESV = 0.496 nL are the largest and smallest imaged cavity volume (`params.IMAGED_V_MAX_NL`, `IMAGED_V_MIN_NL`); ESV* = 0.4 EDV because (EDV - ESV) / EDV < 0.6.
- **Time grid:** 0 to 400 ms in 5 ms steps, as the FE sweep; later times are passive.
- **Outside the trained range:** a state whose pressure lies below P_MIN or above P_MAX is clipped to the limit and
  flagged: 642 states below P_MIN and 468 above P_MAX (25 % of the table). A volume without a single state inside the
  range is left out (the largest and the smallest of the 54; the table has 52 volumes, 0.31 to 1.15 nL). These entries are
  volumes that the coupled cycle does not visit at those times; the step replay counts the cycle samples whose interpolation
  stencil touches a flagged entry (see Verification).

## Step table: table files
As in the FE example: the table is interpolated in time (cubic spline) to heartFEM's time grid, cut to one cycle (0 to
550 ms), split into a passive part P_passive(V) and an active part P_active(V, t) = P - P_passive, and passed to
heartFEM's own `generateLVtable` (original source in `heartfem_mu/data/generateLVtable.py`, executed unchanged except
for one guard, see `run_generate_lv_table`). It writes the ngspice `table2d` files: volume in nL, time in s, pressure in
mmHg.

## Steps circuit, circuit_check: the circuit
The circuit, its parameters (`params.CIRCUIT`) and the circuit code are those of the FE example, unchanged (the template
`heartfem_mu/data/atrium_wk.cir` and the functions of `heartfem_mu/windkessel.py`). Pressure is the voltage (mmHg), flow
the current (nL/s), volume the charge (nL).

```
 venous pool --R_in (no valve)--> ATRIUM --R_AV + AV valve--> VENTRICLE --R_out + outflow valve--> arterial pool
   C_ven  ^                    PINN table                     elastance                             C_art   |
          +------------------------------------ R_sys (systemic resistance) <------------------------------+
```

| element | law |
|---|---|
| venous and arterial pool | C dP/dt = inflow - outflow |
| R_in, R_sys | Q = ΔP / R |
| R_AV + AV valve, R_out + outflow valve | Q = ΔP / R if ΔP > 0, else 0 (ideal diode in series with the resistance) |
| atrium | P_A = P_passive(V_A) + P_active(V_A, t), from the PINN table; dV_A/dt = Q_in - Q_AV |
| ventricle | P_V = E(t) (V_V - V0); dV_V/dt = Q_AV - Q_out |

The inlet has no valve, as in the embryonic heart and in Cairelli's circuit. The atrium therefore has no isovolumic
phase, and part of the blood flows back into the veins while it contracts. Inertances are zero.

**Atrium.** heartFEM's table source (`LV.cir`, source mode 3). Because the pressure depends only on the current
(V, t), the look-up is exact for any volume history.

**Ventricle.** Time-varying elastance (Suga & Sagawa 1974) in the form of the left ventricle of Korakianitis &
Shi 2006:

```
P_V = E(t) (V_V - V0),      E(t) = E_min + (E_max - E_min) e(tau),      tau = (t - t_on) mod T
e = 1/2 (1 - cos(pi tau / t_rise))                    0 <= tau < t_rise
e = 1/2 (1 + cos(pi (tau - t_rise) / t_fall))         t_rise <= tau < t_rise + t_fall
e = 0                                                 otherwise
```

**Solution.** ngspice transient analysis, 10 us step, 30 cycles from the initial state in `params.CIRCUIT`; the last
cycle is analysed. Ventricular timing is taken from the valve events (end-diastole = AV valve closure, end-systole =
outflow valve closure), not from volume extrema, because the volume is exactly flat while both valves are closed.

### Parameters (`params.CIRCUIT`)
Units: R in mmHg s/nL, C in nL/mmHg, E in mmHg/nL, V in nL, t in ms from the atrial activation onset.

| parameter | value | role |
|---|---|---|
| R_in | 1.215 | veins -> atrium: atrial refilling, backflow |
| R_AV | 0.530 | atrium -> ventricle: atrial emptying, a-wave |
| R_out | 0.697 | ventricle -> arteries |
| R_sys | 5.50 | arteries -> veins: mean arterial pressure |
| C_art | 0.168 | arterial pool: pulse pressure |
| C_ven | 3.59 | venous pool (Cairelli's 3 dpf circuit) |
| E_min, E_max | 3.00, 50.5 | ventricular stiffness, relaxed and contracted |
| V0 | 0.118 | ventricular volume at zero pressure |
| t_on, t_rise, t_fall | 145, 233, 24 ms | ventricular activation |
| P_ven0 | 1.53 mmHg | initial venous pressure; sets the total blood volume (7.31 nL) |
| atrium_scale | 1 | gain of the active table: the model is used as it is |
| valves, inertances | ideal, 0 | Cairelli's circuit |

These are the values of the FE example. They were calibrated once, with the FE table, to measured zebrafish hemodynamics
(ventricular volumes, pressures and phase durations at 3 dpf; `../FE_two_fibers/README.md`, "Where the calibrated values
come from"). Nothing was re-calibrated for the PINN, and no tuning code is part of this example.

## Step replay: PINN replay
The atrial volume of the last cycle is sampled at `N_REPLAY` = 52 times, t_j = (j + 0.5) · 550 / 52 ms. For each state
(V_j, t_j): the pressure by bisection as in the step pvt, the displacement u = network output at (P_j, t_j), and the
stresses:
- F from the P2 displacement at the quadrature points;
- the hydrostatic pressure p is recovered as the P1 field that best balances the P2 weak momentum residual (least
  squares; the penalty pressure itself is noisy);
- sigma = F S F^T / J - p I with S the passive + active second Piola-Kirchhoff stress; cell averages; components in
  the current cardiac frame exactly as the FE (r = F^-T eR, c = F eC made normal to r, l = r x c).

This is the post-processing of the volume-driven PINN example, unchanged (`pinn.state_fields`). The replay also verifies
the coupling: the pressure of each state must equal the table pressure the circuit used.

## Output (`output/`)
Same layout, file names and formats as `../FE_two_fibers/output`, so the two can be compared file by file.

Figures and the metric table, at the top of `output/`:

| file | content |
|---|---|
| `pv_loops.png` | P-V loops of the atrium (PINN) and the ventricle, with the valve events |
| `flows.png` | flows through the inlet, the AV valve, the outflow valve and the systemic resistance over the cycle |
| `pressures_volumes.png` | the four pressures and the two chamber volumes over the cycle |
| `pvt_table.png` | the P-V-t table read by the circuit: three 3-D views, the working range, and a map (white line: P = 0; hatched: outside the trained pressure range) |
| `pv_cycle.mp4` | the coupled cycle state by state (52 states from the atrial activation onset): the deforming wall on a long-axis section through the apex coloured by the fiber stress, next to the atrial P-V loop (same layout as the movie of the volume-driven packages) |
| `metrics.csv` | metrics of the last cycle next to the measured values |

Data of each step:

| folder | content |
|---|---|
| `mesh/` | `t0.vtu` (tetrahedra, eC / eL / eR per cell), `parameter.txt` |
| `model.pt`, `train_log.csv`, `train_info.json` | the trained network (weights, configuration with its pressure range, material), the objective per step, time and step counts |
| `pvt/` | `job0of1.npz`: P_kPa[volume, time] in the format of the FE sweep files, plus `clipped_below` / `clipped_above`; `job0of1.log` |
| `table/` | heartFEM files `atrium_Press_*.txt`; ngspice tables `atrium_lvcirtable.txt` (active), `atrium_lvcirtablebase.txt` (passive), `atrium_lvcirtabletr.txt`; `table_check.npz` |
| `circuit/` | `run.cir` (the filled netlist), `run.log`, `result.npz` (last two cycles: t, P_la, V_la, P_lv, V_lv, P_art, P_ven, flows, ventricular elastance; la = atrium, lv = ventricle), `metrics.json`, `numerics_check.json` |
| `replay/` | `summary.csv` (one row per state), `pv_results.npz` (u, sigma, sigma_ff (family 1), sigma_ff2 (family 2), sigma_CC / LL / RR, p, Ct (Σ_i w_i Ct_i), Ct_f1, Ct_f2 per state, mesh, fibers f0, f0_2), `vtu/cycle.pvd` for ParaView, `replay_vs_circuit.csv`, `fields_summary.csv` / `.txt`, `run_info.json` (with the range check) |

`pv_results.npz` holds the displacement at the mesh vertices (u, mm) and cell averages of the stresses (kPa), in the
node and cell order of `mesh/t0.vtu`, as the FE file. The tetrahedra depend on the number of mesher threads
(`MESH_THREADS`): this example uses 36, the FE example `../FE_two_fibers` 1, so node coordinates differ by round-off
(4e-8 um) and the tetrahedra are numbered differently; compare cell data by cell vertices, not by cell index.

## Result
Last of 30 cycles (`output/pv_loops.png`, `flows.png`, `pressures_volumes.png`, `metrics.csv`), next to the FE example
`../FE_two_fibers` (same circuit). Measured values: Salehin et al. 2021 (3 dpf); ventricular volumes: Cairelli et al.
2024.

| | quantity | PINN (this example) | FE (`../FE_two_fibers`) | measured (3 dpf) |
|---|---|---|---|---|
| atrium | volume max / min | 0.840 / 0.485 nL (ejection fraction 42 %) | 0.840 / 0.489 nL (ejection fraction 42 %) | - |
|  | pressure max / min | 4.03 / -0.22 mmHg | 4.11 / -0.22 mmHg | - |
|  | a-wave | 3.30 mmHg | 3.37 mmHg | - |
| ventricle | EDV / ESV | 0.567 / 0.229 nL (ejection fraction 60 %) | 0.565 / 0.229 nL (ejection fraction 60 %) | 0.57 / 0.23 |
|  | EDP / peak / ESP | 1.35 / 7.47 / 5.37 mmHg | 1.34 / 7.45 / 5.35 mmHg | 1.36 / 7.52 / - |
|  | filling / isovolumic contraction / ejection / isovolumic relaxation | 280 / 51 / 180 / 20 ms | 280 / 51 / 180 / 20 ms | 280 / 50 / 180 / 21 |
|  | filling during the atrial contraction | 84 % | 83 % | mainly atrial |
| circulation | arterial pressure min / mean / max | 4.17 / 4.89 / 5.56 mmHg | 4.16 / 4.88 / 5.55 mmHg | mean 4.90 |
|  | venous pressure | 1.51 mmHg | 1.51 mmHg | - |
|  | stroke volume, cardiac output | 0.338 nL, 0.61 nL/s | 0.336 nL, 0.61 nL/s | - |
|  | backflow atrium -> veins | 0.080 nL (24 % of the AV flow) | 0.079 nL (23 % of the AV flow) | - |

The phases of the cycle and their description are those of the FE example (`../FE_two_fibers/README.md`, Result): the
valve events fall on the same milliseconds (end-diastole 131 ms, start of ejection 201 ms, end-systole 381 ms, AV
opening 401 ms).

PINN fields of the coupled cycle (`output/replay/fields_summary.txt`), next to the FE example's:

```
cavity V 0.485-0.840 nL, cube-root shortening 16.7 %; wall-mean ls 1.826-2.084 um (shortening 12.3 %); cell ls 5-95 % over the cycle 1.740-2.231 um
u_max 15.9 um (cavity radius 49-59 um, wall ~10 um)
J: volume mean 0.9951-1.0006, cells 0.833-1.177; away from the base |J - 1| > 0.2 in at most 0.00 % of the wall
at peak P (0.533 kPa, t 48 ms): wall-mean sigma_CC 1.73, sigma_ff 1.73, sigma_LL 0.93, sigma_RR -0.27 kPa; in-plane mean 1.33 vs Laplace P r / 2h 1.48 kPa
```
FE (`../FE_two_fibers/output/replay/fields_summary.txt`):
```
cavity V 0.489-0.840 nL, cube-root shortening 16.5 %; wall-mean ls 1.831-2.084 um (shortening 12.1 %); cell ls 5-95 % over the cycle 1.745-2.232 um
u_max 16.0 um (cavity radius 49-59 um, wall ~10 um)
J: volume mean 0.9989-1.0004, cells 0.832-1.177; away from the base |J - 1| > 0.2 in at most 0.00 % of the wall
at peak P (0.543 kPa, t 48 ms): wall-mean sigma_CC 1.72, sigma_ff 1.72, sigma_LL 0.92, sigma_RR -0.27 kPa; in-plane mean 1.32 vs Laplace P r / 2h 1.51 kPa
```

## Accuracy against the FE example
Computed from `output/` of this example and `../FE_two_fibers/output`. Displacement errors are relative L2 norms over
the mesh nodes; stress errors are volume-weighted relative L2 norms over the cells >= 10 um from the base. Nodes are
matched by coordinates and cells by their vertices.

**Coupled cycle.** Each model in its own coupled cycle (its own table, the same circuit):

| quantity | PINN - FE |
|---|---|
| 28 circuit metrics | largest relative differences: la_a_wave -2.2 %, la_Pmax -1.8 %, la_Pmin +1.8 %, la_inflow_back +1.1 %; all others within 1.0 % |
| 14 event times and phase durations | 13 identical to the ms, the others within 1 ms |
| atrial pressure over the cycle | rms 0.0211 mmHg, max 0.0760 mmHg at 49 ms (0.5 % and 1.8 % of its range) |
| atrial volume over the cycle | rms 0.0018 nL, max 0.0037 nL at 124 ms (0.5 % and 1.1 % of its range) |
| ventricular pressure | rms 0.0136 mmHg, max 0.0267 mmHg at 248 ms (0.2 % and 0.4 % of its range) |

**Pressure table.** Entry by entry on the common grid:

| entries | number | PINN - FE |
|---|---|---|
| near the coupled cycle (what the circuit's interpolation can touch), P_FE < 0 | 131 | rms 0.0098 kPa, max 0.0388 kPa |
| near the coupled cycle, P_FE >= 0 | 286 | rms 0.0169 kPa, max 0.0452 kPa |
| all entries inside the trained pressure range | 2711 | rms 0.0102 kPa, max 0.0503 kPa |

**Fields.** The network evaluated at the 52 coupled states (V, t) of the FE example, so that both models solve the same states (10 states with P < 0):

| quantity | 42 states with P >= 0 | 10 states with P < 0 |
|---|---|---|
| displacement u | 2.0 % | 4.5 % |
| sigma_ff = sigma_CC (family 1, 0°) | 3.4 % | 7.3 % |
| sigma_ff2 = sigma_LL (family 2, 90°) | 4.5 % | 11.5 % |
| sigma_RR | 5.1 % | 12.8 % |
| hydrostatic pressure p | 3.3 % | 7.5 % |

State by state: cavity pressure rms 0.0146 kPa; displacement median 1.8 %, max nodal error 1.72 um; with one norm for
the whole cycle u 2.3 %, sigma_ff 6.9 %. Wall-mean sigma_ff at peak pressure: 1.730 vs 1.718 kPa.

The network is less accurate in the suction phase (P < 0, see Limitations) and under-predicts the peak atrial pressure
by 1.8 % (incompressibility penalty with two contracting families, as in `../../volume_constrain/PINN_two_fibers`).

## Verification
- **Two fiber families.** With `TWO_FIBERS = False` the table and the replay of a one-family network are reproduced (table within 2.4e-7 kPa, replay pressures identical, stresses within 5e-5 kPa, float32); the material code is tested in `../../volume_constrain/PINN_two_fibers` (Verification).
- **Table vs network.** In the replay, the network pressure of the 52 coupled states differs from the table pressure used by the circuit by at most 0.0044 mmHg (rms 0.0017; one family: 0.0026 / 0.0012). The look-up, the interpolation and the circuit's volume are consistent with the network.
- **Range check.** None of the 52 states lies outside the trained pressure range. 54 of the 550 samples of the cycle have one corner of their interpolation stencil at a table entry clipped at P_MIN (-0.05 kPa), where the relaxed wall would reach a lower pressure; a lower P_MIN would remove the clip and needs a new training.
- **Periodic steady state.** The last two cycles differ by 1.2e-06 nL in the chamber volumes.
- **Numerics** (step `circuit_check`). Half the time step: metrics change by <= 1.7e-04; twice the cycles: <= 2e-07; the event times do not change (`output/circuit/numerics_check.json`; one family 1.5e-04 / 2e-07).
- **Circuit.** The circuit template and code are those of `../FE_two_fibers`; conservation and the element laws are verified there (`../FE_two_fibers/README.md`, Verification).
- **Another training.** GPU training is not bit-reproducible. Two independent trainings of this recipe agree within 0.1 % in every circuit metric, with identical valve event times.

## Limitations
The limitations of the model and of the circuit are those of the FE example (`../FE_two_fibers/README.md`: developmental
stage of the measurements, atrial pressure level, lumped ventricle, arterial compliance, cycle length, idealised fibers,
stress-free reference). Specific to the PINN:

- **Suction phase (P < 0).** Under suction the relaxed wall is very soft and has a limit point: in the FE the
  passive P(V) falls to -0.026 kPa at 0.54 nL and the volume then jumps to 0.50 nL. The network has the pressure as an
  input and is trained on the rectangle [P_MIN, P_MAX] x t, which at the passive times contains loads beyond that limit
  point. Pressure and volume stay accurate there (one family: table 0.0012 kPa rms; the coupled cycle is not affected), but the
  shape of the wall is less accurate in the states with P < 0 than elsewhere. One family (a training of this recipe):
  11 states (t = 196 - 301 ms), displacement 6.9 % (0.27 um rms; worst state 13 % at 238 ms) and stresses 12 - 15 %,
  against 2.8 % and 3 - 5 % in the other 41 states. Two families: 10 states (t = 196 - 291 ms), displacement 4.6 % and
  stresses 7 - 13 %, against 1.9 % and 3.4 - 5 % (see Accuracy); the pressure there is less accurate than with one
  family (table 0.010 kPa rms near the cycle).
  Development studies located the cause in the training loads beyond the limit point (a passive-only network trained
  without them reaches 1 - 3 % in these states). Removing them from the full training needs a lower pressure limit
  that depends on the activation time; this was not adopted.
- **Trained range.** P in [-0.05, 1.6] kPa at every activation time. Table entries outside are clipped and flagged;
  P_MIN has no margin against the lowest table entry near the coupled cycle (see Verification). A circuit that drives
  the atrium to lower pressures or to volumes outside 0.31 - 1.15 nL needs a network trained on a wider range.
- **Contracted states.** At the smallest volumes (0.47 - 0.50 nL, high tension) dP/dV is 6 - 7 kPa/nL, so a volume
  error of 0.3 - 0.5 % of the network becomes a pressure error of up to 0.02 kPa there.
- **Penalty versus exact constraints.** Incompressibility and the rigid-body constraints are penalties (FE:
  multipliers). In the contracted states two cells at the base / endocardium corner are inverted in the
  linear-tetrahedron value of J; the FE has its own extremes in the same cells (`../FE_two_fibers/README.md`).
- **Base region.** Stresses within about 10 um of the base reflect the boundary condition and reach unphysical values
  in a few cells at the corner; all wall means and all comparisons exclude these cells (`BASE_EXCLUDE_UM`).
- **One geometry.** The network is the solution operator of this mesh, these fibers, these material constants and this
  activation law. Any change of them requires retraining (a model stores its fibers and is refused by a different
  `params.py`). Circuit changes do not.
- **Run-to-run variation.** Training is not bit-reproducible; see "Another training" under Verification.

## References
- Cairelli A. G. et al. (2024) Role of tissue biomechanics in the formation and function of myocardial trabeculae in
  zebrafish embryos. J Physiol 602(4):597-617. doi:10.1113/JP285490
- Guccione J. M., Waldman L. K., McCulloch A. D. (1993) Mechanics of active contraction in cardiac muscle: Part II -
  cylindrical models of the systolic left ventricle. J Biomech Eng 115:82-90.
- heartFEM (Lee lab; W. X. Chan): `lcleeHeart` forms (`forms_shavik`, `simpleActiveMaterial`), `ngspice_py`
  (`LV.cir`, `3dpf_new.cir`, `generateLVtable.py`).
- Korakianitis T., Shi Y. (2006) A concentrated parameter model for the human cardiovascular system including heart
  valve dynamics and atrioventricular interaction. Med Eng Phys 28:613-628.
- Peralta M. et al. (2013) Heartbeat-driven pericardiac fluid forces contribute to epicardium morphogenesis. Curr Biol
  23(18):1726-1735.
- Peralta M., González-Rosa J. M., Marques I. J., Mercader N. (2014) The epicardium in the embryonic and adult
  zebrafish. J Dev Biol 2:101-116.
- Salehin N. et al. (2021) Assessing pressure-volume relationship in developing heart of zebrafish in-vivo. Ann Biomed
  Eng 49:2080-2093.
- Suga H., Sagawa K. (1974) Instantaneous pressure-volume relationships and their ratio in the excised, supported
  canine left ventricle. Circ Res 35:117-126.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`); third-party notices (heartFEM, three.js) in
`../../../THIRD_PARTY_NOTICES.md`.