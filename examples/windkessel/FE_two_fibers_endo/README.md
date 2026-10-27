# windkessel · FE · two fiber families · myocardium + endocardium

**FE atrium with an endocardial layer in a closed-loop circulation (Windkessel coupling).**

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

This example is `../FE_two_fibers` with one change: the wall is **myocardium + endocardium** (`WALL = "myo_endo"`).
A 2 um endocardial layer is grown from the myocardial inner surface into the cavity (`ENDO_GROW = True`) and solved
together with the myocardium; the cavity is the lumen inside the layer. The layer can instead come with the input
surface (`ENDO_GROW = False`, see Input). The mesh of the myocardium, its material and activation, the two
fiber families (circumferential and longitudinal, half each), the circuit and its parameters are those of
`../FE_two_fibers`, so the difference between the two examples is the effect of the layer.

## This example
```
FE_two_fibers_endo/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  all settings: run settings (`TASK = "windkessel"`, `SOLVER = "FE"`, resources), then the model
├── input/     raw data: t12.stl (the reference frame), base_plane.json
└── output/    result figures at the top, the data of every step in sub-folders (mesh, pvt, table, circuit, replay)
```
The code is the library `../../../heartfem_mu/` (`preprocess.py`, `myo_endo.py`, `model.py`, `fe.py`, `windkessel.py`,
`plots.py`, `data/atrium_wk.cir`, `data/generateLVtable.py`); the example folder holds the data, the settings and the
results. `params.py` holds every setting of this example: the run settings (mode, resources, conda environments) at the
top, then the model (with the wall section `WALL`, `ENDO_*`), the numerics and the output options, each with its meaning.

The git repository holds the small files of `output/` (figures, movies, tables, json, logs). The large ones (`mesh/`,
every `.vtu` and `.npz` file) are in the release archive `heartfem_mu_v1.1.0_outputs_windkessel_FE_two_fibers_endo.tar.gz`
(see Examples in `../../../README.md`); `python main.py` recomputes all of them from `input/`.

## Run
```bash
python main.py                  # every step, input/ -> output/
```
The two conda environments (`CONDA_ENV`, `CONDA_ENV_FENICS` in `params.py`) are installed once with `bash ../../../environments/install.sh` (see Install in `../../../README.md`).

The steps (`python main.py --list`); each runs in its own process in the conda environment it needs:

| step | does | code (`heartfem_mu/`) | runs in |
|---|---|---|---|
| `preprocess` | raw surfaces -> tetrahedral mesh, local frame, cut surfaces | `preprocess.main` | `CONDA_ENV`, `MESH_THREADS` threads |
| `endo_layer` | endocardial layer grown on the mesh | `myo_endo.grow_layer` | `CONDA_ENV` |
| `pvt` | FE P(V, t) sweep | `windkessel.fe_sweep` | `CONDA_ENV_FENICS`, `SWEEP_JOBS` MPI jobs x `SWEEP_RANKS` ranks |
| `table` | heartFEM P-V-t table files | `windkessel.build_table` | `CONDA_ENV_FENICS` |
| `figures_table` | pvt_table.png | `plots.windkessel_figures` | `CONDA_ENV` |
| `circuit` | closed-loop circuit in ngspice | `windkessel.circuit` | `CONDA_ENV_FENICS` |
| `circuit_check` | half time step, twice the cycles | `windkessel.circuit_check` | `CONDA_ENV_FENICS` |
| `figures_cycle` | P-V loops, flows, metrics | `plots.windkessel_figures` | `CONDA_ENV` |
| `replay` | coupled volume back through the FE model | `windkessel.fe_replay` | `CONDA_ENV_FENICS`, MPI, `MPI_RANKS` ranks |
| `figures_fields` | fields summary, pv_cycle.mp4 | `plots.windkessel_figures` | `CONDA_ENV` |

`python main.py <step> ...` runs only the named steps, `python main.py --from <step>` a step and the following ones
(after a change of the layer settings: `python main.py --from endo_layer`). With `ENDO_GROW = False` there is no step
`endo_layer`: the step `preprocess` (`myo_endo.preprocess_layers`) meshes both layers of the input surface.
Resources (`params.py`): `CPUS`, the cores the steps may use (a `taskset` list; `None`: any core), and `MPI_RANKS`
(replay) and `SWEEP_JOBS` x `SWEEP_RANKS` (sweep); the FE steps run one thread per MPI rank.

## Input (`input/`)
| file | content |
|---|---|
| `t12.stl` | Surface of the reference frame, in um: the closed inner and outer surfaces of the myocardium of a 48 hpf zebrafish atrium. It is frame t12 of the 52 tracked frames of one cycle of a 4D (3D + time) recording of the heart at 48 hpf. |
| `base_plane.json` | Saved landmarks for `t12.stl`: the base cut plane and the apex. Placed by hand once in the landmark browser app (`heartfem_mu/landmarks.py`, `USE_SAVED_LANDMARKS = False`); the default run only reads it. |

The coupling uses the images for two things only:
- the mesh, from the reference frame t12 (the only frame in `input/`);
- the range of cavity volumes that the pressure table must cover. It is built from the largest and smallest cavity
  volume of the imaged cycle, which are two settings in `params.py` (`IMAGED_V_MAX_NL` = 0.9491 nL, frame t43, and
  `IMAGED_V_MIN_NL` = 0.4960 nL, frame t5). They are the volumes inside the imaged (myocardial) inner surface.

The imaged volume curve is not prescribed anywhere.

**Input with both layers (`ENDO_GROW = False`).** The STL file holds three closed, nested surfaces of the reference
frame: the lumen (the layer's inner surface), the myocardial inner surface and the epicardium (told apart by their
enclosed volumes). `base_plane.json` as above, with the apex on the epicardium (`USE_SAVED_LANDMARKS = True`); the base
plane must cut each surface in one closed loop. `ENDO_THICKNESS_UM` and `ENDO_SUBLAYERS` are then not read, and
`IMAGED_V_MAX_NL` / `IMAGED_V_MIN_NL` stay the volumes inside the myocardial inner surface.

## FE model of the atrium (`heartfem_mu/fe.py`, `heartfem_mu/myo_endo.py`)
The mechanics are heartFEM's (`lcleeHeart`): a Fung-type hyperelastic, incompressible wall with heartFEM's
time-varying active fiber stress. Lengths are in mm in the solver, stresses in kPa.

### Myocardium (as `../FE_two_fibers`)
- **Geometry.** The myocardial wall is frame t12 cut at the base plane, meshed with linear tetrahedra (5 um). Frame t12
  is the stress-free reference configuration. Local frame per cell from two Laplace problems: eL (longitudinal), eR
  (radial), eC = eR x eL. Fibers: family i, f_i = cos θ_i eC + sin θ_i eL, sheet s0 = eR, normal n_i = f_i x s0; two
  families, `FIBER_ANGLES_DEG` = ((0°, 0°), (90°, 90°)), circumferential and longitudinal, with shares
  w_i = `FIBER_WEIGHTS` = 0.5 / 0.5. Every angle is an (endo, epi) pair, linear in the transmural depth of the
  myocardium (from the myocardium / endocardium interface to the epicardium).
- **Passive** (heartFEM `forms_shavik.PassiveMatSEF`): W = Σ_i w_i C (e^(Q_i) - 1) - p (J - 1), with
  Q_i = C_ff E_ff² + C_ss E_ss² + C_nn E_nn² + C_ns E_ns² + C_fs E_fs² + C_fn E_fn² in the frame (f_i, s0, n_i); p
  enforces J = 1.
- **Active stress** (heartFEM `simpleActiveMaterial`, trackphase off):
  S_act = Σ_i w_i T_MAX · Ct_i(t, ls_i) · Ca0² / (Ca0² + ECa(ls_i)²) · f_i ⊗ f_i, with the current sarcomere length
  ls_i = LR |F f_i|, ECa = Ca0max / sqrt(exp(B (ls - l0)) - 1), and

  ```
  Ct = 1/2 (1 - cos(pi t / t0))            0 <= t < t0
  Ct = 1/2 (1 + cos(pi (t - t0) / tr))     t0 <= t < t0 + tr,     tr = m ls + b
  Ct = 0                                    afterwards
  ```
  t is the time since the atrial activation onset; tr depends on the local sarcomere length, so Ct is a field.

| setting | value | source |
|---|---|---|
| passive C, C_ff, C_ss = C_nn, C_ns, C_fs = C_fn | 0.1 kPa, 29.9, 13.3, 26.6, 53.2 | zebrafish ventricle literature (Salehin et al. 2021); heartFEM law |
| Ca0 = Ca0max, B, l0 | 4.35, 4.75 /um, 1.58 um | heartFEM defaults |
| LR (sarcomere length of the reference state) | 2.0 um | tuned on this atrium |
| T_MAX | 20 kPa | tuned on this atrium |
| t0 (time to peak tension) | 120 ms | tuned on this atrium |
| tr = m ls + b | m = 73 ms/um, b = -33 ms | m: Cairelli's heartFEM runs; b tuned on this atrium |
| cycle length | 550 ms | Cairelli et al. 2024, 3 dpf (see Limitations) |

"Tuned on this atrium" refers to the volume-driven P-V loop of the myocardium-only model, the example
`../../volume_constrain/FE_two_fibers`.

### Endocardial layer (`WALL = "myo_endo"`)
- **Mesh** (step `endo_layer`). Every node and tetrahedron of the myocardial mesh is kept; a layer of thickness
  `ENDO_THICKNESS_UM` = 2 um is grown from the myocardial inner surface into the cavity. Every inner-surface node moves
  along its normal (area-weighted face normals; on the base ring the direction is projected into the base plane, so the
  layer's base stays on the plane) in `ENDO_SUBLAYERS` = 1 step, and every surface triangle sweeps a prism, split into
  three tetrahedra. The diagonal of every side face runs from the lower-numbered surface node at the bottom to the
  other node at the top, so neighboring prisms share their side triangles: the mesh is conforming and the two regions
  share the interface nodes (perfect bonding). A layer cell takes the local frame of the myocardial tetrahedron it
  grows from. The step checks that every tetrahedron is positive, that the old inner surface is interior and that the
  layer volume equals the difference of the volumes enclosed by the two surfaces (`output/mesh/endo_layer_info.json`).
  Here: 12402 layer tetrahedra on the 19472 myocardial ones, layer volume 0.0749 nL (11 % of the reference cavity).
- **Mesh from an input with both layers** (`ENDO_GROW = False`, step `preprocess`). As for the myocardium: the three
  surfaces are cut at the base plane; the five surface patches (lumen, myocardial inner surface, epicardium and the two
  base annuli) are remeshed by gmsh at `MESH_SIZE`, one surface per patch, and the two volumes are filled with
  tetrahedra that share the remeshed myocardial inner surface (conforming, perfect bonding). The local frame is computed
  on the myocardium alone (the Laplace problems of `preprocess.py` between its inner surface and the epicardium); a
  layer cell takes the frame of the myocardial tetrahedron of the nearest interface face. The same checks as above;
  `endo_layer_info.json` also gives the layer thickness (distance of the lumen nodes to the myocardial inner surface).
- **Material.** The laws of the myocardium (above) with the layer's values in `ENDO_MATERIAL`: C, C_ij, T_MAX and its
  fiber settings (angles as (lumen side, myocardial side) pairs). They enter the forms as per-cell (DG0) fields; the
  activation timing (t0, tr, Ca0, B, l0, LR) is shared. This example: the myocardial passive values and fibers and
  T_MAX = 0 (the endocardium does not contract): the layer is a non-contracting shell of the myocardial passive law.
- **Hydrostatic pressure.** `ENDO_PRESSURE_JUMP = True`: one P1 pressure field per region, p = χ p_myo + (1 - χ)
  p_endo (χ = 1 in the myocardium). Traction continuity at the interface requires [p] = [τ_nn], so p jumps where the
  two incompressible materials differ; a single continuous P1 field cannot represent the jump (errors of 5-6 % in the
  cavity pressure when the layer's C differs by a factor of 10, see Verification). The p rows that a region does not
  use are zero and get a unit diagonal.
- **Cavity.** P is the Lagrange multiplier of V(u) = V_target, V(u) = -1/3 ∫ (X + u) · cof(F) N dS over the luminal
  surface (the layer's inner surface): the cavity is the lumen, the blood volume that the circuit exchanges. The layer is
  incompressible, so V_lumen = V_inside the myocardium - V_layer (0.0749 nL) at every state.
- **Boundary conditions.** As `../FE_two_fibers`: the base (myocardium and layer) moves in its plane only
  (u · n = 0); rigid translation in the base plane and rigid rotation are removed by multipliers (origin of the base
  frame: the myocardial base nodes); no epicardial spring (`K_SPRING` = 0).
- **Discretisation.** u in P2, p_myo and p_endo in P1 (Taylor-Hood in each region), P and the rigid-body multipliers
  as global unknowns; Newton with a direct solver (MUMPS) and automatic sub-stepping.

A **state** is a pair (cavity volume V, activation time t). The model is quasi-static and has no history variable, so
its solution, and in particular the cavity pressure, depends only on the current (V, t). This is what allows the table
look-up below.

## Step pvt: pressure table P(V, t)
As heartFEM's `getLVbehavior`: for every volume of the grid, a passive solve at that volume, then the activation time is
advanced at fixed volume and the cavity pressure is recorded.

- **Volume grid** (heartFEM `LVbehaviorRun` rule): 50 log-spaced volumes from 0.9 ESV* to 1.1 EDV, plus 1.1x and
  1.3x the largest and 0.9x and 0.7x the smallest; 54 volumes, 0.22 to 1.25 nL. EDV and ESV are the largest and
  smallest imaged cavity volume minus the layer volume (0.874 and 0.421 nL: the imaged volumes are inside the
  myocardium, the cavity is the lumen); ESV* = 0.4 EDV because (EDV - ESV) / EDV < 0.6.
- **Time grid:** 0 to 400 ms in 5 ms steps. As in heartFEM, a volume stops two steps after the tension of the
  myocardium is off (between 240 and 290 ms); from then on the pressure is the passive pressure.
- **Not converged:** the two volumes 0.315 and 0.220 nL fail at 25 and 20 ms (the wall collapses under high tension)
  and are left out. The table has 52 volumes, 0.28 to 1.25 nL; the coupled atrium stays between 0.44 and 0.74 nL. The
  layer stabilizes the small volumes: without it (`../FE_two_fibers`) more of them fail.

## Step table: table files
The sweep is interpolated in time (cubic spline) to heartFEM's time grid, cut to one cycle (0 to 550 ms; heartFEM writes
two cycles, the circuit reads the cyclic time only), split into a passive part P_passive(V) and an active part
P_active(V, t) = P - P_passive, and passed to heartFEM's own `generateLVtable` (original source in
`heartfem_mu/data/generateLVtable.py`, executed unchanged except for one guard, see `windkessel.run_generate_lv_table`).
It writes the ngspice `table2d` files: 52 volumes x 553 times, volume in nL, time in s, pressure in mmHg.

## Steps circuit, circuit_check: the circuit (`heartfem_mu/data/atrium_wk.cir`)
The circuit and all its parameters are those of `../FE_two_fibers` (its README describes the elements, the parameters
and their calibration); the atrial element reads this example's table. Pressure is the voltage (mmHg), flow the
current (nL/s), volume the charge (nL); the atrial volume is the lumen.

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

The calibration of the circuit (ventricular and arterial targets only) was made with the myocardium-only atrium and is
not repeated here: the total blood volume (set by `P_ven0`) is the same, so the 0.075 nL that the layer takes from the
atrial cavity are held elsewhere in the circulation. ngspice transient analysis, 10 us step, 30 cycles; the last cycle
is analysed; ventricular timing from the valve events.

## Step replay: FE replay
The atrial volume of the last cycle is sampled at 52 times, t_j = (j + 0.5) · 550 / 52 ms, and each state (V_j, t_j) is
solved with the FE model. This gives the displacement and stresses of the coupled cycle. It also verifies the coupling:
the FE pressure of each state must equal the table pressure the circuit used.

## Output (`output/`)
Figures and the metric table, at the top of `output/`:

| file | content |
|---|---|
| `pv_loops.png` | P-V loops of the atrium (FE, lumen volume) and the ventricle, with the valve events |
| `flows.png` | flows through the inlet, the AV valve, the outflow valve and the systemic resistance over the cycle |
| `pressures_volumes.png` | the four pressures and the two chamber volumes over the cycle |
| `pvt_table.png` | the P-V-t table read by the circuit: three 3-D views, the working range, and a map (white line: P = 0) |
| `pv_cycle.mp4` | the coupled cycle state by state (52 states from the atrial activation onset): the deforming wall (myocardium and layer) on a long-axis section through the apex coloured by the fiber stress, next to the atrial P-V loop |
| `metrics.csv` | metrics of the last cycle next to the measured values |

Data of each step:

| folder | content |
|---|---|
| `mesh/` | `t0.vtu` (the myocardium: tetrahedra, eC / eL / eR per cell), `parameter.txt`; `t0_endo.vtu` (myocardium + layer, with `region` = 1 myocardium, 2 endocardial layer; the myocardial nodes and cells first), `endo_layer_info.json` (layer sizes, volumes and checks) |
| `pvt/` | `job*.npz`: P_kPa[volume, time] (lumen volumes), myocardial-mean Ct, Newton iterations, stop time, per job; `job*.log` |
| `table/` | heartFEM files `atrium_Press_*.txt`; ngspice tables `atrium_lvcirtable.txt` (active), `atrium_lvcirtablebase.txt` (passive), `atrium_lvcirtabletr.txt`; `table_check.npz` |
| `circuit/` | `run.cir`, `run.log`, `result.npz` (last two cycles: t, P_la, V_la, P_lv, V_lv, P_art, P_ven, flows, ventricular elastance), `metrics.json`, `numerics_check.json` |
| `replay/` | `summary.csv` (one row per state; wall means of the myocardium and, in `*_endo_mean_kPa`, of the layer), `pv_results.npz` (u, sigma, sigma_ff, sigma_ff2, sigma_CC / LL / RR, p, Ct per state, mesh, fibers, `region`), `vtu/cycle.pvd` for ParaView (cell data `region`), `replay_vs_circuit.csv`, `fields_summary.csv` / `.txt` |

## Result
Last of 30 cycles (`output/pv_loops.png`, `flows.png`, `pressures_volumes.png`, `metrics.csv`). This example:
myocardium + endocardium; the other column is the example `../FE_two_fibers` (myocardium only, otherwise the same model
and circuit). Measured values: Salehin et al. 2021 (3 dpf); ventricular volumes: Cairelli et al. 2024.

| | quantity | myocardium + endocardium | myocardium only | measured (3 dpf) |
|---|---|---|---|---|
| atrium (FE) | volume max / min | 0.740 / 0.438 nL lumen (inside the myocardium 0.815 / 0.513 nL; ejection fraction 41 %) | 0.840 / 0.489 nL (ejection fraction 42 %) | - |
|  | pressure max / min | 3.66 / -0.24 mmHg | 4.11 / -0.22 mmHg | - |
|  | a-wave | 2.85 mmHg | 3.37 mmHg | - |
| ventricle | EDV / ESV | 0.554 / 0.227 nL (ejection fraction 59 %) | 0.565 / 0.229 nL (ejection fraction 60 %) | 0.57 / 0.23 |
|  | EDP / peak / ESP | 1.31 / 7.30 / 5.26 mmHg | 1.34 / 7.45 / 5.35 mmHg | 1.36 / 7.52 / - |
|  | filling / isovolumic contraction / ejection / isovolumic relaxation | 285 / 51 / 180 / 18 ms | 280 / 51 / 180 / 20 ms | 280 / 50 / 180 / 21 |
|  | filling during the atrial contraction | 76 % | 83 % | mainly atrial |
| circulation | arterial pressure min / mean / max | 4.10 / 4.80 / 5.45 mmHg | 4.16 / 4.88 / 5.55 mmHg | mean 4.90 |
|  | venous pressure | 1.54 mmHg | 1.51 mmHg | - |
|  | stroke volume, cardiac output | 0.327 nL, 0.59 nL/s | 0.336 nL, 0.61 nL/s | - |
|  | backflow atrium -> veins | 0.063 nL (19 % of the AV flow) | 0.079 nL (23 % of the AV flow) | - |

The layer takes 0.075 nL of the atrial cavity and resists the contraction: the atrium holds less blood and contracts
with a lower pressure (-11 % peak, -15 % a-wave) at a similar ejection fraction. Ventricular volumes and pressures, the
stroke volume and the arterial pressure change by 1-3 %; the share of the ventricular filling during the atrial
contraction drops from 83 to 76 %. The total blood volume is the same; the venous pool holds the difference.

The cycle, in time from the atrial activation onset:

| time (ms) | phase | AV valve | outflow valve |
|---|---|---|---|
| 0 - 134 | atrial contraction fills the ventricle (atrial pressure up to 3.66 mmHg at 48 ms); backflow to the veins from 17 to 96 ms | open | closed |
| 134 - 201 | isovolumic contraction; the ventricle stiffens from ~150 ms | closed | closed |
| 201 - 381 | ejection; ventricular pressure peaks at 258 ms | closed | open |
| 381 - 399 | isovolumic relaxation | closed | closed |
| 399 - 550 | passive filling of the ventricle through the relaxed atrium | open | closed |

The atrial pressure dips below zero while the atrium relaxes (-0.24 mmHg at 216 ms); the atrium refills from 123 ms to
the end of the cycle.

FE fields of the coupled cycle (`output/replay/fields_summary.txt`; sarcomere length and wall means of the myocardium,
cells >= 10 um from the base):

| quantity | myocardium + endocardium | myocardium only |
|---|---|---|
| largest displacement | 14.6 um (cavity radius 47 - 56 um) | 16.0 um (49 - 59 um) |
| myocardial sarcomere length, wall mean | 1.85 - 2.07 um (cells, 5 - 95 %: 1.79 - 2.20 um); family-1 shortening 10.4 % | 1.83 - 2.08 um (1.75 - 2.23 um); 12.1 % |
| myocardial wall-mean stresses at peak pressure (48 ms) | sigma_ff = sigma_CC = 1.73 kPa, sigma_LL = 1.03 kPa, sigma_RR = -0.29 kPa | 1.72, 0.92, -0.27 kPa |
| myocardial wall-mean sigma_CC at the end of the atrial emptying (122 ms) | 2.63 kPa | 0.93 kPa |
| endocardial layer, mean stresses | at peak pressure sigma_CC -1.42, sigma_LL -1.69, sigma_RR -0.54 kPa; most compressed at 122 ms, sigma_CC -8.86 kPa | - |
| in-plane mean stress vs Laplace P r / 2h at peak pressure | 0.94 vs 1.06 kPa (whole wall, layer + myocardium) | 1.32 vs 1.51 kPa |
| J = det F | volume mean 0.9985 - 1.0003; away from the base, \|J - 1\| > 0.2 in 0.00 % of the wall | 0.9989 - 1.0004; 0.00 % |

In late contraction the layer, the innermost shell of the wall, is shortened most and thickens, and the exponential
law with the myocardial passive values stiffens strongly: the compressed layer pushes back, and the myocardium carries
more tension for a given cavity pressure (wall-mean sigma_CC 2.63 against 0.93 kPa at 122 ms).

**The layer's load at the same myocardial deformation.** The coupled cycle of `../FE_two_fibers` replayed with this
example's model, with the lumen volume = that cycle's volume - 0.0749 nL (so the myocardium encloses the same volume at
every state; 13 of the 52 states): the cavity pressure is up to 0.3 mmHg (40 %) higher during the filling and up to
2.3 mmHg lower in late contraction (myocardial volume down to 0.72 of the reference), with layer-mean sigma_CC down to
-9.7 kPa. With the layer's C ten times smaller the drop is 0.5 mmHg and the filling pressure +5 %.

## Verification
- **Physics of the layer: hemisphere against an exact solution.** A hemispherical shell (inner radius 50 um, wall
  10 um, layer 2 um, mesh 5 um) cut at the equator: there the base condition of this model (in-plane sliding) is the
  symmetry condition of a full sphere, which has a 1-D solution of the same laws (incompressible, spherically
  symmetric, P = ∫ 2 (σ_t - σ_r) / r dr over the layer and the myocardium; two fiber families with equal shares keep the
  tangential stress equibiaxial). At eight states (passive inflation, contraction, relaxation) the FE cavity pressure
  differs from the 1-D solution by at most 0.12 % without the layer, 0.12 % with this example's layer, 0.14 % with the
  layer's C x 10 and 0.12 % with C x 0.1; on a 3.5 um mesh 0.06 - 0.07 % (second order). The layer stresses follow the
  1-D solution in tension and in compression. With one continuous pressure field (`ENDO_PRESSURE_JUMP = False`) the
  C x 10 and C x 0.1 cases are off by 6.3 and 5.5 %.
- **Two-region code against the myocardium code.** With `ENDO_MATERIAL` = the myocardial values (T_MAX too) and one
  continuous p, the two-region forms give the single-region forms on the same mesh to 7e-15 (this atrium, 8 states)
  and 6e-16 (hemisphere).
- **Layer resolution.** Two element layers through the 2 um layer instead of one change the cavity pressure by at most
  0.008 mmHg and the myocardial stresses by <= 1 % (13 states of the coupled cycle; the layer's effect is up to 2.3 mmHg).
- **Pressure formulation on this atrium.** One continuous p instead of one per region changes the cavity pressure by at
  most 0.005 mmHg: with the myocardial passive values on both sides the pressure is nearly continuous; the per-region
  pressure matters when the layer's values differ.
- **Layer mesh** (step endo_layer, `output/mesh/endo_layer_info.json`). All layer tetrahedra positive (smallest
  volume 0.45 x the median, quality >= 0.39); the layer volume equals the difference of the enclosed volumes to 3e-15.
- **Layer from input surfaces** (`ENDO_GROW = False`). Hemisphere from three closed spheres (lumen 48, myocardial
  inner surface 50, epicardium 60 um): the cavity pressure differs from the 1-D solution by at most 0.13 % (layer as
  here), 0.16 % (C x 10) and 0.12 % (C x 0.1) on a 5 um mesh, 0.07 % and 0.08 % on a 3.5 um mesh. With the myocardial
  values in the layer and one continuous p, the two-region forms give the single-region forms to 2e-15 (hemisphere)
  and 6e-15 (this atrium, meshed from input surfaces of the same geometry as the grown layer). The step `preprocess`
  gives the same mesh in every run.
- **Table vs FE.** In the replay, the FE pressure of the 52 coupled states differs from the table pressure used by the
  circuit by at most 0.0026 mmHg (rms 0.0012).
- **Numerics** (step `circuit_check`). Half the time step: metrics change by <= 1.5e-04; twice the cycles: <= 2e-07;
  the event times do not change (`output/circuit/numerics_check.json`).
- **Myocardium-only wall unchanged.** With `WALL` absent or `"myo"` the library takes the myocardium-only path of
  `../FE_two_fibers` unchanged (its results are reproduced to round-off).

## Limitations
- **The layer's material.** The layer has the passive values and the anisotropy of the myocardium and no active
  tension: the endocardium is treated as a non-contracting shell of the myocardial passive law. The endocardium of the
  embryo is an endothelial monolayer on the cardiac jelly; its stiffness was not measured here and is probably lower,
  and its anisotropy is not that of the myofibers. The effect of the layer scales with its stiffness: in a test with
  the layer's C ten times smaller, the drop of the cavity pressure in late contraction at the same myocardial volume
  went from 2.3 to 0.5 mmHg. An isotropic layer with the same law is set by C_FF = C_SS = C_NN = b and
  C_NS = C_FS = C_FN = 2 b in `ENDO_MATERIAL` (then Q = b tr(E²)).
- **Thickness.** 2 um, an endothelial monolayer; the cardiac jelly between endocardium and myocardium is not modelled.
- **Imaged surface.** The inner surface of `t12.stl` is taken as the myocardial inner surface (the layer is added
  inside it); the imaged volumes are compared with the volume inside the myocardium (lumen + layer).
- **Circuit.** Calibrated with the myocardium-only atrium (see `../FE_two_fibers`) and not recalibrated; the
  developmental stage, the ventricle element, the relaxation of the ventricle, the arterial compliance and the cycle
  length have the limitations listed there.
- **FE model.** Idealised fibers (two families, circumferential and longitudinal, equal shares; not measured);
  stress-free reference = image frame t12; the base is fixed normal to the cut plane; tr(ls) uses the instantaneous
  sarcomere length. At the base / endocardium corner the boundary condition produces local extremes of stress.

## References
- Cairelli A. G. et al. (2024) Role of tissue biomechanics in the formation and function of myocardial trabeculae in
  zebrafish embryos. J Physiol 602(4):597-617. doi:10.1113/JP285490
- Guccione J. M., Waldman L. K., McCulloch A. D. (1993) Mechanics of active contraction in cardiac muscle: Part II -
  cylindrical models of the systolic left ventricle. J Biomech Eng 115:82-90.
- heartFEM (Lee lab; W. X. Chan): `lcleeHeart` forms (`forms_shavik`, `simpleActiveMaterial`), `ngspice_py`
  (`LV.cir`, `3dpf_new.cir`, `generateLVtable.py`).
- Korakianitis T., Shi Y. (2006) A concentrated parameter model for the human cardiovascular system including heart
  valve dynamics and atrioventricular interaction. Med Eng Phys 28:613-628.
- Salehin N. et al. (2021) Assessing pressure-volume relationship in developing heart of zebrafish in-vivo. Ann Biomed
  Eng 49:2080-2093.
- Suga H., Sagawa K. (1974) Instantaneous pressure-volume relationships and their ratio in the excised, supported
  canine left ventricle. Circ Res 35:117-126.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`); third-party notices (heartFEM, three.js) in
`../../../THIRD_PARTY_NOTICES.md`.
