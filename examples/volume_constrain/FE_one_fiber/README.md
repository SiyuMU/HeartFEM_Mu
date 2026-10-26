# volume_constrain · FE · one fiber family

**FE P-V cycle of the zebrafish embryonic atrium (volume-driven).**

A finite-element model of the beating zebrafish embryonic atrium (48 hpf), driven by the imaged cavity volume. The
model prescribes the cavity volume of each of the 52 tracked image frames of one cardiac cycle. For every frame it
computes:
- the cavity **pressure** (the pressure-volume loop);
- the wall **displacement**;
- the wall **stresses** (Cauchy tensor, fiber stress, cardiac-frame components);
- the local **activation** state Ct.

The mechanics are heartFEM's (Lee lab, `lcleeHeart`): a Fung-type hyperelastic, incompressible myocardium with
heartFEM's time-varying active fiber stress. The volume is enforced by a Lagrange multiplier, which is the cavity
pressure.

This example has **one fiber family** (`TWO_FIBERS = False`), circumferential (0°). The example `../FE_two_fibers` is
the same model with two families, circumferential and longitudinal, each with half of the myocardium.

## This example
```
FE_one_fiber/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  all settings: run settings (`TASK = "volume_constrain"`, `SOLVER = "FE"`, resources), then the model
├── input/     raw data: t0.stl ... t51.stl (the 52 tracked frames), base_plane.json
└── output/    mesh/ (step preprocess) and the results of the cycle
```
The code is the library `../../../heartfem_mu/` (`preprocess.py`, `model.py`, `fe.py`, `volume_constrain.py`, `plots.py`); the example folder holds the data, the settings and the results. `params.py` holds every setting of this example: the run settings (mode, resources, conda environments) at the top, then the model, the numerics and the output options, each with its meaning.

The git repository holds the small files of `output/` (figures, movies, tables, json, logs). The large ones (`mesh/`,
every `.vtu`, `.npz` and `.pt` file) are in the release archive `heartfem_mu_v1.0.0_outputs_volume_constrain.tar.gz`
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
| `solve` | volume-driven FE cycle of the tracked frames | `volume_constrain.fe_cycle` | `CONDA_ENV_FENICS`, MPI, `MPI_RANKS` ranks |
| `plot` | pv_loop.png/.pdf, pv_cycle.mp4 | `plots.pv_figures` | `CONDA_ENV` |

`python main.py <step> ...` runs only the named steps, `python main.py --from <step>` a step and the following ones.
Resources (`params.py`): `CPUS`, the cores the steps may use (a `taskset` list; `None`: any core), and `MPI_RANKS`; the
FE steps run one thread per MPI rank.

## Input (`input/`)
| file | content |
|---|---|
| `t0.stl ... t51.stl` | Tracked surfaces of the 52 frames of one cycle of a 48 hpf zebrafish atrium, in um. Each file holds the closed endocardial and epicardial surfaces; all frames share the same triangle connectivity. Source: a 4D (3D + time) recording of the heart at 48 hpf (52 time points per cycle). |
| `base_plane.json` | Saved landmarks: the reference frame (t12), the base cut plane and the apex. Placed by hand once in the landmark browser app (`heartfem_mu/landmarks.py`, `USE_SAVED_LANDMARKS = False`), which writes this file; the default run only reads it. It cannot be recomputed automatically, so it is kept with the raw data rather than in `output/`. |

## Model

### Geometry and kinematics
- The wall is the frame t12 cut at the base plane, meshed with linear tetrahedra (mesh size 5 um).
- Frame t12 is the stress-free reference configuration: X = mesh coordinates, u = 0. Lengths are in mm in the
  solver, stresses in kPa.
- Deformation: F = I + grad u, J = det F, C = F^T F, E = (C - I) / 2.
- Local frame per cell, from two Laplace problems: eL (longitudinal, base -> apex), eR (radial, wall normal) and
  eC = eR x eL.
- Fibers. Family i: f_i = cos θ_i eC + sin θ_i eL, sheet s0 = eR, normal n_i = f_i × s0.
  - Every angle is an (endo, epi) pair, linear in the transmural depth d_endo / (d_endo + d_epi); the defaults are the same at every depth.
  - Default (`TWO_FIBERS = True`): two families, `FIBER_ANGLES_DEG` = ((0°, 0°), (90°, 90°)), circumferential and
    longitudinal, with shares w_i = `FIBER_WEIGHTS` = 0.5 / 0.5 (sum 1).
  - `TWO_FIBERS = False`: one family at `FIBER_ANGLE_DEG` = (0°, 0°), circumferential (the example
    `../FE_one_fiber`).

### Passive myocardium (heartFEM `forms_shavik.PassiveMatSEF`)
W = C (e^Q - 1) - p (J - 1), with

Q = C_ff E_ff² + C_ss E_ss² + C_nn E_nn² + C_ns E_ns² + C_fs E_fs² + C_fn E_fn²

where E_ab = a · E b in the (f0, s0, n0) frame and p is the pressure field that enforces J = 1.

With two fiber families the energy is the share-weighted sum (a mixture of the two orthotropic materials): W = Σ_i w_i C
(e^(Q_i) - 1) - p (J - 1), with Q_i evaluated in the frame (f_i, s0, n_i) of family i.

### Active stress (heartFEM `simpleActiveMaterial`, trackphase off)
S_act = T_MAX · Ct(t, ls) · Ca0² / (Ca0² + ECa(ls)²) · f0 ⊗ f0 (second Piola-Kirchhoff, along the reference fiber)

- ls = LR |F f0| is the current sarcomere length.
- ECa = Ca0max / sqrt(exp(B (ls - l0)) - 1), with ls - l0 floored at 0.003 um. Below l0 the fiber produces
  almost no tension.
- tr = m ls + b is the relaxation time. It is a field: it depends on the local sarcomere length and therefore on the
  solution.
- Ct depends on the activation time t (ms since onset). The onset is found from the tracking: the frame of maximum
  cavity volume (t43) + `ONSET_AFTER_ED_FRAMES` = 0.5, i.e. frame 43.5; frame k is at t = mod(k − 43.5, 52) × 550 / 52 ms.

  | phase | Ct |
  |---|---|
  | 0 <= t < t0 | 0.5 (1 - cos(π t / t0)) |
  | t0 <= t < t0 + tr | 0.5 (1 + cos(π (t - t0) / tr)) |
  | afterwards | 0 |

So every point of the wall has its own activation curve. The rise is the same everywhere. Relaxation is slower where the
sarcomeres are longer. In this atrium the relaxation coincides with refilling, so tr grows during relaxation: the wall
mean goes from 91 to 115 ms (one family). `output/pv_results.npz` stores Ct per cell and frame (two families: also
Ct_f1, Ct_f2). Ct decreases monotonically during relaxation: with two families no cell's Ct_i rises from one relaxation
frame to the next (one family: largest rise 0.006, in 0.01 % of the wall volume).

With two fiber families each family has its own sarcomere length ls_i = LR |F f_i|, relaxation time tr_i = m ls_i + b
and activation Ct_i = Ct(t, ls_i): S_act = Σ_i w_i T_MAX Ct_i g(ls_i) f_i ⊗ f_i. T_MAX is the tension of the whole
myocardium (the shares sum to 1). Each family's stress depends only on its own stretch, so the active stress still
derives from an energy (the PINN examples use it).

### Cavity and boundary conditions
- Cavity volume: V(u) = -1/3 ∫_endo (X + u) · cof(F) N dS (heartFEM `V0constrainedE`).
- The prescribed volume of frame k is V_k = V0(mesh) × V_img,k / V_img,t12. The image volumes come from the
  endocardial cut surfaces, closed by cones to the base-plane origin. The cavity pressure P is the Lagrange multiplier
  of V(u) = V_k.
- The base is fixed normal to its plane (u · n_base = 0).
- Five Lagrange multipliers remove the rigid in-plane translations and the rotations.
- **No epicardial spring** (default; optional: `K_SPRING` > 0 restores heartFEM's epicardial spring, traction
  −K (u · N) N with N the reference epicardial normal). At 48 hpf the heart beats inside a fluid-filled pericardial cavity (Peralta et al. 2013).
  The epicardium has not yet formed: it reaches the heart from ~72 hpf and covers it by ~120 hpf (Serluca 2008;
  Peralta et al. 2014). There is no pericardial restraint comparable to the adult pericardial sac, which epicardial
  springs represent in adult-heart models (e.g. Pfaller et al. 2019). Cairelli et al. (2024) used weak 90 Pa springs
  "emulating the effect of the tissues surrounding the heart" without a measurement.

### Weak form and solver
- The residual is R = d/dw[∫ W dx + volume constraint + rigid-body terms] + ∫ F S_act : grad v dx.
- Unknowns: u (P2), p (P1), the cavity pressure P (one number) and five rigid-body multipliers.
- The Jacobian is the exact derivative of R. It includes dCt/du through tr(ls).
- Newton iterations use a MUMPS direct solver and step halving.
- Frames are solved in time order from t12, each starting from the previous solution.
- The activation time is continued across the cycle boundary without jumping. If a frame fails, it is reached in
  sub-steps.
- Convergence: |R| < 1e-9 R_ref, and the volume is met to 1e-8 relative.

## Settings (`params.py`) and their sources
| quantity | value | source |
|---|---|---|
| mesh size, mesher threads | 5 um, 36 (`MESH_THREADS`: the tetrahedra depend on gmsh's thread count) | mesh study (same mesh): 5 vs 3 um changes P by <= 0.30 % of the peak and u by 0.75-2.0 % (relative L2); depth-binned wall-band stress means agree within 2 % between 2 and 3 um |
| fibers | two families (0°, 0°) / (90°, 90°) as (endo, epi) (circumferential / longitudinal), shares 0.5 / 0.5; sheet radial | `TWO_FIBERS`, `FIBER_ANGLES_DEG`, `FIBER_WEIGHTS`; no atrial fiber data at this stage, so no transmural variation (one family at `FIBER_ANGLE_DEG` = (0°, 0°): `TWO_FIBERS = False`) |
| C | 0.1 kPa | passive law of the PINN; with t12 as the stress-free state it gives EDP 0.26 kPa (1.95 mmHg), close to Cairelli's 1.36 mmHg |
| C_ff, C_ss = C_nn, C_ns, C_fs = C_fn | 29.9, 13.3, 26.6, 53.2 | heartFEM / Cairelli et al. 2024 |
| Ca0 = Ca0max, B, l0 | 4.35, 4.75 /um, 1.58 um | heartFEM defaults (Guccione et al. 1993) |
| lr | 2.0 um | tuned (heartFEM default 1.85 um): end-systole is reachable without suction |
| T_max | 20 kPa | tuned to the image volumes (peak P ~1.25 kPa) |
| cycle length | 550 ms | Cairelli et al. 2024 (3 dpf; their 2 dpf value is 700 ms, see Limitations) |
| t0 | 120 ms | tuned: maximum to minimum volume takes 148 ms in the images (Cairelli's 175 ms is too long) |
| m | 73 ms/um | Cairelli's heartFEM runs (heartFEM's BCL-scaled default 720 ms/um is infeasible here: tr <= 0 or > 600 ms) |
| b | -33 ms | tuned so that the P-V loop matches the previous constant-tr (100 ms) reference (b -36 / -30 give 4.4 / 5.3 % instead of 3.4 %) |
| onset | frame 43.5 = maximum-volume frame + 0.5 | the maximum volume (found from the tracking) is at t43; `ONSET_AFTER_ED_FRAMES` = 0.5 |
| epicardial spring | 0 (none) | `K_SPRING`; no epicardium at 48 hpf (see *Model*); optional heartFEM spring |

## Output (`output/`)
- **`mesh/`** (step `preprocess`):
  - `t0.vtu`: the wall mesh of t12, in um, with cell data eC, eL, eR.
  - `parameter.txt`: the reference frame.
  - `surface_cut/`: all frames cut at the base plane, plus `correspondence.npz`.
  - `preprocess_info.json`.
- **`pv_results.npz`** (mm, kPa, nL, ms; frame k = `t{k}.stl`):
  - Per frame: `frame`, `t_ms`, `t_active_ms` (activation time t), `V_nL`, `V_ratio` (V/V_t12), `P_kPa`,
    `T_mean_kPa` (wall mean of T_MAX Ct).
  - Displacement: `u` (52, nodes, 3), in mm, in the `mesh/t0.vtu` node order.
  - Per cell in the `mesh/t0.vtu` cell order:
    - `Ct` (52, cells): the activation (two families: the share-weighted mean Σ_i w_i Ct_i).
    - `Ct_f1`, `Ct_f2` (two families only): the activation of each family.
    - `sigma` (52, cells, 3, 3): the Cauchy stress, in kPa.
    - `sigma_ff`: the stress along the current fiber (two families: family 1, 0°); `sigma_ff2` (two families only):
      along the current fiber of family 2 (90°).
    - `sigma_CC`, `sigma_LL`, `sigma_RR`, `sigma_CL`, `sigma_CR`, `sigma_LR`: components in the current cardiac
      frame. The radial direction is the wall normal F^-T eR; the circumferential direction is F eC made normal to it;
      the longitudinal direction is their cross product.
    - `p`: the hydrostatic pressure field.
  - Mesh:
    - `points` (mm), `cells`, `f0` (fiber of family 1), `f0_2` (two families only: family 2), `eC`, `eL`, `eR`,
      `cell_volume`.
    - `far_from_base`: cells at least 10 um from the base plane. The base/endocardium corner has a stress
      concentration caused by the boundary condition.
  - `material`: the settings used (JSON), including `fiber_angles_deg` and `fiber_weights`.
- **`summary.csv`:** one row per frame with:
  - t, t_active, V, P (kPa and mmHg), the wall-mean tension, max |u|;
  - Newton iterations and wall time;
  - the wall means (cells far from the base) of sigma_ff, sigma_CC, sigma_LL and sigma_RR (and sigma_ff2 with two
    families).
- **`vtu/cycle.pvd`** (+ `frame00..51.vtu`): the time series for ParaView, in um. Point data `u_um`. The stresses and
  Ct are cell data.
- **`pv_loop.png` / `pv_loop.pdf`:** the P-V loop, with P(t), V(t) and the wall-mean tension (same layout as the PINN
  example `../PINN_one_fiber`).
- **`pv_cycle.mp4`:** one cycle from the maximum volume.
  - Left: the deforming wall on a long-axis section, with mesh lines, coloured by sigma_ff.
  - Right: the P-V loop, with a marker on the current frame.
- **`params_used.py`, `run_info.json`:** the settings and the run statistics.

## Result
All 52 frames (`output/pv_loop.png`, `summary.csv`). This example: one fiber family; the other column is the example
`../FE_two_fibers`.

| quantity | two families (0° / 90°, 0.5 / 0.5; `../FE_two_fibers`) | one family (0°; `../FE_one_fiber`) |
|---|---|---|
| P at maximum volume (t43, end-diastole) | 0.238 kPa (1.79 mmHg) | 0.260 kPa (1.95 mmHg) |
| peak P (t51) | 1.348 kPa (10.11 mmHg) | 1.250 kPa (9.38 mmHg) |
| P at minimum volume (t5, end-systole) | 0.193 kPa (1.45 mmHg) | 0.283 kPa (2.12 mmHg) |
| minimum P | 0.024 kPa (t13) | 0.026 kPa (t13) |
| end of activation (wall-mean T < 1e-3 kPa) | t14, 238 ms after onset | t14, 238 ms after onset |
| wall-mean T at t12 (stress-free mesh frame) | 0.77 kPa | 0.56 kPa |
| largest displacement | 19.0 um (t47) | 23.6 um (t5) |
| peak wall-mean fiber stress (>= 10 um from the base) | family 1 (0°) 4.26 kPa (t51), family 2 (90°) 2.55 kPa (t50) | 5.83 kPa (t50) |
| peak wall-mean sigma_CC / sigma_LL | 4.26 / 2.48 kPa | 5.83 / 1.37 kPa |
| wall-mean Ct at t5 / t9 (relaxation) | 0.88 / 0.32 (family 1), 0.87 / 0.29 (family 2) | 0.85 / 0.24 |

Half of the tension acts along the long axis: the circumferential stress falls, the longitudinal stress rises, the peak
pressure rises by 8 % and the wall moves less. Both families relax more slowly than the single family: with half the
circumferential tension the circumferential fibers shorten less at the same volume, so their sarcomere length and tr = m
ls + b are larger.

- The loop runs counterclockwise: contraction from t43 to t5, then relaxation with refilling, then passive filling.
- There are no isovolumic phases, because the image volume changes in every frame. An atrium has no inlet valve.
- At t12 (the stress-free mesh frame) some tension remains: wall-mean T is 0.77 kPa, P is 0.049 kPa and max |u| is
  6.1 um (one family: 0.56 kPa, 0.045 kPa, 4.8 um; its wall-mean tr at t12 is ~104 ms, more than t - t0 = 97 ms).

## Verification
- **Two fiber families.** `TWO_FIBERS = False` reproduces the one-family results bit for bit (P within 4e-16 kPa; u, sigma_ff, p and Ct identical); two families at the same angle (0° / 0°, any shares) equal one family; shares (1, 0) equal one family (the weight-0 family contributes nothing); swapping the two families gives the same P, u and sigma, with sigma_ff / sigma_ff2 and Ct_f1 / Ct_f2 swapped. All 52 frames converge with two families (174 Newton iterations).
- **Against heartFEM.** The active law, the volume constraint and the forms equal heartFEM's own code to round-off (1e-10 kPa in P, 1e-8 relative in u and stresses), checked with heartFEM's `lcleeHeart` model class on this mesh.
- **Path independence.** The quasi-static solution depends only on (V, t) of the frame, not on the solve path: in a loop-closure test (with a constant relaxation time tr) the frame t12, solved again at the end of the cycle, was identical.

## Limitations
- **Cycle length.** The data are 48 hpf (2 dpf), but the cycle length 550 ms is Cairelli et al.'s 3 dpf value; their
  2 dpf value is 700 ms. t0, T_max and b were tuned with 550 ms, so the time axis (ms) scales with this choice.
- **Material.** The material and active-law constants come from the zebrafish ventricle literature. Only T_max, lr,
  t0 and b were adapted to this atrium.
- **Fibers** are idealised, not measured: two families (circumferential and longitudinal) with equal shares; the
  angles and shares are settings, not data.
- **Reference state.** The stress-free state is the image frame t12 (the frame closest to zero pressure), not an
  unloaded geometry computed from the end-diastolic pressure.
- **Base constraint.** The base is fixed normal to the cut plane. The real attachments at the venous pole and the
  atrioventricular canal are not modelled.
- **Relaxation time.** tr(ls) uses the instantaneous sarcomere length (heartFEM with trackphase off). It is not
  integrated along the deformation history.
- **Coupling.** The volume is prescribed from the images; there is no coupling to the circulation. The same FE model
  coupled to a lumped (Windkessel) circulation is the example `../../windkessel/FE_one_fiber`.

## References
- Cairelli A. G. et al. (2024) Role of tissue biomechanics in the formation and function of myocardial trabeculae in
  zebrafish embryos. J Physiol 602(4):597-617. doi:10.1113/JP285490
- Guccione J. M., Waldman L. K., McCulloch A. D. (1993) Mechanics of active contraction in cardiac muscle: Part II -
  cylindrical models of the systolic left ventricle. J Biomech Eng 115:82-90. doi:10.1115/1.2895474
- heartFEM (Lee lab; W. X. Chan): `lcleeHeart` forms (`forms_shavik`, `simpleActiveMaterial`), `ngspice_py`.
- Peralta M. et al. (2013) Heartbeat-driven pericardiac fluid forces contribute to epicardium morphogenesis. Curr Biol
  23(18):1726-1735.
- Peralta M., González-Rosa J. M., Marques I. J., Mercader N. (2014) The epicardium in the embryonic and adult
  zebrafish. J Dev Biol 2:101-116.
- Serluca F. C. (2008) Development of the proepicardial organ in the zebrafish. Dev Biol 315:18-27.
- Pfaller M. R. et al. (2019) The importance of the pericardium for cardiac biomechanics. Biomech Model Mechanobiol
  18:503-529. doi:10.1007/s10237-018-1098-4
- Salehin N. et al. (2021) Assessing pressure-volume relationship in developing heart of zebrafish in-vivo. Ann Biomed
  Eng 49:2080-2093.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`); third-party notices (heartFEM, three.js) in
`../../../THIRD_PARTY_NOTICES.md`.