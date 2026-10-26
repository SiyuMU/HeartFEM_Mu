# invert_para · FE · one fiber family

**Parameter inversion by FE: C and T_MAX of the zebrafish embryonic atrium from two tracked frames.**

From the tracked wall of two frames of one cardiac cycle of a 48 hpf zebrafish atrium, ED (end-diastole: maximum volume,
passive) and PS (peak systolic pressure: contracting), and the cavity pressure given at both frames, this example finds
two parameters of the heartFEM mechanics:

- **C**, the scale of the passive Fung law (kPa);
- **T_MAX**, the peak active fiber tension (kPa).

The fiber architecture is predefined, not inferred: two fiber families at 0° and 90° (circumferential and longitudinal)
with equal shares, or one family at a chosen angle. Every candidate is an exact FE solution of the mechanics; the
parameters are chosen so that the simulated wall is as close as possible to the tracked wall (surface distance) with the
imaged cavity volume. The example `../PINN_one_fiber` solves the same problem with a parametric physics-informed network
instead of FE.

This example has **one fiber family** (`TWO_FIBERS = False`), circumferential (0°). The example `../FE_two_fibers` is
the same model with two families, circumferential and longitudinal, each with half of the myocardium.

## This example
```
FE_one_fiber/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  all settings: run settings (`TASK = "invert_para"`, `SOLVER = "FE"`, resources), then the model
├── input/     raw data: t12.stl (reference), t43.stl (ED), t51.stl (PS), base_plane.json
├── cache/     intermediate files: mesh/ (step preprocess)
└── output/    results: result.json, evaluation.json, ED.vtu, PS.vtu, comparison.png
```
The code is the library `../../../heartfem_mu/` (`preprocess.py`, `model.py`, `fe.py`, `invert_para.py`, `plots.py`); the example folder holds the data, the settings and the results. `params.py` holds every setting of this example: the run settings (mode, resources, conda environments) at the top, then the model, the numerics and the output options, each with its meaning.

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
| `invert` | search C, T_MAX with the FE model | `invert_para.fe_inversion` | `CONDA_ENV_FENICS`, MPI, `MPI_RANKS` ranks |
| `evaluate` | comparison with the tracking | `invert_para.evaluate` | `CONDA_ENV` |

`python main.py <step> ...` runs only the named steps, `python main.py --from <step>` a step and the following ones.
Resources (`params.py`): `CPUS`, the cores the steps may use (a `taskset` list; `None`: any core), and `MPI_RANKS`; the
FE steps run one thread per MPI rank.

## Input (`input/`)
| file | content |
|---|---|
| `t12.stl` | Reference frame (stress-free state, the mesh): closed endocardial and epicardial surfaces, um. |
| `t43.stl` | ED frame: maximum volume, before the activation onset. |
| `t51.stl` | PS frame: 79 ms after the activation onset, the frame of peak pressure of the volume-driven model. |
| `base_plane.json` | Saved landmarks of the reference frame: base cut plane and apex (placed once in the landmark browser app (`heartfem_mu/landmarks.py`)). |

The three frames are frames 12, 43 and 51 of the 52 tracked frames of one cycle of a 4D (3D + time) recording of the
heart at 48 hpf; they share one triangle connectivity (point-to-point correspondence). The file names give the frame
numbers set in `params.py`.

## Pipeline
1. **`preprocess`**: cuts the three input surfaces at the saved base plane, keeps their point-to-point
   correspondence (`cache/mesh/surface_cut/`), builds the 5 um tetrahedral mesh of the reference frame t12 and the
   local frame eC / eL / eR per cell (`cache/mesh/t0.vtu`).
2. **`invert`** (MPI): builds the FE model on that mesh with the fibers of `params.py`; finds C0 and T0
   from the imaged ED and PS cavity volumes (geometry start); runs stage A (ED, C) and stage B (PS, T_MAX) from each
   start; keeps the start with the lowest J over both frames; solves both frames at the result and writes
   `output/result.json`, `ED.vtu`, `PS.vtu`.
3. **`evaluate`**: compares the result with the tracking (surface distances, cavity volumes, Dice), writes
   `output/evaluation.json` and `comparison.png`, and adds the per-vertex distance to the tracking to `ED.vtu`, `PS.vtu`.

## Settings (`params.py`)
Usually only the frames, `PS_ACTIVATION_MS`, the two pressures and the fibers need to be set for new data; the rest are
model constants and numerical settings.

| group | setting | meaning |
|---|---|---|
| folders | `INPUT_DIR`, `CACHE_DIR`, `OUTPUT_DIR` | raw data; intermediate files (mesh); results |
| preprocess | `MESH_SIZE` | tetrahedral mesh size (5 um) |
| | `MESH_THREADS` | threads of the gmsh mesher (1 here): the tetrahedra depend on the thread count |
| | `USE_SAVED_LANDMARKS` | `True`: base plane and apex from `input/base_plane.json`; `False`: place them in the browser app |
| frames | `REF_FRAME`, `ED_FRAME`, `PS_FRAME` | frame numbers of the reference (stress-free, the mesh), ED (passive) and PS (contracting) input files |
| | `PS_ACTIVATION_MS` | time of the PS frame after the activation onset (80 ms); must be below `T0_MS`; gives Ct at PS (0.75) |
| | `MM_PER_UNIT` | input coordinates are um, the model works in mm and kPa |
| pressures | `P_ED_KPA`, `P_PS_KPA` | given cavity pressure at ED and PS (kPa); see *Inverse problem* |
| material (fixed) | `C_FF` ... `C_FN` | exponents of the Fung law (relative stiffness of the fiber, sheet, normal and shear directions); C, the overall scale, is inferred |
| | `CA0`, `CA0MAX`, `B`, `L0` | length dependence g(ls) of the active tension (heartFEM) |
| | `LR` | sarcomere length in the reference state (2.0 um) |
| | `T0_MS` | time from the activation onset to the peak of Ct (120 ms) |
| | `K_SPRING` | heartFEM epicardial spring K (kPa/mm); 0 = none (default) |
| fibers (fixed) | `TWO_FIBERS` | `True`: two families (default); `False`: one family |
| | `FIBER_ANGLE_DEG` | (endo, epi) angle of the single family (0° = circumferential eC, 90° = longitudinal eL; default (0°, 0°)) |
| | `FIBER_ANGLES_DEG`, `FIBER_WEIGHTS` | (endo, epi) angles and shares of the two families (((0°, 0°), (90°, 90°)), (0.5, 0.5)) |
| misfit | `SIGMA_V` | cavity-volume tolerance (0.01 = 1 %) |
| | `SIGMA_S_REL` | surface-distance tolerance as a fraction of L (0.02: 1.76 um here) |
| | `W_EPI` | weight of the epicardium relative to the endocardium (0.2) |
| | `BASE_EXCLUDE_REL` | surface points within this fraction of L of the base plane are not compared (0.1139: 10 um here) |
| search | `START_FACTORS` | starts = geometry start × these factors on (C, T_MAX); `((1.0, 1.0),)` runs one start (about 3× faster) |
| | `XATOL`, `FATOL`, `MAXFEV` | Nelder-Mead stop: simplex size (ln units), change of J, evaluation cap per stage |
| | `STEP` | initial simplex step (ln units, a factor e^0.5 = 1.65) |

## Model
The mechanics are those of the volume-driven FE of the same atrium (`../../volume_constrain/FE_one_fiber`), with the
cavity pressure given instead of the cavity volume and a predefined fiber architecture:

- **Geometry:** the reference frame t12 cut at the base plane, linear tetrahedra (5 um) carrying a quadratic (P2)
  displacement; t12 is the stress-free state. Local frame per cell from two Laplace problems: eL (longitudinal), eR
  (radial), eC = eR x eL.
- **Fibers** (`params.py`, not inferred): family i has the direction f_i = cos(a_i) eC + sin(a_i) eL, sheet s0 = eR,
  normal n_i = f_i x s0, and the share w_i; a_i is linear in the transmural depth d_endo / (d_endo + d_epi) between its
  (endo, epi) values (as the examples `../../volume_constrain` and `../../windkessel`).
  - `TWO_FIBERS = True` (default): a = (0°, 0°) and (90°, 90°), the same at every depth, w = 0.5 / 0.5
    (`FIBER_ANGLES_DEG`, `FIBER_WEIGHTS`);
  - `TWO_FIBERS = False`: one family at `FIBER_ANGLE_DEG` = (0°, 0°) (circumferential).
- **Passive:** W = sum_i w_i C (e^Q_i - 1) - p (J - 1), Q_i = C_ff E_ff² + C_ss E_ss² + C_nn E_nn² + C_ns E_ns² +
  C_fs E_fs² + C_fn E_fn² in the frame (f_i, s0, n_i) (heartFEM `forms_shavik`); C inferred, the exponents C_ff ...
  C_fn fixed (29.9, 13.3, 13.3, 26.6, 53.2, 53.2).
- **Active:** S_act = sum_i w_i T_MAX Ct(t, ls_i) Ca0² / (Ca0² + ECa(ls_i)²) f_i ⊗ f_i, ls_i = LR |F f_i| (heartFEM
  `simpleActiveMaterial`, each family with its own sarcomere length); T_MAX inferred.
- **Activation:** ED is passive (Ct = 0). PS lies `PS_ACTIVATION_MS` = 80 ms after the activation onset, on the rise of
  the heartFEM activation curve (t < t0 = 120 ms), where Ct = ½ (1 − cos(π t / t0)) = 0.75 does not depend on the
  sarcomere length and is the same everywhere; the relaxation law therefore plays no part. 80 ms ≈ 7.5 frames of
  550 / 52 ms after the onset at frame 43.5 (the onset of the volume-driven model `../../volume_constrain`, which
  needs the whole activation curve over the 52 frames: maximum-volume frame t43 + 0.5; keep both consistent).
- **Cavity:** the given pressure P does the work -P V(u), V(u) = -1/3 int_endo (X + u) . cof(F) N dS.
- **Boundary conditions:** base u · n = 0; no rigid translation in the base plane and no rigid rotation (multipliers);
  no epicardial spring by default (at 48 hpf the heart beats in a fluid-filled pericardial cavity and has no epicardium
  yet). Optional: `K_SPRING` > 0 (kPa/mm) restores heartFEM's epicardial spring, traction −K (u · N) N (N the reference
  epicardial normal). The PINN example takes the spring relative to C
  (`K_SPRING_OVER_C`, see its README).

## Inverse problem
**Given pressures.** No measurement of the atrial pressure exists at 48 hpf (pressure recording was not feasible at
that stage; Hu et al. 2000). The values are derived from the nearest measurements:
- P_ED = 1.36 mmHg = 0.1813 kPa: ventricular end-diastolic pressure of the 3 dpf zebrafish (Salehin et al. 2021); at
  end-diastole the AV canal is open, so atrial and ventricular pressure are equal;
- P_PS = 7.52 mmHg × 0.68 / 2.49 = 2.05 mmHg = 0.2738 kPa: 3 dpf ventricular peak systolic pressure (Salehin et al.
  2021) times the atrial-to-ventricular peak-pressure ratio of the adult zebrafish (Hu et al. 2001, 2000).

**Misfit.** For each frame:
1. the tracked frame is aligned to the tracked reference frame by the best rigid fit (the model has no rigid motion);
2. surface distance, endo and epi separately: symmetric point-to-plane distance, area-weighted rms over the points
   farther than 0.114 L from the base plane;
3. cavity volume against the image volume V_img = V0 × V_img,k / V_img,ref.

```
J = sum over ED, PS of ((V - V_img) / (0.01 V_img))^2 + (d_endo / sigma_s)^2 + 0.2 (d_epi / sigma_s)^2,   sigma_s = 0.02 L
```
with L = V0^(1/3), the cube root of the reference cavity volume (87.8 um here: sigma_s = 1.76 um). All length
tolerances are fractions of L, so the misfit does not depend on the size of the heart.

**Why C and T_MAX can be separated.** The shape depends only on P / C and T_MAX Ct / C. The passive ED frame (Ct = 0)
therefore fixes P_ED / C, and with P_ED given, C; the PS frame then fixes T_MAX. Only the ratios are seen by the
images: C and T_MAX scale with the given pressures.

**Search** (step `invert`):
1. *geometry start*, no guess: C0 such that the passive wall reaches the imaged ED volume at P_ED; T0 such that the
   wall at (P_PS, t_PS) reaches the imaged PS volume with C0 (secant iterations on ln C and ln T_MAX);
2. for every start (C0, T0) × `START_FACTORS` ((1, 1), (0.5, 2), (2, 0.5)):
   - *A*: ED frame only (T_MAX has no effect), C free;
   - *B*: PS frame only, T_MAX free (C from A).

Each stage is a Nelder-Mead search on ln C or ln T_MAX, stopping when the simplex is smaller than 2e-4 and J changes by
less than 1e-4 (at most 250 evaluations). Every evaluation solves the FE model by continuation from the last converged
state of the same frame. Each start is then scored by J over both frames; the lowest gives the result.

## Output (`output/`)
| file | content |
|---|---|
| `result.json` | C, T_MAX, T_MAX Ct at PS, fibers; misfit J; convergence and spread of the starts; per start the start point, each stage (J, evaluations, convergence) and the result; per frame the activation time, pressure, cavity and image volume; evaluations, time |
| `evaluation.json` | per frame: endo / epi surface distance (rms, signed mean, max), cavity volume vs image, Dice (wall, cavity) |
| `ED.vtu`, `PS.vtu` | the deformed wall at each frame, in the coordinates of `input/t43.stl` / `input/t51.stl` (opens over them, e.g. in ParaView); point data `displacement_um`, `distance_to_tracking_um` (signed, > 0 where the model lies farther from the cavity than the tracking; NaN inside the wall); cell data `fiber_1` (, `fiber_2`) |
| `comparison.png` | long-axis section through the apex at ED and PS, tracking vs model, with C and T_MAX |

`cache/mesh/` holds the preprocess products (`t0.vtu` with eC / eL / eR per cell, `parameter.txt`, `surface_cut/`,
`preprocess_info.json`).

## Result
This example's run (`output/`), one fiber family 0°, pressures 0.1813 / 0.2738 kPa:

| | C (kPa) | T_MAX (kPa) | J | evaluations A / B |
|---|---|---|---|---|
| geometry start | 0.06967 | 4.0029 | | |
| start (1, 1) | 0.07060 | 4.0162 | 33.490 | 26 / 26 |
| start (0.5, 2) | 0.07058 | 4.0158 | 33.491 | 27 / 28 |
| start (2, 0.5) | 0.07058 | 4.0160 | 33.491 | 30 / 26 |
| **result** | **0.07060** | **4.0162** | **33.490** | |

All stages of all starts stopped by the tolerances. The active tension at PS is T_MAX × Ct = 4.016 × 0.75 = 3.01 kPa.
Against the tracking (`output/evaluation.json`, `comparison.png`):

| | ED (t43) | PS (t51) |
|---|---|---|
| endocardial distance rms (signed mean, max) | 5.73 um (+0.42, 24.5) | 7.34 um (+1.17, 27.4) |
| epicardial distance rms (signed mean, max) | 5.21 um (+1.24, 22.4) | 7.41 um (+3.07, 24.8) |
| cavity volume vs image | 0.94733 vs 0.94912 nL (-0.19 %) | 0.69366 vs 0.69424 nL (-0.08 %) |
| DSC wall / cavity | 0.523 / 0.855 | 0.413 / 0.807 |

The example `../FE_two_fibers` (two families 0° / 90°): C 0.07704 kPa, T_MAX 4.3682 kPa, J 25.72; ED endo / epi 5.63 /
5.30 um, PS 5.88 / 6.04 um. At PS two families fit the tracking better (endocardial rms 5.9 um with two families, 7.3 um
with one), with the same two unknowns.

## Verification
Tests with the same forms, misfit and search, at an activation time with Ct(PS) = 0.742 (79.3 ms; this scales T_MAX by
0.75 / 0.742 and changes nothing else):

- **Forward model.** With one family at 0° the FE solution equals that of a single-fiber implementation (displacement difference 2e-15, relative).
- **Recovery of known parameters** (FE solutions used as "tracking", from the geometry start and the two other starts): one family, (C, T_MAX) = (0.0759, 5.544): C 0.07590, T_MAX 5.5438 - 5.5443; (0.05, 5.0) with Gaussian noise of 0.02 L (1.8 um) on the surfaces: 0.04998 - 0.04999, 5.0001 - 5.0004; two families, (0.077, 4.41): 0.07699 - 0.07700, 4.4100 - 4.4102. All within 0.05 %.
- **Stages A + B vs a final joint search.** A third stage over both frames with both parameters free changed C by 0.7 % with one family (J by 0.02 %) and nothing with two families; it was dropped.
- **Uniqueness.** In every run the three starts reach the same point.

## Limitations
- **Pressures are assumed.** C and T_MAX scale with the given pressures (×0.8 and ×1.2 change both by exactly that
  factor); only P / C and T_MAX / C are fixed by the images.
- **T_MAX depends on the activation timing.** The images fix the active tension at PS, T_MAX × Ct; T_MAX follows with
  Ct = 0.75 from `PS_ACTIVATION_MS` = 80 ms and t0 = 120 ms (75 ms, i.e. an onset half a frame later: Ct 0.69, T_MAX
  +8.5 %). C does not depend on it.
- **Fibers are assumed.** T_MAX depends on the fiber architecture (one family at 0°, two families 0° / 90°, and a
  single inferred angle of 31.4° gave 4.06, 4.41 and 5.54 kPa at Ct 0.742 in the development study); C much less
  (0.071 - 0.077 kPa).
- **Stages A and B fit the frames one after the other** (C from ED, then T_MAX from PS) and do not re-adjust C to PS.
  A final joint search over both frames changed C by 0.7 % with one family at 0° (J by 0.02 %) and not at all with two
  families (development study).
- **Remaining misfit (about 5 - 6 um)** is not reduced by these two parameters. It is largest near the base: the
  tracked wall crosses the base plane by up to about 20 um, while the model's base moves in its plane only (u · n = 0).
  The tracked wall volume also changes by ±12 % over the cycle, which an incompressible wall cannot follow; the wall
  Dice is therefore bounded (about 0.94 at ED even with a perfect shape).
- **Search:** Nelder-Mead is local; the geometry start and the three starts guard against local minima.

## References
- Hu N., Sedmera D., Yost H. J., Clark E. B. (2000) Structure and function of the developing zebrafish heart. Anat Rec
  260:148-157.
- Hu N., Yost H. J., Clark E. B. (2001) Cardiac morphology and blood pressure in the adult zebrafish. Anat Rec 264:1-12.
- Salehin N. et al. (2021) Assessing pressure-volume relationship in developing heart of zebrafish in-vivo. Ann Biomed
  Eng 49:2080-2093.
- heartFEM (Lee lab; W. X. Chan): `lcleeHeart` forms (`forms_shavik`, `simpleActiveMaterial`).
- Guccione J. M., Waldman L. K., McCulloch A. D. (1993) Mechanics of active contraction in cardiac muscle: Part II -
  cylindrical models of the systolic left ventricle. J Biomech Eng 115:82-90.
- Nelder J. A., Mead R. (1965) A simplex method for function minimization. Comput J 7:308-313.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`); third-party notices (heartFEM, three.js) in
`../../../THIRD_PARTY_NOTICES.md`.