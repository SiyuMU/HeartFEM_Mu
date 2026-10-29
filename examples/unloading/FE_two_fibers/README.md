# unloading · FE · two fiber families

**Unloaded (stress-free) geometry and passive stiffness C from an end-diastolic wall.**

A wall imaged at end-diastole (ED) is loaded by the ED cavity pressure; its stress-free state is not imaged. From the
ED wall, the ED pressure P_ED and the cavity volume V0 of the stress-free state, this example finds

- the **unloaded geometry** X: the wall that, inflated to P_ED, gives the ED wall (a tetrahedral mesh with its fibers,
  with the nodes and tetrahedra of the ED mesh);
- **C**, the scale of the passive Fung law (kPa), which together with V0 makes this unloaded state unique.

The method is inverse elastostatics on the ED mesh: the forward FE problem of `../../invert_para` (fe.PressureProblem)
written on the unknown unloaded geometry and solved for it in one Newton system. Every result is checked by the forward
problem itself: the unloaded wall is loaded from zero to P_ED, and it must reproduce the ED wall. The fiber
architecture is predefined on the unloaded geometry, not inferred.

This example has **two fiber families** (`TWO_FIBERS = True`): circumferential (0°) and longitudinal (90°), each with
half of the myocardium. The example `../FE_one_fiber` is the same model with one circumferential family
(`TWO_FIBERS = False`).

## This example
```
FE_two_fibers/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  all settings: run settings (`TASK = "unloading"`, `SOLVER = "FE"`, resources), then the model
├── input/     raw data: ed.stl (the ED wall), base_plane.json (its base plane and apex)
├── cache/     preprocess products: mesh/ (the ED mesh), recomputed by `python main.py`
└── output/    results: result.json, unloaded.vtu
```
The code is the library `../../../heartfem_mu/` (`preprocess.py`, `model.py`, `fe.py`, `unloading.py`,
`unloading_fe.py`); the example folder holds the data, the settings and the results. `params.py` holds every setting of
this example: the run settings (task, solver, CPU cores, MPI ranks, conda environments) and the model. The git
repository holds `input/` and `output/`; `cache/` is not tracked.

## Run
```bash
python main.py                  # every step, input/ -> output/
```
The two conda environments (`CONDA_ENV`, `CONDA_ENV_FENICS` in `params.py`) are installed once with `bash ../../../environments/install.sh` (see Install in `../../../README.md`).

The steps (`python main.py --list`); each runs in its own process in the conda environment it needs:

| step | does | code (`heartfem_mu/`) | runs in |
|---|---|---|---|
| `preprocess` | raw surface -> tetrahedral mesh of the ED wall, local frame | `preprocess.main` | `CONDA_ENV`, `MESH_THREADS` threads |
| `unload` | C and the unloaded geometry (inverse elastostatics), check | `unloading.fe_unload` | `CONDA_ENV_FENICS`, MPI, `MPI_RANKS` ranks (and one helper process in `CONDA_ENV`) |

`python main.py <step> ...` runs only the named steps, `python main.py --from <step>` a step and the following ones
(after a change of `V0_NL`, `P_ED_KPA`, the material or the fibers: `python main.py unload`). Resources (`params.py`):
`CPUS`, the cores the steps may use (a `taskset` list; `None`: any core), and `MPI_RANKS`; the FE step runs one thread
per MPI rank.

## Input (`input/`)
| file | content |
|---|---|
| `ed.stl` | The ED wall, um: one closed surface already cut at the base plane (endocardium, epicardium and the flat base in the plane). Here the surface of an ED wall made with the same model: the mesh of the tracked frame t12 of a 48 hpf zebrafish atrium (the stress-free reference of `../../invert_para`) inflated to P_ED with two fiber families 0° / 90°. |
| `base_plane.json` | Saved landmarks of this surface: the base plane (the base of `ed.stl` lies in it) and the apex on the ED surface. |

The input surface can also be a tracked frame as in the other tasks (two closed shells, the endocardial and the
epicardial surface): the preprocess then cuts it at the base plane. For new data: one STL file of the ED wall in
`input/` (any name; `t<k>.stl` gives the frame number k), its `base_plane.json` (for two shells also placed in the
browser app, `USE_SAVED_LANDMARKS = False`), and `P_ED_KPA`, `V0_NL` in `params.py`.

## Pipeline
1. **`preprocess`**: builds the 5 um tetrahedral mesh of the ED wall (gmsh remeshes the surface at that size; a
   two-shell surface is first cut at the base plane) with its local frame eC / eL / eR per cell (`cache/mesh/t0.vtu`),
   as for every other task.
2. **`unload`** (MPI): builds the inverse problem on the ED mesh and runs the loop over the local frame, the base
   origin and C (below); checks the result with the forward problem; computes d ln C / d ln V0; writes
   `output/result.json` and `output/unloaded.vtu`.

## Settings (`params.py`)
Usually only `P_ED_KPA`, `V0_NL` and the fibers need to be set for new data; the rest are model constants and
numerical settings.

| group | setting | meaning |
|---|---|---|
| folders | `INPUT_DIR`, `CACHE_DIR`, `OUTPUT_DIR` | raw data; preprocess products (`CACHE_DIR/mesh`); results |
| preprocess | `MESH_SIZE` | tetrahedral mesh size (5 um) |
| | `MESH_THREADS` | threads of the gmsh mesher (1 here): the tetrahedra depend on the thread count |
| | `USE_SAVED_LANDMARKS` | `True`: base plane and apex from `input/base_plane.json`; `False`: place them in the browser app |
| | `MM_PER_UNIT` | input coordinates are um, the model works in mm and kPa |
| given | `P_ED_KPA` | cavity pressure at ED (0.1813 kPa); see *Unloading problem* |
| | `V0_NL` | cavity volume of the unloaded wall, closed by the base plane (0.6771818 nL); see *Unloading problem* |
| material (fixed) | `C_FF` ... `C_FN` | exponents of the Fung law (relative stiffness of the fiber, sheet, normal and shear directions); C, the overall scale, is the result |
| | `CA0`, `CA0MAX`, `B`, `L0`, `LR` | active law of heartFEM; inactive at ED (Ct = 0), they set only the residual scale (Newton tolerance) of the forward problem, as in `../../invert_para` |
| | `K_SPRING` | heartFEM epicardial spring; unloading supports 0 only (no spring) |
| fibers (fixed) | `TWO_FIBERS` | `True`: two families (this example); `False`: one family (`../FE_one_fiber`) |
| | `FIBER_ANGLES_DEG`, `FIBER_WEIGHTS` | (endo, epi) angles and shares of the two families (((0°, 0°), (90°, 90°)), (0.5, 0.5)) |
| search | `C_START_KPA` | C of the first iteration (0.4 kPa); the unloaded wall of the first iteration starts at the ED wall |
| | `SECANT_SLOPE0`, `MAX_STEP_LN_C` | d ln V0 / d ln C of the first C step (0.2); largest step of ln C per iteration (0.9, a factor 2.5) |
| | `TOL_X_UM` | stop when the unloaded wall changes by less than this per iteration (1e-4 um, max over the nodes) ... |
| | `V0_TOL_REL`, `TOL_LN_C` | ... and its cavity volume meets `V0_NL` to 1e-8 or the next step of ln C is below 1e-6 |
| | `MAX_ITERATIONS` | cap of the loop (100) |
| check | `CHECK_LOAD_STEPS` | the unloaded wall is loaded from zero to P_ED in this many equal steps (10) |

## Model
The mechanics are those of `../../invert_para/FE_two_fibers`, passive (ED, Ct = 0), on the unloaded geometry:

- **Geometry:** the unloaded wall X with the topology of the ED mesh (5 um linear tetrahedra, straight cells) and a
  quadratic (P2) displacement u to ED. Local frame per cell from two Laplace problems on X (the functions of
  `preprocess.py`): eL (longitudinal), eR (radial), eC = eR x eL.
- **Fibers** (`params.py`, not inferred): family i has the direction f_i = cos(a_i) eC + sin(a_i) eL on X, sheet
  s0 = eR, normal n_i = f_i x s0, and the share w_i; a_i is linear in the transmural depth d_endo / (d_endo + d_epi):
  here a = (0°, 0°) and (90°, 90°), w = 0.5 / 0.5.
- **Passive:** W = sum_i w_i C (e^Q_i - 1) - p (J - 1), Q_i = C_ff E_ff² + C_ss E_ss² + C_nn E_nn² + C_ns E_ns² +
  C_fs E_fs² + C_fn E_fn² in the frame (f_i, s0, n_i) (heartFEM `forms_shavik`); C is the result, the exponents
  C_ff ... C_fn are fixed (29.9, 13.3, 13.3, 26.6, 53.2, 53.2).
- **Cavity:** the given pressure P_ED does the work -P V(u), V(u) = -1/3 int_endo (X + u) . cof(F) N dS.
- **Boundary conditions:** base u · n = 0; no rigid translation in the base plane and no rigid rotation (multipliers;
  the rotation constraints use the base origin of X); no epicardial spring. With no spring the deformed shapes depend
  on P_ED / C only.

## Unloading problem
**Given values.** P_ED = 1.36 mmHg = 0.1813 kPa: ventricular end-diastolic pressure of the 3 dpf zebrafish (Salehin et
al. 2021); at end-diastole the AV canal is open, so atrial and ventricular pressure are equal (as `../../invert_para`).
V0 = 0.6771818 nL: the cavity of the t12 mesh of `../../invert_para`, closed by the base plane. V0 is not measured: any
value below the ED cavity can be set, and C follows from it.

**Why V0 is needed.** For every C there is an unloaded wall that inflates exactly to the ED wall (a softer C, a smaller
unloaded wall). The ED wall and P_ED fix only this one-parameter family; the volume V0 selects C and X.

**Method** (`heartfem_mu/unloading.py`, `unloading_fe.py`): inverse elastostatics on the fixed ED mesh x. The unknowns
are the forward displacement u (P2), the pressure p (P1), the five rigid-body multipliers and the inverse displacement
U (P1, vertex values); the unloaded wall is X = x - U. The residual of the forward problem on X is pulled back to the ED
mesh (dV_X = det A dv, Grad_X = grad_x A^-1, N dA_X = cof(A) n da, A = I - grad U, constant per cell) and U = u is
imposed at the vertices (x_v = X_v + u(X_v)). These are the discrete equations of the forward problem on X, solved for
X as one Newton system (MUMPS). The local frame of X (and the fibers on it) and the base origin of X are lagged, and C
is found in the same loop: iteration k computes the frame and origin of X_k, solves the inverse problem at C_k
(continuation from the previous solution), and takes a secant step on ln C toward V0(X) = V0_NL.

**Check.** The result is loaded with the forward problem from zero to P_ED in 10 equal pressure steps (no warm start)
and must reproduce the ED wall (`reaches_ed`: max |x_ED - X - u| < 1e-3 um). This is not a formality: the equilibrium
is not unique in general. Below a critical V0 the inverse problem still has a solution, but loading that unloaded wall
from zero ends in another state; `result.json` then has `reaches_ed: false`.

## Output (`output/`)
| file | content |
|---|---|
| `result.json` | C, V0 (and its error), P_ED, ED cavity of the mesh, the check (`reaches_ed`, residual, loaded ED cavity), d ln C / d ln V0 at the final frame, wall volumes, displacement unloaded -> ED, the iterations (C, change of X, V0, Newton steps, frame diagnostics) |
| `unloaded.vtu` | the unloaded wall: the nodes and tetrahedra of the ED mesh at the unloaded positions (coordinates of `input/ed.stl`, um); cell data eC, eL, eR, depth, fiber_1, fiber_2; point data `u_to_ed_um` (the displacement of the check: X + u is the ED wall) |

## Result
This example's run (`output/`), two fiber families 0° / 90° (shares 0.5 / 0.5), P_ED 0.1813 kPa, V0 0.6771818 nL:

| | |
|---|---|
| **C** | **0.07891 kPa** (0.0789126) |
| V0 of the unloaded wall | 0.6771818 nL (relative error 3.5e-8) |
| ED cavity: mesh / the check | 0.944862 / 0.945218 nL |
| check: max \|x_ED - X - u\| | 9.0e-7 um (`reaches_ed`: true) |
| d ln C / d ln V0 | 6.78 (1 % in V0 gives about 7 % in C) |
| wall volume: ED / unloaded | 0.43082 / 0.43053 nL |
| displacement unloaded -> ED | max 16.2 um, rms 7.3 um |
| mesh | 5895 nodes, 19138 tetrahedra (5 um) |
| loop | 8 iterations, 29 Newton steps |

The ED cavity of the mesh is that of its straight-sided tetrahedra; the check reports the cavity of the quadratic FE
displacement. The example `../FE_one_fiber` (one family 0°) gives C 0.07006 kPa on the same ED wall.

## Verification
The method was verified in development tests with the same code: the recovery of a known unloaded state from an ED
wall made by the forward model (C and every node), the same wall remeshed, the frame recomputed on the unloaded
geometry against the frame of the ED wall, and the agreement with the fixed point X <- x_ED - u_C(X) (Sellier).

## Limitations
- **P_ED and V0 are assumed.** With no spring the shapes depend on P_ED / C only: C scales exactly with P_ED. C depends
  strongly on V0 (`dlnC_dlnV0` in `result.json`); below a critical V0 there is no reachable unloaded state (check).
- **Fibers are assumed** on the unloaded geometry (two families 0° / 90° here; one family 0° in `../FE_one_fiber`).
- **Base:** u · n = 0 keeps the base of the unloaded wall in the base plane of the ED wall; a tracked base rim that
  moves out of that plane cannot be represented.
- **Passive, quasi-static, no epicardial spring** (`K_SPRING = 0` only); the residual stresses of a real unloaded heart
  are not modeled (the unloaded state is stress-free).
- **Mesh:** the remeshing of the ED surface changes its cavity volume slightly; that error enters C times
  d ln C / d ln V0.

## References
- Salehin N. et al. (2021) Assessing pressure-volume relationship in developing heart of zebrafish in-vivo. Ann Biomed
  Eng 49:2080-2093.
- heartFEM (Lee lab; W. X. Chan): `lcleeHeart` forms (`forms_shavik`).
- Govindjee S., Mihalic P. A. (1996) Computational methods for inverse finite elastostatics. Comput Methods Appl Mech
  Eng 136:47-57.
- Sellier M. (2011) An iterative method for the inverse elasto-static problem. J Fluids Struct 27:1461-1470.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`); third-party notices (heartFEM, three.js) in
`../../../THIRD_PARTY_NOTICES.md`.
