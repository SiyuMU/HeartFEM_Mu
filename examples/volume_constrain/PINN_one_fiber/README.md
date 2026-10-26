# volume_constrain · PINN · one fiber family

**PINN P-V cycle of the zebrafish embryonic atrium (coordinate-free physics-informed network).**

This example solves the same problem as the FE example `../FE_one_fiber`: the imaged zebrafish embryonic atrium
(48 hpf) over one cardiac cycle, with the cavity volume of each of the 52 tracked image frames prescribed. For
every frame it computes:
- the cavity **pressure** (the P-V loop);
- the **displacement**;
- the **stresses**.

The FE solves a nonlinear problem for each frame. Here, one network is trained once by minimizing the potential energy
of the same mechanics, without any FE data. It returns the equilibrium displacement for any cavity pressure P and time t
(the activation time of the FE: ms since the onset of contraction). With the network trained, the volume constraint of
each frame is a one-dimensional root find: all 52 frames take less than a second.

This example has **one fiber family** (`TWO_FIBERS = False`), circumferential (0°). The example `../PINN_two_fibers` is
the same model with two families, circumferential and longitudinal, each with half of the myocardium.

## This example
```
PINN_one_fiber/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  all settings: run settings (`TASK = "volume_constrain"`, `SOLVER = "PINN"`, resources), then the model
├── input/     raw data: t0.stl ... t51.stl (the 52 tracked frames), base_plane.json
└── output/    mesh/ (step preprocess), model.pt and the training log (step train), the results of the cycle
```
The code is the library `../../../heartfem_mu/` (`preprocess.py`, `model.py`, `pinn.py`, `volume_constrain.py`, `plots.py`); the example folder holds the data, the settings and the results. `params.py` holds every setting of this example: the run settings (mode, resources, conda environments) at the top, then the model, the numerics and the output options, each with its meaning.

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
| `train` | train the (P, t) network (energy minimization) | `pinn.train_load_network` | `CONDA_ENV`, `GPU` |
| `solve` | volume-constrained cycle with the network | `volume_constrain.pinn_cycle` | `CONDA_ENV`, `GPU` |
| `plot` | pv_loop.png/.pdf, pv_cycle.mp4 | `plots.pv_figures` | `CONDA_ENV` |

`python main.py <step> ...` runs only the named steps, `python main.py --from <step>` a step and the following ones.
Resources (`params.py`): `GPU`, the GPU of the training and the solves (the `nvidia-smi` index; `None`: the CPU,
training about ten times slower), and `CPUS`, the cores the steps may use (a `taskset` list; `None`: any core).

## Input (`input/`)
`input/` and `output/mesh/` are identical to `../FE_one_fiber`: same frames, landmarks, mesh, and node and cell order.

| file | content |
|---|---|
| `t0.stl ... t51.stl` | Tracked surfaces of the 52 frames of one cycle, in um. Each file holds the closed endocardial and epicardial surfaces; all frames share the same triangle connectivity. |
| `base_plane.json` | Saved landmarks: the reference frame (t12), the base cut plane and the apex (placed once in the landmark browser app (`heartfem_mu/landmarks.py`)). |

## Model

### 1. Mechanics (identical to the FE example)
The wall of the reference frame t12 is the unloaded state. Loads: cavity pressure P on the endocardium and the active
fiber stress at time t. The equilibrium displacement u minimizes the total potential energy

```
Pi(u; P, t) = int_wall [ C (e^Q - 1) + Psi_act(lambda; t) ] dV               passive Fung law + heartFEM active stress
              + incompressibility                                           (see 2.)
              - P (V_cav(u) - V_cav(0))                                     work of the cavity pressure
subject to    u . n = 0 on the base plane, no rigid translation in the base plane, no rigid rotation
```

- **Passive:** Q = C_ff E_ff^2 + C_ss E_ss^2 + C_nn E_nn^2 + C_ns (E_ns^2 + E_sn^2) + C_fs (...) + C_fn (...), E the
  Green-Lagrange strain in the fiber (f), sheet (s) and normal (n) directions. C = 0.1 kPa, C_ff 29.9, C_ss = C_nn 13.3,
  C_ns 26.6, C_fs = C_fn 53.2.
- **Fibers:** family i: f_i = cos θ_i eC + sin θ_i eL, sheet eR, normal n_i = f_i x eR. Default two families,
  `FIBER_ANGLES_DEG` = ((0°, 0°), (90°, 90°)) with shares w_i = `FIBER_WEIGHTS` = 0.5 / 0.5; `TWO_FIBERS = False`:
  one family at `FIBER_ANGLE_DEG` = (0°, 0°) (circumferential). Every angle is an (endo, epi) pair, linear in the transmural depth d_endo / (d_endo + d_epi) (the
  defaults are the same at every depth).
- **Two fiber families** (as the FE): the passive energy is the share-weighted sum Σ_i w_i C (e^(Q_i) - 1), Q_i in the
  frame of family i, and the active energy Σ_i w_i Psi_act(lambda_i; t) with each family's own stretch
  lambda_i = |F f_i| (own ls_i, tr_i and Ct_i). The formulas below are those of one family.
- **No epicardial spring** (as the FE: at 48 hpf the heart has no epicardium yet). Optional: `K_SPRING` > 0 restores
  heartFEM's epicardial spring, K/2 ∫_epi (u · N)² dA in the training energy (traction −K (u · N) N); the network must
  then be trained again.
- **Active stress** (heartFEM `simpleActiveMaterial`): S_act = T_max Ct(t, ls) g(ls) f0 x f0 (second Piola-Kirchhoff),
  with fiber stretch lambda = |F f0|, sarcomere length ls = lr lambda, g = Ca0^2 / (Ca0^2 + ECa(ls)^2),
  ECa = Ca0max / sqrt(exp(B (ls - l0)) - 1) (ls - l0 floored at 0.003 um); Ca0 = Ca0max 4.35, B 4.75, l0 1.58 um,
  lr 2.0 um, T_max 20 kPa.
- **Activation Ct(t, ls)**, t = time since the onset of contraction (onset = frame of maximum tracked cavity volume,
  t43, + `ONSET_AFTER_ED_FRAMES` 0.5 = frame 43.5; frame k at t = mod(k - 43.5, 52) x 550/52 ms):

  | phase | Ct |
  |---|---|
  | 0 <= t < t0 (120 ms) | 0.5 (1 - cos(pi t / t0)), the same everywhere |
  | t0 <= t < t0 + tr(ls) | 0.5 (1 + cos(pi (t - t0) / tr)), with the relaxation time tr = m ls + b (m 73 ms/um, b -33 ms) |
  | afterwards | 0 |

  tr is a field: relaxation is slower where the sarcomeres are longer.
- **Active energy.** For fixed t the active stress depends on E only through lambda, so it has a potential:
  Psi_act(lambda; t) = T_max int_1^lambda Ct(t, lr s) g(lr s) s ds, with dPsi_act/dE_ff = T_max Ct g, exactly the FE's
  S_act (checked to 1e-10 kPa).
  - Rise (t < t0): Ct does not depend on ls, and the integral has a closed form.
  - Relaxation: Ct(t, lr s) = 0 below s* = (t - t0 - b) / (m lr) and is C1 there. The integral therefore runs over
    [max(1, s*), max(lambda, s*)], split at the kink of g (s = (l0 + 0.003) / lr). It is evaluated with 6-point
    Gauss-Legendre per smooth piece; the stress error is below 1e-6 kPa.
  - Check without the network (direct minimization over the P2 nodal displacements at the FE frame loads): the cavity
    volume agrees with the FE within 0.13 % in every tested frame, rise, relaxation and passive alike (one family).
  - Two families: the autograd derivative of the mixed energy equals the mixed analytic stress used by the step `solve`
    to 3e-12 kPa (passive) and 1e-9 kPa (active) on the FE's own deformations of every cell and frame.

### 2. Discretization
The displacement lives in the same space as the FE: **quadratic (P2) tetrahedra** on the 5 um mesh
(5708 vertices + edge midpoints = 35706 nodes, 19472 tets).
- **Deformation gradient:** F = I + sum_a u_a (x) grad N_a, with the P2 shape-function gradients of the unloaded mesh,
  precomputed once. The network never sees coordinates.
- **Quadrature:** 8-point degree-3 Stroud conical rule per tetrahedron (`QUAD_ORDER = 2`; 156k points).
- **Incompressibility:** penalty KAPPA/2 sum_a m_a Jbar_a^2, with Jbar_a the lumped P1 projection of J - 1 onto vertex a,
  m_a the vertex volume and KAPPA = 1000 kPa. It acts on the same P1 pressure space as the FE's Taylor-Hood constraint,
  avoiding the locking of a pointwise penalty. The FE enforces J = 1 exactly with a Lagrange multiplier.
- **Cavity volume:** exact for the curved P2 endocardium: V = 1/3 int (x - o) . n dA, closed by cones to the base
  origin o, with a 7-point degree-5 triangle rule.
- **Rigid-body terms:** exact integrals over the P2 volume. Rigid translation in the base plane
  and rigid rotation are removed by a penalty (RIGID_PENALTY = 1000 kPa; the FE uses Lagrange multipliers).
- **Base:** u . n = 0 is built into the network output (section 3).
- **Overflow guard:** the Fung exponential is continued C^2-smoothly by a quadratic above Q = 25. Solutions stay far
  below that value; only wild optimizer trial steps reach it, which then stay finite instead of overflowing.

### 3. Network
```
features(P, t) -> 5 x 128 tanh MLP -> h -> linear layer        -> u_node  (35706 x 3)
                                       -> linear layer (8 um)  -> coarse u on 1713 clusters -> interpolate -> u_8
                                       -> linear layer (20 um) -> coarse u on  222 clusters -> interpolate -> u_20
                                       -> linear layer (50 um) -> coarse u on   29 clusters -> interpolate -> u_50
U(P, t) = u_node + u_8 + u_20 + u_50
u(P, t) = 0.01 mm x (U(P, t) - U(0, 0)),   then the base-normal component is removed on the base nodes
```
- **Inputs:** cavity pressure P (kPa) and time t in [0, 550) ms, the same t as in the FE's Ct(t, ls): ms since the onset
  of contraction (frame k: t = mod(k - 43.5, 52) x 550/52 ms, the FE's `t_active_ms`). The network sees six features,
  all computed from heartFEM's formula:
  - P / 1.6;
  - Ct(t, lr lambda_k) for the fiber stretches lambda_k = 0.8, 1.0, 1.2, 1.4, 1.6.

  On the rise the five Ct values are equal (Ct does not depend on ls). In relaxation they differ: their spread encodes how
  the relaxation depends on the sarcomere length. Once even lambda = 1.6 has relaxed (t > 320.6 ms) they are all 0, so
  every passive time gives exactly the same input as t = 0, the unloaded state. Nothing is cut off or mapped by hand,
  and the energy always uses the exact Ct(t, ls) of each quadrature point with its own ls. The five lambda_k only
  describe the time t to the network; they are not the fiber stretches of the solution, and they are the same for one
  or two fiber families.
  (Development versions used the time with a cutoff at 300 ms, or periodic sin/cos features of t; the
  first relies on an assumed maximum fiber stretch, the second drifted between the sampled passive times.)
- **Coordinate-free:** the input is only the load (P, t). The output is the displacement of every P2 node. The network
  is the solution operator (P, t) -> u of this one geometry. The last layer is linear, so for fixed hidden features the
  energy is an FE energy in the nodal unknowns. One energy evaluation costs about 10 ms per load case (most of it the
  material law at 156k points; the network itself is about 1 ms).
- **Multilevel output (preconditioner):** the mesh vertices are grouped into voxel clusters of 8, 20 and 50 um (about
  3 clusters across the chamber at 50 um). Each extra linear head predicts one displacement per cluster centre. A hat
  kernel max(0, 1 - d / (2 h)), normalized per node (partition of unity), interpolates it to every node. The four parts
  are summed into one displacement field; the energy is computed once, on the sum.
  - The nodal output alone can already represent every P2 field. The coarse heads therefore do not change the
    solution, only the parametrization.
  - Purpose: a gradient step on a nodal value only sees its neighbouring cells, so global deformations (the chamber
    inflating or contracting) converge slowly. A coarse parameter moves a whole region at once, and its gradient is
    the sum of the nodal gradients of that region. The large-scale modes therefore converge first, much like a
    multigrid hierarchy. This halves the training time at +6 % cost per evaluation.
  - The split between the parts is not unique and has no meaning on its own; only the sum is the displacement.
- **Unloaded state and base:** subtracting U(0, 0) makes u = 0 exactly at zero load. The base-normal component is
  removed on the base vertices and midpoints, so u . n = 0 holds exactly.
- **Size:** 14.7 M weights, almost all in the nodal output layer (128 x 35706 x 3). Saved model: 59 MB.

### 4. Training (step `train`, no FE data)
The loss is the potential energy Pi of section 1. Minimizing it over the network weights, for many loads at once, makes
the network output the equilibrium for every load.

**Stage 1: Adam warm-up** (1000 steps):
- 8 random loads per step: P uniform; activation time on the rise from a tension level uniform in [0, T_max] (35 %),
  uniform in the relaxation window (120, 320] ms (35 %), or passive, uniform in [320, 550) ms (30 %);
- a load ramp over 200 steps (loads scaled from 0 to full);
- a KAPPA ramp from 10 to 1000 over 400 steps;
- learning rate 1e-5 -> 2e-6, gradient clipping at norm 1.

The nodal output only tolerates a small learning rate (above about 1e-5 per-weight Adam steps become node-by-node noise
and the exponential law overflows). Adam therefore only brings the network into the right basin.

**Stage 2: full-batch L-BFGS** (this does most of the work):
- **Fixed load grid.** L-BFGS needs a deterministic objective, so random sampling is replaced by a fixed grid:
  - n pressures, log-spaced and dense at small P: P_i = 0.02 (e^(a_i ln 81) - 1), a_i in [0, 1];
  - times t:
    - rise: the exact times at which Ct(t) = T_j / T_max for n_T tension levels T_j log-spaced in [0, T_max] (0.5 kPa
      scale). On the rise Ct does not depend on ls, so these times follow from the formula without approximation;
      t = 0 (the passive state) is one of them;
    - relaxation: round(4/3 n_T) times uniform in (120, 320] ms, where the wall relaxes (fibers with stretch 0.8-1.6
      finish between 204 and 321 ms). No reference relaxation time is assumed;
    - passive times need no points: their features equal those of t = 0;
  - all pairs except (0, t = 0): 35 / 83 loads for n = 4 / 6; the last stage uses 8 pressures x 14 times
    (6 tension levels + 8 relaxation times) = 111 loads.

  The soft passive filling at small P and the small tensions at the beginning and end of the twitch are the most
  nonlinear regimes.
- **Objective:** (1/N) sum_i Pi(u(P_i, t_i)) / w_i.
  - The weights w_i = |Pi_i|^0.5 (normalized) are taken from the energies at the start of each grid stage and then
    kept fixed. High loads have energies 100-1000 x larger than small loads.
  - Without weights the small loads are ignored. With w = |Pi| (power 1) the active, high-load states are neglected
    (P RMS 0.033 kPa). Power 0.5 balances the two.
- **L-BFGS settings** (PyTorch `torch.optim.LBFGS`):
  - lr 1, strong-Wolfe line search, 20 iterations per outer step.
  - **History 100.** The last 100 pairs (parameter change, gradient change) approximate the inverse Hessian, so each
    step is close to a Newton step. 100 instead of 50 is about 1.2 x faster and costs about 6 GB more GPU memory:
    each pair is 2 x 14.7 M floats.
  - The loads are processed in chunks of 11 to bound memory. Gradients are accumulated, so the step is still exact
    full-batch.
- **Staged load grids** (GRID_STAGES): n = 4 from outer step 1, 6 from step 42, then 8 pressures x 6 tension levels
  from step 101 (35 -> 83 -> 111 loads).
  - Early iterations on few loads are 3 x cheaper, while the network learns the global (P, t) response.
  - Earlier switches (outer 30 / 75) were slower; 6 relaxation times instead of 8 left 4.6 % u error in the relaxation
    frames (8: 2.7 %).
  - At each switch the weights are recomputed from the current energies and the L-BFGS history is cleared, because it
    belongs to the old objective.
  - Each switch costs a few minutes of slower progress, so there are only two, both early. A later switch or a direct
    4 x 4 -> 8 x 8 jump was slower.
- **Whitened output parametrization on the finest grid** (WHITEN 1e-5, from outer step WHITEN_FROM_OUTER = 101):
  - All output layers act on the difference of the last hidden features, u = W A (h - h0), with A = I at first.
  - Over the training loads the 128 hidden features are far from equally strong: their singular values decay by five
    orders of magnitude (the 20th is below 1 % of the largest). The curvature of the objective with respect to W scales
    with the square of these values, so displacement content that depends on fine differences between loads (small
    loads, the stretch-dependent relaxation) converges very slowly. This was the long tail of the training.
  - When the finest grid starts, A is set to (G / max eig + WHITEN I)^(-1/2), where G is the Gram matrix of h - h0 over
    the loads of the grid, and W is replaced by W A_old A_new^(-1). The displacement field is unchanged (to 1e-8 mm);
    only the coordinates of the optimization change, so that weak feature directions get as much curvature as the
    dominant ones. The hidden layers are evaluated in float64 on this path, so the amplification does not amplify
    float32 rounding.
  - Effect: from the same trained state the target is reached after ~18 outer steps instead of more than 54 (paired
    test), and after ~30 outer steps the accuracy equals that of the plain recipe at convergence.
  - It must not start earlier: whitening the features of a barely trained network (right after Adam) slowed training
    and spoiled the interpolation between the grid loads, and a switch at outer step ~50 left the relaxation frames
    behind (5 % u error).
- **Restarts and stopping:**
  - An outer step without progress (relative decrease < 1e-9) clears the curvature history and counts as a stall.
  - On the finest grid training stops when the objective has flattened: relative decrease over the last STOP_WINDOW = 10
    outer steps below STOP_TOL = 1e-4, about 30 outer steps after the whitening. Training on brings nothing (the
    displacement error is saturated and the pressure error slowly rises again).
  - 5 stalls in a row also end training on the finest grid; on a coarser grid they move on to the next grid.
  - TRAIN_HOURS (2 h) is a safety cap.

A warm start from the previous (P, T) model was worse than training
from scratch.

### 5. Volume-constrained cycle (step `solve`)
- **Targets:** V_k = V0 x V_k^image / V_t12^image, with V0 the network's cavity volume at u = 0 (= the FE's V0) and the
  image volumes from the cut surfaces of `output/mesh`.
- **Time:** t_k = mod(k - 43.5, 52) x 550 / 52 ms, as in the FE (`t_active_ms`).
- **Pressure:** with t_k fixed, V_PINN(P, t_k) increases monotonically with P. Bisection on [0, 1.6] kPa (40 halvings)
  gives P_k, for all frames in one batch. Frames outside the trained range would be flagged and clipped; none are.
- **Displacement:** u_k = u(P_k, t_k), the network output.
- **Stresses:**
  - F from the P2 displacement at the quadrature points.
  - The penalty pressure is noisy, so the hydrostatic pressure p is recovered as the P1 field that best balances the
    P2 weak momentum residual (least squares, base-normal reaction rows dropped).
  - sigma = F S F^T / J - p I, with S the passive + active second Piola-Kirchhoff stress.
  - Cell averages over the quadrature points (the FE stores DG0 projections) of sigma, F, sigma_ff, p and Ct.
  - Wall-mean tension T_mean = T_max x (volume mean of Ct(t_k, ls) over the wall), as the FE.
  - Cardiac-frame components exactly as the FE: r = F^-T eR, c = F eC made normal to r, l = r x c.

## Settings (`params.py`)
| group | settings |
|---|---|
| mesh, frames, units | MESH_SIZE 5 um, MESH_THREADS 36, USE_SAVED_LANDMARKS, N_FRAMES 52, MM_PER_UNIT 1e-3 |
| mechanics (= FE) | fibers (TWO_FIBERS, FIBER_ANGLES_DEG, FIBER_WEIGHTS; one family: FIBER_ANGLE_DEG), Fung constants, active law (CA0, CA0MAX, B, L0, LR), T_MAX, BCL_MS, T0_MS, TR_M_MS_PER_UM, TR_B_MS, ONSET_AFTER_ED_FRAMES, K_SPRING (0: no epicardial spring) |
| network input | (P, t), t = activation time as in the FE; FEATURE_STRETCHES (0.8, 1.0, 1.2, 1.4, 1.6) |
| discretization | KAPPA 1000, RIGID_PENALTY 1000, QUAD_ORDER 2 |
| network | P_MAX 1.6, WIDTH/DEPTH 128/5, LEVELS_UM (8, 20, 50), U_SCALE_MM 0.01, SEED 0 |
| Adam | ADAM_STEPS 1000, ADAM_LR 1e-5 -> 2e-6, ADAM_LOADS 8, ramps 200/400, KAPPA_START 10 |
| L-BFGS | GRID_STAGES ((4, 1), (6, 42), (8, 6, 101)), GRID_P0 0.02, GRID_T0 0.5, GRID_RELAX_FACTOR 4/3, GRID_RELAX_END_MS 320, WEIGHT_POWER 0.5, LBFGS_HISTORY 100, LBFGS_INNER 20, LBFGS_CHUNK 11, WHITEN 1e-5, WHITEN_FROM_OUTER 101, STOP_WINDOW 10, STOP_TOL 1e-4, TRAIN_HOURS 2.0 |
| cycle, output | BISECTION_STEPS 40, BASE_EXCLUDE_UM 10, FFMPEG, MOVIE_FPS |

GRID_STAGES counts L-BFGS outer steps, not minutes, so the schedule does not depend on the GPU speed.

## Output (`output/`)
Same files, names, units and layout as `../FE_one_fiber/output`, plus the training files:
- **`mesh/`** (step `preprocess`): `t0.vtu` (wall mesh of t12, um, cell data eC, eL, eR), `parameter.txt`,
  `surface_cut/` (all frames cut at the base plane, `correspondence.npz`), `preprocess_info.json`.
- **`model.pt`**: the trained network (config, weights, material, fibers, quadrature order, training time).
  **`train_log.csv`**: objective per Adam / L-BFGS step. **`train_info.json`**: time, step counts, grid stages.
- **`pv_results.npz`** (mm, kPa, nL, ms; frame k = `t{k}.stl`):
  - Per frame: `P_kPa`, `V_nL`, `V_ratio`, `T_mean_kPa` (wall mean of T_max Ct), `t_ms`, `t_active_ms`, `frame`.
  - Displacement: `u` (52, nodes, 3), in the `mesh/t0.vtu` node order.
  - Per cell, in the `mesh/t0.vtu` cell order: `sigma` (Cauchy), `sigma_ff` (family 1), `sigma_ff2` (family 2, two
    families only), `sigma_CC/LL/RR/CL/CR/LR` (current cardiac frame), `p` (recovered hydrostatic pressure), `Ct`
    (activation; two families: Σ_i w_i Ct_i) and `Ct_f1`, `Ct_f2` (each family, two families only).
  - Mesh: `points`, `cells`, `f0` (family 1), `f0_2` (family 2), `eC`, `eL`, `eR`, `cell_volume`, `far_from_base`; and
    `material` (with the fibers).
- **`summary.csv`**: one row per frame (P, V, T_mean, max |u|, time, wall means of sigma_ff/CC/LL/RR (and sigma_ff2)
  >= 10 um from the base).
- **`vtu/cycle.pvd`** (+ `frame00..51.vtu`), in um: point data `u_um`; cell data the stresses and Ct. Open in ParaView.
- **`pv_loop.png`** (300 dpi) and **`pv_loop.pdf`** (vector): (a) the P-V loop, (b-d) P(t), V(t) and the wall-mean tension.
- **`pv_cycle.mp4`**: one cycle from the maximum volume. Left: the deforming wall on a long-axis section through the
  apex, coloured by sigma_ff. Right: the P-V loop with the current frame.
- **`params_used.py`**, **`run_info.json`**.

## Result
All 52 frames (`output/pv_loop.png`, `summary.csv`). This example: one fiber family; the other column is the example
`../PINN_two_fibers`.

| quantity | two families (0° / 90°, 0.5 / 0.5; `../PINN_two_fibers`) | one family (0°; `../PINN_one_fiber`) |
|---|---|---|
| P at maximum volume (t43, end-diastole) | 0.235 kPa (1.76 mmHg) | 0.260 kPa (1.95 mmHg) |
| peak P (t51) | 1.354 kPa (10.16 mmHg) | 1.245 kPa (9.34 mmHg) |
| P at minimum volume (t5, end-systole) | 0.223 kPa (1.68 mmHg) | 0.292 kPa (2.19 mmHg) |
| minimum P | 0.027 kPa (t13) | 0.027 kPa (t13) |
| end of activation (wall-mean T < 1e-3 kPa) | t14, 238 ms after onset | t14, 238 ms after onset |
| wall-mean T at t12 (stress-free mesh frame) | 0.78 kPa | 0.56 kPa |
| largest displacement | 18.9 um (t47) | 24.0 um (t5) |
| peak wall-mean fiber stress (>= 10 um from the base) | family 1 (0°) 4.26 kPa (t51), family 2 (90°) 2.56 kPa (t50) | 5.79 kPa (t50) |
| peak wall-mean sigma_CC / sigma_LL | 4.26 / 2.49 kPa | 5.78 / 1.40 kPa |
| wall-mean Ct at t5 / t9 (relaxation) | 0.88 / 0.32 (family 1), 0.87 / 0.29 (family 2) | 0.85 / 0.24 |

The loop runs counterclockwise. There are no isovolumic phases, because the image volume changes in every frame.

## Accuracy against the FE example
Errors against `../FE_one_fiber/output` (the same mesh, node and cell order, and frames). Displacement and stresses are
relative L2 errors per frame; stresses are volume-weighted over the cells >= 10 um from the base and averaged over the
frames.

| quantity | PINN | FE |
|---|---|---|
| P at t43 / t51 / t5 / t13 | 0.260 / 1.245 / 0.291 / 0.027 kPa | 0.260 / 1.250 / 0.283 / 0.026 kPa |
| cavity pressure P error | RMS 0.0031 kPa, max 0.009 kPa (0.7 % of the peak, t5) |  |
| relaxation frames t3-t13 | P RMS 0.0047 kPa, u error 3.3 % |  |
| displacement u error | median 1.3 %, mean 2.3 %; max nodal error 1.78 um | FE max \|u\| 23.6 um |
| sigma_ff / sigma_CC error | 2.0 % / 2.0 % |  |
| sigma_LL / sigma_RR error | 2.5 % / 3.3 % |  |
| hydrostatic pressure p error | 2.4 % |  |
| activation Ct per cell | volume-mean error <= 0.0016 in every frame |  |
| wall-mean tension | within 0.031 kPa |  |

Another training of the same recipe had P RMS 0.0031 kPa, u median / mean 1.2 % / 2.3 %, sigma_ff 2.0 %; GPU training is
not bit-reproducible, so two trainings differ by this much.

## Verification
- **Two fiber families.** With `TWO_FIBERS = False` the post-processing of a one-family network reproduces its output (P identical, u within 1e-9 mm, stresses within 5e-5 kPa, float32); the two-family energy with both families at the same angle equals the one-family energy (1e-7, float32) and a weight-0 family contributes nothing; the derivative of the mixed energy equals the mixed stress of the step `solve` to 3e-12 kPa (passive) and 1e-9 kPa (active) on the FE's deformations of all cells and frames.
- **Active law.** dPsi_act/dE_ff equals the FE's active stress S_act to 1e-10 kPa.
- **Energy without the network.** Minimizing the same energy directly over the P2 nodal displacements at the loads of FE frames gives the FE cavity volume within 0.13 % in every tested frame (rise, relaxation and passive).
- **Whitening.** Setting the whitened parametrization changes the displacement field by at most 1e-8 mm.
- **Trained range.** No frame lies outside the trained pressure range (`P_clipped` = 0 for all 52 frames in `output/summary.csv`).
- **Run-to-run spread.** GPU training is not bit-reproducible. Independent trainings of this recipe give, against the FE, P RMS 0.0113 - 0.0114 kPa and a mean u error of 1.7 % with two families, P RMS 0.0031 - 0.0032 kPa and a mean u error of 2.2 - 2.3 % with one family.

## Limitations
- **Trained range:** P in [0, 1.6] kPa, all t in [0, 550) ms. Outside the pressure range the network extrapolates; retrain
  with a larger P_MAX if needed. The Ct features cover fiber stretches up to 1.6 (passive filling reaches 1.32).
- **One geometry:** the network is the solution operator of this mesh, these fibers and these material constants. A new geometry,
  mesh, fibers, material or activation law requires retraining.
- **Penalty versus exact constraints:** incompressibility and rigid-body constraints are penalties, whereas the FE uses
  multipliers. With one family this leaves a bias of about 1 % in P at the peak. With two families the finite KAPPA
  costs about 2-3 % of the peak pressure at peak tension (see Accuracy; a larger KAPPA reduces it but trains longer).
- **Run-to-run variation:** GPU training is not bit-reproducible. Independent runs differ by about 0.001 kPa in P and
  0.5-1 % in u.
- **Base region:** stresses within about 10 um of the base reflect the boundary condition. Wall means in
  `summary.csv` exclude them.

## References
- Cairelli A. G. et al. (2024) Role of tissue biomechanics in the formation and function of myocardial trabeculae in
  zebrafish embryos. J Physiol 602(4):597-617. doi:10.1113/JP285490
- Guccione J. M., Waldman L. K., McCulloch A. D. (1993) Mechanics of active contraction in cardiac muscle: Part II -
  cylindrical models of the systolic left ventricle. J Biomech Eng 115:82-90. doi:10.1115/1.2895474
- heartFEM (Lee lab; W. X. Chan): `lcleeHeart` forms (`forms_shavik`, `simpleActiveMaterial`).
- Peralta M. et al. (2013) Heartbeat-driven pericardiac fluid forces contribute to epicardium morphogenesis. Curr Biol
  23(18):1726-1735.
- Salehin N. et al. (2021) Assessing pressure-volume relationship in developing heart of zebrafish in-vivo. Ann Biomed
  Eng 49:2080-2093.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`); third-party notices (heartFEM, three.js) in
`../../../THIRD_PARTY_NOTICES.md`.