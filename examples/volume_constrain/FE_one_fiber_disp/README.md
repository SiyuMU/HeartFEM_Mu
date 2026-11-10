# volume_constrain · FE · one fiber family · displacement constraint

**FE P-V cycle of the zebrafish embryonic atrium (volume-driven), with the wall pulled toward the tracked wall.**

The example `../FE_one_fiber` (same input, mesh, material, activation and volume constraint; its README describes the
model) with one addition, switched on by `DISP_CONSTRAIN = True` in `params.py`: every node of the endocardium and the
epicardium is pulled toward the position of its material point in the tracked frame (base flattened, see Method),

    traction  t = DISP_WEIGHT (u_img - u)   on the endocardium and epicardium      (energy  DISP_WEIGHT/2 ∫ |u - u_img|² dS)

with `u_img` the tracked displacement of the node and `DISP_WEIGHT = 300` kPa/mm (1 um off the tracked wall -> 0.3 kPa),
and the prescribed cavity volumes are those of the base-flattened tracked surfaces. Without the switch
(`DISP_CONSTRAIN = False` or absent) the run is the volume_constrain FE of `../FE_one_fiber`.

The traction is a force the real heart does not have ("image force"). The weight sets the balance: a larger weight brings
the simulated wall closer to the tracking, but the image force grows and, through the volume constraint, raises the
cavity pressure.

## This example
```
FE_one_fiber_disp/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  the settings of ../FE_one_fiber + DISP_CONSTRAIN = True, DISP_WEIGHT = 300.0 (last section)
├── input/     raw data: t0.stl ... t51.stl (the 52 tracked frames), base_plane.json (as ../FE_one_fiber)
└── output/    mesh/ (+ disp_targets.npz), the results of the cycle, disp_compare.mp4
```
Code: `../../../heartfem_mu/disp_constrain.py` (steps targets, compare), the image term in `fe.py` (`VolumeProblem`,
`image_w`), `volume_constrain.py`, `plots.py` (`disp_compare_movie`).

The git repository holds the small files of `output/` (figures, movies, tables, json, logs). The large ones (`mesh/`
with `disp_targets.npz`, every `.vtu` and `.npz` file) are in the release archive
`heartfem_mu_v1.3.0_outputs_volume_constrain_FE_one_fiber_disp.tar.gz` (see Examples in `../../../README.md`);
`python main.py` recomputes all of them from `input/`.

## Run
```bash
python main.py                  # every step, input/ -> output/
```

| step | does | code (`heartfem_mu/`) | runs in |
|---|---|---|---|
| `preprocess` | raw surfaces -> tetrahedral mesh, local frame, cut surfaces | `preprocess.main` | `CONDA_ENV`, `MESH_THREADS` threads |
| `targets` | base-flattened tracking: displacement u_img of every endocardial / epicardial node, cavity volumes | `disp_constrain.targets` | `CONDA_ENV` |
| `solve` | volume-driven FE cycle with the image term | `volume_constrain.fe_cycle` | `CONDA_ENV_FENICS`, MPI, `MPI_RANKS` ranks |
| `compare` | wall Dice per frame; disp_compare.mp4: simulation over the tracking on four sections | `disp_constrain.compare`, `plots.disp_compare_movie` | `CONDA_ENV`, `MPI_RANKS` processes |
| `plot` | pv_loop.png/.pdf, pv_cycle.mp4 | `plots.pv_figures` | `CONDA_ENV` |

## Method
- **Tracked wall** (step `targets`). The cut surfaces of the preprocess (`output/mesh/surface_cut/t*.stl`: endocardium and
  epicardium cut at the base plane of t12 and followed through the cycle by the tracking; all frames share the
  triangles), with the base flattened, because the model holds its base in the base plane (u . n = 0) while the rim of the
  tracking moves out of it (here by up to 23.9 um):
  1. rigid alignment of every frame onto t12 (least-squares rotation + translation of all cut-surface vertices; the model
     has no rigid motion; here up to 4.3 deg);
  2. the rim moved back into the base plane along its normal: every vertex gets the normal correction h of the nearest rim
     vertex of its surface, weighted by phi(d) = (1 + cos(pi d / L)) / 2 for d < L, 0 below (d: depth below the base
     plane in t12; L = `FLAT_DEPTH_UM` = 20 um in `disp_constrain.py`); in-plane positions unchanged.

         x~_k = y_k - phi(d) h_k n,      y_k = R_k x_k + c_k
- **Targets and volumes** (step `targets`). Each endocardial / epicardial node of the mesh (the reference frame t12) is
  attached to the closest point of the same flattened surface of t12, as barycentric coordinates in one triangle, and the
  point is followed through the flattened frames: u_img = x~_k - x~_t12 (0 at t12). The cavity volume of every frame is
  that of the flattened endocardium (closed by cones to the base-plane origin, as for `../FE_one_fiber`): the flattening
  removes the part of the filling that bulges up through the base plane (here V/V_t12 at most 1.294 instead of 1.402).
- **Solve**. The residual of `../FE_one_fiber` + DISP_WEIGHT ∫ (u - u_img) · v dS over the endocardium and epicardium
  (u_img a P1 field, changed along the same continuation as the volume and the activation time), with the flattened
  volumes. Everything else is unchanged: Newton with the exact Jacobian, MUMPS, the same tolerances.
- **Compare** (step `compare`), every frame: the wall Dice on 1 um voxels below the base plane, no further alignment (the
  bases agree by construction): tracked wall = inside the epicardium and outside the endocardium of the flattened
  surfaces, each closed by a fan over its rim; simulated wall = inside the closed boundary of the deformed mesh. Then the
  movie.

## Output (`output/`)
The files of `../FE_one_fiber` (same names and contents; `material` in `pv_results.npz` also holds
`disp_constrain.weight_kPa_per_mm`; `V_ratio` and `V_nL` are the flattened volumes), and:
- **`disp_compare.mp4`**: one cycle from the maximum volume, four sections side by side: three through the long axis and
  the base centre at 0, 60 and 120 deg around it (dotted: the base plane) and one short-axis section 50 um below the base
  plane. The simulated wall coloured by the fiber stress (-10..10 kPa, fixed for every weight) with a black outline, over
  the flattened tracked wall of the frame (grey band, solid / dashed epi / endo outline); title: frame, cavity pressure
  and wall Dice.
- `mesh/disp_targets.npz` (step targets, input of the step solve): `u_img` (52, nodes, 3) in mm in the `mesh/t0.vtu`
  node order (0 on interior and base nodes), `data_nodes`, `V_ratio` (flattened cavity volume / t12), the removed rigid
  motion and the rim height before the flattening per frame.

## Result
All 52 frames (`output/disp_compare.mp4`, the log of the step compare, `summary.csv`, `pv_loop.png`), next to the same
model without the constraint: `../FE_one_fiber` (volumes of the cut surfaces, not flattened) and `DISP_WEIGHT = 0`
(the flattened volumes of this example); every Dice against the flattened tracking as above:

| quantity | `../FE_one_fiber` | `DISP_WEIGHT = 0` | `DISP_WEIGHT = 300` kPa/mm (this example) |
|---|---|---|---|
| wall Dice, mean over the cycle (lowest, frame) | 0.587 (0.362, t49) | 0.618 (0.432, t49) | 0.856 (0.726, t1) |
| peak cavity pressure | 1.250 kPa (t51) | 0.972 kPa (t50) | 1.710 kPa (t49) |
| P at maximum volume (t43) | 0.260 kPa | 0.148 kPa | 0.984 kPa |
| image force, RMS over endo + epi and the cycle | - | 0 | 1.13 kPa (66 % of the peak pressure) |

The wall follows the tracking closely in every frame (grey gaps nearly gone in the movie); the price is the image force,
which carries a large part of the load: the cavity pressure rises most in the passive filling (P at maximum volume x 6.6)
and the stress map changes (Limitations). Solve: 9.7 min on 8 MPI ranks (204 Newton iterations, 52 frames).

## Verification
- Targets, volumes, FE cycle and wall Dice reproduce an independent development implementation of the same method to
  round-off: u_img within 1e-13 um, volumes identical, P within 3e-14 kPa, displacements identical, wall Dice identical
  to 5 digits.
- `DISP_WEIGHT = 0` equals the volume_constrain FE with the flattened volumes and no constraint term (P within 3e-15 kPa,
  displacements and stresses identical). `DISP_CONSTRAIN = False` (or absent) runs `../FE_one_fiber` unchanged: the
  code path without the switch is not changed.
- The other modes refuse `DISP_CONSTRAIN = True` (PINN, windkessel, invert_para, unloading); every other example runs
  as before (same steps, same settings).

## Limitations
- The image force has no physical source and at this weight it dominates: its RMS over endo + epi and the cycle is 66 %
  of the peak cavity pressure, P drops below zero after the peak (-0.49 kPa at t4), and the fibre stress at the maximum
  volume (passive) is about 8 times that of the same model without the force. The stresses of this run show what the
  tracking demands, not the heart's stress; compare with `DISP_WEIGHT = 0` (or a small weight). Larger weights (1000) did
  not converge in the development runs.
- The tracked correspondence of points along the surface (tangential) is that of the tracking, not necessarily the
  material motion; the targets use the full displacement vector.
- The tracked wall volume changes over the cycle (flattened: 0.84-1.03 x t12, lowest at the end of contraction) while the
  model wall is incompressible: the wall Dice cannot reach 1.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`).
