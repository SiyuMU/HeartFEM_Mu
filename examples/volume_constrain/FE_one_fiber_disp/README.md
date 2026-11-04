# volume_constrain · FE · one fiber family · displacement constraint

**FE P-V cycle of the zebrafish embryonic atrium (volume-driven), with the wall pulled toward the tracked wall.**

The example `../FE_one_fiber` (same input, mesh, material, activation and volume constraint; its README describes the
model) with one addition, switched on by `DISP_CONSTRAIN = True` in `params.py`: every node of the endocardium and the
epicardium is pulled toward the position of its material point in the tracked frame,

    traction  t = DISP_WEIGHT (u_img - u)   on the endocardium and epicardium      (energy  DISP_WEIGHT/2 ∫ |u - u_img|² dS)

with `u_img` the tracked displacement of the node and `DISP_WEIGHT = 30` kPa/mm (1 um off the tracked wall -> 0.03 kPa).
Without the switch (`DISP_CONSTRAIN = False` or absent) the run is the volume_constrain FE of `../FE_one_fiber`.

The traction is a force the real heart does not have ("image force"). The weight sets the balance: a larger weight brings
the simulated wall closer to the tracking, but the image force grows and, through the volume constraint, raises the
cavity pressure. Every run therefore reports the image force next to the cavity pressure.

## This example
```
FE_one_fiber_disp/
├── README.md  this file
├── main.py    runs the pipeline: `python main.py`
├── params.py  the settings of ../FE_one_fiber + DISP_CONSTRAIN = True, DISP_WEIGHT = 30.0 (last section)
├── input/     raw data: t0.stl ... t51.stl (the 52 tracked frames), base_plane.json (as ../FE_one_fiber)
└── output/    mesh/, the results of the cycle, disp_constrain/
```
Code: `../../../heartfem_mu/disp_constrain.py` (steps targets, compare), the image term in `fe.py` (`VolumeProblem`,
`image_w`), `volume_constrain.py`, `plots.py` (`disp_constrain_figure`).

## Run
```bash
python main.py                  # every step, input/ -> output/
```

| step | does | code (`heartfem_mu/`) | runs in |
|---|---|---|---|
| `preprocess` | raw surfaces -> tetrahedral mesh, local frame, cut surfaces | `preprocess.main` | `CONDA_ENV`, `MESH_THREADS` threads |
| `targets` | tracked displacement u_img of every endocardial / epicardial node | `disp_constrain.targets` | `CONDA_ENV` |
| `solve` | volume-driven FE cycle with the image term | `volume_constrain.fe_cycle` | `CONDA_ENV_FENICS`, MPI, `MPI_RANKS` ranks |
| `compare` | simulation vs tracking per frame: Dice, surface distance, misfit, image force | `disp_constrain.compare` | `CONDA_ENV`, `MPI_RANKS` processes |
| `plot` | pv_loop.png/.pdf, pv_cycle.mp4 | `plots.pv_figures` | `CONDA_ENV` |

## Method
- **Targets** (step `targets`). The tracked frames share one triangle order. Each endocardial / epicardial node of the
  mesh (the reference frame t12) is attached to the closest point of the same surface of the tracked t12, as
  barycentric coordinates in one triangle, and the point is followed through the cycle. The rigid motion of the tracking
  is removed per frame (least-squares rotation + translation of all tracked vertices onto t12; the model has no rigid
  motion). u_img = position in frame k - position in t12 (0 at t12).
- **Solve**. The residual of `../FE_one_fiber` + DISP_WEIGHT ∫ (u - u_img) · v dS over the endocardium and epicardium
  (u_img a P1 field, changed along the same continuation as the volume and the activation time). Everything else is
  unchanged: Newton with the exact Jacobian, MUMPS, the same tolerances.
- **Compare** (step `compare`), every frame, on the part of the wall at least `BASE_EXCLUDE_UM` = 10 um below the base
  plane: wall Dice on 1 um voxels (tracked wall: inside the epicardium and outside the endocardium of the closed input
  surfaces, rigidly aligned as above; simulated wall: inside the closed boundary of the deformed mesh); surface
  distance of the simulated endo / epi nodes to the tracked surface of the same side; misfit to the targets and image
  force DISP_WEIGHT |u_img - u|, area-weighted rms over the endo + epi nodes.

## Output (`output/`), in addition to those of `../FE_one_fiber`
- **`disp_constrain/targets.npz`**: `u_img` (52, nodes, 3) in mm in the `mesh/t0.vtu` node order (0 on interior and
  base nodes), `data_nodes`, the removed rigid motion per frame.
- **`disp_constrain/compare.csv`**: per frame: t, P, wall Dice, surface distance (median, 95 %), misfit (vector,
  normal component), image force (rms).
- **`disp_constrain/compare.json`**: the cycle summary; **`disp_constrain/disp_constrain.png`**: the four curves.
- `pv_results.npz` and the other files as in `../FE_one_fiber`; `material` holds `disp_constrain.weight_kPa_per_mm`.

## Result
All 52 frames (`output/disp_constrain/compare.json`, `compare.csv`, `disp_constrain.png`, `pv_loop.png`), next to the
same model without the constraint (`../FE_one_fiber`, evaluated with the same comparison):

| quantity | without constraint (`../FE_one_fiber`) | DISP_WEIGHT = 30 kPa/mm (this example) |
|---|---|---|
| wall Dice, mean over the cycle (lowest, frame) | 0.625 (0.418, t49) | 0.772 (0.556, t50) |
| surface distance to the tracking, mean of the frame medians | 3.18 um | 1.91 um |
| misfit to the targets, rms (vector / normal component) | 10.26 / 5.07 um | 6.65 / 3.49 um |
| image force, rms | 0 | 0.199 kPa (13 % of the peak pressure) |
| peak cavity pressure (t51) | 1.250 kPa | 1.566 kPa |
| P at maximum volume (t43) | 0.260 kPa | 0.520 kPa |

The wall follows the tracking more closely in every frame; the image force raises the cavity pressure, most in the
passive filling (P at maximum volume doubles). Solve: 8.6 min on 8 MPI ranks (194 Newton iterations, 52 frames).

## Verification
- `DISP_CONSTRAIN = False` (or absent) and `DISP_CONSTRAIN = True` with `DISP_WEIGHT = 0` reproduce `../FE_one_fiber`
  to round-off (P within 4e-15 kPa, u within 2e-14 um, stresses within 3e-8 kPa).
- With `DISP_CONSTRAIN = True` the same weight works with two fiber families (`../FE_two_fibers` settings, weight 30:
  wall Dice mean 0.790, image force 13 % of the peak pressure).
- The other modes refuse `DISP_CONSTRAIN = True` (PINN, windkessel, invert_para, unloading); every other example runs
  as before (same steps, same settings).

## Limitations
- The image force has no physical source: the stresses of this run contain its effect. Use a small weight, or compare
  with the run without the constraint (`../FE_one_fiber`).
- The tracked correspondence of points along the surface (tangential) is that of the tracking, not necessarily the
  material motion; the targets use the full displacement vector.
- The tracked wall volume changes over the cycle (0.90-1.14 x t12) while the model wall is incompressible: the
  wall Dice cannot reach 1.

## License
MIT License, Copyright (c) 2026 Siyu Mu (`../../../LICENSE`).
