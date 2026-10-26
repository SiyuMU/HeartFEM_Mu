# Third-party notices

HeartFEM_Mu (MIT License, Copyright (c) 2026 Siyu Mu, see `LICENSE`) contains or derives from the following works.

## heartFEM

W. X. Chan (Lee lab), `heartFEM` (`lcleeHeart`, `ngspice_py`). Used in HeartFEM_Mu:
- `heartfem_mu/data/generateLVtable.py`: heartFEM's original source file, unchanged;
- `heartfem_mu/data/atrium_wk.cir`: the circuit, built from heartFEM's `LV.cir` (P-V-t table source, clock, volume
  integrators) and `3dpf_new.cir` (valves, vein sub-circuit, compliances, link resistances);
- `heartfem_mu/fe.py`, `heartfem_mu/pinn.py`: the passive (Fung) and active (`simpleActiveMaterial`) laws, the
  volume-controlled formulation and the epicardial spring, re-implemented after heartFEM's `lcleeHeart` forms.

```
MIT License
Copyright (c) 2019 W. X. Chan
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## three.js

`heartfem_mu/data/assets/three.core.min.js`, `three.module.min.js`, `OrbitControls.js` (the browser app of
`heartfem_mu/landmarks.py`): three.js, MIT License, Copyright © 2010-2026 three.js authors; the full text is in
`heartfem_mu/data/assets/LICENSE.txt`.

## Dependencies

The conda environments (`environments/`) install, among others, legacy FEniCS 2019.1 (LGPL-3.0), PETSc and MUMPS,
MPICH, ngspice, gmsh (GPL-2.0-or-later, called as an external program), PyTorch, VTK, NumPy, SciPy, matplotlib, trimesh,
meshio and the ffmpeg binary of imageio-ffmpeg. They are not part of this repository; each keeps its own license.
