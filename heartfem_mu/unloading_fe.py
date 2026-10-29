"""FE solver of the task unloading (dolfin, MPI; the method is described in unloading.py): the forward problem
fe.PressureProblem on a movable reference (the check), the inverse problem on the ED mesh, the loop over frame, base
origin and C, and the result files."""
import json
import operator
import shutil
import tempfile
import time
from functools import reduce
from types import SimpleNamespace

import dolfin
import numpy as np
import ufl
from dolfin import (Constant, DirichletBC, FacetNormal, FiniteElement, Function, FunctionSpace, Identity, MixedElement,
                    SpatialCoordinate, TestFunction, TrialFunction, VectorElement, VectorFunctionSpace, as_backend_type,
                    as_vector, assemble, cross, derivative, det, dx, exp, grad, inner, inv, split, vertex_to_dof_map)
from scipy.spatial import cKDTree
from ufl import cofac

from . import fe, model, unloading
from .fe import COMM, RANK, log

TIE_FCP = {"representation": "quadrature"}  # uflacs (FFC 2019.1) generates uncompilable code for the vertex rule


def allmax(x):
    return COMM.allreduce(float(np.max(x)) if np.size(x) else 0.0, op=fe.pyMPI.MAX)


class Frames:
    """Local frame of a geometry on all ranks (computed by the helper of rank 0)."""

    def __init__(self, cfg, work):
        self.client = unloading.FrameClient(cfg, work) if RANK == 0 else None

    def __call__(self, points_um, topo):
        out = self.client(points_um, topo) if RANK == 0 else None
        return COMM.bcast(out, root=0)

    def calls(self):
        return COMM.bcast(self.client.calls if RANK == 0 else None, root=0)

    def close(self):
        if RANK == 0:
            self.client.close()


class Reference:
    """fe.PressureProblem on a movable reference with the fixed topology `topo` (points um): a reference is set by
    moving the mesh vertices and refilling the fiber fields."""

    def __init__(self, cfg, topo, X, lrc, mesh_h5):
        self.cfg, self.topo = cfg, topo
        g = self.geometry(X, lrc)
        self.R = R = fe.base_frame(g)
        if RANK == 0:
            fe.write_mesh(g, R, mesh_h5)
        COMM.barrier()
        self.pb = pb = fe.PressureProblem(cfg, COMM, g, R, mesh_h5)
        dist, self.vidx = cKDTree((g["points"] - g["base_origin"]) @ R.T).query(pb.mesh.coordinates())
        assert allmax(dist) < 1e-9
        self.zero = pb.w.vector().copy()
        self.zero.zero()

    def geometry(self, X, lrc):
        """The input dict of fe.PressureProblem (fibers from the transmural depth, as for invert_para) on X (um)."""
        t, points = self.topo, X * self.cfg.MM_PER_UNIT
        return {"points": points, "cells": t.cells, "facets": t.facets, "facet_labels": t.facet_labels,
                "base_origin": points[t.base_nodes].mean(0), "base_normal": -t.n_out,
                **{k: np.asarray(lrc[k], float) for k in ("eC", "eL", "eR")},
                "depth": model.transmural_depth(points, t.cells, t.endo, t.epi)}

    def fiber_values(self, g):
        """eC, eL, eR in the base frame and per family (f_i, n_i = f_i x eR), as in fe.PressureProblem."""
        eC, eL, eR = (g[k] @ self.R.T for k in ("eC", "eL", "eR"))
        families = []
        for a, _ in model.fiber_angles(self.cfg, g["depth"]):
            a = a[:, None]
            f = np.cos(a) * eC + np.sin(a) * eL
            f /= np.linalg.norm(f, axis=1, keepdims=True)
            families.append((f, np.cross(f, eR)))
        return eC, eL, eR, families

    def set_reference(self, X, lrc):
        g = self.geometry(X, lrc)
        pb = self.pb
        Xb = (g["points"] - g["base_origin"]) @ self.R.T
        pb.mesh.coordinates()[:] = Xb[self.vidx]
        eC, eL, eR, families = self.fiber_values(g)
        pb.eC, pb.eL, pb.eR = eC, eL, eR
        pb.set_cell_values(pb.s0, eR)
        for (f, n), ff, nn in zip(families, pb.fibers, pb.normals):
            pb.set_cell_values(ff, f)
            pb.set_cell_values(nn, n)
        pb.g = g
        if RANK == 0:
            pb.tree = cKDTree(Xb)  # vertex matching of nodal_u_um

    def check(self, C, X, lrc, x_ed, steps):
        """Load X (frame lrc) from the unloaded state to P_ED in `steps` equal pressure steps (each a march with step
        halving; no warm start, which could land on another equilibrium): (residual per node um, u um, ED volume nL),
        or None if the loading fails."""
        pb = self.pb
        self.set_reference(X, lrc)
        pb.set_vector(self.zero)
        prev = np.array([0.0, 0.0, C, 0.0])
        for k in range(1, steps + 1):
            state = np.array([self.cfg.P_ED_KPA * k / steps, 0.0, C, 0.0])
            ok, _, _ = pb.march(prev, state)
            if not ok:
                log(f"   check: loading failed at P = {state[0]:.4f} kPa")
                return None
            prev = state
        u = COMM.bcast(pb.nodal_u_um(), root=0)
        return np.linalg.norm(x_ed - X - u, axis=1), u, pb.volume_nL()


class Inverse(fe.Newton):
    """The inverse problem on the ED mesh x_ed of `ref` (a Reference built on x_ed: same base frame, multiplier scales
    and residual scale R_ref)."""

    def __init__(self, cfg, topo, x_ed, lrc, ref, mesh_h5):
        self.cfg, self.topo, self.x_ed = cfg, topo, x_ed
        g = ref.geometry(x_ed, lrc)
        self.ref, self.R = ref, ref.R
        if RANK == 0:
            fe.write_mesh(g, self.R, mesh_h5)
        COMM.barrier()
        mesh, marker = fe.read_mesh(COMM, mesh_h5)
        self.mesh = mesh
        self.gcell = np.array(mesh.topology().global_indices(3))
        L = float(np.ptp(g["points"], 0).max())
        DG0v = VectorFunctionSpace(mesh, "DG", 0)
        families = model.fiber_shares(cfg)
        self.s0 = s0 = Function(DG0v)
        self.fibers = [Function(DG0v) for _ in families]
        self.normals = [Function(DG0v) for _ in families]
        weights = [w_ for _, w_ in families]

        def mix(terms):
            """sum_i w_i terms_i over the fiber families (one family: the term itself, as fe.PressureProblem)."""
            if len(terms) == 1:
                return terms[0]
            return reduce(operator.add, [Constant(wi) * t for wi, t in zip(weights, terms)])
        ds = dolfin.ds(domain=mesh, subdomain_data=marker)
        n, x = FacetNormal(mesh), SpatialCoordinate(mesh)
        self.Pc, self.Cc = Constant(0.0), Constant(0.1)
        self.ubar = Constant((0.0, 0.0, 0.0))  # mean of U over the base nodes: X - origin(X) = x - U + ubar
        cell = mesh.ufl_cell()
        self.W = W = FunctionSpace(mesh, MixedElement([VectorElement("CG", cell, 2), FiniteElement("CG", cell, 1),
                                                       MixedElement([FiniteElement("Real", cell, 0)] * 5),
                                                       VectorElement("CG", cell, 1)]))
        self.bcs = [DirichletBC(W.sub(0).sub(2), Constant(0.0), marker, fe.BASE_ID),
                    DirichletBC(W.sub(3).sub(2), Constant(0.0), marker, fe.BASE_ID)]
        self.w = w = Function(W)
        wt, dw = TestFunction(W), TrialFunction(W)
        u, p, c, U = split(w)
        z = split(wt)[3]
        Uf = Function(VectorFunctionSpace(mesh, "CG", 1))  # stands for U in the forward energy (not a variation)
        I = Identity(3)
        A = I - grad(Uf)
        dA = det(A)
        X = x - Uf + self.ubar
        F = I + grad(u) * inv(A)
        J = det(F)
        E = 0.5 * (F.T * F - I)
        Cs = [Constant(v) for v in (cfg.C_FF, cfg.C_SS, cfg.C_NN, cfg.C_NS, cfg.C_FS, cfg.C_FN)]

        def fung_exp(f0, n0):
            """e^Q - 1 in the frame (f0, s0, n0) of one fiber family (as fe.PressureProblem)."""
            Eff, Ess, Enn = inner(f0, E * f0), inner(s0, E * s0), inner(n0, E * n0)
            Efs, Efn, Ens = inner(f0, E * s0), inner(f0, E * n0), inner(n0, E * s0)
            Q = Cs[0] * Eff ** 2 + Cs[1] * Ess ** 2 + Cs[2] * Enn ** 2 + Cs[3] * Ens ** 2 + Cs[4] * Efs ** 2 + Cs[5] * Efn ** 2
            return exp(Q) - 1.0
        W_fung = mix([fung_exp(f, nn) for f, nn in zip(self.fibers, self.normals)])
        dsendo = ds(fe.ENDO_ID)
        V_u = -Constant(1.0 / 3.0) * inner(u + X, cofac(F) * cofac(A) * n)
        h = (assemble(Constant(1.0) * dx(mesh)) / mesh.num_entities_global(3)) ** (1 / 3)
        sc = Constant(2 * 0.1 * cfg.C_FF / h ** 2)  # as fe.PressureProblem (ED mesh)
        energy = (self.Cc * W_fung - p * (J - 1.0)) * dA * dx - self.Pc * V_u * dsendo \
            + inner(as_vector([c[0], c[1], 0.0]) * sc, u) * dA * dx \
            + inner(as_vector([c[3], c[4], c[2]]) * sc / L, cross(X, u)) * dA * dx
        self.R_form = ufl.replace(derivative(energy, w, wt), {Uf: U})
        self.Jac = derivative(self.R_form, w, dw)
        tie = Constant(float(sc)) * inner(U - u, z) * dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
        self.T = assemble(derivative(tie, w, dw), form_compiler_parameters=TIE_FCP)  # linear: assembled once
        self.vol_form = ufl.replace(V_u * dsendo, {Uf: U})
        self.A = self.ksp = None
        self.R_ref = ref.pb.R_ref
        self.zero = w.vector().copy()
        self.zero.zero()
        V1 = VectorFunctionSpace(mesh, "CG", 1)
        self.v1 = Function(V1)
        self.v2d = vertex_to_dof_map(V1).reshape(-1, 3)
        self.vown = (self.v2d < self.v1.vector().local_size()).all(1)
        self.tree = cKDTree((g["points"] - g["base_origin"]) @ self.R.T) if RANK == 0 else None

    # ---- linear algebra (fe.Newton with the tie rows added)
    def residual(self):
        b = assemble(self.R_form)
        b.axpy(1.0, self.T * self.w.vector())
        for bc in self.bcs:
            bc.apply(b)
        return b

    def newton(self, tol, max_iter=25):
        from petsc4py import PETSc
        b = self.residual()
        res, it = b.norm("l2"), 0
        dw = self.w.vector().copy()
        while res > tol and it < max_iter:
            it += 1
            if self.A is None:
                self.A = assemble(self.Jac)
            else:
                assemble(self.Jac, tensor=self.A)
            self.A.axpy(1.0, self.T, True)
            for bc in self.bcs:
                bc.apply(self.A)
            if self.ksp is None:
                self.ksp = PETSc.KSP().create(self.mesh.mpi_comm())
                self.ksp.setType("preonly")
                self.ksp.getPC().setType("lu")
                self.ksp.getPC().setFactorSolverType("mumps")
                self.ksp.setOperators(as_backend_type(self.A).mat())
            b *= -1.0
            self.ksp.solve(as_backend_type(b).vec(), as_backend_type(dw).vec())
            saved, step, prev = self.w.vector().copy(), 1.0, res
            while True:  # halve the step while the residual grows
                self.w.vector().axpy(step, dw)
                b = self.residual()
                res = b.norm("l2")
                if not (res > 2.0 * prev or np.isnan(res)):
                    break
                self.set_vector(saved)
                step *= 0.5
                if step < 1e-3:
                    return False, it, res
        return res <= tol, it, res

    # ---- fields
    def set_cell_values(self, f, values):
        n = f.vector().local_size()
        V = f.function_space()
        dofs = np.array([V.dofmap().cell_dofs(c) for c in range(self.mesh.num_cells())]).reshape(self.mesh.num_cells(), -1)
        vals = values[self.gcell].reshape(len(dofs), -1)
        own = dofs < n
        arr = np.zeros(n)
        arr[dofs[own]] = vals[own]
        f.vector().set_local(arr)
        f.vector().apply("insert")

    def set_frame(self, X, lrc):
        """Fiber fields of the reference X (um) with the local frame lrc, and the base origin of X."""
        g = self.ref.geometry(X, lrc)
        _, _, eR, families = self.ref.fiber_values(g)
        self.set_cell_values(self.s0, eR)
        for (f, nn), ff, fn in zip(families, self.fibers, self.normals):
            self.set_cell_values(ff, f)
            self.set_cell_values(fn, nn)
        Ub = ((self.x_ed - X) * self.cfg.MM_PER_UNIT) @ self.R.T
        ubar = Ub[self.topo.base_nodes].mean(0)
        self.ubar.assign(Constant((float(ubar[0]), float(ubar[1]), 0.0)))

    def vertex_um(self, k):
        """Vertex values (um, original frame, all ranks) of the sub-function k (0: u, 3: U)."""
        self.v1.interpolate(self.w.sub(k))
        parts = COMM.gather((self.mesh.coordinates()[self.vown], self.v1.vector().get_local()[self.v2d[self.vown]]),
                            root=0)
        out = None
        if RANK == 0:
            out = np.full((len(self.x_ed), 3), np.nan)
            for xx, a in parts:
                dist, idx = self.tree.query(xx)
                assert dist.max() < 1e-9
                out[idx] = a
            assert not np.isnan(out).any()
            out = (out @ self.R) / self.cfg.MM_PER_UNIT
        return COMM.bcast(out, root=0)

    def volume_nL(self):
        return assemble(self.vol_form) * 1e3

    # ---- solves
    def set_state(self, s):
        self.Pc.assign(float(s[0]))
        self.Cc.assign(float(s[1]))

    def march(self, start, target, h=1.0):
        """Continuation in s = (P, C) from the converged state `start` to `target`, halving the step on failure."""
        start, target = np.asarray(start, float), np.asarray(target, float)
        s, total = 0.0, 0
        while True:
            s_new = 1.0 if 1.0 - (s + h) < 0.2 * h else s + h
            self.set_state(start + s_new * (target - start))
            saved = self.w.vector().copy()
            final = s_new == 1.0
            ok, its, res = self.newton((1e-9 if final else 1e-6) * self.R_ref)
            total += its
            if ok:
                if final:
                    return True, total, res
                s, h = s_new, (min(1.0, 1.5 * h) if its <= 5 else h)
            else:
                self.set_vector(saved)
                h *= 0.5
                if h < 1.0 / 1024:
                    return False, total, res

    def solve(self, C, warm=None):
        """(solution vector, Newton iterations) at (P_ED, C), or None. warm = (vector, C_w): Newton at the target, then
        continuation in C from C_w; otherwise (or on failure) continuation in P from the unloaded state."""
        target = np.array([self.cfg.P_ED_KPA, C])
        its, ok = 0, False
        if warm is not None:
            vec, C_w = warm
            self.set_vector(vec)
            self.set_state(target)
            ok, n, _ = self.newton(1e-9 * self.R_ref)
            its += n
            if not ok and C_w != C:
                self.set_vector(vec)
                ok, n, _ = self.march(np.array([self.cfg.P_ED_KPA, C_w]), target)
                its += n
        for h in (() if ok else (1.0, 0.125)):
            self.set_vector(self.zero)
            ok, n, _ = self.march(np.array([0.0, C]), target, h=h)
            its += n
            if ok:
                break
        return (self.w.vector().copy(), its) if ok else None

    def reference(self):
        """(X um, max |U - u| um at the vertices) of the current solution; raises if a cell of X is inverted."""
        U, u = self.vertex_um(3), self.vertex_um(0)
        X = self.x_ed - U
        if (np.sign(unloading.signed_volumes(X, self.topo.cells)) != self.topo.sign0).any():
            raise RuntimeError("inverted cells in the unloaded geometry")
        return X, float(np.abs(U - u).max())


# ==================== the step ====================
def search(cfg, inv, frames, x_ed, V_target):
    """The loop over frame, base origin and C (unloading.py). Returns (C, X, frame of X, solution vector, history)."""
    topo, X, z, slope, warm, prev, hist = inv.topo, x_ed.copy(), float(np.log(cfg.C_START_KPA)), cfg.SECANT_SLOPE0, \
        None, None, []
    for k in range(cfg.MAX_ITERATIONS):
        C = float(np.exp(z))
        t0 = time.time()
        lrc, diag = frames(X, topo)
        inv.set_frame(X, lrc)
        out = inv.solve(C, warm)
        if out is None:
            raise RuntimeError(f"the inverse problem did not converge at C = {C:.6g} kPa (iteration {k})")
        vec, its = out
        warm = (vec, C)
        Xn, tie = inv.reference()
        dX = float(np.linalg.norm(Xn - X, axis=1).max())
        V0 = unloading.cavity_volume_nL(Xn, topo)
        f = float(np.log(V0 / V_target))
        hist.append({"iteration": k, "C_kPa": C, "dX_max_um": dX, "V0_nL": V0, "ln_V0_over_target": f, "newton": its,
                     "tie_max_um": tie, "seconds": round(time.time() - t0, 1), "frame": diag})
        log(f"iteration {k:2d}: C {C:.8f} kPa, max |dX| {dX:.3e} um, V0 {V0:.6f} nL ({f:+.2e}), Newton {its} "
            f"({hist[-1]['seconds']} s)")
        X = Xn
        if dX < cfg.TOL_X_UM and abs(f) < cfg.V0_TOL_REL:
            return C, X, lrc, vec, hist
        if prev is not None and abs(z - prev[0]) > 1e-12:
            s_ = (f - prev[1]) / (z - prev[0])
            slope = s_ if 0.02 <= s_ <= 2 else slope  # d ln V0 / d ln C > 0; a secant outside this range is not used
        prev = (z, f)
        step = float(np.clip(-f / slope, -cfg.MAX_STEP_LN_C, cfg.MAX_STEP_LN_C))
        if abs(step) < cfg.TOL_LN_C:
            if dX < cfg.TOL_X_UM:
                return C, X, lrc, vec, hist
            step = 0.0
        z += step
    raise RuntimeError(f"no convergence in MAX_ITERATIONS = {cfg.MAX_ITERATIONS} iterations")


def unload(cfg):
    t_start = time.time()
    if cfg.K_SPRING != 0:
        raise ValueError("unloading supports K_SPRING = 0 only (no epicardial spring)")
    w = model.read_wall(SimpleNamespace(mesh=cfg.mesh, MM_PER_UNIT=1.0))  # the ED mesh (step preprocess) in input units
    x_ed = w["points"]
    topo = unloading.Topology(w)
    saved = json.loads((cfg.input / "base_plane.json").read_text())
    d_apex = topo.set_apex(np.asarray(saved["apex"], float), x_ed)
    V_target, V_ed = float(cfg.V0_NL), unloading.cavity_volume_nL(x_ed, topo)
    if not 0 < V_target < V_ed:
        raise ValueError(f"V0_NL = {V_target} nL must lie between 0 and the ED cavity volume of the mesh {V_ed:.6f} nL")
    log(f"unloading: ED mesh {len(x_ed)} nodes, {len(topo.cells)} tetrahedra; ED cavity {V_ed:.6f} nL; target V0 "
        f"{V_target:.6f} nL; P_ED {cfg.P_ED_KPA} kPa; apex {d_apex:.2e} um from the mesh; "
        f"fibers {'two families' if cfg.TWO_FIBERS else 'one family'}")
    cfg.output.mkdir(parents=True, exist_ok=True)
    work = COMM.bcast(tempfile.mkdtemp(prefix=".unloading-", dir=cfg.cache) if RANK == 0 else None, root=0)
    frames = Frames(cfg, work)
    try:
        lrc_ed, _ = frames(x_ed, topo)
        ref = Reference(cfg, topo, x_ed, lrc_ed, f"{work}/forward.h5")
        inv = Inverse(cfg, topo, x_ed, lrc_ed, ref, f"{work}/inverse.h5")
        C, X, _, vec, hist = search(cfg, inv, frames, x_ed, V_target)
        V0 = unloading.cavity_volume_nL(X, topo)
        lrc, frame_diag = frames(X, topo)  # the frame of the unloaded geometry: the check and the output fibers
        # sensitivity at the final frame: one inverse solve at C e^0.01
        z = float(np.log(C))
        out = inv.solve(float(np.exp(z + 0.01)), (vec, C))
        sens = None
        if out is not None:
            V1 = unloading.cavity_volume_nL(inv.reference()[0], topo)
            sens = 0.01 / float(np.log(V1 / V0))
        # check: the forward problem on X, loaded from zero
        t0 = time.time()
        chk = ref.check(C, X, lrc, x_ed, cfg.CHECK_LOAD_STEPS)
        r = None if chk is None else float(chk[0].max())
        reaches = r is not None and r < unloading.CHECK_TOL_UM
        log(f"check: " + ("loading from the unloaded wall failed" if chk is None else
                          f"max |x_ED - X - u| {r:.3e} um, ED cavity {chk[2]:.6f} nL") + f" ({time.time() - t0:.1f} s)")
        u = chk[1] if chk is not None else x_ed - X
        result = {
            "C_kPa": C, "V0_nL": V0, "V0_target_nL": V_target, "V0_rel_error": V0 / V_target - 1,
            "P_ED_kPa": cfg.P_ED_KPA, "V_ed_mesh_nL": V_ed, "reaches_ed": reaches, "check_max_residual_um": r,
            "check_V_ed_nL": None if chk is None else chk[2], "dlnC_dlnV0": sens,
            "wall_volume_ed_nL": unloading.wall_volume_nL(x_ed, topo.cells),
            "wall_volume_unloaded_nL": unloading.wall_volume_nL(X, topo.cells),
            "displacement_to_ed_um": {"max": float(np.linalg.norm(x_ed - X, axis=1).max()),
                                      "rms": float(np.sqrt(np.mean(np.sum((x_ed - X) ** 2, 1))))},
            "fibers": "two families" if cfg.TWO_FIBERS else "one family", "frame_unloaded": frame_diag,
            "iterations": len(hist), "newton_steps": sum(h["newton"] for h in hist), "frame_calls": frames.calls(),
            "nodes": len(x_ed), "tetrahedra": len(topo.cells), "seconds": round(time.time() - t_start, 1),
            "history": hist}
        if RANK == 0:
            depth = model.transmural_depth(X, topo.cells, topo.endo, topo.epi)
            cell = {k: lrc[k] for k in ("eC", "eL", "eR")}
            cell["depth"] = depth
            for i, (a, _) in enumerate(model.fiber_angles(cfg, depth), 1):
                f = np.cos(a)[:, None] * lrc["eC"] + np.sin(a)[:, None] * lrc["eL"]
                cell[f"fiber_{i}"] = f / np.linalg.norm(f, axis=1, keepdims=True)
            unloading.write_mesh(cfg.output / "unloaded.vtu", X, topo.cells, cell, {"u_to_ed_um": u})
            (cfg.output / "result.json").write_text(json.dumps(result, indent=1))
        log(f"unloading: C = {C:.6f} kPa, V0 {V0:.6f} nL (target {V_target:.6f}), d ln C / d ln V0 = "
            f"{sens if sens is None else round(sens, 2)}; the unloaded wall loaded to P_ED "
            + ("reproduces the ED wall" if reaches else "does NOT reproduce the ED wall (see unloading.py)")
            + f"; {result['seconds']} s")
    finally:
        frames.close()
        if RANK == 0:
            shutil.rmtree(work, ignore_errors=True)
