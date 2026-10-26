"""FE solver of the heart-chamber wall (legacy FEniCS 2019 dolfin, MPI; imported only in CONDA_ENV_FENICS).

Problem (lengths mm, stresses kPa). Unknowns: displacement u (P2), pressure p (P1), five rigid-body multipliers, and in
the volume-driven problem the cavity pressure P (one number).
    Passive energy: C (e^Q - 1) - p (J - 1), with Q = sum C_ij E_ij^2 in the fiber, sheet (eR) and normal frame; fiber =
                    eC rotated toward eL by the fiber angle (params: (endo, epi), linear in transmural depth).
    Active stress:  S_act = T_MAX Ct(t, ls) Ca0^2 / (Ca0^2 + ECa^2) f0 x f0 (heartFEM simpleActiveMaterial, trackphase
                    off), with ls = LR |F f0| (current sarcomere length), ECa = Ca0max / sqrt(exp(B (ls - l0)) - 1),
                    ls - l0 floored at 0.003, and the field
                        tr = m ls + b                                  (relaxation time, depends on the solution)
                        Ct = 0.5 (1 - cos(pi t / t0))                  for 0 <= t < t0
                        Ct = 0.5 (1 + cos(pi (t - t0) / tr))           for t0 <= t < t0 + tr,   0 afterwards.
    Two fiber families (params.TWO_FIBERS): family i has its own angle, frame (f_i, eR, f_i x eR), ls_i, tr_i and Ct_i;
                    W_pas = sum_i w_i C (e^Q_i - 1), S_act = sum_i w_i S_act,i (shares w_i = FIBER_WEIGHTS, sum 1).
                    sigma_ff = family 1, sigma_ff2 = family 2; Ct = sum_i w_i Ct_i.
    Boundaries:     base u.n = 0; no rigid translation in the base plane and no rotation; epicardial spring
                    K (u.N) N with K = K_SPRING (heartFEM; default 0: none).
Two problems:
    VolumeProblem    (volume_constrain, windkessel) the cavity volume is prescribed: P is the Lagrange multiplier of
                     V(u) = V_target, V(u) = -1/3 int_endo (X + u) . cof(F) N dS. A state is the pair (V, activation
                     time t); the model is quasi-static and has no history variable.
    PressureProblem  (invert_para) the cavity pressure is given and the volume is an output; C and T_MAX are assignable
                     constants (parameter sets are solved without recompiling); the activation level Ct of a frame is a
                     constant (0 at ED, the rise value at PS); a state is reached by continuation in s = (P, Ct, C, T_MAX).
"""
import operator
import time
from functools import reduce
from pathlib import Path

import numpy as np

from dolfin import *  # noqa: F401,F403
import dolfin
from mpi4py import MPI as pyMPI
from scipy.spatial import cKDTree
from ufl import cofac

from . import model

COMM = pyMPI.COMM_WORLD
RANK = COMM.rank
BASE_ID, ENDO_ID, EPI_ID = model.BASE_ID, model.ENDO_ID, model.EPI_ID
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["ghost_mode"] = "shared_facet"
set_log_level(LogLevel.WARNING)


def log(*args):
    if RANK == 0:
        print(*args, flush=True)


# ==================== 1. inputs (numpy) ====================
def geometry(cfg):
    """Mesh (mm), boundary facets (base / endo / epi), base plane, local frame; fibers (volume_constrain, windkessel) or
    the transmural depth (invert_para: fibers built by PressureProblem); frame volume ratios (volume_constrain)."""
    w = model.read_wall(cfg)
    points, cells = w["points"], w["cells"]
    labels = {tuple(f): ENDO_ID for f in np.sort(w["endo"], 1)}
    labels.update({tuple(f): EPI_ID for f in np.sort(w["epi"], 1)})
    labels.update({tuple(f): BASE_ID for f in np.sort(w["bnd"][w["on_base"]], 1)})
    g = {"points": points, "cells": cells, "facets": np.array(list(labels.keys())),
         "facet_labels": np.array(list(labels.values())), "base_origin": points[w["base_nodes"]].mean(0),
         "base_normal": -w["n_out"], **{k: w[k] for k in ("eC", "eL", "eR")}}
    depth = model.transmural_depth(points, cells, w["endo"], w["epi"])
    if cfg.TASK == "invert_para":
        g["depth"] = depth
    else:
        g.update(model.fiber_fields(cfg, w, depth))
    if cfg.TASK == "volume_constrain":
        vols, ref = model.frame_volumes(cfg), model.reference_frame(cfg)
        g.update(V_ratio=vols / vols[ref], reference_frame=ref)
    return g


def base_frame(g):
    """Rotation to the base frame: e3 = -base_normal, base plane z' = 0, wall at z' < 0."""
    e3 = -g["base_normal"] / np.linalg.norm(g["base_normal"])
    trial = np.eye(3)[np.argmin(np.abs(np.eye(3) @ e3))]
    e1 = trial - (trial @ e3) * e3
    e1 /= np.linalg.norm(e1)
    return np.array([e1, np.cross(e3, e1), e3])


def write_mesh(g, R, path):
    """Rank 0: dolfin mesh in the base frame with facet markers -> path (HDF5)."""
    Xp = (g["points"] - g["base_origin"]) @ R.T
    mesh = Mesh(pyMPI.COMM_SELF)
    editor = MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)
    editor.init_vertices(len(Xp))
    editor.init_cells(len(g["cells"]))
    for i, x in enumerate(Xp):
        editor.add_vertex(i, x)
    for i, c in enumerate(g["cells"]):
        editor.add_cell(i, c)
    editor.close()
    mesh.init(2, 0)
    index = {tuple(f): i for i, f in enumerate(np.sort(np.asarray(mesh.topology()(2, 0)(), dtype=np.int64).reshape(-1, 3), 1))}
    marker = MeshFunction("size_t", mesh, 2, 0)
    for f, label in zip(g["facets"], g["facet_labels"]):
        marker.array()[index[tuple(f)]] = label
    h5 = HDF5File(mesh.mpi_comm(), str(path), "w")
    h5.write(mesh, "mesh")
    h5.write(marker, "mesh/facets")
    h5.close()


def setup(cfg, mesh_h5):
    """Geometry on all ranks and the serial mesh file `mesh_h5` (temporary, read back distributed); returns (g, R)."""
    g = geometry(cfg) if RANK == 0 else None
    g = COMM.bcast(g, root=0)
    R = base_frame(g)
    if RANK == 0:
        Path(mesh_h5).parent.mkdir(parents=True, exist_ok=True)
        write_mesh(g, R, mesh_h5)
    COMM.barrier()
    return g, R


def read_mesh(comm, mesh_h5):
    mesh = Mesh(comm)
    h5 = HDF5File(comm, str(mesh_h5), "r")
    h5.read(mesh, "mesh", False)
    marker = MeshFunction("size_t", mesh, 2)
    h5.read(marker, "mesh/facets")
    h5.close()
    return mesh, marker


class Newton:
    """Newton (MUMPS LU, residual-halving line search) and the continuation of a problem with the forms R_form and Jac,
    the solution w, the boundary conditions bcs and the residual scale R_ref."""

    def set_vector(self, vec):
        self.w.vector().zero()
        self.w.vector().axpy(1.0, vec)

    def residual(self):
        b = assemble(self.R_form)
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


# ==================== 2. volume-driven problem (volume_constrain, windkessel) ====================
class VolumeProblem(Newton):
    def __init__(self, cfg, comm, g, R, mesh_h5):
        self.cfg = cfg
        L = float(np.ptp(g["points"], 0).max())
        mesh, marker = read_mesh(comm, mesh_h5)
        self.mesh = mesh
        self.gcell = np.array(mesh.topology().global_indices(3))  # local cell -> t0.vtu cell
        DG0v = VectorFunctionSpace(mesh, "DG", 0)
        s0 = self.cell_field(DG0v, g["s0"] @ R.T)
        self.fibers = [self.cell_field(DG0v, f @ R.T) for f in g["fibers"]]
        normals = [self.cell_field(DG0v, n @ R.T) for n in g["normals"]]
        self.f0 = self.fibers[0]
        weights = g["weights"]

        def mix(terms):
            """sum_i w_i terms_i over the fiber families (one family: the term itself, unchanged form)."""
            if len(terms) == 1:
                return terms[0]
            return reduce(operator.add, [Constant(float(wi)) * x for wi, x in zip(weights, terms)])
        ds = dolfin.ds(domain=mesh, subdomain_data=marker)
        N, X = FacetNormal(mesh), SpatialCoordinate(mesh)
        self.ta, self.Vt = Constant(0.0), Constant(0.0)  # activation time (ms), target cavity volume (mm^3)

        Ve = VectorElement("CG", mesh.ufl_cell(), 2)
        Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
        Re = FiniteElement("Real", mesh.ufl_cell(), 0)
        self.W = W = FunctionSpace(mesh, MixedElement([Ve, Qe, Re, MixedElement([Re] * 5)]))
        self.bcs = [DirichletBC(W.sub(0).sub(2), Constant(0.0), marker, BASE_ID)]
        self.w = w = Function(W)
        wt, dw = TestFunction(W), TrialFunction(W)
        u, p, P, c = split(w)
        v = TestFunctions(W)[0]
        self.u, self.p, self.P = u, p, P

        I = Identity(3)
        F = I + grad(u)
        J = det(F)
        E = 0.5 * (F.T * F - I)
        self.F, self.J = F, J
        Cs = [Constant(x) for x in (cfg.C, cfg.C_FF, cfg.C_SS, cfg.C_NN, cfg.C_NS, cfg.C_FS, cfg.C_FN)]

        def sym(a, b):
            return outer(a, b) + outer(b, a)

        def passive(f0, n0):
            """Fung energy C (e^Q - 1) and its S = dW/dE in the frame (f0, s0, n0) of one fiber family."""
            Eff, Ess, Enn = inner(f0, E * f0), inner(s0, E * s0), inner(n0, E * n0)
            Efs, Efn, Ens = inner(f0, E * s0), inner(f0, E * n0), inner(n0, E * s0)
            Q = Cs[1] * Eff ** 2 + Cs[2] * Ess ** 2 + Cs[3] * Enn ** 2 + Cs[4] * Ens ** 2 + Cs[5] * Efs ** 2 + Cs[6] * Efn ** 2
            S = Cs[0] * exp(Q) * (Cs[1] * Eff * sym(f0, f0) + Cs[2] * Ess * sym(s0, s0) + Cs[3] * Enn * sym(n0, n0)
                                  + Cs[4] * Ens * sym(n0, s0) + Cs[5] * Efs * sym(f0, s0) + Cs[6] * Efn * sym(f0, n0))
            return Cs[0] * (exp(Q) - 1.0), S

        t0, ta = Constant(cfg.T0_MS), self.ta

        def active(f0):
            """heartFEM Ct(t, ls) and the active stress per unit tension of one fiber family (its own ls and tr)."""
            ls = sqrt(dot(f0, F.T * F * f0)) * cfg.LR
            ECa = cfg.CA0MAX / sqrt(exp(cfg.B * conditional(le(ls, cfg.L0 + 0.003), 0.003, ls - cfg.L0)) - 1)
            tr = Constant(cfg.TR_M_MS_PER_UM) * ls + Constant(cfg.TR_B_MS)
            w1 = conditional(lt(ta, t0), 1.0, 0.0) * pi * ta / t0
            w2 = conditional(le(t0, ta), 1.0, 0.0) * conditional(lt(ta, t0 + tr), 1.0, 0.0) * pi * (ta - t0 + tr) / tr
            return 0.5 * (1 - cos(w1 + w2)), cfg.CA0 ** 2 / (cfg.CA0 ** 2 + ECa ** 2) * outer(f0, f0)

        pas = [passive(f, n) for f, n in zip(self.fibers, normals)]
        act = [active(f) for f in self.fibers]
        W_pas = mix([w_ for w_, _ in pas])
        self.S_pas = mix([S for _, S in pas])
        self.Ct_family = [ct for ct, _ in act]
        self.Ct = mix(self.Ct_family)  # share-weighted activation (one family: its Ct)
        S_unit = mix([su for _, su in act])  # used for R_ref only
        self.S_act = mix([Constant(cfg.T_MAX) * ct * su for ct, su in act])

        dsendo = ds(ENDO_ID)
        V_u = -Constant(1.0 / 3.0) * inner(u + X, cofac(F) * N)
        self.vol_form = V_u * dsendo
        self.wall_volume = assemble(Constant(1.0) * dx(mesh))
        h = (self.wall_volume / mesh.num_entities_global(3)) ** (1 / 3)
        sc = Constant(2 * cfg.C * cfg.C_FF / h ** 2)  # scales the multiplier rows only
        energy = (W_pas - p * (J - 1.0)) * dx \
            + P * self.Vt / Constant(assemble(Constant(1.0) * dsendo)) * dsendo - P * V_u * dsendo \
            + inner(as_vector([c[0], c[1], 0.0]) * sc, u) * dx + inner(as_vector([c[3], c[4], c[2]]) * sc / L, cross(X, u)) * dx
        F_act = inner(F * self.S_act, grad(v)) * dx
        self.R_form = derivative(energy, w, wt) + F_act
        if cfg.K_SPRING > 0:  # epicardial spring of heartFEM: traction -K (u . N) N
            self.R_form = self.R_form + Constant(cfg.K_SPRING) * inner(dot(u, N) * N, v) * ds(EPI_ID)
        self.Jac = derivative(self.R_form, w, dw)
        self.A = self.ksp = None
        self.V0 = assemble(self.vol_form)
        b = assemble(derivative(-V_u * dsendo, w, wt) + inner(F * S_unit, grad(v)) * dx)  # |R(u = 0; P = 1, T = 1)|
        for bc in self.bcs:
            bc.apply(b)
        self.R_ref = b.norm("l2")

    def cell_field(self, V, values):
        """DG0 function from per-cell values given in t0.vtu cell order."""
        f = Function(V)
        n = f.vector().local_size()
        dofs = np.array([V.dofmap().cell_dofs(c) for c in range(self.mesh.num_cells())]).reshape(self.mesh.num_cells(), -1)
        vals = values[self.gcell].reshape(len(dofs), -1)
        own = dofs < n
        arr = np.zeros(n)
        arr[dofs[own]] = vals[own]
        f.vector().set_local(arr)
        f.vector().apply("insert")
        return f

    def set_target(self, V, x):
        """x = activation time unwrapped along the solve order (continuation never jumps a cycle); the forms see
        mod(x, BCL)."""
        self.Vt.assign(float(V))
        self.ta.assign(float(x) % self.cfg.BCL_MS)

    def march(self, start, target, h=1.0):
        """Reach target = (V, unwrapped t_a) from the converged start, with sub-steps if Newton fails."""
        s = 0.0
        while True:
            s_new = 1.0 if 1.0 - (s + h) < 0.2 * h else s + h
            V, x = (start[i] + s_new * (target[i] - start[i]) for i in (0, 1))
            self.set_target(V, x)
            saved = self.w.vector().copy()
            final = s_new == 1.0
            ok, its, res = self.newton((1e-9 if final else 1e-6) * self.R_ref)
            if ok and final:
                ok = abs(assemble(self.vol_form) - V) <= 1e-8 * abs(V)
            if ok:
                if final:
                    return True, its, res
                s, h = s_new, (min(1.0, 1.5 * h) if its <= 5 else h)
            else:
                self.set_vector(saved)
                h *= 0.5
                if h < 1.0 / 4096:
                    return False, its, res


def solve_states(cfg, g, R, mesh_h5, order, x, volume, out, name, describe, row):
    """Solve the states k in `order`, each from the previous solution (the first from the reference state u = 0), and
    gather u and the stresses on rank 0 after each state into model.States(out, name).
    x[k]: activation time unwrapped along the solve order; volume(pb, k): target cavity volume (mm^3);
    describe(k, V): start of the log line; row(k, P, V, T_mean, u_max_um, newton_its, seconds): summary row.
    Returns (States, info) on rank 0."""
    pb = VolumeProblem(cfg, COMM, g, R, mesh_h5)
    mesh = pb.mesh
    log(f"{COMM.size} ranks, {pb.W.dim()} unknowns, V0 {pb.V0 * 1e3:.5f} nL")
    t_solve = time.time()
    # Cell averages (DG0 projections) gathered in t0.vtu cell order.
    DG0, TG0 = FunctionSpace(mesh, "DG", 0), TensorFunctionSpace(mesh, "DG", 0)
    q, qt = TestFunction(DG0), TestFunction(TG0)
    ds_ = np.array([DG0.dofmap().cell_dofs(c)[0] for c in range(mesh.num_cells())])
    dt_ = np.array([TG0.dofmap().cell_dofs(c) for c in range(mesh.num_cells())])
    vol_vec = assemble(q * dx)
    own = ds_ < vol_vec.local_size()
    vol = vol_vec.get_local()[ds_[own]]

    def gather(idx, arr, n):
        parts = COMM.gather((idx, arr), root=0)
        if RANK:
            return None
        out_ = np.empty((n,) + arr.shape[1:])
        for i_, a in parts:
            out_[i_] = a
        return out_

    def cells(arr):
        return gather(pb.gcell[own], arr, len(g["cells"]))

    def avg(expr):
        return cells(assemble(expr * q * dx).get_local()[ds_[own]] / vol)

    def avg_tensor(expr):
        A = cells((assemble(inner(expr, qt) * dx).get_local()[dt_[own]] / vol[:, None]).reshape(-1, 3, 3))
        return None if RANK else np.einsum("ji,cjk,kl->cil", R, A, R)  # base frame -> original frame

    F, p = pb.F, pb.p
    sigma = F * (pb.S_pas + pb.S_act) * F.T / pb.J - p * Identity(3)

    def fiber_stress(f0):
        fib = F * f0 / sqrt(inner(F * f0, F * f0))
        return inner(fib, sigma * fib)
    sigma_ff = fiber_stress(pb.f0)  # family 1
    cell_volume = cells(vol)
    # Nodal displacement: u interpolated into P1 (its values at the vertices); vertices are matched to t0.vtu nodes by
    # coordinates (the distributed mesh renumbers vertices).
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    u1 = Function(V1)
    v2d = vertex_to_dof_map(V1).reshape(-1, 3)
    vown = (v2d < u1.vector().local_size()).all(1)
    tree = cKDTree((g["points"] - g["base_origin"]) @ R.T) if RANK == 0 else None

    def nodal_u():
        u1.interpolate(pb.w.sub(0))
        parts = COMM.gather((mesh.coordinates()[vown], u1.vector().get_local()[v2d[vown]]), root=0)
        if RANK:
            return None
        out_ = np.full((len(g["points"]), 3), np.nan)
        for x_, a in parts:
            dist, idx = tree.query(x_)
            assert dist.max() < 1e-9, "vertex not found in the input mesh"
            out_[idx] = a
        assert not np.isnan(out_).any()
        return out_ @ R  # base frame -> original frame
    states = model.States(cfg, g, cell_volume, out, name) if RANK == 0 else None

    current, prev = (pb.V0, float(x[order[0]])), None
    for k in order:
        target, t0 = (volume(pb, k), float(x[k])), time.time()
        vec0 = pb.w.vector().copy()
        if prev is not None:  # linear extrapolation from the last two states, kept if it lowers the residual
            pb.set_target(*target)
            r0 = pb.residual().norm("l2")
            pb.set_vector(vec0 + (vec0 - prev))
            r1 = pb.residual().norm("l2")
            if np.isnan(r1) or r1 > r0:
                pb.set_vector(vec0)
        ok, its, rr = pb.march(current, target)
        if not ok:
            pb.set_vector(vec0)
            ok, its, rr = pb.march(current, target, h=0.25)
        if not ok:
            raise RuntimeError(f"{name} {k} did not converge")
        P = assemble(pb.P * dx(mesh)) / pb.wall_volume
        V = assemble(pb.vol_form)
        T_mean = cfg.T_MAX * assemble(pb.Ct * dx(mesh)) / pb.wall_volume
        log(f"{describe(k, V)}  mean T {T_mean:6.3f} kPa  ->  P {P:+.4f} kPa  "
            f"({its} Newton its, |R|/R_ref {rr / pb.R_ref:.0e}, {time.time() - t0:.0f} s)")
        u = nodal_u()
        fields = {"sigma": avg_tensor(sigma), "F": avg_tensor(F), "sigma_ff": avg(sigma_ff), "p": avg(p), "Ct": avg(pb.Ct)}
        if len(pb.fibers) == 2:  # family 2 fiber stress and the activation of each family
            fields.update(sigma_ff2=avg(fiber_stress(pb.fibers[1])), Ct_f1=avg(pb.Ct_family[0]), Ct_f2=avg(pb.Ct_family[1]))
        if RANK == 0:
            states.add(k, row(k, P, V, T_mean, 1e3 * np.linalg.norm(u, axis=1).max(), its, time.time() - t0), u, fields)
        prev, current = vec0, target
    info = {"mpi_ranks": COMM.size, "unknowns": pb.W.dim(), "nodes": len(g["points"]), "tetrahedra": len(g["cells"]),
            "solve_s": round(time.time() - t_solve, 1)}
    return states, info


def material_record(cfg, g, onset=None):
    """Material constants stored in pv_results.npz (JSON)."""
    m = {"C": cfg.C, "C_ff": cfg.C_FF, "C_ss": cfg.C_SS, "C_nn": cfg.C_NN, "C_ns": cfg.C_NS, "C_fs": cfg.C_FS,
         "C_fn": cfg.C_FN, "Ca0": cfg.CA0, "Ca0max": cfg.CA0MAX, "B": cfg.B, "l0": cfg.L0, "lr": cfg.LR,
         "T_max": cfg.T_MAX, "t0_ms": cfg.T0_MS, "tr_m_ms_per_um": cfg.TR_M_MS_PER_UM, "tr_b_ms": cfg.TR_B_MS,
         "BCL_ms": cfg.BCL_MS, **({"onset_frame": onset} if onset is not None else {}), "K_spring": cfg.K_SPRING,
         "fiber_angles_deg": [list(a) for a in model.fiber_families(cfg)[0]], "fiber_weights": g["weights"].tolist()}
    return m


# ==================== 3. pressure-driven problem (invert_para) ====================
class PressureProblem(Newton):
    """u (P2), p (P1), five rigid-body multipliers; given cavity pressure Pc; C, T_max assignable; fixed fibers."""

    def __init__(self, cfg, comm, g, R, mesh_h5):
        self.cfg, self.g, self.R = cfg, g, R
        L = float(np.ptp(g["points"], 0).max())
        mesh, marker = read_mesh(comm, mesh_h5)
        self.mesh = mesh
        self.gcell = np.array(mesh.topology().global_indices(3))
        DG0v = VectorFunctionSpace(mesh, "DG", 0)
        self.DG0v = DG0v
        self.eC, self.eL, self.eR = (g[k] @ R.T for k in ("eC", "eL", "eR"))
        self.s0 = s0 = Function(DG0v)
        self.set_cell_values(s0, self.eR)
        self.families = model.fiber_shares(cfg)
        self.fibers, normals = [], []
        for a, _ in model.fiber_angles(cfg, g["depth"]):  # per cell, linear in the transmural depth
            a = a[:, None]
            f = np.cos(a) * self.eC + np.sin(a) * self.eL
            f /= np.linalg.norm(f, axis=1, keepdims=True)
            ff, nn = Function(DG0v), Function(DG0v)
            self.set_cell_values(ff, f)
            self.set_cell_values(nn, np.cross(f, self.eR))
            self.fibers.append(ff)
            normals.append(nn)
        self.f0 = self.fibers[0]
        weights = [w_ for _, w_ in self.families]

        def mix(terms):
            """sum_i w_i terms_i over the fiber families (one family: the term itself, unchanged form)."""
            if len(terms) == 1:
                return terms[0]
            return reduce(operator.add, [Constant(wi) * x for wi, x in zip(weights, terms)])
        ds = dolfin.ds(domain=mesh, subdomain_data=marker)
        N, X = FacetNormal(mesh), SpatialCoordinate(mesh)
        self.Ct, self.Pc = Constant(0.0), Constant(0.0)
        self.Cc, self.Tc = Constant(0.1), Constant(1.0)  # set by set_state()
        Ve = VectorElement("CG", mesh.ufl_cell(), 2)
        Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
        Re = FiniteElement("Real", mesh.ufl_cell(), 0)
        self.W = W = FunctionSpace(mesh, MixedElement([Ve, Qe, MixedElement([Re] * 5)]))
        self.bcs = [DirichletBC(W.sub(0).sub(2), Constant(0.0), marker, BASE_ID)]
        self.w = w = Function(W)
        wt, dw = TestFunction(W), TrialFunction(W)
        u, p, c = split(w)
        v = TestFunctions(W)[0]
        I = Identity(3)
        F = I + grad(u)
        J = det(F)
        E = 0.5 * (F.T * F - I)
        Cs = [Constant(x) for x in (cfg.C_FF, cfg.C_SS, cfg.C_NN, cfg.C_NS, cfg.C_FS, cfg.C_FN)]

        def fung_exp(f0, n0):
            """e^Q - 1 in the frame (f0, s0, n0) of one fiber family."""
            Eff, Ess, Enn = inner(f0, E * f0), inner(s0, E * s0), inner(n0, E * n0)
            Efs, Efn, Ens = inner(f0, E * s0), inner(f0, E * n0), inner(n0, E * s0)
            Q = Cs[0] * Eff ** 2 + Cs[1] * Ess ** 2 + Cs[2] * Enn ** 2 + Cs[3] * Ens ** 2 + Cs[4] * Efs ** 2 + Cs[5] * Efn ** 2
            return exp(Q) - 1.0

        def active_unit(f0):
            """Ct g(ls) f0 x f0 of one fiber family (its own ls), per unit T_MAX."""
            ls = sqrt(dot(f0, F.T * F * f0)) * cfg.LR
            ECa = cfg.CA0MAX / sqrt(exp(cfg.B * conditional(le(ls, cfg.L0 + 0.003), 0.003, ls - cfg.L0)) - 1)
            return self.Ct, cfg.CA0 ** 2 / (cfg.CA0 ** 2 + ECa ** 2) * outer(f0, f0)
        W_fung = mix([fung_exp(f, n) for f, n in zip(self.fibers, normals)])
        act = [active_unit(f) for f in self.fibers]
        S_act_unit = mix([ct * su for ct, su in act])   # active stress per unit T_MAX
        S_unit = mix([su for _, su in act])             # R_ref only
        dsendo = ds(ENDO_ID)
        V_u = -Constant(1.0 / 3.0) * inner(u + X, cofac(F) * N)
        self.vol_form = V_u * dsendo
        self.wall_volume = assemble(Constant(1.0) * dx(mesh))
        h = (self.wall_volume / mesh.num_entities_global(3)) ** (1 / 3)
        sc = Constant(2 * 0.1 * cfg.C_FF / h ** 2)  # scales the multiplier rows only
        energy = (self.Cc * W_fung - p * (J - 1.0)) * dx - self.Pc * V_u * dsendo \
            + inner(as_vector([c[0], c[1], 0.0]) * sc, u) * dx + inner(as_vector([c[3], c[4], c[2]]) * sc / L, cross(X, u)) * dx
        F_act = inner(F * self.Tc * S_act_unit, grad(v)) * dx
        self.R_form = derivative(energy, w, wt) + F_act
        if cfg.K_SPRING > 0:  # epicardial spring of heartFEM: traction -K (u . N) N
            self.R_form = self.R_form + Constant(cfg.K_SPRING) * inner(dot(u, N) * N, v) * ds(EPI_ID)
        self.Jac = derivative(self.R_form, w, dw)
        self.A = self.ksp = None
        self.V0 = assemble(self.vol_form)
        b = assemble(derivative(-V_u * dsendo, w, wt) + inner(F * S_unit, grad(v)) * dx)
        for bc in self.bcs:
            bc.apply(b)
        self.R_ref = b.norm("l2")
        # vertex displacement gathering (vertices matched to the input mesh by coordinates)
        V1 = VectorFunctionSpace(mesh, "CG", 1)
        self.u1 = Function(V1)
        self.v2d = vertex_to_dof_map(V1).reshape(-1, 3)
        self.vown = (self.v2d < self.u1.vector().local_size()).all(1)
        self.tree = cKDTree((g["points"] - g["base_origin"]) @ R.T) if RANK == 0 else None

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

    def set_state(self, s):
        """s = (P kPa, Ct, C kPa, T_max kPa)."""
        self.Pc.assign(float(s[0]))
        self.Ct.assign(float(s[1]))
        self.Cc.assign(float(s[2]))
        self.Tc.assign(float(s[3]))

    def march(self, start, target, h=1.0):
        """Continuation from the converged state `start` to `target` (4-vectors), halving the step on failure."""
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

    def nodal_u_um(self):
        self.u1.interpolate(self.w.sub(0))
        parts = COMM.gather((self.mesh.coordinates()[self.vown], self.u1.vector().get_local()[self.v2d[self.vown]]), root=0)
        if RANK:
            return None
        out = np.full((len(self.g["points"]), 3), np.nan)
        for x, a in parts:
            dist, idx = self.tree.query(x)
            assert dist.max() < 1e-9
            out[idx] = a
        assert not np.isnan(out).any()
        return (out @ self.R) * 1e3  # base frame -> original frame, mm -> um

    def volume_nL(self):
        return assemble(self.vol_form) * 1e3


class FrameSolver:
    """One PressureProblem, a stored converged solution per frame (warm start for the next parameter set)."""

    def __init__(self, cfg, frames, P_kPa, ct, work):
        self.mesh_h5 = Path(work).resolve() / "_mesh.h5"  # temporary: the serial mesh, read back distributed
        g, R = setup(cfg, self.mesh_h5)
        self.pb = PressureProblem(cfg, COMM, g, R, self.mesh_h5)
        self.g = g
        self.frames = list(frames)
        self.P = {k: float(P_kPa[k]) for k in self.frames}
        self.ct = dict(ct)
        zero = self.pb.w.vector().copy()
        zero.zero()
        self.store = {k: (zero.copy(), None) for k in self.frames}  # (solution, state)

    def target(self, k, theta):
        C, T = theta
        return np.array([self.P[k], self.ct[k], C, T])

    def solve(self, k, theta):
        """Displacement (um, rank 0) and cavity volume (nL) of frame k for theta = (C, T_max)."""
        pb = self.pb
        vec, state = self.store[k]
        tgt = self.target(k, theta)
        t0 = time.time()
        if state is None:  # first solve: from the unloaded, inactive reference
            state = np.array([0.0, 0.0, theta[0], theta[1]])
            pb.set_state(state)
        pb.set_vector(vec)
        ok, its, res = pb.march(state, tgt)
        if not ok:
            pb.set_vector(vec)
            ok, its, res = pb.march(state, tgt, h=0.125)
        if not ok:
            log(f"   frame t{k} theta {theta}: no convergence")
            return None, np.nan, its, time.time() - t0
        self.store[k] = (pb.w.vector().copy(), tgt)
        return pb.nodal_u_um(), pb.volume_nL(), its, time.time() - t0
