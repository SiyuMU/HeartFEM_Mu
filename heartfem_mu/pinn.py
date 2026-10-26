"""PINN solver of the heart-chamber wall (torch; CONDA_ENV, GPU if visible).

The network maps the loads to the displacement of every node of the quadratic (P2) tetrahedral mesh - the same
displacement space as the FE. There is no coordinate input: the deformation gradient F is evaluated from the P2
shape-function gradients of the unloaded mesh. Training uses no FE or image data; it minimizes the total potential energy
of the same mechanics as the FE (lengths mm, stresses kPa):

    Pi(u; P, t) = int [ C (e^Q - 1) + Psi_act(lambda; t) ] dV              passive Fung + heartFEM active stress
                  + KAPPA/2 sum_a m_a Jbar_a^2                              incompressibility on the nodal P1 projection
                  - P (V_cav(u) - V_cav(0))                                  cavity pressure (curved P2 endocardium)
                  + rigid-body penalty                                       no rigid motion
                  + K_SPRING/2 int_epi (u . N)^2 dA                          epicardial spring of heartFEM (if > 0; default 0)
    Psi_act(lambda; t) = T_MAX int_1^lambda Ct(t, LR s) g(LR s) s ds, so that dPsi_act/dE_ff = T_MAX Ct(t, ls) g(ls), the
    heartFEM active stress with the field relaxation time tr = m ls + b; u . n = 0 on the base. With two fiber families
    the passive and active terms are the share-weighted sums over the families.
Two networks (features -> tanh MLP -> linear P2 nodal output + multilevel coarse output, u = U_SCALE (NN(x) - NN(0)) so
that u = 0 unloaded, whitened output parametrization):
    LoadPINN   (volume_constrain, windkessel) (P, t) -> u: cavity pressure P (kPa) and activation time t (ms since the
               onset, as the FE); features P / P_MAX and heartFEM's Ct(t, lr lambda_k) at the fixed fiber stretches
               FEATURE_STRETCHES (all equal on the rise; all 0 once even lambda = 1.6 has relaxed, so every passive time
               gives the same input as t = 0). windkessel trains P in [P_MIN, P_MAX] (P_MIN < 0).
    ParamPINN  (invert_para) (p, a) -> u in units of C, p = P / C, a = T_MAX Ct(t) / C: with C, T_MAX and P scaled
               together the displacement does not change, and on the Ct rise (and when passive) Ct does not depend on the
               fiber length, so one network serves every (C, T_MAX, P) and both frames.
Training: Adam warm-up on random loads (load and kappa ramps), then full-batch L-BFGS (history LBFGS_HISTORY) on fixed
log-spaced load grids refined in stages (GRID_STAGES, by L-BFGS outer step), each load weighted by
1 / |Pi(stage start)|^WEIGHT_POWER. On the finest grid the output layers are re-parametrized with whitened hidden features
(WHITEN: same displacement field, better-conditioned optimization) and training stops when the objective has stopped
decreasing (STOP_WINDOW / STOP_TOL), after 5 outer steps without progress, or at TRAIN_HOURS.
"""
import csv
import json
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.special import roots_jacobi, roots_legendre
from torch import nn

from . import model

EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))  # edge (midpoint) order inside a tetrahedron


# ==================== 1. mesh and inputs (numpy) ====================
def geometry(cfg):
    """Mesh (mm), P1 gradients, boundaries (base / endocardium / epicardium), base plane, local frame and transmural
    depth; fibers (volume_constrain, windkessel; invert_para builds them per point, param_fibers); frame volume ratios
    (volume_constrain)."""
    w = model.read_wall(cfg)
    points, cells = w["points"], w["cells"]
    corners = points[cells]
    D = np.stack((corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]), -1)
    volume = np.abs(np.linalg.det(D)) / 6
    inverse = np.linalg.inv(D)
    grad_N = np.concatenate((-inverse.sum(1, keepdims=True), inverse), 1)  # P1 shape gradients (E, 4, 3)
    endo, epi = w["endo"], w["epi"]
    # Endocardial triangles wound into the wall (positive closed cavity volume); epicardial normals point outward.
    endo_nodes, triangles = np.unique(endo[:, [0, 2, 1]], return_inverse=True)
    epi_normals = np.cross(points[epi[:, 1]] - points[epi[:, 0]], points[epi[:, 2]] - points[epi[:, 0]])
    depth = model.transmural_depth(points, cells, endo, epi)
    g = {"points": points, "cells": cells, "volume": volume, "grad_N": grad_N, "centroids": corners.mean(1),
         **{k: w[k] for k in ("eC", "eL", "eR")}, "depth": depth, "base_nodes": w["base_nodes"],
         "base_origin": points[w["base_nodes"]].mean(0), "base_normal": -w["n_out"],
         "endo_nodes": endo_nodes, "endo_triangles": triangles.reshape(-1, 3), "epi_faces": epi,
         "epi_normals": epi_normals / np.linalg.norm(epi_normals, axis=1, keepdims=True),
         "epi_areas": np.linalg.norm(epi_normals, axis=1) / 2}
    if cfg.TASK != "invert_para":
        g.update(model.fiber_fields(cfg, w, depth))
    if cfg.TASK == "volume_constrain":
        vols, ref = model.frame_volumes(cfg), model.reference_frame(cfg)
        g.update(V_ratio=vols / vols[ref], reference_frame=ref)
    return g


# ==================== 2. material laws (torch) ====================
@dataclass
class Material:
    C: float
    C_ff: float
    C_ss: float
    C_nn: float
    C_ns: float
    C_fs: float
    C_fn: float
    Ca0: float
    Ca0max: float
    B: float
    l0: float
    lr: float
    kappa: float
    T_max: float = 0.0
    t0: float = 0.0
    tr_m: float = 0.0
    tr_b: float = 0.0
    k_spring: float = 0.0  # kPa/mm; 0 = no epicardial spring
    active_quadrature: int = 16  # Gauss points for the active energy when no closed form applies

    @classmethod
    def of(cls, cfg):
        """The constants of params.py (volume_constrain, windkessel)."""
        return cls(cfg.C, cfg.C_FF, cfg.C_SS, cfg.C_NN, cfg.C_NS, cfg.C_FS, cfg.C_FN, cfg.CA0, cfg.CA0MAX, cfg.B, cfg.L0,
                   cfg.LR, cfg.KAPPA, cfg.T_MAX, cfg.T0_MS, cfg.TR_M_MS_PER_UM, cfg.TR_B_MS, cfg.K_SPRING)

    @classmethod
    def in_units_of_C(cls, cfg):
        """invert_para: the network is trained for C = 1 (the tension is an input)."""
        return cls(1.0, cfg.C_FF, cfg.C_SS, cfg.C_NN, cfg.C_NS, cfg.C_FS, cfg.C_FN, cfg.CA0, cfg.CA0MAX, cfg.B, cfg.L0,
                   cfg.LR, cfg.KAPPA_OVER_C, t0=cfg.T0_MS)


Q_CAP = 25.0  # passive law continued quadratically above this Q (only reached by wild optimizer trial steps)


def green_lagrange(F):
    return 0.5 * (F.transpose(-1, -2) @ F - torch.eye(3, dtype=F.dtype, device=F.device))


def fiber_stretch(F, f0):
    return (F @ f0[..., None]).squeeze(-1).norm(dim=-1)


def project(E, a, b):
    return (a * (E @ b[..., None]).squeeze(-1)).sum(-1)


def passive_q(E, f0, s0, n0, mat):
    return (mat.C_ff * project(E, f0, f0) ** 2 + mat.C_ss * project(E, s0, s0) ** 2
            + mat.C_nn * project(E, n0, n0) ** 2 + mat.C_ns * project(E, n0, s0) ** 2
            + mat.C_fs * project(E, f0, s0) ** 2 + mat.C_fn * project(E, f0, n0) ** 2)


def passive_energy(E, f0, s0, n0, mat):
    """C (e^Q - 1) for Q <= Q_CAP, continued C^2-smoothly by a quadratic above (keeps trial steps finite)."""
    Q = passive_q(E, f0, s0, n0, mat)
    d = torch.relu(Q - Q_CAP)
    return mat.C * (torch.exp(torch.clamp(Q, max=Q_CAP)) * (1 + d + 0.5 * d * d) - 1)


def passive_stress(E, f0, s0, n0, mat):
    """S_pas = dW/dE = C e^Q dQ/dE."""
    def sym(a, b):
        return a[..., :, None] * b[..., None, :] + b[..., :, None] * a[..., None, :]
    dq = (mat.C_ff * project(E, f0, f0)[..., None, None] * sym(f0, f0)
          + mat.C_ss * project(E, s0, s0)[..., None, None] * sym(s0, s0)
          + mat.C_nn * project(E, n0, n0)[..., None, None] * sym(n0, n0)
          + mat.C_ns * project(E, n0, s0)[..., None, None] * sym(n0, s0)
          + mat.C_fs * project(E, f0, s0)[..., None, None] * sym(f0, s0)
          + mat.C_fn * project(E, f0, n0)[..., None, None] * sym(f0, n0))
    return mat.C * torch.exp(passive_q(E, f0, s0, n0, mat))[..., None, None] * dq


def active_factor(lam, mat):
    """g(lambda) = Ca0^2 / (Ca0^2 + ECa50^2), with heartFEM's 0.003 um floor on l - l0."""
    dl = torch.clamp(mat.lr * lam - mat.l0, min=0.003)
    return mat.Ca0 ** 2 / (mat.Ca0 ** 2 + mat.Ca0max ** 2 / torch.expm1(mat.B * dl))


def active_energy_T(lam, T, mat):
    """Psi_act = T int_1^lambda g(s) s ds, so dPsi/dE = T g(lambda) f0 x f0 (closed form when Ca0 = Ca0max)."""
    a = mat.B * mat.lr
    s_star = (mat.l0 + 0.003) / mat.lr
    if mat.Ca0 == mat.Ca0max and s_star < 1:
        def G(s):
            return 0.5 * s ** 2 + np.exp(mat.B * mat.l0) * torch.exp(-a * s) * (s / a + 1 / a ** 2)

        def G_float(s):
            return 0.5 * s ** 2 + np.exp(-mat.B * (mat.lr * s - mat.l0)) * (s / a + 1 / a ** 2)

        g0 = -np.expm1(-0.003 * mat.B)
        above = G(torch.clamp(lam, min=s_star)) - G_float(1.0)
        below = G_float(s_star) - G_float(1.0) + 0.5 * g0 * (lam ** 2 - s_star ** 2)
        return T * torch.where(lam >= s_star, above, below)
    nodes, weights = np.polynomial.legendre.leggauss(mat.active_quadrature)
    nodes = torch.as_tensor(nodes, dtype=lam.dtype, device=lam.device)
    weights = torch.as_tensor(weights, dtype=lam.dtype, device=lam.device)
    half = 0.5 * (lam - 1)[..., None]
    s = 1 + half * (nodes + 1)
    return T * (half * weights * active_factor(s, mat) * s).sum(-1)


def ct_rise(t, mat):
    """Ct on the rise, 0 <= t < t0 (the same everywhere)."""
    return 0.5 * (1 - torch.cos(np.pi * torch.clamp(t, 0.0, mat.t0) / mat.t0))


def ct_field(t, lam, mat):
    """heartFEM Ct(t, ls), ls = lr lambda: rise 0.5 (1 - cos(pi t / t0)); relaxation 0.5 (1 + cos(pi (t - t0) / tr)) with
    tr = m ls + b while t < t0 + tr; 0 afterwards. t broadcasts against lambda."""
    tr = mat.tr_m * mat.lr * lam + mat.tr_b
    relax = (t - mat.t0).clamp(min=0.0)
    on = relax < tr
    ct_relax = 0.5 * (1 + torch.cos(np.pi * torch.where(on, relax / tr.clamp(min=1e-6), torch.ones_like(tr))))
    return torch.where(t < mat.t0, ct_rise(t, mat).expand_as(ct_relax), torch.where(on, ct_relax, torch.zeros_like(ct_relax)))


RELAX_QUADRATURE = 6  # Gauss points per smooth piece of the relaxation-phase active energy (stress error < 1e-6 kPa)


def active_energy(lam, t, mat):
    """Psi_act(lambda; t) = T_MAX int_1^lambda Ct(t, lr s) g(lr s) s ds, so dPsi/dE_ff = T_MAX Ct(t, ls) g(ls) (the FE's S_act).
    Rise (t < t0): Ct = Ct(t) uniform -> closed form. Relaxation: Ct(t, lr s) = 0 for s <= s* = (t - t0 - b) / (m lr) and
    is C1 there, so the integral runs over [max(1, s*), max(lambda, s*)], split at the kink of g (Gauss-Legendre on smooth pieces).
    lam (M, n), t (M, 1)."""
    out = torch.zeros_like(lam)
    rise = (t < mat.t0)
    if bool(rise.any()):
        out = torch.where(rise, active_energy_T(lam, mat.T_max * ct_rise(t, mat), mat), out)
    if bool((~rise).any()):
        s_star = (t - mat.t0 - mat.tr_b) / (mat.tr_m * mat.lr)
        a = torch.clamp(s_star, min=1.0).expand_as(lam)
        b = torch.maximum(lam, s_star)
        # g(lr s) has a kink at s_g = (l0 + 0.003) / lr (floor of ls - l0): split there so both pieces are smooth
        c = torch.minimum(torch.maximum(torch.full_like(b, (mat.l0 + 0.003) / mat.lr), torch.minimum(a, b)),
                          torch.maximum(a, b))
        nodes, weights = np.polynomial.legendre.leggauss(RELAX_QUADRATURE)
        nodes = torch.as_tensor(nodes, dtype=lam.dtype, device=lam.device)
        weights = torch.as_tensor(weights, dtype=lam.dtype, device=lam.device)
        tq = torch.maximum(t, torch.full_like(t, mat.t0))[..., None]
        relax_energy = 0.0
        for lo, hi in ((a, c), (c, b)):
            half = (0.5 * (hi - lo))[..., None]
            s = lo[..., None] + half * (nodes + 1)
            f = ct_field(tq, s, mat) * active_factor(s, mat) * s
            relax_energy = relax_energy + mat.T_max * (half * weights * f).sum(-1)
        out = torch.where(rise, out, relax_energy)
    return out


def active_stress(lam, t, f0, mat):
    return (mat.T_max * ct_field(t, lam, mat) * active_factor(lam, mat))[..., None, None] * f0[..., :, None] * f0[..., None, :]


def projected_volume_change(J, weight, projection):
    """Lumped P1 projection of J - 1 onto the mesh vertices: bar_a = int N_a (J - 1) dV / m_a. J: (..., Q points)."""
    local = ((weight * (J - 1))[..., None] * projection["N"]).reshape(*J.shape[:-1], -1)
    nodal = torch.zeros(*J.shape[:-1], len(projection["node_volume"]), dtype=J.dtype, device=J.device)
    return nodal.index_add(-1, projection["cells"].reshape(-1), local) / projection["node_volume"]


# ==================== 3. P2 discretization ====================
def tet_rule(n):
    """Tetrahedron rule (barycentric points, weights summing to 1): n = 1 the 4-point degree-2 Gauss rule; n >= 2 the
    Stroud conical product rule with n^3 points, degree 2n - 1 (positive weights)."""
    if n == 1:
        a, b = 0.5854101966249685, 0.1381966011250105
        return np.array([[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]]), np.full(4, 0.25)

    def jacobi01(n, alpha):
        x, w = roots_jacobi(n, alpha, 0.0) if alpha > 0 else roots_legendre(n)
        return (1 + x) / 2, w / 2 ** (alpha + 1)
    u, wu = jacobi01(n, 2)
    v, wv = jacobi01(n, 1)
    s, ws = jacobi01(n, 0)
    U, V, S = np.meshgrid(u, v, s, indexing="ij")
    W = np.einsum("i,j,k->ijk", wu, wv, ws)
    x, y, z = U, (1 - U) * V, (1 - U) * (1 - V) * S
    return np.stack((1 - x - y - z, x, y, z), -1).reshape(-1, 4), (W / W.sum()).ravel()


def tri_rule():
    """Dunavant 7-point degree-5 triangle rule: barycentric points (7, 3), weights summing to 1."""
    a1, b1, w1 = 0.059715871789770, 0.470142064105115, 0.132394152788506
    a2, b2, w2 = 0.797426985353087, 0.101286507323456, 0.125939180544827
    pts = [(1 / 3, 1 / 3, 1 / 3), (a1, b1, b1), (b1, a1, b1), (b1, b1, a1), (a2, b2, b2), (b2, a2, b2), (b2, b2, a2)]
    return np.array(pts), np.array([0.225, w1, w1, w1, w2, w2, w2])


def p2_tet_shapes(lam):
    """Values (Q, 10) and barycentric derivatives (Q, 10, 4) of the P2 tetrahedron shapes (vertices, then EDGES)."""
    N, dN = np.zeros((len(lam), 10)), np.zeros((len(lam), 10, 4))
    for k in range(4):
        N[:, k] = lam[:, k] * (2 * lam[:, k] - 1)
        dN[:, k, k] = 4 * lam[:, k] - 1
    for e, (i, j) in enumerate(EDGES):
        N[:, 4 + e] = 4 * lam[:, i] * lam[:, j]
        dN[:, 4 + e, i] = 4 * lam[:, j]
        dN[:, 4 + e, j] = 4 * lam[:, i]
    return N, dN


def p2_tri_shapes(lam):
    """6-node triangle (a, b, c, ab, bc, ca) at barycentric (1 - xi - eta, xi, eta): values (Q, 6), d/d(xi, eta) (Q, 6, 2)."""
    L1, L2, L3 = lam[:, 0], lam[:, 1], lam[:, 2]
    N = np.stack((L1 * (2 * L1 - 1), L2 * (2 * L2 - 1), L3 * (2 * L3 - 1), 4 * L1 * L2, 4 * L2 * L3, 4 * L3 * L1), -1)
    dNdL = np.zeros((len(lam), 6, 3))
    for m in range(3):
        dNdL[:, m, m] = 4 * lam[:, m] - 1
    dNdL[:, 3, 0], dNdL[:, 3, 1] = 4 * L2, 4 * L1
    dNdL[:, 4, 1], dNdL[:, 4, 2] = 4 * L3, 4 * L2
    dNdL[:, 5, 2], dNdL[:, 5, 0] = 4 * L1, 4 * L3
    return N, dNdL @ np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


def p2_mesh(g, quad_order):
    """P2 degrees of freedom (vertices, then edge midpoints) and the volume rule of geometry() data."""
    points, cells = g["points"], g["cells"]
    nv = len(points)
    pairs = np.sort(np.stack([cells[:, [i, j]] for i, j in EDGES], 1), -1)
    edges, edge_id = np.unique(pairs.reshape(-1, 2), axis=0, return_inverse=True)
    lookup = {tuple(e): nv + k for k, e in enumerate(edges)}

    def six(tri):
        a, b, c = tri
        return [a, b, c, lookup[tuple(sorted((a, b)))], lookup[tuple(sorted((b, c)))], lookup[tuple(sorted((c, a)))]]

    base = set(g["base_nodes"].tolist())
    lam, w = tet_rule(quad_order)
    N, dN = p2_tet_shapes(lam)
    return {"dof_points": np.concatenate((points, points[edges].mean(1))),
            "cell_dofs": np.concatenate((cells, nv + edge_id.reshape(len(cells), 6)), 1), "n_vertices": nv,
            "base_dofs": np.array(sorted(base | {nv + k for k, (a, b) in enumerate(edges) if a in base and b in base})),
            "endo6": np.array([six(t) for t in g["endo_nodes"][g["endo_triangles"]]]),
            "epi6": np.array([six(t) for t in g["epi_faces"]]),
            "lam": lam, "w": w, "N": N, "grad": np.einsum("qam,emk->eqak", dN, g["grad_N"]),
            "X": np.einsum("qm,emk->eqk", lam, points[cells]), "W": g["volume"][:, None] * w[None, :]}


class Geometry:
    """Tensors of the P2 mesh for the energy and the stresses, on one device."""

    def __init__(self, g, p2, device):
        def f(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)

        def i(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
        E, Q = p2["grad"].shape[:2]
        self.E, self.Q = E, Q
        self.cell_dofs, self.grad, self.Nq = i(p2["cell_dofs"]), f(p2["grad"]), f(p2["N"])
        self.X, self.W = f(p2["X"]).reshape(E * Q, 3), f(p2["W"]).reshape(E * Q)
        rep = lambda a: f(np.repeat(a, Q, 0))  # per cell -> per point
        if "fibers" in g:  # fixed fiber families per point
            self.f0, self.s0, self.n0 = rep(g["f0"]), rep(g["s0"]), rep(g["n0"])
            self.fibers, self.normals = [rep(a) for a in g["fibers"]], [rep(a) for a in g["normals"]]  # per family
            self.fiber_weights = [float(w) for w in g["weights"]]
        else:  # invert_para: local frame and depth per point, the fibers follow from them (param_fibers)
            self.eC, self.eL, self.eR = rep(g["eC"]), rep(g["eL"]), rep(g["eR"])
            self.depth = rep(g["depth"])  # transmural depth per point (cell value), for the fiber angles
        node_volume = np.bincount(g["cells"].ravel(), np.repeat(g["volume"] / 4, 4), len(g["points"]))
        self.projection = {"cells": i(np.repeat(g["cells"], Q, 0)), "N": f(np.tile(p2["lam"], (E, 1))),
                           "node_volume": f(node_volume)}
        lam3, w3 = tri_rule()
        Nt, dNt = p2_tri_shapes(lam3)
        self.tri_N, self.tri_dN, self.tri_w = f(Nt), f(dNt), f(w3 / 2)  # reference triangle area 1/2
        self.endo6, self.epi6 = i(p2["endo6"]), i(p2["epi6"])
        self.dof_points = f(p2["dof_points"])
        self.epi_normal, self.epi_area = f(g["epi_normals"]), f(g["epi_areas"])
        self.base_origin = f(g["base_origin"])
        n = f(g["base_normal"])
        self.base_normal = n / n.norm()
        self.wall_volume = float(g["volume"].sum())
        self.length_scale = float(np.ptp(g["centroids"], 0).max() / 2)
        self.reference_volume = self.cavity_volume(torch.zeros(1, len(p2["dof_points"]), 3, device=device))[0]

    def gradients(self, U):
        """U (M, n_dofs, 3) -> grad u (M, E*Q, 3, 3) at the volume points."""
        g = torch.einsum("meai,eqaj->meqij", U[:, self.cell_dofs], self.grad)
        return g.reshape(U.shape[0], self.E * self.Q, 3, 3)

    def point_values(self, U):
        return torch.einsum("meai,qa->meqi", U[:, self.cell_dofs], self.Nq).reshape(U.shape[0], -1, 3)

    def cavity_volume(self, U):
        """(M,) cavity volume of the curved P2 endocardium closed by cones to the base origin: 1/3 int (x - o) . n dA."""
        x = (self.dof_points + U)[:, self.endo6] - self.base_origin
        xq = torch.einsum("mtai,qa->mtqi", x, self.tri_N)
        dxi = torch.einsum("mtai,qa->mtqi", x, self.tri_dN[..., 0])
        deta = torch.einsum("mtai,qa->mtqi", x, self.tri_dN[..., 1])
        return ((xq * torch.cross(dxi, deta, dim=-1)).sum(-1) * self.tri_w).sum((-1, -2)) / 3

    def spring_energy(self, U):
        """(M,) 1/2 int_epi (u . N)^2 dA over the reference P2 epicardium (N its normal): energy of the heartFEM
        epicardial spring per unit stiffness."""
        if not hasattr(self, "epi_dA"):
            x = self.dof_points[self.epi6]
            dxi = torch.einsum("tai,qa->tqi", x, self.tri_dN[..., 0])
            deta = torch.einsum("tai,qa->tqi", x, self.tri_dN[..., 1])
            nA = torch.cross(dxi, deta, dim=-1)
            self.epi_dA = nA.norm(dim=-1) * self.tri_w
            self.epi_N = nA / nA.norm(dim=-1, keepdim=True)
        uq = torch.einsum("mtai,qa->mtqi", U[:, self.epi6], self.tri_N)
        return 0.5 * (((uq * self.epi_N).sum(-1)) ** 2 * self.epi_dA).sum((-1, -2))


def rigid_terms(U, geo, W):
    """(M,) squared mean in-plane translation and rotation of the wall, scaled by the wall volume and size (the rigid-body
    penalty is 1/2 RIGID x this: no rigid translation in the base plane and no rotation)."""
    u = geo.point_values(U)
    n, L = geo.base_normal, geo.length_scale
    in_plane = u - (u @ n)[..., None] * n
    translation = (W[:, None] * in_plane).sum(-2)
    rotation = (W[:, None] * torch.cross((geo.X - geo.base_origin).expand_as(u), u, dim=-1)).sum(-2)
    return (translation ** 2).sum(-1) / (geo.wall_volume * L ** 2) + (rotation ** 2).sum(-1) / (geo.wall_volume * L ** 4)


def mix(weights, terms):
    """sum_i w_i terms_i over the fiber families (one family: the term itself, unchanged)."""
    return terms[0] if len(terms) == 1 else sum(w * x for w, x in zip(weights, terms))


# ==================== 4. energies ====================
def energy(U, geo, mat, P, t, rigid_penalty, incompressibility=True):
    """LoadPINN: total potential energy terms (M,) of nodal P2 displacements U (M, n_dofs, 3) for loads P (kPa), t (ms) (M,)."""
    F = torch.eye(3, dtype=U.dtype, device=U.device) + geo.gradients(U)
    E = green_lagrange(F)
    J = torch.linalg.det(F)
    W = geo.W
    terms = {"passive": mix(geo.fiber_weights, [(W * passive_energy(E, f, geo.s0, n, mat)).sum(-1)
                                                for f, n in zip(geo.fibers, geo.normals)]),
             "active": mix(geo.fiber_weights, [(W * active_energy(fiber_stretch(F, f), t[:, None], mat)).sum(-1)
                                               for f in geo.fibers])}
    if incompressibility:
        bar = projected_volume_change(J, W, geo.projection)
        terms["incompressibility"] = (geo.projection["node_volume"] * 0.5 * mat.kappa * bar ** 2).sum(-1)
    volume = geo.cavity_volume(U)
    terms["pressure"] = -P * (volume - geo.reference_volume)
    if mat.k_spring > 0:  # epicardial spring of heartFEM: traction -K (u . N) N
        terms["spring"] = mat.k_spring * geo.spring_energy(U)
    if rigid_penalty > 0:
        terms["rigid"] = 0.5 * rigid_penalty * rigid_terms(U, geo, W)
    terms["total"] = sum(terms.values())
    terms["cavity_volume"] = volume.detach()
    return terms


def param_fibers(cfg, geo):
    """ParamPINN: [(f_i, s0, n_i, w_i)] per fiber family: f_i = cos(a_i) eC + sin(a_i) eL, a_i linear in the transmural
    depth between the (endo, epi) angles, sheet s0 = eR, n_i = f_i x s0; arrays (E*Q, 3)."""
    out = []
    for (endo, epi), w in model.fiber_shares(cfg):
        a = torch.deg2rad((1 - geo.depth.double()) * endo + geo.depth.double() * epi)[:, None]  # float64 trigonometry
        f0 = torch.cos(a).to(geo.eC.dtype) * geo.eC + torch.sin(a).to(geo.eC.dtype) * geo.eL
        f0 = f0 / f0.norm(dim=-1, keepdim=True)
        out.append((f0, geo.eR, torch.cross(f0, geo.eR, dim=-1), w))
    return out


def param_energy(cfg, U, geo, mat, p, a, kappa):
    """ParamPINN: normalized total potential energy (M,) and the cavity volume for loads (p, a) (M,), in units of C:
    Pi / C = int sum_i w_i [(e^Q_i - 1) + Psi_act(lambda_i; a)] dV + (KAPPA/C)/2 sum m Jbar^2 - p (V - V0) + rigid / C
             (+ K_SPRING_OVER_C / 2 int_epi (u . N)^2 dA, only if > 0)."""
    F = torch.eye(3, dtype=U.dtype, device=U.device) + geo.gradients(U)
    E = green_lagrange(F)
    J = torch.linalg.det(F)
    W = geo.W
    total = 0.0
    for f0, s0, n0, w in geo.families:
        lam = (F @ f0[..., None]).squeeze(-1).norm(dim=-1)
        part = (W * passive_energy(E, f0, s0, n0, mat)).sum(-1) + (W * active_energy_T(lam, a[:, None], mat)).sum(-1)
        total = total + (part if w == 1.0 else w * part)
    if cfg.K_SPRING_OVER_C > 0:  # epicardial spring of heartFEM: traction -K (u . N) N, K = K_SPRING_OVER_C C
        total = total + cfg.K_SPRING_OVER_C * geo.spring_energy(U)
    bar = projected_volume_change(J, W, geo.projection)
    total = total + (geo.projection["node_volume"] * 0.5 * kappa * bar ** 2).sum(-1)
    volume = geo.cavity_volume(U)
    total = total - p * (volume - geo.reference_volume)
    total = total + 0.5 * cfg.RIGID_OVER_C * rigid_terms(U, geo, W)
    return total, volume


# ==================== 5. networks ====================
def coarse_interpolation(points, spacing):
    """Sparse partition-of-unity interpolation (rows, cols, values, n_coarse) from voxel-cluster centres at the given
    spacing to the points: hat kernel max(0, 1 - d / (2 spacing)), normalized per point."""
    from scipy.spatial import cKDTree
    key = np.floor((points - points.min(0)) / spacing).astype(np.int64)
    _, cluster = np.unique(key, axis=0, return_inverse=True)
    cluster = cluster.ravel()
    centres = np.zeros((cluster.max() + 1, 3))
    np.add.at(centres, cluster, points)
    centres /= np.bincount(cluster)[:, None]
    radius = 2 * spacing
    rows, cols, vals = [], [], []
    for i, js in enumerate(cKDTree(centres).query_ball_point(points, radius)):
        js = np.asarray(js, dtype=np.int64)
        w = np.maximum(0.0, 1 - np.linalg.norm(centres[js] - points[i], axis=1) / radius) if len(js) else np.zeros(0)
        if w.sum() <= 0:
            js, w = np.array([np.argmin(np.linalg.norm(centres - points[i], axis=1))]), np.ones(1)
        rows.append(np.full(len(js), i)), cols.append(js), vals.append(w / w.sum())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), len(centres)


class _Network(nn.Module):
    """tanh MLP of the input features, linear nodal P2 output plus multilevel coarse output (coarse displacements on voxel
    clusters interpolated to the DOFs: same representable fields, better-conditioned optimization), u = U_SCALE (NN(x) -
    NN(0)), u . n = 0 on the base vertices and midpoints. Output parametrization: u = W A (h - h0) for all output layers,
    h the last hidden features and h0 those of the unloaded state; A = I until refresh_whitening() sets it to the inverse
    square root of the feature Gram matrix."""

    def build(self, p2, base_normal, n_features, width, depth, levels, bias):
        points = p2["dof_points"]
        self.n_dofs = len(points)
        layers, size = [], n_features
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.Tanh()]
            size = width
        self.hidden = nn.Sequential(*layers)
        for layer in self.hidden:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.out = nn.Linear(size, 3 * self.n_dofs, bias=bias)
        nn.init.zeros_(self.out.weight)  # u = 0 at the start
        if bias:
            nn.init.zeros_(self.out.bias)
        self.coarse_out, self.interpolations = nn.ModuleList(), []
        for spacing in levels:
            rows, cols, vals, n_coarse = coarse_interpolation(points, spacing)
            layer = nn.Linear(size, 3 * n_coarse, bias=bias)
            nn.init.zeros_(layer.weight)
            if bias:
                nn.init.zeros_(layer.bias)
            self.coarse_out.append(layer)
            self.interpolations.append(torch.sparse_coo_tensor(np.stack((rows, cols)), vals.astype(np.float32),
                                                               (self.n_dofs, n_coarse)).coalesce())
        return size

    def hidden_difference(self, x1, x2):
        """h(x) - h(0) of the last hidden layer, (M, WIDTH), evaluated in float64 (the whitening amplifies its weak
        directions, so float32 rounding of the difference would be amplified too; the cost is negligible)."""
        z = torch.zeros(1, dtype=x1.dtype, device=x1.device) if self.zero_row else torch.zeros_like(x1)
        x = torch.cat((self.features(x1, x2), self.features(z, z))).double()  # last rows: unloaded state
        for layer in self.hidden:
            x = torch.tanh(x) if isinstance(layer, nn.Tanh) else nn.functional.linear(x, layer.weight.double(), layer.bias.double())
        return x[:-1] - x[-1:] if self.zero_row else x[:len(x1)] - x[len(x1):]

    @torch.no_grad()
    def refresh_whitening(self, x1, x2, eps):
        """A <- (G / max eig + eps I)^(-1/2), G the Gram matrix of the hidden differences over the loads. The output
        weights are transformed (W <- W A_old A_new^-1), so the displacement field does not change: only the coordinates
        of the optimization do (feature directions with little variation over the loads get as much curvature as the
        dominant ones)."""
        phi = self.hidden_difference(x1, x2)
        lam, V = torch.linalg.eigh(phi.T @ phi)
        lam = lam.clamp(min=0.0) / lam.max()
        new = (V / torch.sqrt(lam + eps)) @ V.T
        transfer = self.whiten_A @ ((V * torch.sqrt(lam + eps)) @ V.T)
        for layer in (self.out, *self.coarse_out):
            layer.weight.copy_((layer.weight.double() @ transfer).to(layer.weight.dtype))
        self.whiten_A.copy_(new)

    def forward(self, x1, x2):
        """Loads (M,), (M,) -> U (M, n_dofs, 3) mm."""
        M = len(x1)
        phi = (self.hidden_difference(x1, x2) @ self.whiten_A.T).to(x1.dtype)  # A (h - h0); the output biases cancel
        u = nn.functional.linear(phi, self.out.weight).reshape(M, self.n_dofs, 3)
        for i, layer in enumerate(self.coarse_out):
            if self.interpolations[i].device != u.device:
                self.interpolations[i] = self.interpolations[i].to(u.device)
            w = nn.functional.linear(phi, layer.weight).reshape(M, -1, 3).permute(1, 0, 2).reshape(-1, 3 * M)
            coarse = torch.sparse.mm(self.interpolations[i], w).reshape(self.n_dofs, M, 3).permute(1, 0, 2)
            u = u + (self.config["level_scale"] * coarse if "level_scale" in self.config else coarse)
        n = self.base_normal
        u = u - (self.base_mask[:, None] * (u @ n)[..., None]) * n
        return self.config["u_scale"] * u


class LoadPINN(_Network):
    """(P, t) -> P2 nodal displacements (volume_constrain, windkessel). Input: cavity pressure P (kPa) and activation time
    t (ms since onset, as in the FE). Features: P / P_MAX and heartFEM's Ct(t, lr lambda_k) at the fixed fiber stretches
    lambda_k = FEATURE_STRETCHES."""
    zero_row = True

    def __init__(self, p2, base_origin, base_normal, p_scale, u_scale, width, depth, levels, t0, stretches, lr, tr_m,
                 tr_b, level_scale=1.0, p_min=None):
        super().__init__()
        self.config = {"p_scale": float(p_scale), **({"p_min": float(p_min)} if p_min is not None else {}),
                       "u_scale": float(u_scale), "width": int(width), "depth": int(depth),
                       "levels": [float(v) for v in levels], "level_scale": float(level_scale), "t0": float(t0),
                       "stretches": [float(v) for v in stretches], "lr": float(lr), "tr_m": float(tr_m),
                       "tr_b": float(tr_b)}
        size = self.build(p2, base_normal, 1 + len(stretches), width, depth, levels, bias=True)
        normal = torch.as_tensor(np.asarray(base_normal), dtype=torch.float32)
        self.register_buffer("base_normal", normal / normal.norm())
        self.register_buffer("base_origin", torch.as_tensor(np.asarray(base_origin), dtype=torch.float32))
        mask = torch.zeros(self.n_dofs)
        mask[torch.as_tensor(p2["base_dofs"])] = 1.0
        self.register_buffer("base_mask", mask)
        self.register_buffer("whiten_A", torch.eye(size, dtype=torch.float64))

    @classmethod
    def of(cls, cfg, p2, g):
        return cls(p2, g["base_origin"], g["base_normal"], p_scale=cfg.P_MAX, u_scale=cfg.U_SCALE_MM, width=cfg.WIDTH,
                   depth=cfg.DEPTH, levels=tuple(v * cfg.MM_PER_UNIT for v in cfg.LEVELS_UM), t0=cfg.T0_MS,
                   stretches=cfg.FEATURE_STRETCHES, lr=cfg.LR, tr_m=cfg.TR_M_MS_PER_UM, tr_b=cfg.TR_B_MS,
                   p_min=getattr(cfg, "P_MIN", None))

    def features(self, P, t):
        """(M, 1 + K): P / P_MAX and Ct(t, lr lambda_k), k = 1..K."""
        c = self.config
        ta = t[..., None]
        lam = torch.as_tensor(c["stretches"], dtype=P.dtype, device=P.device)
        tr = c["tr_m"] * c["lr"] * lam + c["tr_b"]
        rise = 0.5 * (1 - torch.cos(np.pi * torch.clamp(ta, max=c["t0"]) / c["t0"]))
        relax_t = torch.clamp(ta - c["t0"], min=0.0)
        relax = torch.where(relax_t < tr, 0.5 * (1 + torch.cos(np.pi * torch.clamp(relax_t / tr, max=1.0))),
                            torch.zeros_like(relax_t))
        ct = torch.where(ta < c["t0"], rise.expand_as(relax), relax)
        return torch.cat(((P / c["p_scale"])[..., None], ct), -1)


class ParamPINN(_Network):
    """(p, a) -> P2 nodal displacements in units of C (invert_para); features p / P_MAX_N, a / A_MAX_N; output layers
    without bias."""
    zero_row = False

    def __init__(self, p2, base_normal, width, depth, levels, u_scale, p_max, a_max, fibers, k_spring_over_c):
        super().__init__()
        self.config = {"width": width, "depth": depth, "levels": list(levels), "u_scale": u_scale,
                       "p_max": p_max, "a_max": a_max, "fibers": fibers, "k_spring_over_c": float(k_spring_over_c)}
        size = self.build(p2, base_normal, 2, width, depth, levels, bias=False)
        normal = torch.as_tensor(np.asarray(base_normal), dtype=torch.float32)
        self.register_buffer("base_normal", normal / normal.norm())
        mask = torch.zeros(self.n_dofs)
        mask[torch.as_tensor(p2["base_dofs"])] = 1.0
        self.register_buffer("base_mask", mask)
        self.register_buffer("whiten_A", torch.eye(size, dtype=torch.float64))

    @classmethod
    def of(cls, cfg, p2, g):
        return cls(p2, g["base_normal"], width=cfg.WIDTH, depth=cfg.DEPTH,
                   levels=tuple(v * cfg.MM_PER_UNIT for v in cfg.LEVELS_UM), u_scale=cfg.U_SCALE_MM, p_max=cfg.P_MAX_N,
                   a_max=cfg.A_MAX_N, fibers=model.fiber_shares(cfg), k_spring_over_c=cfg.K_SPRING_OVER_C)

    def features(self, p, a):
        c = self.config
        return torch.stack((p / c["p_max"], a / c["a_max"]), -1)


# ==================== 6. training ====================
class Clock:
    """Optimization time only (paused while monitoring or saving)."""

    def __init__(self, device):
        self.device, self.total, self.t0 = device, 0.0, None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self.t0 = time.time()

    def stop(self):
        self._sync()
        self.total += time.time() - self.t0


def optimize(cfg, net, device, total_energy, random_loads, grid_loads, kappa_start, kappa, normalize, log_file):
    """Adam warm-up on random loads (load ramp; kappa from kappa_start to kappa), then staged full-batch L-BFGS on the
    load grids of GRID_STAGES with the stop rules (module docstring).
    total_energy(U, x1, x2, kappa): energy (M,) of the loads (x1, x2) (kappa None: the final value);
    random_loads(m, ramp): Adam loads; grid_loads(stage, index): the loads of a grid stage.
    Returns (clock, outer steps, stop reason)."""
    clock = Clock(device)
    log = open(log_file, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["stage", "step", "clock_s", "objective", "kappa"])

    # Stage 1: Adam on random loads with load and kappa ramps.
    opt = torch.optim.Adam(net.parameters(), lr=cfg.ADAM_LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(cfg.ADAM_LR_END / cfg.ADAM_LR) ** (1 / max(cfg.ADAM_STEPS, 1)))
    for step in range(1, cfg.ADAM_STEPS + 1):
        clock.start()
        ramp = min(1.0, step / cfg.LOAD_RAMP_STEPS)
        k = kappa_start * (kappa / kappa_start) ** min(1.0, step / cfg.KAPPA_RAMP_STEPS)
        X1, X2 = random_loads(cfg.ADAM_LOADS, ramp)
        opt.zero_grad()
        loss = 0.0
        for c in torch.split(torch.arange(cfg.ADAM_LOADS, device=device), cfg.ADAM_CHUNK):
            part = total_energy(net(X1[c], X2[c]), X1[c], X2[c], k).sum() / (cfg.ADAM_LOADS * normalize)
            part.backward()
            loss += part.item()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        clock.stop()
        if step % 100 == 0:
            writer.writerow(["adam", step, f"{clock.total:.2f}", loss, k])
            print(f"adam {step:5d} | loss {loss:+.5e} | kappa {k:.0f} | clock {clock.total:.0f}s", flush=True)

    # Stage 2: full-batch L-BFGS on fixed log-spaced load grids, refined in stages (GRID_STAGES).
    def energies():
        with torch.no_grad():
            return torch.cat([total_energy(net(X1[c], X2[c]), X1[c], X2[c], None) for c in chunks])

    def objective():
        lbfgs.zero_grad()
        total = 0.0
        for c in chunks:
            part = (total_energy(net(X1[c], X2[c]), X1[c], X2[c], None) / scale[c]).sum() / len(X1)
            part.backward()
            total += part.item()
        return total

    def new_lbfgs():
        return torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=cfg.LBFGS_INNER, history_size=cfg.LBFGS_HISTORY,
                                 line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

    def start_stage(stage, index):
        """New load grid: per-load weights from the current energies, fresh curvature history, stall counter reset; from
        WHITEN_FROM_OUTER on also the whitened output parametrization for the loads of the grid."""
        nonlocal X1, X2, chunks, scale, lbfgs, previous, stalls
        X1, X2 = grid_loads(stage, index)
        chunks = torch.split(torch.arange(len(X1), device=device), cfg.LBFGS_CHUNK)
        if cfg.WHITEN > 0 and outer >= cfg.WHITEN_FROM_OUTER:
            net.refresh_whitening(X1, X2, cfg.WHITEN)
        current_energy = energies().abs().clamp(min=1e-12)
        scale = current_energy ** cfg.WEIGHT_POWER
        scale = scale / scale.mean() * current_energy.mean() ** (1 - cfg.WEIGHT_POWER)
        lbfgs = new_lbfgs()
        previous, stalls = (energies() / scale).mean().item(), 0
        trace.clear()

    X1 = X2 = chunks = scale = lbfgs = None
    previous, stalls, outer = 0.0, 0, 0
    trace, converged = [], False  # objective after each outer step of the current stage (STOP_WINDOW rule)
    stages = list(cfg.GRID_STAGES)
    while clock.total < cfg.TRAIN_HOURS * 3600 and (stalls < 5 or stages) and not converged:
        outer += 1
        clock.start()
        if stages and (outer >= stages[0][-1] or stalls >= 5):  # scheduled refinement, or a coarse grid converged early
            stage = stages.pop(0)  # (n, first outer step) or (n_1, n_2, first outer step)
            start_stage(stage, len(cfg.GRID_STAGES) - len(stages) - 1)
            print(f"L-BFGS outer {outer}: load grid {'x'.join(str(v) for v in stage[:-1])} ({len(X1)} loads), "
                  f"objective {previous:+.8f}, clock {clock.total:.0f}s", flush=True)
        lbfgs.step(objective)
        current = (energies() / scale).mean().item()
        if current < previous - 1e-9 * abs(previous):
            stalls = 0
        else:  # no progress: restart the curvature history
            lbfgs = new_lbfgs()
            stalls += 1
        previous = min(previous, current)
        trace.append(current)
        if cfg.STOP_WINDOW and not stages and len(trace) > cfg.STOP_WINDOW:  # finest grid: the objective has flattened
            converged = (trace[-1 - cfg.STOP_WINDOW] - current) < cfg.STOP_TOL * abs(current)
        clock.stop()
        writer.writerow(["lbfgs", outer, f"{clock.total:.2f}", current, kappa])
        log.flush()
        if outer % 10 == 0 or outer == 1:
            print(f"lbfgs {outer:4d} | objective {current:+.10f} | clock {clock.total:.0f}s", flush=True)
    log.close()
    return clock, outer, "objective window" if converged else "stalls" if stalls >= 5 else "time cap"


def device_of():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def rise_time(T, mat):
    """Activation time on the rise where Ct(t) = T / T_MAX (exact: on the rise Ct does not depend on ls)."""
    return mat.t0 / np.pi * torch.arccos(1 - 2 * torch.clamp(T / mat.T_max, 0.0, 1.0))


def load_grid(cfg, n, device, n_t=None, n_negative=0):
    """LoadPINN L-BFGS loads -> (P, t): n log-spaced pressures in [0, P_MAX] (scale GRID_P0, dense at small P), with
    windkessel also n_negative log-spaced in [P_MIN, 0) with the same scale, x activation times
    - rise: the exact times of n_t (default n) tension levels log-spaced in [0, T_MAX] (includes t = 0);
    - relaxation: round(GRID_RELAX_FACTOR n_t) times uniform on (t0, GRID_RELAX_END_MS];
    without (P, t) = (0, 0). Passive times need no grid points: their features equal those of t = 0."""
    mat = Material.of(cfg)
    axis = torch.linspace(0, 1, n, device=device, dtype=torch.float64)
    P_axis = cfg.GRID_P0 * torch.expm1(axis * np.log1p(cfg.P_MAX / cfg.GRID_P0))
    axis = torch.linspace(0, 1, n_t or n, device=device, dtype=torch.float64)
    T_levels = cfg.GRID_T0 * torch.expm1(axis * np.log1p(cfg.T_MAX / cfg.GRID_T0))
    n_relax = int(round(cfg.GRID_RELAX_FACTOR * len(T_levels)))
    relax = torch.linspace(mat.t0, cfg.GRID_RELAX_END_MS, n_relax + 1, device=device, dtype=torch.float64)[1:]
    t_axis = torch.cat((rise_time(T_levels, mat), relax))
    if not n_negative:
        P_, t_ = (v.reshape(-1)[1:].float() for v in torch.meshgrid(P_axis, t_axis, indexing="ij"))
        return P_, t_
    axis = torch.linspace(1, 0, n_negative + 1, device=device, dtype=torch.float64)[:-1]  # from P_MIN up to 0 (excluded)
    P_axis = torch.cat((-cfg.GRID_P0 * torch.expm1(axis * np.log1p(-cfg.P_MIN / cfg.GRID_P0)), P_axis))
    P_, t_ = (v.reshape(-1) for v in torch.meshgrid(P_axis, t_axis, indexing="ij"))
    loaded = (P_ != 0) | (t_ != 0)  # all pairs except the unloaded state
    return P_[loaded].float(), t_[loaded].float()


def load_random(cfg, m, ramp, device):
    """LoadPINN Adam loads: P uniform in ramp [P_MIN, P_MAX] (P_MIN = 0 without negative pressures); activation time on
    the rise from a tension level uniform in [0, ramp T_MAX] (35 %), on the relaxation uniform in [END - ramp (END - t0),
    END] (35 %, small tension first; END = GRID_RELAX_END_MS) or passive, uniform in [END, BCL] (30 %)."""
    mat = Material.of(cfg)
    r = torch.rand(m, device=device)
    u = torch.rand(m, device=device)
    rise = rise_time(torch.rand(m, device=device) * ramp * cfg.T_MAX, mat)
    relax = cfg.GRID_RELAX_END_MS - torch.rand(m, device=device) * ramp * (cfg.GRID_RELAX_END_MS - mat.t0)
    passive = cfg.GRID_RELAX_END_MS + torch.rand(m, device=device) * (cfg.BCL_MS - cfg.GRID_RELAX_END_MS)
    t = torch.where(u < 0.35, rise, torch.where(u < 0.7, relax, passive))
    if not hasattr(cfg, "P_MIN"):
        return r * ramp * cfg.P_MAX, t
    floor = torch.full_like(t, cfg.P_MIN)
    return r * ramp * (cfg.P_MAX - floor) + ramp * floor, t


def train_load_network(cfg):
    """Step `train` of volume_constrain / windkessel: OUTPUT_DIR/model.pt, train_log.csv, train_info.json."""
    torch.manual_seed(cfg.SEED)
    device = device_of()
    out_file = cfg.output / "model.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    g = geometry(cfg)
    p2 = p2_mesh(g, cfg.QUAD_ORDER)
    net = LoadPINN.of(cfg, p2, g).to(device)
    geo = Geometry(g, p2, device)
    material = Material.of(cfg)
    print(f"mesh {len(g['cells'])} tets, {len(p2['dof_points'])} P2 DOFs ({p2['n_vertices']} vertices), {geo.Q} points per "
          f"tet | {sum(p.numel() for p in net.parameters())} weights | device {device}", flush=True)

    def total_energy(U, P, T, kappa):
        mat = material if kappa is None else replace(material, kappa=kappa)
        return energy(U, geo, mat, P, T, cfg.RIGID_PENALTY)["total"]

    def grid_loads(stage, index):  # (n, first outer step) or (n_P, n_T, first outer step)
        n_negative = cfg.GRID_NEGATIVE[index] if hasattr(cfg, "GRID_NEGATIVE") else 0
        return load_grid(cfg, stage[0], device, stage[1] if len(stage) == 3 else None, n_negative)

    clock, outer, stopped_by = optimize(cfg, net, device, total_energy, lambda m, ramp: load_random(cfg, m, ramp, device),
                                        grid_loads, cfg.KAPPA_START, cfg.KAPPA, geo.wall_volume,
                                        out_file.parent / "train_log.csv")
    torch.save({"config": net.config, "state_dict": net.state_dict(), "material": vars(material).copy(),
                "quad_order": cfg.QUAD_ORDER, "train_clock_s": clock.total, "lbfgs_outer": outer,
                "fibers": model.fiber_config(cfg), "date": time.strftime("%Y-%m-%d %H:%M")}, out_file)
    info = {"train_clock_s": round(clock.total, 1), "lbfgs_outer_steps": outer, "adam_steps": cfg.ADAM_STEPS,
            "stopped_by": stopped_by, "whiten": cfg.WHITEN, "whiten_from_outer": cfg.WHITEN_FROM_OUTER,
            "grid_stages": [list(s) for s in cfg.GRID_STAGES], "lbfgs_history": cfg.LBFGS_HISTORY,
            "device": str(device), "weights": sum(p.numel() for p in net.parameters())}
    (out_file.parent / "train_info.json").write_text(json.dumps(info, indent=1))
    print(f"saved {out_file}: {json.dumps(info)}", flush=True)


def param_setup(cfg, device):
    g = geometry(cfg)
    p2 = p2_mesh(g, cfg.QUAD_ORDER)
    geo = Geometry(g, p2, device)
    geo.families = param_fibers(cfg, geo)
    return g, p2, geo


def param_grid(cfg, n_p, n_a, device):
    """ParamPINN L-BFGS loads: n_p x n_a log-spaced (p, a) (scales GRID_P0_N, GRID_A0_N, dense at small loads) without
    the unloaded state."""
    ax = lambda n: torch.linspace(0, 1, n, device=device, dtype=torch.float64)  # noqa: E731
    p_axis = cfg.GRID_P0_N * torch.expm1(ax(n_p) * np.log1p(cfg.P_MAX_N / cfg.GRID_P0_N))
    a_axis = cfg.GRID_A0_N * torch.expm1(ax(n_a) * np.log1p(cfg.A_MAX_N / cfg.GRID_A0_N))
    p, a = (v.reshape(-1) for v in torch.meshgrid(p_axis, a_axis, indexing="ij"))
    keep = (p > 0) | (a > 0)
    return p[keep].float(), a[keep].float()


def param_random(cfg, m, ramp, device):
    """ParamPINN Adam loads: p and a log-uniform-like (dense at small loads); a = 0 for a quarter of them."""
    r1, r2 = torch.rand(m, device=device), torch.rand(m, device=device)
    p = ramp * cfg.GRID_P0_N * torch.expm1(r1 * np.log1p(cfg.P_MAX_N / cfg.GRID_P0_N))
    a = ramp * cfg.GRID_A0_N * torch.expm1(r2 * np.log1p(cfg.A_MAX_N / cfg.GRID_A0_N)) * (torch.rand(m, device=device) < 0.75)
    return p, a


def train_param_network(cfg):
    """Step `train` of invert_para: CACHE_DIR/model/ (model.pt, train_log.csv, train_info.json)."""
    out = cfg.cache / "model"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.SEED)
    device = device_of()
    g, p2, geo = param_setup(cfg, device)
    net = ParamPINN.of(cfg, p2, g).to(device)
    base = Material.in_units_of_C(cfg)
    print(f"mesh {len(g['cells'])} tets, {len(p2['dof_points'])} P2 DOFs | {sum(q.numel() for q in net.parameters())} weights "
          f"| device {device}", flush=True)

    def total_energy(U, p, a, kappa):
        return param_energy(cfg, U, geo, base, p, a, cfg.KAPPA_OVER_C if kappa is None else kappa)[0]

    def grid_loads(stage, index):  # (n_p, n_a, first outer step)
        return param_grid(cfg, stage[0], stage[1], device)

    def random_loads(m, ramp):
        return param_random(cfg, m, ramp, device)

    clock, outer, stopped_by = optimize(cfg, net, device, total_energy, random_loads, grid_loads, cfg.KAPPA_START_OVER_C,
                                        cfg.KAPPA_OVER_C, geo.wall_volume, out / "train_log.csv")
    torch.save({"config": net.config, "state_dict": net.state_dict(), "outer": outer, "train_clock_s": clock.total}, out / "model.pt")
    info = {"train_clock_s": round(clock.total, 1), "lbfgs_outer_steps": outer, "stopped_by": stopped_by,
            "grid_stages": cfg.GRID_STAGES, "domain": {"p": cfg.P_MAX_N, "a": cfg.A_MAX_N}, "fibers": model.fiber_shares(cfg),
            "device": str(device)}
    (out / "train_info.json").write_text(json.dumps(info, indent=1))
    print("saved", out / "model.pt", json.dumps(info), flush=True)


# ==================== 7. trained networks ====================
def load_model(cfg, device):
    """LoadPINN of OUTPUT_DIR/model.pt: (net, geometry, p2, material, geometry() data, checkpoint)."""
    import inspect
    checkpoint = torch.load(cfg.output / "model.pt", map_location=device, weights_only=False)
    trained = checkpoint.get("fibers", {"angles_deg": [[0.0, 0.0]], "weights": [1.0]})  # older models: one family 0 / 0
    if trained != model.fiber_config(cfg):
        raise ValueError(f"model trained with fibers {trained}, params.py has {model.fiber_config(cfg)}")
    g = geometry(cfg)
    p2 = p2_mesh(g, checkpoint.get("quad_order", cfg.QUAD_ORDER))
    known = inspect.signature(LoadPINN).parameters
    net = LoadPINN(p2, g["base_origin"], g["base_normal"], **{k: v for k, v in checkpoint["config"].items() if k in known})
    net = net.to(device)
    missing = net.load_state_dict(checkpoint["state_dict"], strict=False)  # models trained before the whitening: A = I
    assert not missing.unexpected_keys and set(missing.missing_keys) <= {"whiten_A"}, missing
    spring = checkpoint["material"].get("k_spring", 0.0)
    if spring != cfg.K_SPRING:
        raise ValueError(f"model trained with K_SPRING = {spring}, params.py has {cfg.K_SPRING}: retrain")
    mat = Material(**{k: v for k, v in checkpoint["material"].items() if k in Material.__dataclass_fields__})
    return net.eval(), Geometry(g, p2, torch.device(device)), p2, mat, g, checkpoint


def load_param_model(cfg, device):
    """ParamPINN of CACHE_DIR/model/model.pt: (net, geometry() data, p2, geometry)."""
    g, p2, geo = param_setup(cfg, device)
    ck = torch.load(cfg.cache / "model" / "model.pt", map_location=device, weights_only=False)
    # models stored before the (endo, epi) angles have one angle per family: the same angle at every depth
    fam = [(((float(a), float(a)) if np.ndim(a) == 0 else tuple(map(float, a))), float(w)) for a, w in ck["config"]["fibers"]]
    if fam != [(tuple(a), float(w)) for a, w in model.fiber_shares(cfg)]:
        raise ValueError(f"model trained for fibers {fam}, params.py has {model.fiber_shares(cfg)}")
    spring = float(ck["config"].get("k_spring_over_c", 0.0))
    if spring != cfg.K_SPRING_OVER_C:
        raise ValueError(f"model trained with K_SPRING_OVER_C = {spring}, params.py has {cfg.K_SPRING_OVER_C}: retrain")
    net = ParamPINN.of(cfg, p2, g).to(device)
    net.config.update(ck["config"])
    net.load_state_dict(ck["state_dict"])
    return net.eval(), g, p2, geo


def predict(net, geo, theta, P_kPa, ct, n_vertices):
    """ParamPINN: vertex displacement (um, numpy) and cavity volume (nL) for theta = (C, T_max) at pressure P and
    activation level Ct."""
    C, T = theta
    dev = geo.dof_points.device
    p = torch.tensor([float(P_kPa / C)], dtype=torch.float32, device=dev)
    a = torch.tensor([float(T * ct / C)], dtype=torch.float32, device=dev)
    with torch.no_grad():
        U = net(p, a)
        V = geo.cavity_volume(U)[0].item() * 1e3
    return U[0, :n_vertices].cpu().numpy() * 1e3, V


# ==================== 8. stresses of a state (LoadPINN) ====================
def recover_pressure(U, geo, mat, P, t, g, p2):
    """P1 nodal hydrostatic pressure q (n_vertices,) balancing the P2 weak residual of displacement U (1, n_dofs, 3).

    R0 = gradient w.r.t. the nodal displacements of the energy without the incompressibility and rigid terms;
    G[3I+i, b] = sum_q w_q J_q (F_q^-T grad N_I(q))_i N_b(q); q = argmin |R0 - G q| over the rows that are not base-normal
    reactions (normal equations, sparse direct solve).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    U = U.detach().clone().requires_grad_(True)
    P_, t_ = torch.tensor([P], dtype=U.dtype, device=U.device), torch.tensor([t], dtype=U.dtype, device=U.device)
    e = energy(U, geo, mat, P_, t_, rigid_penalty=0.0, incompressibility=False)["total"].sum()
    R0 = torch.autograd.grad(e, U)[0][0].detach().cpu().double().numpy().ravel()
    with torch.no_grad():
        F = (torch.eye(3, device=U.device) + geo.gradients(U))[0].cpu().double().numpy()
    J = np.linalg.det(F)
    E, Q = geo.E, geo.Q
    dof_ids = np.repeat(p2["cell_dofs"], Q, 0)  # (E*Q, 10)
    grad_N = p2["grad"].reshape(E * Q, 10, 3)
    force = (p2["W"].ravel() * J)[:, None, None] * np.einsum("qij,qaj->qai", np.linalg.inv(F).transpose(0, 2, 1), grad_N)
    rows = (3 * dof_ids[:, :, None] + np.arange(3)).reshape(E * Q, 30)
    pN = np.tile(p2["lam"], (E, 1))
    pc = np.repeat(g["cells"], Q, 0)
    n_dofs, nv = len(p2["dof_points"]), p2["n_vertices"]
    G = sp.csr_matrix(((force.reshape(E * Q, 30, 1) * pN[:, None, :]).ravel(),
                       (np.repeat(rows[:, :, None], 4, 2).ravel(), np.repeat(pc[:, None, :], 30, 1).ravel())),
                      shape=(3 * n_dofs, nv))
    n = g["base_normal"] / np.linalg.norm(g["base_normal"])
    keep = sp.identity(3 * n_dofs, format="lil")
    for a in p2["base_dofs"]:
        keep[3 * a:3 * a + 3, 3 * a:3 * a + 3] = np.eye(3) - np.outer(n, n)
    keep = keep.tocsr()
    A, b = keep @ G, keep @ R0
    return spsolve((A.T @ A).tocsc(), A.T @ b)


def cell_fields(U, geo, mat, t, q):
    """Cell averages over the quadrature points (FE: DG0 projections) of sigma, F, sigma_ff, p and Ct, as numpy; and the
    wall-mean tension T_MAX Ct."""
    with torch.no_grad():
        F = torch.eye(3, device=U.device) + geo.gradients(U)[0]
        J = torch.linalg.det(F)
        E = green_lagrange(F)
        tt = torch.tensor(t, dtype=F.dtype, device=U.device)  # activation time t (ms since onset)
        lams = [fiber_stretch(F, f) for f in geo.fibers]
        cts = [ct_field(tt, lam, mat) for lam in lams]
        ct = mix(geo.fiber_weights, cts)  # share-weighted activation (one family: its Ct)
        S = mix(geo.fiber_weights, [passive_stress(E, f, geo.s0, n, mat) + active_stress(lam, tt, f, mat)
                                    for f, n, lam in zip(geo.fibers, geo.normals, lams)])
        qn = torch.as_tensor(q, dtype=F.dtype, device=F.device)
        p = (qn[geo.projection["cells"]] * geo.projection["N"]).sum(-1)
        sigma = F @ S @ F.transpose(-1, -2) / J[:, None, None] - p[:, None, None] * torch.eye(3, device=F.device)

        def fiber_stress(f0):
            fib = (F @ f0[..., None]).squeeze(-1)
            fib = fib / fib.norm(dim=-1, keepdim=True)
            return (fib * (sigma @ fib[..., None]).squeeze(-1)).sum(-1)
        sff = fiber_stress(geo.f0)  # family 1
        W = geo.W.reshape(geo.E, geo.Q)

        def avg(a):
            a = a.reshape(geo.E, geo.Q, *a.shape[1:])
            w = W.reshape(geo.E, geo.Q, *([1] * (a.dim() - 2)))
            return ((a * w).sum(1) / w.sum(1)).cpu().numpy().astype(np.float64)
        T_mean = float(mat.T_max * (geo.W * ct).sum() / geo.W.sum())
        out = {"sigma": avg(sigma), "F": avg(F), "sigma_ff": avg(sff), "p": avg(p), "Ct": avg(ct)}
        if len(geo.fibers) == 2:  # family 2 fiber stress and the activation of each family
            out.update(sigma_ff2=avg(fiber_stress(geo.fibers[1])), Ct_f1=avg(cts[0]), Ct_f2=avg(cts[1]))
        return out, T_mean


def state_fields(net, geo, mat, g, p2, P, t, device):
    """Network displacement at (P, t), its cavity volume, the vertex displacements, and the cell fields of the state."""
    with torch.no_grad():
        U = net(torch.tensor([P], device=device), torch.tensor([t], device=device))
        V = geo.cavity_volume(U)[0].item()
    u = U[0, :p2["n_vertices"]].cpu().numpy().astype(np.float64)  # vertex displacements (= the FE's P1 interpolation of P2)
    q = recover_pressure(U, geo, mat, P, t, g, p2)
    f, T = cell_fields(U, geo, mat, t, q)
    return V, u, f, T
