from __future__ import annotations

"""Hydraulic 0-D network solver
- Mass balance at nodes; energy balance on branches.
- Branch types: pipe | pump | check   (check is a smooth one-way branch, P0).
- Node types: junction | pressure | flow | mixed.
- Robust friction: Churchill; smooth behavior near Q≈0.
- Monotone pump curve via PCHIP with C¹ plateaus.
- Optional sparse Jacobian + Newton/backtracking (--sparse).

Leonardo Signori
"""

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root

# NEW: sparse imports
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ------------------------- costanti globali -------------------------
G = 9.81                         # [m/s²]
DEFAULT_EPS = 4.5e-5             # roughness [m]
Q_EPS = 1e-9                     # threshold flow ≈ 0 for regularization

# [P0] Utilities: smooth abs, sign, Churchill friction
def smooth_abs(x: float, eps: float) -> float:
    return math.sqrt(x*x + eps*eps)

def sign(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0

def friction_factor_churchill(Re: float, eps_over_D: float) -> float:
    if Re <= 0.0:
        Re = 1e-16
    A = (2.457 * math.log((7.0/Re)**0.9 + 0.27 * eps_over_D))**16
    B = (37530.0/Re)**16
    return 8.0 * ((8.0/Re)**12 + 1.0/((A + B)**1.5))**(1.0/12.0)

# NEW (P0): smooth rectifier used by "check" branch
def softplus(x: float, beta: float = 40.0) -> float:
    t = beta * x
    if t > 20.0:    # avoid overflow, becomes ~identity
        return x
    if t < -20.0:   # avoid underflow, becomes ~0
        return 0.0
    return math.log1p(math.exp(t)) / beta

def dsoftplus(x: float, beta: float = 40.0) -> float:
    t = beta * x
    if t > 20.0:
        return 1.0
    if t < -20.0:
        return 0.0
    e = math.exp(-t)
    return 1.0 / (1.0 + e)   # sigmoid(beta*x)

# ===================================================================
# Curva pompa: spline H(Q)
# ===================================================================
class PumpCurve:
    """Monotone H(Q) with PCHIP and C¹ plateaus beyond catalog."""
    def __init__(self, points: List[List[float]]):
        if len(points) < 2:
            raise ValueError("Pump curve requires at least 2 (Q,H) points.")
        q, h = zip(*sorted(points))
        # enforce non-increasing H
        for i in range(1, len(h)):
            if h[i] > h[i-1] + 1e-6:
                raise ValueError("Pump curve must be non-increasing in H(Q).")
        self.Qmin, self.Qmax = float(q[0]), float(q[-1])
        self.Hmax, self.Hmin = float(h[0]), float(h[-1])
        self._p = PchipInterpolator(q, h, extrapolate=True)

    def __call__(self, Q: float) -> float:
        if Q <= self.Qmin: return self.Hmax
        if Q >= self.Qmax: return self.Hmin
        return float(self._p(Q))

    def dHdQ(self, Q: float) -> float:
        if Q <= self.Qmin or Q >= self.Qmax: return 0.0
        return float(self._p.derivative()(Q))

# ===================================================================
# Data-classes base
# ===================================================================
@dataclass
class Fluid:
    rho: float   # densità [kg/m³]
    mu: float    # viscosità dinamica [Pa·s]
    g: float = G

@dataclass
class Node:
    id: str
    type: str             # Allowed: junction | pressure | flow | mixed
    P: float | None = None  # Pressure [Pa] if fixed (or derived from H)
    Q: float | None = None  # External demand/injection [m^3/s] (positive = leaving node)
    z: float = 0.0          # Geometric elevation [m]

@dataclass
class Branch:
    id: str
    from_node: str
    to_node: str
    type: str             # pipe | pump | check
    params: Dict[str, object] = field(default_factory=dict)
    curve: PumpCurve | None = None

    # ------------ Helper -------------------------------------------
    def is_pipe(self): return self.type == "pipe"
    def is_pump(self): return self.type == "pump"
    def is_check(self): return self.type == "check"

    # ------------ ΔP(Q) --------------------------------------------
    def delta_p(self, Q: float, f: Fluid) -> float:
        """Caduta/guadagno di pressione lungo il ramo.
        Segno positivo → perdita di carico (P_from > P_to).
        For 'check': Δp is same model as pipe, but Q must be the rectified (nonnegative) physical flow.
        """
        if self.is_pipe() or self.is_check():
            L, D = self.params["L"], self.params["DN"] / 1000.0
            eps = self.params.get("epsilon", DEFAULT_EPS)
            k = self.params.get("k_local", 0.0)
            A = math.pi * D ** 2 / 4
            V = Q / A
            Vabs = smooth_abs(V, 1e-9)  # [P0] smooth near 0
            Re = f.rho * Vabs * D / f.mu
            fch = friction_factor_churchill(Re, eps / D)
            K = 0.5 * f.rho * (fch * L / D + k)
            return K * Vabs * V

        if self.is_pump():
            H = self.curve(Q)
            return -f.rho * f.g * H

        raise ValueError(self.type)

    # --- Derivative d(Δp)/dQ ---
    def d_delta_p(self, Q: float, f: Fluid) -> float:
        if self.is_pipe() or self.is_check():
            L, D = self.params["L"], self.params["DN"] / 1000.0
            eps = self.params.get("epsilon", DEFAULT_EPS)
            k = self.params.get("k_local", 0.0)
            A = math.pi * D ** 2 / 4
            V = Q / A
            Vabs = smooth_abs(V, 1e-9)
            Re = f.rho * Vabs * D / f.mu
            fch = friction_factor_churchill(Re, eps / D)
            K = 0.5 * f.rho * (fch * L / D + k)
            dV_dQ = 1.0 / A
            dVabs_dQ = (V * dV_dQ) / max(Vabs, 1e-16)
            return K * (dVabs_dQ * V + Vabs * dV_dQ)

        if self.is_pump():
            return -f.rho * f.g * self.curve.dHdQ(Q)

        raise ValueError(self.type)

# ===================================================================
# Rete idraulica
# ===================================================================
class HydraulicNetwork:
    # [P1] auto-scale helper
    def _choose_default_scales(self):
        # ---- Flow scale (m^3/s)
        if self._Qext:
            Q_char = sum(abs(q) for q in self._Qext.values())
        else:
            Q_char = 1e-4  # safe default when no external flows are specified

        # ---- Pressure scale (Pa)
        rho_g = self.f.rho * self.f.g

        # Hydrostatic range from elevations
        z_vals = [n.z for n in self.nodes.values()]
        dz = (max(z_vals) - min(z_vals)) if z_vals else 0.0
        Pz = abs(rho_g * dz)

        # Max pump head present in the network (in meters) -> convert to Pa
        Hpump = 0.0
        for b in self.br:
            if b.is_pump() and hasattr(b, "curve") and b.curve is not None:
                # PumpCurve carries Hmax/Hmin already
                Hpump = max(Hpump, getattr(b.curve, "Hmax", 0.0))
        Ppump = rho_g * Hpump
        # Choose a conservative pressure scale
        P_char = max(1e4, Pz, Ppump)
        print(f"[LOG] Auto-scales: P_scale={P_char:.1e} Pa, Q_scale={Q_char:.1e} m³/s", file=sys.stderr)
        return P_char, Q_char


    # -------------------- checks ---------------------------
    def mass_balance_residuals(self, Q: dict) -> dict:
        """Compute mass residuals at each *unknown-pressure* node:
        residual = sum_in - sum_out + Qext  (mirrors solver sign convention).
        Excludes pressure/mixed nodes from the report, as they act as boundaries.
        """
        res_all = {nid: 0.0 for nid in self.nodes}
        for b in self.br:
            q = Q[b.id]
            res_all[b.from_node] -= q
            res_all[b.to_node] += q
        for nid, qext in self._Qext.items():
            res_all[nid] += qext
        # keep only nodes that are in the unknown set (junction/flow)
        keep = set(n.id for n in self._unk_nd)
        return {nid: val for nid, val in res_all.items() if nid in keep}

    def energy_balance_residuals(self, Q: dict, P: dict) -> dict:
        """Compute per-branch energy residual in pressure units [Pa]:
        r_b = P_from - P_to + rho g (z_from - z_to) - Δp_model(Q)."""
        res = {}
        for b in self.br:
            zf = self.nodes[b.from_node].z
            zt = self.nodes[b.to_node].z
            res[b.id] = P[b.from_node] - P[b.to_node] + self.f.rho * self.f.g * (zf - zt) - b.delta_p(Q[b.id], self.f)
        return res

    """Solver stazionario 0-D con scaling residui, Jacobiano analitico, e opzione sparsa."""
    def __init__(self, fluid: Fluid, nodes: Dict[str, Node], branches: List[Branch]):
        self.f, self.nodes, self.br = fluid, nodes, branches
        self._iQ = {b.id: i for i, b in enumerate(branches)}
        self._unk_br = branches
        self._unk_nd = [
            n for n in nodes.values()
            if (n.type in ("junction","flow","mixed") and n.P is None)
        ]
        self._iP = {n.id: i for i, n in enumerate(self._unk_nd)}
        self._Pfix = {n.id: n.P for n in nodes.values() if n.P is not None}
        self._Qext = {
            n.id: n.Q for n in nodes.values()
            if (n.type in ("flow","mixed") and n.Q is not None)
        }

    # -- JSON parsing --
    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        fluid = Fluid(**d["fluid"])
        # --- nodes ---
        nodes_raw = d["nodes"]
        nodes = {}
        for nid, nd in nodes_raw.items():
            nd = dict(nd)
            t = str(nd.get("type", "junction")).lower()
            if t not in ("junction","pressure","flow","mixed"):
                raise ValueError(f"Unsupported node type: {t}")
            nd["type"] = t
            # Elevation (m)
            nd["z"] = float(nd.get("z", 0.0))
            # Accept H (head) input; derive P if only H provided
            H = nd.pop("H", None)
            if H is not None and nd.get("P") is None:
                nd["P"] = fluid.rho * fluid.g * (float(H) - nd["z"])
            nodes[nid] = Node(id=nid, **nd)
        # --- branches ---
        brs: List[Branch] = []
        for b in d["branches"]:
            tp = b["type"].lower()
            if tp not in ("pipe","pump","check"):
                raise ValueError(f"Unsupported branch type: {tp}")
            if tp == "pump":
                if "curve" not in b or "points" not in b["curve"]:
                    raise ValueError(f"La pompa '{b['id']}' richiede una curva con punti (Q,H).")
                curve = PumpCurve(b["curve"]["points"])
            else:
                curve = None
            params = {k: v for k, v in b.items() if k not in ("id", "from", "to", "type", "curve")}
            brs.append(Branch(id=b["id"], from_node=b["from"], to_node=b["to"], type=tp, params=params, curve=curve))
        return cls(fluid, nodes, brs)

    # -------------------- stima iniziale ---------------------------
    def build_initial_guess(self, Q_seed: float = 0.02, P_seed: float = 2.0e5) -> np.ndarray:
        nQ, nP = len(self._unk_br), len(self._unk_nd)
        return np.hstack([np.full(nQ, Q_seed), np.full(nP, P_seed)])

    # -------------------- residual & jac ---------------------------
    def _residual(self, x: np.ndarray, *, P_scale: float, Q_scale: float, scale: bool) -> np.ndarray:
        nQ = len(self._unk_br)
        Qv, Pv = x[:nQ], x[nQ:]
        # Build physical flows: rectified for "check"
        Qphys = {}
        for b in self._unk_br:
            qhat = Qv[self._iQ[b.id]]
            if b.is_check():
                beta = float(b.params.get("beta", 40.0))
                Qphys[b.id] = softplus(qhat, beta)
            else:
                Qphys[b.id] = qhat
        P = {**self._Pfix, **{nid: Pv[idx] for nid, idx in self._iP.items()}}
        r = np.zeros_like(x)
        # energia (one per branch) using physical flows
        for b in self._unk_br:
            zf = self.nodes[b.from_node].z
            zt = self.nodes[b.to_node].z
            r[self._iQ[b.id]] = P[b.from_node] - P[b.to_node] + self.f.rho * self.f.g * (zf - zt) - b.delta_p(Qphys[b.id], self.f)
        # massa (one per unknown-P node) using physical flows
        for n in self._unk_nd:
            mass = self._Qext.get(n.id, 0.0)
            for b in self._unk_br:
                q = Qphys[b.id]
                if b.from_node == n.id:
                    mass -= q
                elif b.to_node == n.id:
                    mass += q
            r[nQ + self._iP[n.id]] = mass
        if scale:
            r[:nQ] /= P_scale
            r[nQ:] /= Q_scale
        return r

    def _jac_dense(self, x: np.ndarray, *, P_scale: float, Q_scale: float, scale: bool) -> np.ndarray:
        nQ, nP = len(self._unk_br), len(self._unk_nd)
        N = nQ + nP
        J = np.zeros((N, N))
        # mapping slopes and local Qphys
        dmap = np.ones(nQ)
        Qloc = np.zeros(nQ)
        for b in self._unk_br:
            i = self._iQ[b.id]
            qhat = x[i]
            if b.is_check():
                beta = float(b.params.get("beta", 40.0))
                dmap[i] = dsoftplus(qhat, beta)
                Qloc[i] = softplus(qhat, beta)
            else:
                dmap[i] = 1.0
                Qloc[i] = qhat
        # energia rows
        for b in self._unk_br:
            i = self._iQ[b.id]
            d_dp_dQ = b.d_delta_p(Qloc[i], self.f)   # derivative wrt physical Q
            J[i, i] = - d_dp_dQ * dmap[i]            # chain rule
            frm, to = b.from_node, b.to_node
            if frm in self._iP:
                J[i, nQ + self._iP[frm]] = 1.0
            if to in self._iP:
                J[i, nQ + self._iP[to]] = -1.0
        # massa rows (incidence ± dmap for check)
        for n in self._unk_nd:
            row = nQ + self._iP[n.id]
            for b in self._unk_br:
                col = self._iQ[b.id]
                coef = dmap[col] if b.is_check() else 1.0
                if b.from_node == n.id:
                    J[row, col] = -coef
                elif b.to_node == n.id:
                    J[row, col] = +coef
        if scale:
            S = np.hstack([np.full(nQ, 1.0 / P_scale), np.full(nP, 1.0 / Q_scale)])
            J *= S[:, None]
        return J

    # NEW: sparse Jacobian assembly (CSR)
    def _jac_sparse(self, x: np.ndarray, *, P_scale: float, Q_scale: float, scale: bool) -> csr_matrix:
        nQ, nP = len(self._unk_br), len(self._unk_nd)
        I: List[int] = []
        J: List[int] = []
        V: List[float] = []
        # mapping slopes and local Qphys
        dmap = np.ones(nQ)
        Qloc = np.zeros(nQ)
        for b in self._unk_br:
            i = self._iQ[b.id]
            qhat = x[i]
            if b.is_check():
                beta = float(b.params.get("beta", 40.0))
                dmap[i] = dsoftplus(qhat, beta)
                Qloc[i] = softplus(qhat, beta)
            else:
                dmap[i] = 1.0
                Qloc[i] = qhat
        # energia rows
        for b in self._unk_br:
            i = self._iQ[b.id]
            d_dp_dQ = b.d_delta_p(Qloc[i], self.f)
            val = - d_dp_dQ * dmap[i]
            I.append(i); J.append(i); V.append(val / P_scale if scale else val)
            frm, to = b.from_node, b.to_node
            if frm in self._iP:
                I.append(i); J.append(nQ + self._iP[frm]); V.append(1.0 / P_scale if scale else 1.0)
            if to in self._iP:
                I.append(i); J.append(nQ + self._iP[to]);  V.append(-1.0 / P_scale if scale else -1.0)
        # massa rows
        for n in self._unk_nd:
            row = nQ + self._iP[n.id]
            for b in self._unk_br:
                col = self._iQ[b.id]
                coef = dmap[col] if b.is_check() else 1.0
                val = (-coef) if b.from_node == n.id else ((+coef) if b.to_node == n.id else None)
                if val is not None:
                    I.append(row); J.append(col); V.append(val / Q_scale if scale else val)
        return coo_matrix((V, (I, J)), shape=(nQ + nP, nQ + nP)).tocsr()

    # ---------------------- solve ----------------------------------
    def solve(
        self,
        initial_guess: np.ndarray | None = None,
        *,
        method: str = "lm",
        tol: float = 1e-8,
        scale_residuals: bool = True,
        P_scale: float | None = None,
        Q_scale: float | None = None,
        analytic_jac: bool = True,
        log_physics: bool = True,
        use_sparse: bool = False,          # NEW: enable sparse Jacobian path
        max_iter: int = 50,                # NEW: sparse Newton max iterations
        line_search: bool = True           # NEW: sparse backtracking
    ) -> Dict[str, pd.DataFrame]:
        nQ, nP = len(self._unk_br), len(self._unk_nd)
        x0 = self.build_initial_guess() if initial_guess is None else np.asarray(initial_guess, float)
        if x0.size != nQ + nP:
            raise ValueError("initial_guess ha dimensione sbagliata")
        # [P1] autoscale if None
        if P_scale is None or Q_scale is None:
            P_auto, Q_auto = self._choose_default_scales()
            P_scale = P_auto if P_scale is None else P_scale
            Q_scale = Q_auto if Q_scale is None else Q_scale

        fun = lambda v: self._residual(v, P_scale=P_scale, Q_scale=Q_scale, scale=scale_residuals)

        if not use_sparse:
            # Dense path (SciPy root)
            def attempt(use_jac: bool):
                if use_jac:
                    jac = lambda v: self._jac_dense(v, P_scale=P_scale, Q_scale=Q_scale, scale=scale_residuals)
                    return root(fun, x0, jac=jac, method=method, tol=tol)
                return root(fun, x0, method=method, tol=tol)

            sol = attempt(analytic_jac)
            if not sol.success:
                sol = attempt(False)
            if not sol.success:
                raise RuntimeError(f"Solver failed: {sol.message}")

            x = sol.x
        else:
            # Sparse Newton path with backtracking
            x = x0.copy()
            r = fun(x)
            nrm0 = np.linalg.norm(r)
            for k in range(max_iter):
                J = self._jac_sparse(x, P_scale=P_scale, Q_scale=Q_scale, scale=scale_residuals)
                dx = spsolve(J, -r)
                step = 1.0
                x_trial = x + step * dx
                r_trial = fun(x_trial)
                nrm_trial = np.linalg.norm(r_trial)
                if line_search:
                    nrm = np.linalg.norm(r)
                    # light backtracking (sufficient decrease)
                    while nrm_trial > (0.9 if k > 0 else 1.0) * nrm and step > 1e-3:
                        step *= 0.5
                        x_trial = x + step * dx
                        r_trial = fun(x_trial)
                        nrm_trial = np.linalg.norm(r_trial)
                x, r = x_trial, r_trial
                if nrm_trial < tol * max(1.0, nrm0):
                    break

        Qv, Pv = x[:nQ], x[nQ:]

        # ---------------- logs ------------------------------------
        if log_physics:
            Re_vals = []
            for b in self._unk_br:
                if b.is_pipe() or b.is_check():
                    D = b.params["DN"] / 1000.0
                    A = math.pi * D ** 2 / 4
                    V = Qv[self._iQ[b.id]] / A if abs(Qv[self._iQ[b.id]]) > Q_EPS else 0.0
                    Re_vals.append(self.f.rho * abs(V) * D / self.f.mu)
            if Re_vals:
                print(f"[LOG] max Re = {max(Re_vals):.1f}, min P = {min(Pv) if len(Pv)>0 else min(self._Pfix.values()) :.0f} Pa", file=sys.stderr)

        # Build solution dicts (report raw unknown Qv; for check, physical flow is softplus(Qv))
        flows = []
        for b in self._unk_br:
            qhat = Qv[self._iQ[b.id]]
            qphys = softplus(qhat, float(b.params.get("beta", 40.0))) if b.is_check() else qhat
            flows.append({
                "branch": b.id,
                "from": b.from_node,
                "to": b.to_node,
                "type": b.type,
                "Q_m3_s": qphys,   # report physical (rectified) flow
                "dP_Pa": b.delta_p(qphys, self.f),
                "dz_m": self.nodes[b.from_node].z - self.nodes[b.to_node].z,
            })

        pressures = (
            [{"node": nid, "P_Pa": P, "fixed": True} for nid, P in self._Pfix.items()]
            + [{"node": nid, "P_Pa": Pv[idx], "fixed": False} for nid, idx in self._iP.items()]
        )

        # Checks (use physical flows for diagnostics)
        Qphys_for_checks = {}
        for b in self._unk_br:
            qhat = Qv[self._iQ[b.id]]
            Qphys_for_checks[b.id] = softplus(qhat, float(b.params.get("beta", 40.0))) if b.is_check() else qhat
        Psol = {**self._Pfix, **{nid: Pv[idx] for nid, idx in self._iP.items()}}
        mass_res = self.mass_balance_residuals(Qphys_for_checks)
        energy_res = self.energy_balance_residuals(Qphys_for_checks, Psol)

        if log_physics:
            max_mass = max(abs(v) for v in mass_res.values()) if mass_res else 0.0
            max_energy = max(abs(v) for v in energy_res.values()) if energy_res else 0.0
            print(f"[CHECK] max |mass residual| = {max_mass:.3e} m^3/s", file=sys.stderr)
            print(f"[CHECK] max |energy residual| = {max_energy:.3e} Pa", file=sys.stderr)
        return {
            "flows": pd.DataFrame(flows).set_index("branch"),
            "pressures": pd.DataFrame(pressures).set_index("node"),
        }

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # simple CLI: python DT_v2_updated.py network.json [--sparse] [--no-analytic] [--tol 1e-9]
    if len(sys.argv) < 2:
        print("Usage: python DT_v2_updated.py network.json [--sparse] [--no-analytic] [--tol 1e-9]", file=sys.stderr)
        sys.exit(1)

    json_path = None
    use_sparse = False
    analytic = True
    tol = 1e-8

    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            json_path = arg
        elif arg == "--sparse":
            use_sparse = True
        elif arg == "--no-analytic":
            analytic = False
        elif arg.startswith("--tol"):
            try:
                tol = float(arg.split()[1])  # support "--tol 1e-9"
            except Exception:
                # also support "--tol=1e-9"
                eq = arg.split("=")
                if len(eq) == 2:
                    tol = float(eq[1])

    if json_path is None:
        print("Please provide a JSON file path.", file=sys.stderr)
        sys.exit(1)

    net = HydraulicNetwork.from_json(json_path)
    res = net.solve(log_physics=True, use_sparse=use_sparse, analytic_jac=analytic, tol=tol)
    print("Node pressures (Pa):\n", res["pressures"])
    print("\nBranch flows and drops:\n", res["flows"])
