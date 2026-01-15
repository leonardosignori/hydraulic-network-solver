# hydraulic-network-solver
# Hydraulic 0‑D Network Solver (Python)

Steady-state hydraulic network solver for incompressible flow networks including pipes, pumps, and (optional) check valves. It computes branch flow rates and node pressures by solving mass balance at nodes and energy balance across branches. 

## Physics model

This solver is a 0‑D network model:

- **Mass balance (continuity) at nodes**  
  For each node with unknown pressure, the net flow must balance external demand/injection:
  - Positive external `Q` means flow leaving the node (demand).
  - The solver enforces: sum(in) − sum(out) + Q_ext = 0.  
  (Sign convention matches the implementation.)  
- **Energy balance on each branch** (written in pressure units, Pa)  
  For each branch:
  \[
  P_from - P_to + ρ g (z_from - z_to) - Δp_model(Q) = 0
  \]
  where `z` is elevation (m).

### Pipe pressure drop
For a pipe (and also for a check valve when open), the model uses Darcy–Weisbach in the form:

- Velocity: `V = Q / A`
- Reynolds: `Re = ρ |V| D / μ`
- Friction factor: **Churchill correlation**
- Pressure loss:
  \[
  Δp = \frac{1}{2} ρ ( f \frac{L}{D} + k_{local}) |V| V
  \]

To avoid numerical issues at very small flow, the solver uses a smooth approximation of `|V|` near 0.

### Pump pressure rise
A pump is modeled by a head curve `H(Q)` (meters). The pressure change is:
\[
Δp = - ρ g H(Q)
\]
The curve is interpolated using a monotone **PCHIP** spline and is clamped to constant head outside the provided Q-range (plateaus).

### Check valve (optional)
The `check` branch type is treated as a one-way element by mapping the internal flow variable to a nonnegative “physical” flow using a smooth rectifier (`softplus`), so the solver remains differentiable.

## How the code works

Main components:

- `Fluid`: density `rho` (kg/m³), viscosity `mu` (Pa·s)
- `Node`: nodes can be:
  - `pressure`: fixed pressure boundary (given `P`), or you can provide `H` and it will be converted to `P = ρ g (H - z)`
  - `junction`: unknown pressure node (typical internal node)
  - `flow`: imposes external flow `Q` (demand/injection), pressure is unknown unless `P` is also set
  - `mixed`: supports both `P` and `Q` boundary-style use (see JSON behavior)
- `Branch`: branches can be:
  - `pipe` with `L`, `DN` (mm), optional `epsilon`, optional `k_local`
  - `pump` with a curve `points: [[Q, H], ...]`
  - `check` (like pipe, but flow is rectified)

### Unknowns and equations
The solver builds one unknown `Q` per branch and one unknown `P` per “unknown-pressure” node. It then solves:
- One energy equation per branch
- One mass balance equation per unknown-pressure node

### Numerical method
Two solution paths are available:
- **Dense**: `scipy.optimize.root(...)` with optional analytic Jacobian.
- **Sparse**: custom Newton iterations using a sparse Jacobian and a simple backtracking line search (`--sparse`).

Residual scaling is applied automatically using characteristic pressure/flow scales for better conditioning.

## Input format (JSON)

Example: `ex1.json`

Top-level keys:
- `fluid`: `{ "rho": ..., "mu": ... }`
- `nodes`: dictionary keyed by node id
- `branches`: list of branch objects

### Nodes
Each node can include:
- `type`: `"junction" | "pressure" | "flow" | "mixed"`
- `P`: pressure in Pa (optional)
- `H`: head in m (optional alternative to P; if provided and P is missing, P is derived)
- `Q`: external flow in m³/s (optional; used mainly for `flow` / `mixed`)
- `z`: elevation in m (optional, default 0)

### Branches
Each branch includes:
- `id`, `from`, `to`, `type`
- For `pipe` / `check`:
  - `L` (m)
  - `DN` (mm)
  - `k_local` (optional)
  - `epsilon` (optional)
- For `pump`:
  - `curve.points`: array of `[Q, H]` pairs (Q in m³/s, H in m)

## Run the example
python solver.py ex1.json

Optional flag: --sparse; --no-analytic; --tol=1e-9
