"""
Wall model for compressible turbulent boundary layers over prism-shaped roughness.
Cogo et al., J. Fluid Mech. (under review) 
"""

import numpy as np

# ── DNS reference values (case M2I) ─────────────────────────────────────────
tauw_DNS = 1.1880873065331360e-2
qw_DNS   = 1.0342713513744434e-2

# ── Matching-point quantities (Adimensional, sampled at y+ ~ 300) ──────────────
y_match  = 6.6080252856e-01   # y_m
u_match  = 1.3441701604e+00   # u_m
T_match  = 1.2562289607e+00   # T_m

# ── Boundary-layer edge and wall quantities (Adimensional) ───────────────────────────────────
T_wall   = 1.179256189862286  # T_w
rho_wall = 0.8490377235127163 # rho_w
T_edge   = 1.0167895877e+00   # T_delta
u_edge   = 2.3429738595e+00   # u_delta

# ── Fluid properties ──────────────────────────────────────────────────────────
Pr       = 0.72               # molecular Prandtl number
Cp       = 3.5                # specific heat at constant pressure c_p

# ── Roughness geometry ────────────────────────────────────────────────────────
k        = 0.12               # roughness height (cubical elements)

# ── Temperature-velocity closure: 'ZHANG', 'WALZ', or 'HUANG' ─────────────
# HUANG = Huang & Coleman (1994), Pre = 0.8 — best performer (Table 4)
qw_method = 'HUANG'

# ── Case label ────────────────────────────────────────────────────────────────
case_name = 'M2I'


def bisection_solver(y_match, u_match, T_match, T_wall, T_edge, u_edge,
                     Pr, Cp, rho_wall, k, qw_method):
    """
    Return (y_vec, u_sol, T_sol, tauw, qw, u_tau, b1, b2).
    Implements the roughness geometry (App. A), compressible log-law (Eq. 6.2),
    and T-u closure (Table 3), solved by bisection on u_tau.
    """

    # ── STEP 1: roughness geometry — iterative solution for u_tau/u_k ────────
    # Parameters for cubical elements with streamwise spacing 2k (App. A)
    Lx    = 2 * k          # streamwise element spacing
    lf    = 1 / 9          # frontal solidity lambda_f = A_f / A_T
    a_min = 0.4            # attenuation factor, unsheltered limit (Eq. A8)
    kappa = 0.41           # von Karman constant

    utau_over_uk = 0.1     # initial guess
    for _ in range(10):
        hs        = max(k - utau_over_uk * Lx, 0)             # shelter height  (Eq. A7)
        a         = a_min / (1 - hs / k)                      # attenuation     (Eq. A8)
        d_over_k  = 1 / (1 - np.exp(-2*a)) - 1 / (2*a)       # d/k             (Eq. A5)
        z0_over_k = (1 - d_over_k) * np.exp(                  # z0/k            (Eq. A6)
            -kappa / np.sqrt(0.5/a * lf * (1 - np.exp(-2*a)))
        )
        d = d_over_k * k
        utau_over_uk = kappa / np.log((k - d) / (z0_over_k * k))  # Eq. 6.3

    # ── STEP 2: T-u closure coefficients b1, b2 ──────────────────────────────
    # T_bar = T_w + b1*u_bar + b2*u_bar^2  (Eq. 4.2 / Table 3)
    if qw_method == 'ZHANG':
        # Zhang et al. (2014) GRA — needs both matching point and edge values
        det_A = u_match * u_edge**2 - u_match**2 * u_edge
        b1 = ((T_match - T_wall)*u_edge**2 - (T_edge - T_wall)*u_match**2) / det_A
        b2 = (u_match*(T_edge - T_wall) - u_edge*(T_match - T_wall)) / det_A
    elif qw_method == 'WALZ':
        # Walz (1969) — recovery factor r = Pr^(1/3)
        rf = Pr ** (1/3)
        b1 = (T_match + rf*u_match**2/(2*Cp) - T_wall) / u_match
        b2 = -rf / (2*Cp)
    elif qw_method == 'HUANG':
        # Huang & Coleman (1994) — mixed Prandtl number Pre = 0.8
        rf = 0.8
        b1 = (T_match + rf*u_match**2/(2*Cp) - T_wall) / u_match
        b2 = -rf / (2*Cp)
    else:
        raise ValueError(f"Unknown qw_method '{qw_method}'. Choose ZHANG, WALZ, or HUANG.")

    # ── STEP 3: bisection on u_tau ────────────────────────────────────────────
    # Integrate Eq. 6.2 from y=k to y=y_match; find u_tau s.t. u_sol[-1] = u_match
    utau_lo, utau_hi = 0.01, 0.51
    dy      = 0.0001
    n_pts   = int((y_match - k) / dy) + 2
    u_sol   = np.zeros(n_pts)
    u_lo    = np.zeros(n_pts)
    T_sol   = np.zeros(n_pts)
    rho_sol = np.zeros(n_pts)
    rho_lo  = np.zeros(n_pts)
    y_vec   = np.zeros(n_pts)

    for _ in range(100):
        if abs(utau_hi - utau_lo) < 1e-8:
            break
        u_tau    = (utau_lo + utau_hi) / 2
        u_sol[0] = u_tau   / utau_over_uk   # velocity at roughness crest (Eq. 6.3)
        u_lo[0]  = utau_lo / utau_over_uk
        y = k
        for i in range(n_pts - 1):
            y          += dy
            y_vec[i]    = y
            T_sol[i]    = T_wall + b1*u_sol[i] + b2*u_sol[i]**2     # Eq. 6.4
            T_lo        = T_wall + b1*u_lo[i]  + b2*u_lo[i]**2
            rho_sol[i]  = T_wall * rho_wall / T_sol[i]               # ideal gas
            rho_lo[i]   = T_wall * rho_wall / T_lo
            du          = (u_tau   / kappa) / (y - d) * np.sqrt(rho_wall / rho_sol[i]) * dy
            du_lo       = (utau_lo / kappa) / (y - d) * np.sqrt(rho_wall / rho_lo[i])  * dy
            u_sol[i+1]  = u_sol[i] + du                              # Eq. 6.2 (Euler)
            u_lo[i+1]   = u_lo[i]  + du_lo
        if (u_lo[-1] - u_match) * (u_sol[-1] - u_match) < 0:
            utau_hi = u_tau
        else:
            utau_lo = u_tau

    tauw = rho_wall * u_tau**2             # tau_w = rho_w * u_tau^2

    # Complete temperature at the last grid point
    y_vec[-1] = y
    T_sol[-1] = T_wall + b1*u_sol[-1] + b2*u_sol[-1]**2

    # ── STEP 4: wall heat flux from consistency condition (Eq. 6.5) ───────────
    divisor = rf if qw_method == 'HUANG' else Pr
    qw = Cp * tauw * b1 / divisor

    return y_vec, u_sol, T_sol, tauw, qw, u_tau, b1, b2


# ── Run and print ─────────────────────────────────────────────────────────────
_G, _RST = "\033[32m", "\033[0m"

def _fmt(pct):
    col = _G 
    return f"{col}{pct:+.3f} %{_RST}"

_, _, _, tauw, qw, *_ = bisection_solver(
    y_match, u_match, T_match, T_wall, T_edge, u_edge,
    Pr, Cp, rho_wall, k, qw_method
)

tauw_err = (tauw - tauw_DNS) / tauw_DNS * 100
qw_err   = (qw   - qw_DNS)   / qw_DNS   * 100

print(f"\n  {case_name} [{qw_method}]"
      f"  |  eps_tauw {_fmt(tauw_err)}"
      f"  |  eps_qw {_fmt(qw_err)}\n")
