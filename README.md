# cogo2026-wall-model

Physics-based wall model for compressible turbulent boundary layers over prism-shaped roughness.

Companion code for:
> Cogo, Depieri, Bernardini & Picano — *On the Reynolds analogy for high-speed rough-wall flows: implications for wall modelling*, J. Fluid Mech. (under review)

## What it does

Given mean flow information far from the wall and roughness geometry, the model predicts wall shear stress $\tau_w$ and wall heat flux $q_w$ without resolving the roughness sublayer. It couples three building blocks:

1. **Drag model** — roughness geometry and sheltering, Yang et al. (2016) [Appendix A]
2. **Compressibility correction** — Van Driest (1951) transformation [Eq. 6.2]
3. **Thermal closure** — temperature–velocity relation, three options [Table 3]
   - `BRADSHAW` — Huang & Coleman (1994), $Pr_e = 0.8$ ← recommended
   - `WALZ` — Walz (1969), $r = Pr^{1/3}$
   - `ZHANG` — Zhang et al. (2014), generalised Reynolds analogy

## Usage

Edit the input block at the top of `wall_model.py` to match your case, then run:

```bash
python wall_model.py
```

Output example:

```
  M2I [BRADSHAW]  |  eps_tauw +7.843 %  |  eps_qw +14.295 %
```

## Requirements

```
numpy
```

## Reference DNS data

The default inputs correspond to case M2I from the paper ($M_\infty = 2$, isothermal wall $\Theta = 0.25$). DNS data are available upon reasonable request to the corresponding author.
