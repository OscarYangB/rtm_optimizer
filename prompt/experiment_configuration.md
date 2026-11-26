# Experiment configuration (RTM-Optimizer)

This file is the single place where the physical setup and every experiment's settings are written down. Milestone prompts (for example `milestone1_skills.md`) point here. Conventions for naming, logging and run time are in `generic_skills.md`.

**Keep this file in sync with the code.** Every experiment listed here must be registered in `EXPERIMENTS` in `main.py`, and the shared defaults live in `rtm/config.py` (`RTMConfig`). If a value changes, update this file and the code together, and log the change in `logs/session.log`.

## 1. Physical setup (shared by all experiments)

| Quantity | Value |
|---|---|
| Domain | Rectangle, L = 342.9 mm (x) × W = 241 mm (y) |
| Thickness h | **[TBD]** mm (needed only for flow rate and volume checks) |
| Inlet (gate) | Midpoint of one short edge: (0, W/2). Port diameter **[TBD]** mm. Modelled as a line inlet (edge segment) or as a point/semicircle? **[TBD]** |
| Outlet (vent) | Midpoint of the opposite short edge: (L, W/2). Port diameter **[TBD]** |
| Driving pressure | Inlet at 1 atm = 101,325 Pa (absolute). Vent pressure p_vent is a config parameter with two cases. **Baseline (realistic):** the vacuum pump only reaches −85 kPa gauge, so p_vent = 101,325 − 85,000 ≈ 16,325 Pa (absolute) and Δp ≈ 85 kPa. **Ideal bound:** perfect vacuum, p_vent = 0 Pa (absolute) and Δp ≈ 101.3 kPa. The pressure equation is linear and the geometry is fixed, so both cases give the same front shapes, and the times scale by Δp_ideal/Δp_real ≈ 1.19. Produce the figures for the baseline case and report the ideal-case total fill time alongside. |
| Resin | AE2032-30-X, using the datasheet values for resin PR2032. Infused at **room temperature**, nominal 25 °C (77 °F, the datasheet reference temperature). **Viscosity μ = 1,650 cP = 1.65 Pa·s at 77 °F.** Specific gravity 1.15, so density ρ = 1,150 kg/m³ (needed only for the Reynolds-number check). Assume μ is constant over the ~20 min fill, with no cure. **Caveat:** 1.65 Pa·s is the viscosity of the *resin alone*. The resin mixed with hardener is usually less viscous, and fill time scales directly with μ. Treat 1.65 Pa·s as the current value and replace it with the mixed viscosity if we get one **Mix ratio (as used):** 228 g resin + 69 g hardener, i.e. 100 : 30.3 by weight (≈ 1 : 3.3 hardener : resin, 23 wt% hardener). This matches the "-30" in AE2032-30-X. **[TBD: mixed viscosity at 25 °C (from the system datasheet) or the hardener's product name and its viscosity]**. |
| Reinforcement | **14k carbon fibre fabric, 2 layers.** Weave or architecture (plain, twill or UD) and areal weight **[TBD]**. |
| Permeability | **Not measured on our preform. This is a known limitation (see "Known limitations").** Use the literature values from the sensitivity table supplied by the team (`logs/attachments/2026-11-26_sensitivity_table1_K_phi.png`). The table gives five anisotropic (K_x, K_y) pairs in m²: (2×10⁻¹⁰, 1×10⁻¹⁰), (6×10⁻¹⁰, 5×10⁻¹⁰), (2×10⁻⁹, 1×10⁻⁹), (6×10⁻⁹, 5×10⁻⁹) and (6×10⁻⁸, 5×10⁻⁸). The reference value is the median pair, **K_x = 2×10⁻⁹, K_y = 1×10⁻⁹ m²**, with off-diagonal terms zero. Keep K as config parameters and print the values used on every figure. |
| Porosity φ (= 1 − V_f) | **Not measured on our preform. This is a known limitation.** Use **φ = 0.498** from the same literature table. Keep it as a config parameter and print it on every figure. |
| Experimental data | **Not your concern.** The team handles all experiment data (video, timestamps). For reference only, the experiment filled in about 20 min. Do not load, process or tune to experiment data. |

Do not invent values for the **[TBD]** fields. Where a value is missing, make it a named parameter, list it in a "needed inputs" table and run with a clearly labelled placeholder.

## 2. Shared numerical and run settings

| Shared parameter | Value | Code |
|---|---|---|
| Domain | 342.9 × 241 mm; gate at (0, W/2), vent at (L, W/2); semicircular ports of 10 mm diameter (placeholder) | `L`, `W`, `gate_diameter`, `vent_diameter` |
| Mesh | P1 triangles; element size 4 mm away from the ports and 0.8 mm at the port arcs, graded over 40 mm. For two-region runs the mesh is split along y = W/2 so the K jump lies on element edges. | `h_far`, `h_port`, `refine_radius` |
| Pressures | p_inlet = 101,325 Pa abs; p_vent = 16,325 Pa abs (−85 kPa gauge), so Δp = 85 kPa. The ideal-vacuum fill time (Δp = 101.3 kPa) is reported alongside. | `p_inlet`, `vacuum_gauge`, `p_vent_ideal` |
| Resin | μ = 1.65 Pa·s (resin only, PR2032 at 25 °C); ρ = 1,150 kg/m³ | `mu`, `rho` |
| Porosity | φ = 0.498 (literature table) | `phi` |
| Reference permeability | K_x = 2×10⁻⁹, K_y = 1×10⁻⁹ m², diagonal (median pair in the literature table) | `K_xx`, `K_yy` (`K_REF` in `main.py`) |
| Front tracking | FE/CV (FAN) method, one control volume filled per step, pore volume φ·V_i | `rtm/fan.py` |
| Outputs | isochrones every 60 s; pressure snapshots at 10 / 35 / 65 / 95 % of the area filled; PNG at 200 dpi | `isochrone_interval_s`, `pressure_snapshot_fractions`, `figure_dpi` |

## 3. Experiments

| Experiment | Permeability layout | K_x, K_y [m²] | Purpose |
|---|---|---|---|
| `experiment_1_uniform_K` | uniform | 2.0×10⁻⁹, 1.0×10⁻⁹ everywhere | Baseline. Reference (dashed lines) in the comparison figures of experiments 2 and 3. |
| `experiment_2_two_region_K_20pct_diff` | two regions split along y = W/2 (the gate–vent line) | top 2.0×10⁻⁹, 1.0×10⁻⁹; **bottom 2.4×10⁻⁹, 1.2×10⁻⁹** (bottom 20 % higher in both x and y) | Effect of a 20 % permeability difference between the top and bottom halves (HYPOTHETICAL layout). |
| `experiment_3_two_region_K_100pct_diff` | two regions split along y = W/2 | top 2.0×10⁻⁹, 1.0×10⁻⁹; **bottom 4.0×10⁻⁹, 2.0×10⁻⁹** (bottom 100 % higher, i.e. 2×, in both x and y) | Effect of a 100 % permeability difference between the top and bottom halves (HYPOTHETICAL layout). |

- Run them with `./run.sh --experiment <#>` or `./run.sh --all` (in order; experiment 1 must exist before experiments 2 and 3 can draw their comparison figures).
- Outputs go to `outputs/<experiment>/figures/` and `outputs/<experiment>/data/`. Each run adds a row to `outputs/run_history.csv` (results) and to `outputs/performance_log.csv` (run time).
- Figures for each experiment: fig1 mesh, fig2 pressure stages, fig3 isochrones, fig4 centreline front. Experiments 2 and 3 also get fig5 (front along the top, centre and bottom lines) and fig6 (overlay on experiment 1).
- To add an experiment, give it the next free number, add a row to the table above, and register it in `main.py`. When experiments are abandoned, delete their output folders; do not archive them. Their parameters and results stay recorded in `logs/session.log`, `run_history.csv` and `performance_log.csv`.
