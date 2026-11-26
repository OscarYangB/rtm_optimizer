> **Also follow [`generic_skills.md`](generic_skills.md)** (prompt logging, run-time / performance log, experiment naming convention, folder layout), which applies to every prompt. Take every physical value and experiment setting from **[`experiment_configuration.md`](experiment_configuration.md)**.

## Role

You are a computational fluid dynamics engineer experienced in flow through porous media and liquid composite moulding (RTM / VARTM). You write clean, well-documented FEniCSx (dolfinx) code. Our team will validate the solver ourselves (see "Validation plan"), so your outputs must be easy to compare against other results.

## Goal

Build a 2D PDE solver that simulates resin filling a rectangular fibre preform, so we can reproduce what we saw in our resin-infusion experiment. For Part I, the two outputs that matter are:

1. **How the flow front progresses.** Draw the flow front as a set of lines (isochrones) on the 2D domain, one line per time stamp, and colour each line by its time using a continuous colormap with a colourbar in seconds. Draw one isochrone every **1 min (60 s)**. Our experiment takes about **20 min** to fill, so expect about 20 lines. The number of lines should follow the solver's own fill time; do not fix it at 20. Make the interval a config parameter (`isochrone_interval_s = 60`). Line spacing shows front speed: widely spaced lines mean the front is moving fast, and closely spaced lines mean it is slow.
2. **Fill time**: the total fill time, plus when each part of the domain fills. The colour-coded isochrone plot answers both: the colour of the line passing through a point is the time at which that point filled.

Part II (numerical-method study, optimisation) is out of scope. Do not start it.

## Reference model

Okabe et al., "Multi-objective optimization for resin transfer molding process", *Composites Part A* 92 (2017) 1–9, Sec. 2.1 (`1-s2.0-S1359835X1630327X-main.pdf`).

- Darcy's law: **u = −(K/μ) ∇p**, where u is the Darcy (volume-averaged) velocity, K is the permeability tensor and μ is the resin viscosity.
- Continuity for incompressible resin: **∇·u = 0**, which gives **∇·((K/μ) ∇p) = 0** inside the filled region.
- Boundary conditions:
  - p = p_inlet = 1 atm = 101,325 Pa (absolute) at the gate (inlet). The resin reservoir is open to the atmosphere.
  - ∂p/∂n = 0 at the mould walls.
  - p = p_vent at the moving flow front and at the outlet. The vacuum pulls the air ahead of the front out to the vent, so the front sits at vent pressure (see "Driving pressure" below).
- Front tracking uses the Flow Analysis Network (FAN) / FE–CV method:
  - Each node i has a control volume V_i, a net inflow Q_i and a fill fraction f_i ∈ [0, 1].
  - Fill-fraction update: f_iⁿ⁺¹ = f_iⁿ + (Q_i / V_i) Δtⁿ.
  - Adaptive time step: Δtⁿ = min over partially filled CVs of (1 − f_iⁿ) V_i / Q_i, so that exactly one CV fills per step.
  - The pressure field is solved again only on the filled region after every step.

### Modelling assumptions (state these explicitly in the code and the report)

1. Darcy flow applies: creeping, viscous-dominated flow. The paper reports Re ≈ 0.02 for its experiment. Check Re for our case.
2. The resin is incompressible and Newtonian. Viscosity is constant during filling, so the model ignores cure and temperature effects during infusion.
3. The preform is rigid with uniform porosity φ and uniform permeability K. K may be anisotropic, and off-diagonal terms are zero in the x–y axes.
4. The part is thin, so we use a 2D in-plane model. Through-thickness flow is ignored.
5. **The flow front is a mixture of air and resin.** In the FAN method, this mixed zone is the band of partially filled control volumes (0 < f_i < 1), about one element wide. The air ahead of the front is assumed to be evacuated to p_vent and not compressed or trapped, so p = p_vent on the partially filled and empty nodes. The model ignores capillary pressure, unsaturated flow within the tows, and gravity.
6. There is no race-tracking along the edges. The paper notes this as the main source of disagreement with experiment, so we should expect it and discuss it.
7. **Porosity:** the front advances with the interstitial velocity u/φ, not with the Darcy velocity u. Eq. (7) in the paper takes V_i as the CV volume. We must use the *pore* volume φV_i, or the fill times will be off by a factor of φ.

## Problem definition and experiment configuration

The physical setup (domain, ports, pressures, resin, reinforcement, permeability, porosity, [TBD] fields) and the list of experiments with their settings are in **[`experiment_configuration.md`](experiment_configuration.md)**. Use only the values given there.

## Known limitations

1. **Permeability is unknown.** No permeability test was run on our 2-layer 14k carbon preform, so K comes from a literature table, not from a measurement. This is the largest source of uncertainty in the model:
   - **Absolute times are not reliable.** Fill time scales as 1/K, so an error of 10× in K gives an error of 10× in fill time.
   - **Front *shapes* and *relative* progression (t/t_fill) do not depend on K** as long as K is isotropic and μ is constant. Only the time axis stretches.
   - **Anisotropy is not captured.** If the real fabric has K_xx ≠ K_yy, the true fronts will be elliptical rather than circular near the gate.
2. **Porosity is unknown.** The 2-layer preform was not characterised: there is no porosity or fibre-volume-fraction measurement. φ = 0.498 is a literature value. Fill time scales linearly with φ, so absolute times carry this uncertainty too. A uniform φ does not change front shapes or the t/t_fill progression.
3. **Mixed resin viscosity is not yet known.** The model uses the resin-only datasheet value, and fill time scales as μ.
4. **Race-tracking, local variation in the preform, and compaction of the vacuum bag are not modelled.** The preform is treated as rigid and uniform.

Because of points 1–3, every Milestone 1 figure should put the K, φ and μ it used in the title or caption.

## Starting point

`RTM-Optimizer/main.py` is a dolfinx skeleton. Known gaps:

- It uses a unit square mesh, not the 342.9 × 241 mm rectangle (SI units, metres).
- Inlet and outlet are circles at interior points (0.4, 0.4) and (0.6, 0.6). They should be at the short-edge midpoints.
- It solves a single steady pressure field over the whole domain with p = 0 at the outlet. This is the fully saturated problem, not the filling problem. The solve must cover only the filled region, with p = 0 on the front.
- Velocity is multiplied by an arbitrary factor of 10000, which must be removed. K = 6e-8 m² and μ = 0.5 Pa·s are placeholders.
- TODOs in the file:
  - Compute the CV volumetric flux.
  - Implement FAN (or VOF).
  - Use an anisotropic K tensor.

Keep dolfinx as the FE framework, and build from `main.py`.

## Tasks

1. **Physics setup.** Collect all parameters in one config (dataclass or YAML) using SI units. Document every assumption.
2. **Mesh.** Build the rectangle with refinement near the gate and vent, and tag the inlet, outlet and wall boundaries.
3. **Pressure solver.** Solve ∇·(K/μ ∇p) = 0 on the filled sub-domain with BCs (4)–(6) and an anisotropic K. Recover u = −(K/μ)∇p as a flux that is conservative across CV faces.
4. **Front tracking.** Build the FE/CV control volumes (median-dual) and run the FAN loop with an adaptive Δt and porosity-corrected volumes. Record each node's fill time.
5. **Figures (Milestone 1 output).** See "Milestone 1 deliverables" below.

## Validation plan (done by our team, not by you)

We will validate the solver ourselves in two ways. **Do not carry out any validation, write validation scripts, or report validation results.** Your only job here is to make your outputs easy for us to compare.

1. **Flow-front timestamps vs ANSYS and vs the experiment video.** We will compare the time at which the simulated flow front reaches given positions with:
   - an ANSYS simulation of the same setup. We are writing our own solver because ANSYS runs too slowly for the later optimisation work. ANSYS is the reference, not the tool we will use going forward.
   - the flow front seen in our experiment video.
2. **Method of Manufactured Solutions (MMS).** We will check that the FEA pressure solve is correct, and that it converges at the expected rate, using manufactured solutions.

To support this without doing it yourself:
- Use the same time convention for every output: t = 0 when resin enters at the gate, in seconds. Print the times on the figures.
- Structure the pressure solve as a function that takes the domain or mesh, K, μ, an optional source term f and the Dirichlet/Neumann BC data. That lets us plug in a manufactured solution. Do not write the MMS test itself.

## Milestone 1 deliverables (figures only)

The first milestone produces **figures only**. It does not include a report, a validation, a parameter sweep or a test suite. Save the figures to `RTM-Optimizer/outputs/experiment_<#>_<short_description>/figures/` (see `generic_skills.md`) as PNG at 150 dpi or more, with units on the axes and the parameter values in the caption or title:

1. Geometry and mesh, with the inlet, outlet and walls tagged.
2. The pressure field at a few fill stages.
3. **Main figure: colour-coded flow-front lines.** Draw the isochrones of the node fill times t_fill(x, y) as lines on the rectangle with the inlet and outlet marked. Colour every line by its time from one continuous colormap, add a colourbar in seconds, and put the total fill time in the title.
4. A flow-front position vs time plot along the centreline (inlet → outlet), with the front velocity on the same plot or a second one.

Alongside the figures, provide a runnable entry point that regenerates them all from the config.

## Working rules

- Ask before assuming any **[TBD]** physical value.
- Do not touch experiment data: no loading, overlaying or fitting. The team owns it.
- Stay within Milestone 1: figures only. Do not perform the validation (ANSYS, experiment video or MMS). The team does that.
- Keep units in SI throughout, and comment on units wherever a conversion happens.
- Record every prompt, decision and session summary in `RTM-Optimizer/logs/session.log`, and every run time in `outputs/performance_log.csv` (see `generic_skills.md`).
