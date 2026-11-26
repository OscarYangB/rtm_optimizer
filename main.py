"""RTM Part I - Milestone 1 entry point: regenerates every figure from the config.

    ./run.sh                       # experiment 1 (default)
    ./run.sh --experiment 2        # by number ...
    ./run.sh --experiment experiment_2_two_region_K   # ... or by full name
    ./run.sh --all                 # every registered experiment, in order
    python3 main.py [...]          # if dolfinx is installed locally

Each experiment writes to outputs/experiment_<#>_<short_description>/{figures,data}/ and appends
one row (incl. wall-clock run time) to outputs/run_history.csv. See prompt/generic_skills.md.

Pipeline: config -> gmsh mesh -> P1 Darcy operator (dolfinx) -> FE/CV FAN filling loop ->
figures + raw node fill times for the team's comparisons.
Original skeleton preserved in legacy/main_skeleton.py.
"""
import argparse
import csv
import datetime
import json
import os
import time
import numpy as np
import matplotlib.tri as mtri
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh as dmesh

from rtm.config import RTMConfig, NEEDED_INPUTS, OUTPUTS_ROOT, EXPERIMENT_RE
from rtm.geometry import build_mesh, INLET, OUTLET, WALL, TOP, BOTTOM
from rtm.pressure import assemble_cv_operator
from rtm.fan import run_fan
from rtm import plotting

# Registered experiments: name (experiment_<#>_<short_description>) -> config overrides.
# Add new ones with the next free number; never renumber or reuse an old number.
# Experiment registry - must match prompt/experiment_configuration.md (Section 3).
# K and phi come from the literature sensitivity table (Table 1, see rtm/config.py); not measured.
K_TABLE1 = [(2e-10, 1e-10), (6e-10, 5e-10), (2e-9, 1e-9), (6e-9, 5e-9), (6e-8, 5e-8)]  # (Kx, Ky) m^2
K_REF = (2e-9, 1e-9)       # median Table-1 pair: uniform value and top-half value
EXPERIMENTS = {
    # Uniform anisotropic K = diag(2e-9, 1e-9) m^2 everywhere.
    "experiment_1_uniform_K": dict(),
    # HYPOTHETICAL layout: top half K_REF, bottom half 20 % higher in both x and y
    # -> (2.4e-9, 1.2e-9) m^2. Split along y = W/2. (Was 5 % until Entry 17, 2026-11-26.)
    "experiment_2_two_region_K_20pct_diff": dict(K_top=K_REF,
                                                 K_bottom=(1.2 * K_REF[0], 1.2 * K_REF[1])),
    # HYPOTHETICAL layout: top half K_REF, bottom half 100 % higher (2x) in both x and y
    # -> (4e-9, 2e-9) m^2. Split along y = W/2.
    # (History: bottom 1.5x in x,y (Entry 18); bottom 2x in x only (Entry 19); now Entry 20.)
    "experiment_3_two_region_K_100pct_diff": dict(K_top=K_REF,
                                                  K_bottom=(2.0 * K_REF[0], 2.0 * K_REF[1])),
}
# (run_sweep() below is kept for future sensitivity studies; register e.g. dict(sweep=K_TABLE1).)
for _name in EXPERIMENTS:
    assert EXPERIMENT_RE.match(_name), f"bad experiment name {_name}"
REFERENCE_EXPERIMENT = "experiment_1_uniform_K"   # dashed overlay in fig6 comparisons


def resolve_experiment(key: str) -> str:
    """Accept '2' or the full folder name; return the registered name."""
    if key.isdigit():
        hits = [n for n in EXPERIMENTS if EXPERIMENT_RE.match(n).group(1) == key]
        if len(hits) != 1:
            raise SystemExit(f"no unique experiment numbered {key}: {list(EXPERIMENTS)}")
        return hits[0]
    if key not in EXPERIMENTS:
        raise SystemExit(f"unknown experiment {key!r}; registered: {list(EXPERIMENTS)}")
    return key


def permeability(cfg, msh, cell_tags):
    """UFL permeability tensor [m^2]: uniform 2x2 constant, or cell-wise (DG0) two-region."""
    if not cfg.two_region:
        return cfg.K
    Q = fem.functionspace(msh, ("DG", 0))
    kx, ky = fem.Function(Q), fem.Function(Q)
    for tag, (Kx, Ky) in ((TOP, cfg.K_top), (BOTTOM, cfg.K_bottom)):
        kx.x.array[cell_tags.find(tag)] = Kx
        ky.x.array[cell_tags.find(tag)] = Ky
    assert np.all(kx.x.array > 0), "every cell needs a region tag"
    return ufl.as_matrix([[kx, 0], [0, ky]])   # diagonal, zero off-diagonal (A3)


def append_history(path, row):
    """Append a run row; if the column set changed, rewrite the file with the union header."""
    rows, fields = [], list(row)
    if os.path.exists(path):
        with open(path, newline="") as fh:
            r = csv.DictReader(fh); rows = list(r)
            fields = list(dict.fromkeys((r.fieldnames or []) + list(row)))
    rows.append(row)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)


def main(cfg: RTMConfig = RTMConfig()):
    assert MPI.COMM_WORLD.size == 1, "FAN loop is serial; run without mpirun"
    os.makedirs(cfg.figures_dir, exist_ok=True); os.makedirs(cfg.data_dir, exist_ok=True)
    print(f"=== {cfg.experiment}{' / ' + cfg.sub_run if cfg.sub_run else ''} ===")

    print("Needed inputs (running with placeholders):")
    for name, what, now in NEEDED_INPUTS:
        print(f"  - {name:32s} {what:48s} -> {now}")

    # ---- wall-clock timers [s] (sim run time is tracked for the team, see run_history.csv) ----
    timers = {}
    t_start = time.perf_counter()

    # ---- mesh & P1 space ----
    t0 = time.perf_counter()
    msh, ct, ft = build_mesh(cfg)
    K = permeability(cfg, msh, ct)
    V = fem.functionspace(msh, ("Lagrange", 1))
    fdim = msh.topology.dim - 1
    gate = fem.locate_dofs_topological(V, fdim, ft.find(INLET))
    vent = fem.locate_dofs_topological(V, fdim, ft.find(OUTLET))
    xy = V.tabulate_dof_coordinates()[:, :2]            # [m]
    cells = V.dofmap.list
    tri = mtri.Triangulation(xy[:, 0] * plotting.MM, xy[:, 1] * plotting.MM, cells)
    print(f"Mesh: {xy.shape[0]} nodes, {cells.shape[0]} triangles, "
          f"{gate.size} gate nodes, {vent.size} vent nodes")

    timers["mesh_s"] = time.perf_counter() - t0
    segs = {}
    for tag in (INLET, OUTLET, WALL):
        g = dmesh.entities_to_geometry(msh, fdim, ft.find(tag))
        segs[tag] = msh.geometry.x[g][:, :, :2]
    plotting.fig_mesh(f"{cfg.figures_dir}/fig1_geometry_mesh.png", tri, segs, cfg,
                      xy.shape[0], cells.shape[0])

    # ---- Darcy operator + FAN filling ----
    t0 = time.perf_counter()
    A, area = assemble_cv_operator(V, K, cfg.mu)
    timers["assembly_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = run_fan(A, area, gate, vent, cfg, cfg.pressure_snapshot_fractions)
    timers["fan_solve_s"] = time.perf_counter() - t0   # FAN loop incl. all pressure solves
    t_fill, t_total = res["t_fill"], res["t_total"]
    # Linear pressure problem + fixed geometry: times scale exactly with 1/dp.
    t_ideal = t_total * cfg.dp / cfg.dp_ideal

    # ---- Reynolds check (assumption A1): Re = rho |u| d / mu, max Darcy speed at t = 0+ ----
    from rtm.pressure import darcy_velocity
    p0 = fem.Function(V); p0.x.array[:] = res["p_first"]
    u = darcy_velocity(p0, K, cfg.mu).x.array.reshape(-1, 2)
    u_max = np.linalg.norm(u, axis=1).max()             # [m/s]
    Re_f = cfg.rho * u_max * cfg.fibre_diameter / cfg.mu
    Re_K = cfg.rho * u_max * np.sqrt(cfg.K_max) / cfg.mu
    print(f"Re check: max Darcy speed {u_max:.3e} m/s -> Re(d_f = {cfg.fibre_diameter*1e6:g} um)"
          f" = {Re_f:.2e}, Re(sqrt K) = {Re_K:.2e}  (Darcy valid for Re << 1)")

    # ---- figures ----
    t0 = time.perf_counter()
    plotting.fig_pressure(f"{cfg.figures_dir}/fig2_pressure_stages.png", tri,
                          res["snapshots"], t_fill, cfg)
    plotting.fig_isochrones(f"{cfg.figures_dir}/fig3_isochrones.png", tri, t_fill,
                            t_total, t_ideal, cfg)
    x, t, v = plotting.centreline_front(tri, t_fill, cfg)
    plotting.fig_centreline(f"{cfg.figures_dir}/fig4_centreline_front.png", x, t, v, t_total, cfg)
    if cfg.two_region:
        lanes = [(f"top (y = {0.75*cfg.W*1e3:.0f} mm)", "tab:purple",
                  *plotting.centreline_front(tri, t_fill, cfg, y=0.75 * cfg.W)),
                 (f"centre (y = {0.5*cfg.W*1e3:.1f} mm)", "tab:gray", x, t, v),
                 (f"bottom (y = {0.25*cfg.W*1e3:.0f} mm)", "tab:green",
                  *plotting.centreline_front(tri, t_fill, cfg, y=0.25 * cfg.W))]
        plotting.fig_lanes(f"{cfg.figures_dir}/fig5_front_top_centre_bottom.png", lanes, t_total, cfg)
        ref = f"{OUTPUTS_ROOT}/{REFERENCE_EXPERIMENT}/data/node_fill_times.csv"  # if it exists
        if os.path.exists(ref):
            d = np.loadtxt(ref, delimiter=",", skiprows=1)
            tri_ref = mtri.Triangulation(d[:, 0] * plotting.MM, d[:, 1] * plotting.MM)
            plotting.fig_compare(f"{cfg.figures_dir}/fig6_compare_uniform.png", tri, t_fill, t_total,
                                 tri_ref, d[:, 2], d[:, 2].max(), cfg,
                                 f"uniform K = diag({cfg.K_xx:.0e}, {cfg.K_yy:.0e}) m² (experiment 1)")

    timers["plotting_s"] = time.perf_counter() - t0   # figs 2-4 (fig 1 counted in mesh_s)

    # ---- raw data for the team's own comparisons (same t = 0 convention, SI) ----
    np.savetxt(f"{cfg.data_dir}/node_fill_times.csv", np.column_stack([xy, t_fill]),
               delimiter=",", header="x_m,y_m,t_fill_s", comments="")
    np.savetxt(f"{cfg.data_dir}/centreline_front.csv", np.column_stack([x, t, v]),
               delimiter=",", header="x_m,t_s,front_speed_m_per_s", comments="")
    timers["total_s"] = time.perf_counter() - t_start
    # total SOLVER time = operator assembly + FAN loop (all pressure solves + front tracking);
    # excludes meshing, plotting and file I/O.
    timers["solver_total_s"] = timers["assembly_s"] + timers["fan_solve_s"]
    timers["per_fan_step_ms"] = 1e3 * timers["fan_solve_s"] / max(res["steps"], 1)
    summary = dict(config=cfg.to_dict(), run_time=timers, fill_time_baseline_s=t_total,
                   fill_time_ideal_vacuum_s=t_ideal, fan_steps=res["steps"],
                   n_nodes=int(xy.shape[0]), n_cells=int(cells.shape[0]),
                   centreline_vent_arrival_s=float(t[-1]),
                   Re_fibre=Re_f, Re_sqrtK=Re_K, u_darcy_max_m_per_s=u_max)
    with open(f"{cfg.data_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    # (1) results/parameter history - one row per run; units in every header [..]
    append_history(f"{OUTPUTS_ROOT}/run_history.csv", {
        "timestamp": stamp, "experiment": cfg.experiment, "sub_run": cfg.sub_run or "",
        "n_nodes [-]": int(xy.shape[0]), "n_cells [-]": int(cells.shape[0]),
        "fan_steps [-]": res["steps"], "h_far [m]": cfg.h_far, "h_port [m]": cfg.h_port,
        "K_xx [m^2]": cfg.K_xx, "K_yy [m^2]": cfg.K_yy,
        "K_top (Kx;Ky) [m^2]": "{};{}".format(*cfg.K_top) if cfg.K_top else "",
        "K_bottom (Kx;Ky) [m^2]": "{};{}".format(*cfg.K_bottom) if cfg.K_bottom else "",
        "phi [-]": cfg.phi, "mu [Pa s]": cfg.mu, "dp [Pa]": cfg.dp,
        "fill_time [s]": round(t_total, 2), "fill_time_ideal_vacuum [s]": round(t_ideal, 2)})
    # (2) separate PERFORMANCE log for run-time analysis (wall-clock, perf_counter)
    append_history(f"{OUTPUTS_ROOT}/performance_log.csv", {
        "timestamp": stamp, "experiment": cfg.experiment, "sub_run": cfg.sub_run or "",
        "n_nodes [-]": int(xy.shape[0]), "fan_steps [-]": res["steps"],
        "mesh_time [s]": round(timers["mesh_s"], 3),
        "assembly_time [s]": round(timers["assembly_s"], 3),
        "fan_solve_time [s]": round(timers["fan_solve_s"], 3),
        "solver_total_time [s]": round(timers["solver_total_s"], 3),
        "plotting_time [s]": round(timers["plotting_s"], 3),
        "total_run_time [s]": round(timers["total_s"], 3),
        "time_per_fan_step [ms]": round(timers["per_fan_step_ms"], 3)})
    print(f"Solver run time: {timers['solver_total_s']:.2f} s (assembly {timers['assembly_s']:.2f} s"
          f" + FAN {timers['fan_solve_s']:.2f} s, {timers['per_fan_step_ms']:.2f} ms/step); "
          f"total run time {timers['total_s']:.2f} s -> {OUTPUTS_ROOT}/performance_log.csv")
    print(f"Fill time: baseline {t_total:.1f} s ({t_total/60:.2f} min); "
          f"ideal vacuum {t_ideal:.1f} s ({t_ideal/60:.2f} min)")
    print(f"Figures written to {cfg.figures_dir}/")
    return dict(tri=tri, t_fill=t_fill, t_total=t_total, t_ideal=t_ideal, timers=timers)


def run_sweep(name, pairs):
    """Uniform-K sensitivity sweep: one full run per (Kx, Ky) in runs/K_<Kx>_<Ky>/, then
    summary figures + table in the experiment's own figures/ and data/."""
    runs = []
    for Kx, Ky in pairs:
        cfg = RTMConfig(experiment=name, sub_run=f"K_{Kx:.0e}_{Ky:.0e}", K_xx=Kx, K_yy=Ky)
        r = main(cfg)
        runs.append(dict(Kx=Kx, Ky=Ky, t_fill=r["t_total"], t_total=r["t_total"],
                         t_ideal=r["t_ideal"], tri=r["tri"], t_nodes=r["t_fill"],
                         solver_s=r["timers"]["solver_total_s"]))
    cfg = RTMConfig(experiment=name)
    os.makedirs(cfg.figures_dir, exist_ok=True); os.makedirs(cfg.data_dir, exist_ok=True)
    plotting.fig_sweep_fill_time(f"{cfg.figures_dir}/fig1_fill_time_vs_K.png", runs, cfg)
    shapes = [dict(r, t_fill=r["t_nodes"]) for r in sorted(runs, key=lambda r: r["Kx"] * r["Ky"])]
    plotting.fig_sweep_shapes(f"{cfg.figures_dir}/fig2_normalised_front_shapes.png", shapes, cfg)
    with open(f"{cfg.data_dir}/fill_time_vs_K.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Kx [m^2]", "Ky [m^2]", "K_eff=sqrt(KxKy) [m^2]", "fill_time [s]",
                    "fill_time [min]", "fill_time_ideal_vacuum [s]", "solver_total_time [s]"])
        for r in sorted(runs, key=lambda r: r["Kx"] * r["Ky"]):
            w.writerow([r["Kx"], r["Ky"], f"{np.sqrt(r['Kx']*r['Ky']):.3e}", round(r["t_total"], 2),
                        round(r["t_total"] / 60, 2), round(r["t_ideal"], 2), round(r["solver_s"], 2)])
    print(f"Sweep summary written to {cfg.experiment_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--experiment", default="1", help="number or full experiment_<#>_<desc> name")
    g.add_argument("--all", action="store_true", help="run every registered experiment")
    args = ap.parse_args()
    names = list(EXPERIMENTS) if args.all else [resolve_experiment(args.experiment)]
    for name in names:
        spec = dict(EXPERIMENTS[name])
        if "sweep" in spec:
            run_sweep(name, spec.pop("sweep"))
        else:
            main(RTMConfig(experiment=name, **spec))
