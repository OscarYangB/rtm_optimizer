"""Milestone 1 figures (matplotlib). Solver works in metres; plots use mm (x 1e3)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .geometry import INLET, OUTLET, WALL, TAG_NAMES

MM = 1e3  # m -> mm for display only
TAG_COLOURS = {INLET: "tab:red", OUTLET: "tab:blue", WALL: "0.25"}


def _caption(fig, cfg, extra=""):
    fig.text(0.5, 0.005, cfg.param_caption() + extra, ha="center", va="bottom",
             fontsize=7.5, color="0.25", wrap=True)


def _ports(ax, cfg, legend=True):
    ax.plot(0, cfg.W / 2 * MM, "o", ms=9, mfc="tab:red", mec="k", zorder=5,
            label="inlet (gate), p = 1 atm")
    ax.plot(cfg.L * MM, cfg.W / 2 * MM, "s", ms=9, mfc="tab:blue", mec="k", zorder=5,
            label=f"outlet (vent), p = {cfg.p_vent/1e3:.2f} kPa abs")
    ax.add_patch(Rectangle((0, 0), cfg.L * MM, cfg.W * MM, fill=False, ec="k", lw=1.2))
    if cfg.two_region:  # permeability interface + region labels
        ax.plot([0, cfg.L * MM], [cfg.W / 2 * MM] * 2, color="crimson", ls="--", lw=1.0, zorder=4)
        kw = dict(ha="right", fontsize=8.5, color="crimson", zorder=6,
                  bbox=dict(fc="w", ec="crimson", alpha=0.85, pad=2))
        ax.text(cfg.L * MM - 4, cfg.W * MM - 10,
                f"TOP  Kx, Ky = {cfg.K_top[0]:.3g}, {cfg.K_top[1]:.3g} m²", va="top", **kw)
        ax.text(cfg.L * MM - 4, 10,
                f"BOTTOM  Kx, Ky = {cfg.K_bottom[0]:.3g}, {cfg.K_bottom[1]:.3g} m²", va="bottom", **kw)


def _axes(ax, cfg):
    ax.set_aspect("equal")
    ax.set_xlim(-5, cfg.L * MM + 5); ax.set_ylim(-5, cfg.W * MM + 5)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")


def fig_mesh(path, tri, boundary_segments, cfg, n_nodes, n_cells):
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(13, 5.6),
                                  gridspec_kw=dict(width_ratios=[2.2, 1]))
    for a in (ax, axz):
        a.triplot(tri, lw=0.2, color="0.6")
        for tag, segs in boundary_segments.items():
            for s in segs:
                a.plot(s[:, 0] * MM, s[:, 1] * MM, color=TAG_COLOURS[tag], lw=2.2)
    _axes(ax, cfg)
    r = 25  # zoom half-width [mm]
    axz.set_xlim(-2, 2 * r); axz.set_ylim(cfg.W / 2 * MM - r, cfg.W / 2 * MM + r)
    axz.set_aspect("equal"); axz.set_xlabel("x [mm]"); axz.set_ylabel("y [mm]")
    axz.set_title("zoom: gate refinement", fontsize=10)
    ax.add_patch(Rectangle((-2, cfg.W / 2 * MM - r), 2 * r + 2, 2 * r, fill=False, ec="k", ls="--"))
    handles = [Line2D([], [], color=TAG_COLOURS[t], lw=3, label=f"tag {t}: {TAG_NAMES[t]}")
               for t in (INLET, OUTLET, WALL)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    ax.set_title(f"Geometry & mesh: {cfg.L*MM:.1f} × {cfg.W*MM:.0f} mm, "
                 f"{n_nodes} nodes / {n_cells} P1 triangles")
    _caption(fig, cfg)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=cfg.figure_dpi); plt.close(fig)


def fig_pressure(path, tri, snaps, t_fill, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    norm = Normalize(cfg.p_vent / 1e3, cfg.p_inlet / 1e3)
    for ax, s in zip(axes.flat, snaps):
        p_kpa = s["p"] / 1e3  # Pa -> kPa
        wet = np.isfinite(t_fill) & (t_fill <= s["t"])
        mask = ~wet[tri.triangles].all(axis=1)  # hide cells with any unfilled node
        ax.add_patch(Rectangle((0, 0), cfg.L * MM, cfg.W * MM, fc="0.9", ec="none"))
        tri_m = mtri.Triangulation(tri.x, tri.y, tri.triangles, mask=mask)
        cs = ax.tricontourf(tri_m, p_kpa, levels=np.linspace(norm.vmin, norm.vmax, 21),
                            cmap="plasma", norm=norm)
        ax.tricontour(tri, t_fill, levels=[s["t"]], colors="w", linewidths=1.8)
        _ports(ax, cfg); _axes(ax, cfg)
        ax.set_title(f"t = {s['t']:.0f} s ({s['t']/60:.1f} min), "
                     f"{100*s['frac']:.0f} % of area filled", fontsize=10)
    cax = fig.add_axes([0.87, 0.15, 0.018, 0.7])
    cb = fig.colorbar(cs, cax=cax, ticks=np.arange(20, 101, 10))
    cb.set_label("pressure p [kPa abs]  (grey = unfilled, p = p_vent; white = flow front)")
    fig.suptitle("Pressure field in the filled region at four fill stages (baseline vacuum)",
                 fontsize=13)
    _caption(fig, cfg)
    fig.subplots_adjust(left=0.05, right=0.84, bottom=0.08, top=0.92, hspace=0.25, wspace=0.18)
    fig.savefig(path, dpi=cfg.figure_dpi); plt.close(fig)


def fig_isochrones(path, tri, t_fill, t_total, t_ideal, cfg):
    dt = cfg.isochrone_interval_s
    levels = np.arange(dt, t_total, dt)   # number of lines follows the solver's fill time
    fig, ax = plt.subplots(figsize=(11, 8.4))
    norm = Normalize(0.0, t_total)
    cmap = plt.get_cmap("viridis")
    cs = ax.tricontour(tri, t_fill, levels=levels, cmap=cmap, norm=norm,
                       linewidths=0.9 if len(levels) > 60 else 1.4)
    step = max(1, int(np.ceil(len(levels) / 12)))     # label at most ~12 lines (in min)
    ax.clabel(cs, levels=levels[step - 1::step], fmt=lambda v: f"{v/60:.0f} min",
              fontsize=7.5, inline=True)
    _ports(ax, cfg); _axes(ax, cfg)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("fill time t [s]  (t = 0: resin enters at gate)")
    ax.set_title(f"Flow-front isochrones every {dt:g} s ({len(levels)} lines)\n"
                 f"Total fill time: {t_total:.0f} s = {t_total/60:.1f} min (baseline Δp = "
                 f"{cfg.dp/1e3:.0f} kPa);  ideal vacuum (Δp = {cfg.dp_ideal/1e3:.1f} kPa): "
                 f"{t_ideal:.0f} s = {t_ideal/60:.1f} min", fontsize=11)
    _caption(fig, cfg, "\nLine colour = time at which that point filled; close lines = slow front.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=cfg.figure_dpi); plt.close(fig)


def centreline_front(tri, t_fill, cfg, n=600, y=None):
    """t_fill sampled along a horizontal line y (default W/2, gate arc -> vent arc).
    Returns x [m], t [s], v [m/s]."""
    y = cfg.W / 2 if y is None else y
    x0 = cfg.gate_diameter / 2 if abs(y - cfg.W / 2) < cfg.gate_diameter / 2 else 0.0
    x1 = cfg.L - cfg.vent_diameter / 2 if abs(y - cfg.W / 2) < cfg.vent_diameter / 2 else cfg.L
    x = np.linspace(x0, x1 - 1e-9, n)
    # tri is in mm (display units) -> sample with mm coordinates
    t = mtri.LinearTriInterpolator(tri, t_fill)(x * MM, np.full_like(x, y * MM)).filled(np.nan)
    t = np.maximum.accumulate(np.nan_to_num(t, nan=0.0))  # front position is monotone in t
    # front speed v = dx/dt = 1 / (dt_fill/dx); smooth over ~5 mm to suppress mesh noise
    w = max(3, int(0.005 / (x[1] - x[0])))
    dtdx = np.gradient(np.convolve(t, np.ones(w) / w, mode="same"), x)
    v = np.where(dtdx > 0, 1.0 / np.maximum(dtdx, 1e-300), np.nan)
    v[: w // 2] = np.nan; v[-(w // 2):] = np.nan  # discard convolution edge effects
    return x, t, v


def fig_centreline(path, x, t, v, t_total, cfg):
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(t, x * MM, color="tab:blue", lw=2, label="front position x_f(t)")
    ax.set_xlabel("time t [s]  (t = 0: resin enters at gate)")
    ax.set_ylabel("front position along centreline x_f [mm]", color="tab:blue")
    ax.set_xlim(0, t_total * 1.02); ax.set_ylim(0, cfg.L * MM)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, v * MM, color="tab:orange", lw=1.6, label="front speed dx_f/dt")
    ax2.set_ylabel("front speed dx_f/dt [mm/s]", color="tab:orange")
    ax2.set_yscale("log")
    for k in range(1, int(t_total // 60) + 1):  # minute marks
        ax.axvline(60 * k, color="0.85", lw=0.5, zorder=0)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title(f"Centreline (y = {cfg.W/2*MM:.1f} mm) flow-front progression, gate → vent\n"
                 f"centreline reaches vent at t = {t[-1]:.0f} s; whole part filled at "
                 f"t = {t_total:.0f} s", fontsize=11)
    _caption(fig, cfg, "\nGrey vertical lines: 1 min marks. Speed smoothed over 5 mm.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=cfg.figure_dpi); plt.close(fig)


def fig_lanes(path, lanes, t_total, cfg):
    """Front position (top panel) and speed (bottom panel) along several horizontal lines.
    lanes: list of (label, colour, x [m], t [s], v [m/s])."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True)
    for label, col, x, t, v in lanes:
        ax.plot(t, x * MM, color=col, lw=2, label=f"{label}: reaches x = {x[-1]*MM:.0f} mm at {t[-1]:.0f} s")
        ax2.plot(t, v * MM, color=col, lw=1.5, label=label)
    ax.set_ylabel("front position x_f [mm]"); ax.set_ylim(0, cfg.L * MM); ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax2.set_yscale("log"); ax2.set_ylabel("front speed dx_f/dt [mm/s]"); ax2.grid(alpha=0.3, which="both")
    ax2.set_xlabel("time t [s]  (t = 0: resin enters at gate)"); ax2.set_xlim(0, t_total * 1.02)
    ax2.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Flow-front progression along horizontal lines (gate side → vent side)\n"
                 f"whole part filled at t = {t_total:.0f} s = {t_total/60:.1f} min", fontsize=11)
    _caption(fig, cfg, "\nSpeed smoothed over 5 mm.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=cfg.figure_dpi); plt.close(fig)


def fig_compare(path, tri, t_fill, t_total, tri_ref, t_ref, t_ref_total, cfg, ref_label,
                step_s=120.0):
    """Overlay isochrones of this case (solid, coloured) on a reference case (dashed grey),
    every `step_s` seconds, so asymmetry relative to the uniform case is visible."""
    fig, ax = plt.subplots(figsize=(11, 8.4))
    tmax = max(t_total, t_ref_total)
    levels = np.arange(step_s, tmax, step_s)
    norm = Normalize(0.0, tmax); cmap = plt.get_cmap("viridis")
    ax.tricontour(tri_ref, t_ref, levels=levels, colors="0.45", linestyles="--", linewidths=0.9)
    cs = ax.tricontour(tri, t_fill, levels=levels, cmap=cmap, norm=norm, linewidths=1.6)
    ax.clabel(cs, fmt=lambda v: f"{v/60:.0f} min", fontsize=7.5, inline=True)
    _ports(ax, cfg); _axes(ax, cfg)
    h = [Line2D([], [], color=cmap(0.6), lw=1.6, label=f"{cfg.experiment} (fill {t_total:.0f} s)"),
         Line2D([], [], color="0.45", ls="--", lw=0.9, label=f"{ref_label} (fill {t_ref_total:.0f} s)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=9)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("fill time t [s]")
    ax.set_title(f"Isochrones every {step_s/60:.0f} min: {cfg.experiment} vs uniform K", fontsize=11)
    _caption(fig, cfg, f"\nDashed grey = {ref_label}.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=cfg.figure_dpi); plt.close(fig)


def fig_sweep_fill_time(path, rows, cfg):
    """Total fill time vs permeability (log-log) for the uniform-K sweep.
    rows: list of dicts with Kx, Ky, t_fill [s]. K_eff = sqrt(Kx*Ky) [m^2]."""
    Ke = np.array([np.sqrt(r["Kx"] * r["Ky"]) for r in rows])
    tf = np.array([r["t_fill"] for r in rows])
    o = np.argsort(Ke); Ke, tf = Ke[o], tf[o]; rows = [rows[i] for i in o]
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    kk = np.logspace(np.log10(Ke.min() / 2), np.log10(Ke.max() * 2), 50)
    c = np.exp(np.mean(np.log(tf * Ke)))            # best constant in t = c / K_eff
    ax.loglog(kk, c / kk / 60, color="0.6", ls="--", lw=1.2, label="t_fill ∝ 1/K (slope −1)")
    ax.loglog(Ke, tf / 60, "o-", color="tab:blue", ms=8, lw=1.8, label="FAN simulation (uniform K)")
    for r, k, t in zip(rows, Ke, tf):
        ax.annotate(f"Kx={r['Kx']:.0e}\nKy={r['Ky']:.0e}\n{t/60:.1f} min ({t:.0f} s)",
                    (k, t / 60), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("effective permeability K_eff = √(Kx·Ky)  [m²]")
    ax.set_ylabel("total fill time [min]")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower left")
    ax.set_title("Permeability sensitivity: total fill time vs K (all Table-1 pairs)\n"
                 "fill time ∝ φ·μ / (K·Δp): 10× higher K → 10× shorter fill", fontsize=11)
    _caption(fig, cfg)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=cfg.figure_dpi); plt.close(fig)


def fig_sweep_shapes(path, runs, cfg):
    """Isochrones of t/t_fill (every 10 %) for each sweep member: shape depends only on Kx/Ky."""
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    for ax, r in zip(np.atleast_1d(axes), runs):
        cs = ax.tricontour(r["tri"], r["t_fill"] / r["t_total"], levels=np.arange(0.1, 1.0, 0.1),
                           cmap="viridis", norm=Normalize(0, 1), linewidths=1.2)
        ax.add_patch(Rectangle((0, 0), cfg.L * MM, cfg.W * MM, fill=False, ec="k"))
        ax.plot(0, cfg.W / 2 * MM, "o", mfc="tab:red", mec="k"); ax.plot(cfg.L * MM, cfg.W / 2 * MM, "s", mfc="tab:blue", mec="k")
        ax.set_aspect("equal"); ax.set_xlabel("x [mm]")
        ax.set_title(f"Kx={r['Kx']:.0e}, Ky={r['Ky']:.0e} m²\nKx/Ky = {r['Kx']/r['Ky']:.1f}, "
                     f"t_fill = {r['t_total']/60:.1f} min", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("y [mm]")
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
    fig.colorbar(sm, ax=axes, shrink=0.8, label="normalised fill time t / t_fill [-]")
    fig.suptitle("Front shapes in normalised time (lines every 10 % of t_fill): "
                 "K scales the time axis; only the anisotropy Kx/Ky changes the shape", fontsize=11)
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight"); plt.close(fig)
