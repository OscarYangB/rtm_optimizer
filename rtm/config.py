"""Single source of truth for all RTM Part I parameters (SI units throughout).

Modelling assumptions (Okabe et al., Compos. Part A 92 (2017), Sec. 2.1)
-----------------------------------------------------------------------
A1  Darcy flow (creeping, viscous-dominated). Re is computed and printed at run time.
A2  Incompressible, Newtonian resin; constant viscosity during filling (no cure/temperature).
A3  Rigid preform, uniform porosity phi and uniform permeability K. K may be anisotropic;
    off-diagonal terms are zero in the x-y axes (K_xy kept as a parameter, default 0).
A4  Thin part -> 2D in-plane model, no through-thickness flow. Every quantity is per unit
    thickness, so the thickness h cancels from the fill times (h only matters for Q in m^3/s).
A5  The flow front is an air/resin mixture: the band of partially filled CVs (0 < f < 1).
    The air ahead of the front is evacuated to p_vent (not compressed/trapped), so p = p_vent
    on partially filled and empty nodes. Capillary pressure, intra-tow unsaturated flow and
    gravity are ignored.
A6  No race-tracking along the mould edges.
A7  The front advances with the interstitial velocity u/phi: CV volumes are PORE volumes
    phi * V_i (correction to Okabe Eq. 7, which uses V_i).

Values marked PLACEHOLDER are not measured / not provided ([TBD] in the prompt). They are
named parameters so the team can replace them; every figure prints K, phi and mu.
"""
import re
from dataclasses import dataclass, field, asdict

ATM = 101_325.0  # Pa, 1 standard atmosphere
OUTPUTS_ROOT = "outputs"
# Naming convention (prompt/generic_skills.md): experiment_<#>_<short_description>,
# snake_case description (symbols such as K may keep their case), e.g. experiment_2_two_region_K.
EXPERIMENT_RE = re.compile(r"^experiment_(\d+)_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")


@dataclass
class RTMConfig:
    # ---- geometry [m] (converted from mm: 342.9 mm = 0.3429 m, 241 mm = 0.241 m) ----
    L: float = 0.3429                 # domain length in x [m]
    W: float = 0.241                  # domain width in y [m]
    thickness_h: float | None = None  # [m] [TBD] - not needed for fill times (A4)
    # Gate and vent are modelled as SEMICIRCULAR cut-outs centred on the short-edge midpoints.
    # PLACEHOLDER: port diameters are [TBD]; 10 mm (= 0.010 m) is used until measured.
    gate_diameter: float = 0.010      # [m] PLACEHOLDER
    vent_diameter: float = 0.010      # [m] PLACEHOLDER

    # ---- mesh [m] ----
    h_far: float = 0.004              # element size away from ports [m] (4 mm)
    h_port: float = 0.0008            # element size at the port arcs [m] (0.8 mm)
    refine_radius: float = 0.04       # distance over which size grows h_port -> h_far [m]

    # ---- pressures [Pa, absolute] ----
    p_inlet: float = ATM              # gate open to the atmosphere
    vacuum_gauge: float = 85_000.0    # pump reaches -85 kPa gauge (baseline, realistic)
    p_vent_ideal: float = 0.0         # perfect vacuum bound

    # ---- resin: Aeropoxy PR2032 (resin-only datasheet value at 77 F / 25 C) ----
    mu: float = 1.65                  # [Pa s] = 1650 cP (1 cP = 1e-3 Pa s). Mixed value [TBD]
    rho: float = 1150.0               # [kg/m^3] from specific gravity 1.15 (Re check only)

    # ---- preform: 2 layers 14k carbon (weave/areal weight [TBD]) ----
    # Permeability and porosity from the literature sensitivity table supplied by the team
    # (logs/attachments/2026-11-26_sensitivity_table1_K_phi.png). NOT measured on our preform.
    # Table pairs (Kx, Ky) [m^2]: (2e-10,1e-10) (6e-10,5e-10) (2e-9,1e-9) (6e-9,5e-9) (6e-8,5e-8).
    # Default = the median pair (2e-9, 1e-9).
    K_xx: float = 2.0e-9              # [m^2] literature Table 1
    K_yy: float = 1.0e-9              # [m^2] literature Table 1
    K_xy: float = 0.0                 # [m^2] zero in x-y axes (A3)
    # Optional TWO-REGION case (relaxes A3): diagonal K = (Kx, Ky) per region, K_bottom for
    # y < W/2 and K_top for y > W/2, split along the gate-vent centreline. None -> uniform K.
    K_top: tuple | None = None        # (Kx, Ky) [m^2] HYPOTHETICAL layout, Table-1 values
    K_bottom: tuple | None = None     # (Kx, Ky) [m^2]
    phi: float = 0.498                # [-] porosity (= 1 - V_f), literature Table 1
    fibre_diameter: float = 7.0e-6    # [m] typical carbon filament, Re length scale only

    # ---- outputs ----
    isochrone_interval_s: float = 60.0
    pressure_snapshot_fractions: tuple = (0.10, 0.35, 0.65, 0.95)  # filled-area fractions
    figure_dpi: int = 200
    experiment: str = "experiment_1_uniform_K"   # -> outputs/<experiment>/{figures,data}/
    sub_run: str | None = None        # sweep member -> outputs/<experiment>/runs/<sub_run>/

    def __post_init__(self):
        if not EXPERIMENT_RE.match(self.experiment):
            raise ValueError(f"experiment name {self.experiment!r} must follow "
                             "experiment_<#>_<short_description>")

    # ---------------- derived ----------------
    @property
    def experiment_dir(self) -> str:
        base = f"{OUTPUTS_ROOT}/{self.experiment}"
        return f"{base}/runs/{self.sub_run}" if self.sub_run else base

    @property
    def figures_dir(self) -> str:
        return f"{self.experiment_dir}/figures"

    @property
    def data_dir(self) -> str:
        return f"{self.experiment_dir}/data"

    @property
    def p_vent(self) -> float:
        """Baseline vent pressure [Pa abs] = 101325 - 85000 = 16325 Pa."""
        return self.p_inlet - self.vacuum_gauge

    @property
    def dp(self) -> float:
        return self.p_inlet - self.p_vent

    @property
    def dp_ideal(self) -> float:
        return self.p_inlet - self.p_vent_ideal

    @property
    def two_region(self) -> bool:
        return self.K_top is not None and self.K_bottom is not None

    @property
    def K_max(self) -> float:
        return max(*self.K_top, *self.K_bottom) if self.two_region else max(self.K_xx, self.K_yy)

    @property
    def K(self):
        return [[self.K_xx, self.K_xy], [self.K_xy, self.K_yy]]

    def param_caption(self) -> str:
        k = (f"HYPOTHETICAL layout: K_top = diag({self.K_top[0]:.3g}, {self.K_top[1]:.3g}), "
             f"K_bottom = diag({self.K_bottom[0]:.3g}, {self.K_bottom[1]:.3g}) m² (split y = W/2),  "
             if self.two_region else
             f"K = diag({self.K_xx:.0e}, {self.K_yy:.0e}) m² (lit. Table 1),  ")
        return (k + f"φ = {self.phi:g} (lit.),  μ = {self.mu:g} Pa·s (resin only),  "
                f"Δp = {self.dp/1e3:.1f} kPa (p_vent = {self.p_vent/1e3:.2f} kPa abs),  "
                f"placeholder port Ø = {self.gate_diameter*1e3:g} mm")

    def to_dict(self):
        d = asdict(self)
        d.update(figures_dir=self.figures_dir, data_dir=self.data_dir, p_vent=self.p_vent, dp=self.dp, dp_ideal=self.dp_ideal)
        return d


# Inputs still needed from the team (printed at run time).
NEEDED_INPUTS = [
    ("thickness_h", "Part thickness [mm]", "not needed for fill time (2D, per unit thickness)"),
    ("gate_diameter / vent_diameter", "Port diameters [mm]", "PLACEHOLDER 10 mm"),
    ("inlet model", "Line vs point/semicircle gate", "PLACEHOLDER semicircular cut-out"),
    ("mu", "Mixed resin+hardener viscosity at 25 C", "resin-only 1.65 Pa s"),
    ("K_xx, K_yy", "Measured permeability of 2-layer 14k preform", "literature Table 1: 2e-9, 1e-9 m^2"),
    ("phi", "Measured porosity / fibre volume fraction", "literature Table 1: 0.498"),
    ("weave, areal weight", "Fabric architecture", "unused until K/phi measured"),
]
