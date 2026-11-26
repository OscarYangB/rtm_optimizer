"""Plot the 2D RTM domain with inlet/outlet positions for geometry check.

Utility (not an experiment). Run from anywhere:  python3 utils/plot_geometry.py
Writes outputs/geometry_check.png (path resolved relative to the project root).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

L, W = 342.9, 241.0          # mm
inlet = (0.0, W / 2)         # middle of left short edge
outlet = (L, W / 2)          # middle of right short edge

fig, ax = plt.subplots(figsize=(9, 6.8))
ax.add_patch(Rectangle((0, 0), L, W, fc="#e8eef5", ec="black", lw=1.5, label="Preform / mould cavity"))
ax.plot(*inlet, "o", ms=12, color="tab:red", label=f"Inlet (gate) {inlet[0]:.1f}, {inlet[1]:.1f} mm")
ax.plot(*outlet, "s", ms=12, color="tab:blue", label=f"Outlet (vent) {outlet[0]:.1f}, {outlet[1]:.1f} mm")
ax.annotate("", xy=(L * 0.7, W / 2), xytext=(L * 0.3, W / 2),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
ax.text(L / 2, W / 2 + 6, "main flow direction", ha="center", color="gray")
ax.axhline(W / 2, ls=":", color="gray", lw=0.8)

# dimensions
ax.annotate("", xy=(0, -18), xytext=(L, -18), arrowprops=dict(arrowstyle="<->"))
ax.text(L / 2, -28, f"L = {L} mm", ha="center")
ax.annotate("", xy=(-18, 0), xytext=(-18, W), arrowprops=dict(arrowstyle="<->"))
ax.text(-28, W / 2, f"W = {W} mm", va="center", rotation=90)
ax.text(L / 2, W + 6, "wall: dp/dn = 0", ha="center", color="dimgray")
ax.text(L / 2, 6, "wall: dp/dn = 0", ha="center", color="dimgray")

ax.set_xlim(-45, L + 25); ax.set_ylim(-40, W + 45)
ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
ax.set_title("RTM domain geometry (top view) — inlet/outlet check")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # RTM-Optimizer/
out = os.path.join(ROOT, "outputs", "geometry_check.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {os.path.relpath(out, ROOT)}")
