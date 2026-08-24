"""Figure 6: measured spectral floor against inverse longest box dimension.

Box lengths are those of the boxes actually simulated, from the run logs
(relaxed X-phase cell a = 12.064 A; W-phase relaxed conventional cell).
C = 175.5 meV A is the mean of E_onset * L_long over the three intermediate
X-phase boxes (171.9, 178.1, 176.6 meV A).

Usage: python make_fig_floor.py [--bw]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

BW = "--bw" in sys.argv

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

C = 175.5   # meV A

# (L_long in A from the run logs, measured onset in meV, label, offset)
x_pts = [(24.13, 6.24, "2x2x2", (-44, -3)),
         (36.19, 4.75, "3x3x3", (8, -3)),
         (48.26, 3.69, "4x4x4", (8, -3)),
         (96.51, 1.83, "8x4x4", (8, 0)),
         (120.64, 1.38, "10x4x3", (7, -11))]
w_pts = [(118.03, 1.32, "3x2x1", (-42, 3)),
         (236.06, 0.65, "6x2x1", (8, -4))]

band_c = "0.85" if BW else "#c6dbef"          # light grey / light blue
x_c    = "k"    if BW else "#1f4e79"          # black / deep blue
w_c    = "k"    if BW else "#c0392b"          # black / red
suffix = "_bw" if BW else ""

fig, ax = plt.subplots(figsize=(6.4, 4.6))
x = np.linspace(0.0005, 0.045, 200)

ax.fill_between(x, 169.0 * x, 190.0 * x, color=band_c, lw=0,
                label=r"$hv_T$, measured, 169–190 meV $\mathrm{\AA}$")
ax.plot(x, C * x, color="k", lw=1.0,
        label=r"$E = C/L_{\rm long}$, $C = 175.5$ meV $\mathrm{\AA}$")

for L, o, lab, off in x_pts:
    ax.plot(1 / L, o, "o", ms=6, color=x_c, zorder=5)
    ax.annotate(lab, (1 / L, o), textcoords="offset points",
                xytext=off, fontsize=8)
for L, o, lab, off in w_pts:
    ax.plot(1 / L, o, "s", ms=6, mfc="white", mec=w_c, zorder=5)
    ax.annotate(lab, (1 / L, o), textcoords="offset points",
                xytext=off, fontsize=8)
ax.plot([], [], "o", color=x_c, label="X-phase")
ax.plot([], [], "s", mfc="white", mec=w_c, label="W-phase")

ax.set_xlabel(r"$1/L_{\rm long}$ ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel(r"$E_{\rm onset}$ (meV)")
ax.set_xlim(0, 0.045)
ax.set_ylim(0, 7)
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
plt.tight_layout()
plt.savefig(f"fig_floor_scaling{suffix}.png", dpi=300, bbox_inches="tight")
print(f"wrote fig_floor_scaling{suffix}.png")
