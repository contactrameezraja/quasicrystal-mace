"""Figure: pair distribution functions of the X-phase (fig_gr.png).

Run from the repository root. Reads the archived MSD/g(r) reductions
md_2x2x2_s11_msd_gr.npz and md_3x3x3_s12_msd_gr.npz (Section 3.8 pipeline,
msd_gr.py) and plots the total g(r) of both boxes with the partials of the
702-atom box. No recomputation; archived arrays only.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "font.size": 9})

d2 = np.load("md_2x2x2_s11_msd_gr.npz")
d3 = np.load("md_3x3x3_s12_msd_gr.npz")
r = d3["r"]

fig, axs = plt.subplots(3, 1, figsize=(6.0, 6.2), sharex=True,
                        gridspec_kw=dict(hspace=0.12))
ax = axs[0]
ax.plot(d2["r"], d2["g_total"], color="0.6", lw=0.9,
        label=r"$2\times2\times2$, 208 atoms")
ax.plot(r, d3["g_total"], "k-", lw=1.0,
        label=r"$3\times3\times3$, 702 atoms")
ax.set_ylabel(r"$g(r)$")
ax.legend(frameon=False, fontsize=8)
ax.text(0.02, 0.86, "(a)  total", transform=ax.transAxes, fontsize=8.5)

ax = axs[1]
ax.plot(r, d3["g_Al-Al"], "k-", lw=1.0, label="Al-Al")
ax.plot(r, d3["g_Al-Co"], "k--", lw=0.9, label="Al-Co")
ax.plot(r, d3["g_Al-Ni"], "k:", lw=1.1, label="Al-Ni")
ax.set_ylabel(r"$g_{\alpha\beta}(r)$")
ax.legend(frameon=False, fontsize=8)
ax.text(0.02, 0.86, "(b)  aluminium partials", transform=ax.transAxes,
        fontsize=8.5)

ax = axs[2]
ax.plot(r, d3["g_Co-Co"], "k-", lw=1.0, label="Co-Co")
ax.plot(r, d3["g_Co-Ni"], "k--", lw=0.9, label="Co-Ni")
ax.plot(r, d3["g_Ni-Ni"], "k:", lw=1.1, label="Ni-Ni")
ax.set_ylabel(r"$g_{\alpha\beta}(r)$")
ax.set_xlabel(r"$r$ $(\mathrm{\AA})$")
ax.legend(frameon=False, fontsize=8)
ax.text(0.02, 0.86, "(c)  transition-metal partials", transform=ax.transAxes,
        fontsize=8.5)
ax.set_xlim(1.5, 6.0)

fig.savefig("fig_gr.png", dpi=200, bbox_inches="tight")
print("wrote fig_gr.png")
