"""Stacking-direction gap centres against gap index, Section 4.6.3.

Data are the six recorded gap centres from the band-path extraction at the
stacking zone boundary q = (-0.5, 0.5, 0), Section 3.4 definition (interval
wider than 0.3 meV between consecutive sorted mode energies below 20 meV,
centres reported). Source: the Blythe band-path run of 16 August 2026; the
raw band-structure file is not in the public repository, and the centres
are archived in stacking_bandpath_gaps.txt alongside this script.

The figure compares the recorded centres against two references sharing
the first centre, a uniform (arithmetic) sequence at the mean spacing of
the first five centres, and the geometric sequence a tau hierarchy would
produce, drawn to its third member so it terminates inside the frame.
The first five centres follow the uniform line, successive ratios 1.080
to 1.097; the sixth departs from both references.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = (1 + 5 ** 0.5) / 2

centres = np.loadtxt("stacking_bandpath_gaps.txt")[:, 1]
n = np.arange(1, len(centres) + 1)

step = np.mean(np.diff(centres[:5]))          # 0.431 meV
arith = centres[0] + step * (n - 1)           # uniform reference
ng = np.arange(1, 4)                          # geometric reference to n = 3
geom = centres[0] * TAU ** (ng - 1)           # 4.419, 7.150, 11.569

print("mean spacing of first five: %.3f meV" % step)
print("successive ratios of first five centres:",
      ", ".join("%.3f" % r for r in centres[1:5] / centres[0:4]))

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "ytick.minor.visible": True, "ytick.minor.size": 2,
    "legend.frameon": False,
})

fig, ax = plt.subplots(figsize=(4.4, 3.5))
ax.plot(n, arith, color="black", lw=0.9,
        label="uniform spacing, %.2f meV per step" % step)
ax.plot(ng, geom, color="black", lw=0.9, ls=":", marker="x", ms=5, mew=0.9,
        label=r"geometric, centre $\times\,\tau$ per step")
ax.plot(n, centres, "o", color="black", ms=4.5, mfc="white", mew=0.9,
        label="recorded gap centres")
ax.set_xlim(0.6, 6.4)
ax.set_ylim(4.0, 12.2)
ax.set_xticks(n)
ax.set_xlabel("gap index along the stacking direction")
ax.set_ylabel("gap centre (meV)")
ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.84), fontsize=8)

plt.tight_layout(pad=0.4)
plt.savefig("fig_stacking_centres.png", dpi=300, bbox_inches="tight")
plt.savefig("fig_stacking_centres.pdf", bbox_inches="tight")
print("wrote fig_stacking_centres.png / .pdf")
