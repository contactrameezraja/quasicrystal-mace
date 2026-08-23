"""Figure: workflow diagram (fig2_workflow_fullpage.png), full-A4 layout.

Self-contained drawing; no data dependencies. Gate hexagons, solid process
arrows, dashed prediction/caveat arrows carried against the flow.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9})

fig, ax = plt.subplots(figsize=(8.3, 11.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 132)
ax.axis("off")
FS = 9
FSI = 8.5


def box(cx, cy, w, h, text, fs=FS):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, lw=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def hexa(cx, cy, w, h, text, fs=FS):
    d = h / 2
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
           (cx + w / 2 + d, cy), (cx + w / 2, cy + h / 2),
           (cx - w / 2, cy + h / 2), (cx - w / 2 - d, cy)]
    ax.add_patch(Polygon(pts, fill=False, lw=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def arr(p1, p2, dashed=False):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                 lw=0.9, linestyle=(0, (4, 3)) if dashed else "solid",
                 color="k", shrinkA=0, shrinkB=0))


def dline(p1, p2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k", lw=0.9,
            linestyle=(0, (4, 3)))


box(28, 126, 37, 7.5, "Deposited approximants\nX-phase (26), cubic Al-Co-Ni (60),\nW-phase (265)")
hexa(28, 115.2, 36, 7.5, "Verification: atom count,\ncomposition, volume; failure halts\nbefore any force call, \u00a73.2")
box(28, 104.2, 36, 7.5, "W-phase primitive to conventional\naxes, Eq. (1); fixes box extents\nand band-path labels")
box(28, 94.2, 30, 5.8, "Relaxation and supercell\nconstruction, Eq. (3)")
box(14.5, 83.5, 26, 7.5, "Harmonic route\nforce constants by\nfinite displacement")
box(42, 83.5, 29, 7.5, "Dynamics route\n300 K protocol, \u00a73.5\nthermostat off in production")
box(14.5, 69.5, 26, 9.4, "Diagonalise, Eq. (4)\n$g(E)$, projections\n$P(s)$, Eqs. (5), (6)\nband paths and gaps")
box(42, 69.5, 28, 9.4, "Velocity autocorrelation\nEqs. (7), (8) $\\to g(E)$\nprojections\nsegment statistics")
hexa(42, 57, 20, 7.2, "Vibrating about\nfixed sites?\nEqs. (11), (12)")
box(28, 45.5, 40, 7.5, "Comparison of the two routes\ntheir agreement is a designed\ncheck, not a coincidence")
box(28, 34.5, 36, 7.5, "Spectral floor, Definition 1, and\nthe measured size law, then the $\\tau$\nratio test against its null model, \u00a73.7")
box(28, 23.5, 31, 7.5, "Uncertainty decomposed, \u00a73.9\nstatistical, finite size,\nnumerical precision, model")

RX = 80
box(RX, 126, 28, 5.8, "Three candidate\npotentials, Eq. (2)")
hexa(RX, 117, 22, 5.8, "Relax, and remain\nintact at 300 K?")
box(RX, 108, 26, 5.8, "Primary potential\ncarried forward")
box(RX, 92, 25, 7.5, "Measured elastic\nconstants $\\to$ Eq. (10)\npredicted floor")
ax.text(84, 85.4, "supervisor-directed\nsearch", ha="center", va="center",
        fontsize=FSI, style="italic")
box(RX, 78.5, 27, 9.4, "Monte Carlo\ngeneration, Eq. (13),\n\u00a73.10; melt-quench,\nreplica exchange, fixed site", fs=8.6)
hexa(RX, 68.5, 20, 5.8, "Does the chemistry\nimprove on the input?", fs=8.6)
box(RX, 59.5, 25, 5.8, "Composition retargeting\n\u00a73.2.2, box held fixed", fs=8.6)
ax.text(RX, 54.3, "carries chemical disorder that\nthe reference structures do not",
        ha="center", va="top", fontsize=FSI, style="italic")

arr((28, 122.25), (28, 119.55))
arr((28, 110.85), (28, 107.95))
arr((28, 100.45), (28, 97.1))
arr((21, 91.3), (14.5, 87.25))
arr((35, 91.3), (42, 87.25))
arr((14.5, 79.75), (14.5, 74.2))
arr((42, 79.75), (42, 74.2))
arr((42, 64.8), (42, 61.0))
arr((14.5, 64.8), (21, 49.25))
arr((42, 53.4), (35, 49.25))
arr((28, 41.75), (28, 38.25))
arr((28, 30.75), (28, 27.25))

arr((RX, 123.1), (RX, 120.3))
arr((RX, 114.1), (RX, 110.9))
arr((67, 108), (41, 97.0))
arr((72, 88.25), (72, 83.2))
arr((RX, 73.8), (RX, 71.75))
arr((RX, 65.6), (RX, 62.4))

dline((67.5, 92), (58.2, 92))
dline((58.2, 92), (58.2, 34.5))
arr((58.2, 34.5), (46, 34.5), dashed=True)
dline((67.5, 59.5), (60.6, 59.5))
dline((60.6, 59.5), (60.6, 83.5))
arr((60.6, 83.5), (56.5, 83.5), dashed=True)

ax.plot([12, 20], [13, 13], "k-", lw=0.9)
ax.text(21.5, 13, "process flow", va="center", fontsize=FS)
ax.plot([42, 50], [13, 13], "k", lw=0.9, linestyle=(0, (4, 3)))
ax.text(51.5, 13, "prediction or caveat carried forward", va="center",
        fontsize=FS)
pts = [(13, 6.3), (19, 6.3), (20.5, 8), (19, 9.7), (13, 9.7), (11.5, 8)]
ax.add_patch(Polygon(pts, fill=False, lw=1.0))
ax.text(16, 8, "gate", ha="center", va="center", fontsize=FS)
ax.text(23, 8, "verification point at which failure halts the calculation",
        va="center", fontsize=FS)

fig.savefig("fig2_workflow_fullpage.png", dpi=200, bbox_inches="tight")
print("wrote fig2_workflow_fullpage.png")
