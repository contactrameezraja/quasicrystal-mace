"""Figure: static structure factor of the 6x2x1 W-phase box (fig_sq.png).

Run from the repository root. Reads W-AlCoNi-265-relaxed.vasp, tiles it by
[[6,6,0],[-2,2,0],[0,0,1]] to the 6360-atom production box of Table 3, and
evaluates S(Q) = |sum_j exp(iQ.r_j)|^2 / N^2 only at the box-allowed
wavevectors Q = 2*pi*n/L along each axis. Panel (a), the in-plane long axis;
panel (b), the stacking direction. Dashed red lines are the measured ladder
of Matsuura et al. converted to wavevector by Q = 2*pi*E / 155.9 meV A, the
W-phase prefactor of Section 4.5.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "font.size": 9})

# ---- structure: relaxed primitive cell, tiled to the 6x2x1 box
lines = open("W-AlCoNi-265-relaxed.vasp").read().splitlines()
scale = float(lines[1])
cell = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
n_atoms = sum(int(x) for x in lines[6].split())
assert n_atoms == 265
pos = np.array([[float(x) for x in lines[8 + i].split()[:3]]
                for i in range(n_atoms)])
if lines[7].strip().lower().startswith("d"):
    pos = pos @ cell

M = np.array([[6, 6, 0], [-2, 2, 0], [0, 0, 1]])
A = M @ cell
Ainv = np.linalg.inv(A)
shifts = [t for n1 in range(-4, 9) for n2 in range(-4, 9)
          for t in [n1 * cell[0] + n2 * cell[1]]
          if np.all((t @ Ainv) > -1e-9) and np.all((t @ Ainv) < 1 - 1e-9)]
assert len(shifts) == 24
r = (pos[None, :, :] + np.array(shifts)[:, None, :]).reshape(-1, 3)
N = len(r)
assert N == 6360
B = 2 * np.pi * np.linalg.inv(A).T
q1 = np.linalg.norm(B[0])


def S_at(Q):
    return np.abs(np.exp(1j * (Q @ r.T)).sum(axis=1)) ** 2 / N ** 2


u1 = B[0] / q1
u2 = B[1] / np.linalg.norm(B[1])
qa = np.arange(1, 113) * q1
Sa = S_at(qa[:, None] * u1[None, :])
qb = np.arange(1, 13) * np.linalg.norm(B[1])
Sb = S_at(qb[:, None] * u2[None, :])

rungs = [2.15, 1.33, 0.82, 0.51, 0.31, 0.19, 0.12]
rQ = [2 * np.pi * E / 155.9 for E in rungs]
RED = "#c62828"
SHADE = "#e8eaf0"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 5.6))
floor = 3e-7
xlo = 2.8e-3

ax1.axvspan(xlo, q1, color=SHADE, zorder=0)
mask = Sa > 1e-6
ax1.vlines(qa[mask], floor, Sa[mask], color="k", lw=1.4, zorder=3)
ax1.plot(qa[mask], Sa[mask], "ko", ms=3.2, zorder=4)
ax1.vlines(qa[~mask & (qa < 0.7)], floor, floor * 1.6, color="0.55", lw=0.6)
for E, Q in zip(rungs, rQ):
    ax1.axvline(Q, color=RED, ls="--", lw=0.8, zorder=1, alpha=0.9)
    bg = SHADE if Q < q1 else "white"
    ax1.text(Q, 4.5e-3, f"{E}", rotation=90, ha="center", va="top",
             fontsize=6.5, color=RED,
             bbox=dict(facecolor=bg, edgecolor="none", pad=0.6))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(xlo, 3.0)
ax1.set_ylim(floor, 1e-2)
ax1.set_ylabel(r"$S(Q)/N^{2}$")
ax1.set_title("(a)  in-plane long axis, 6\u00d72\u00d71 W-phase box, 6360 atoms",
              loc="left", fontsize=9)
ax1.text(np.sqrt(xlo * q1), 3e-6,
         "no allowed $Q$\nbelow $2\\pi/L_{\\mathrm{box}}$",
         ha="center", va="center", fontsize=7.5,
         bbox=dict(facecolor=SHADE, edgecolor="none", pad=1.5))
ax1.plot([], [], "ko-", ms=3.2, lw=1.4, label="Bragg reflections of the box")
ax1.plot([], [], color="0.55", lw=0.6, label="allowed $Q$, zero intensity")
ax1.plot([], [], color=RED, ls="--", lw=0.8,
         label="measured ladder [2] at 155.9 meV\u2009\u00c5")
leg = ax1.legend(frameon=False, loc="upper right", fontsize=7.2,
                 bbox_to_anchor=(0.995, 0.97))
for t in leg.get_texts():
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.5))

ax2.vlines(qb, 3e-7, np.maximum(Sb, 3e-7), color="k", lw=1.4, zorder=3)
ax2.plot(qb[Sb > 1e-6], Sb[Sb > 1e-6], "ko", ms=3.2, zorder=4)
for m in range(1, 4):
    ax2.axvline(2 * np.pi * m / 4.051, color="0.45", ls=":", lw=0.7, zorder=1)
ax2.plot([], [], color="0.45", ls=":", lw=0.7,
         label=r"$2\pi m/4.051\ \mathrm{\AA}^{-1}$")
ax2.set_yscale("log")
ax2.set_xlim(0, 5.0)
ax2.set_ylim(3e-7, 2.0)
ax2.set_xlabel(r"$Q$ $(\mathrm{\AA}^{-1})$")
ax2.set_ylabel(r"$S(Q)/N^{2}$")
ax2.set_title("(b)  stacking direction", loc="left", fontsize=9)
leg2 = ax2.legend(frameon=False, loc="center right", fontsize=8)
for t in leg2.get_texts():
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.5))
fig.tight_layout()
fig.savefig("fig_sq.png", dpi=200)
print("wrote fig_sq.png")
