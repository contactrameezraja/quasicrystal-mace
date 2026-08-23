"""Figure: the pseudo-gap prediction evaluated for the Fibonacci chain
(fig_pseudogap_mechanism.png).

Self-contained; generates the 378-atom Fibonacci chain of Figure 1, computes
its Hann-windowed diffracted intensity, and draws the two-branch mixing form
at each pseudo-Brillouin zone boundary G_n/2 with gap widths proportional to
the square root of the computed intensities. An evaluation of the prediction
of Niizeki-Akamatsu, not a diagonalisation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "font.size": 9})
TAU = (1 + np.sqrt(5)) / 2


def fib_word_len(n):
    a, b = "L", "LS"
    while len(b) < n:
        a, b = b, b + a
    return b[:n]


word = fib_word_len(378)
sp = np.array([TAU if c == "L" else 1.0 for c in word])
x = np.concatenate([[0], np.cumsum(sp)[:-1]])
w = np.hanning(len(x))
Qg = np.linspace(0.02, 2.6, 4000)
F = np.abs((w[None, :] * np.exp(1j * np.outer(Qg, x))).sum(axis=1)) ** 2 \
    / (w.sum()) ** 2

Gstar = Qg[np.argmax(F)]
Gs = []
for n in range(3, -1, -1):
    g0 = Gstar * TAU ** (-n)
    i0 = np.argmin(np.abs(Qg - g0))
    j = i0 - 10 + np.argmax(F[i0 - 10:i0 + 10])
    Gs.append(Qg[j])
Is = [F[np.argmin(np.abs(Qg - g))] for g in Gs]
print("reflection ratios:", [round(Gs[i + 1] / Gs[i], 3) for i in range(3)])

v = 1.0
qn = np.array([g / 2 for g in Gs])
qmax = qn[-1] * 1.30
delta = np.maximum(
    np.array([0.12 * np.sqrt(i / max(Is)) * v * qn[-1] for i in Is]),
    0.0045 * v * qmax)
qs = np.linspace(0, qmax, 6000)
wq = v * qs.copy()
for qc, d in zip(qn, delta):
    s = np.sign(qs - qc)
    s[s == 0] = 1
    wq += s * (np.sqrt((v * (qs - qc)) ** 2 + d ** 2) - np.abs(v * (qs - qc)))

fig, (axT, axB) = plt.subplots(2, 1, figsize=(6.0, 5.4),
                               height_ratios=[1, 1.7],
                               gridspec_kw=dict(hspace=0.10))
axT.plot(Qg, F, "k-", lw=0.7)
axT.set_yscale("log")
axT.set_ylim(1e-5, 3e-1)
axT.set_xlim(0, 2 * qmax)
axT.set_ylabel(r"$S(Q)$")
axT.set_xticks([])
axT.tick_params(labelbottom=False)
for i, (g, I) in enumerate(zip(Gs, Is)):
    axT.annotate(rf"$G_{{{i+1}}}$", (g, I * 2.0), ha="center", fontsize=8)
    axT.plot([g, g], [1e-5, I], color="k", ls=":", lw=0.5)
axT.annotate("", xy=(Gs[3], 2.2e-2), xytext=(Gs[2], 2.2e-2),
             arrowprops=dict(arrowstyle="<->", lw=0.7))
axT.annotate(r"$\times\,\tau$", xy=(np.sqrt(Gs[2] * Gs[3]), 3.3e-2),
             ha="center", fontsize=8)
axT.text(0.02, 0.88, "(a)  strong reflections of the Fibonacci chain",
         transform=axT.transAxes, fontsize=8.5)

for qc, d in zip(qn, delta):
    axB.axhspan(v * qc - d, v * qc + d, color="0.88", zorder=0)
    axB.plot([qc, qc], [0, v * qc - d], color="k", ls=":", lw=0.5)
axB.plot(qs, v * qs, "k:", lw=0.6)
axB.plot(qs, wq, "k-", lw=1.2)
for qc, lab in zip(qn, [r"$G_1/2$", r"$G_2/2$", r"$G_3/2$", r"$G_4/2$"]):
    axB.annotate(lab, (qc, -0.045 * v * qmax), ha="center", va="top",
                 fontsize=8, annotation_clip=False)
axB.annotate(r"gap width $\Delta \propto \sqrt{I(G)}$",
             xy=(0.22 * qmax, 0.84 * v * qmax), fontsize=8.5)
axB.set_xlim(0, qmax)
axB.set_ylim(0, v * qmax * 1.05)
axB.set_xlabel(r"$q$", loc="right")
axB.set_ylabel(r"$E$", rotation=0, labelpad=10)
axB.set_xticks([])
axB.set_yticks([])
axB.text(0.02, 0.93,
         "(b)  acoustic branch, pseudo-zone boundaries at $G_n/2$",
         transform=axB.transAxes, fontsize=8.5)
fig.savefig("fig_pseudogap_mechanism.png", dpi=200, bbox_inches="tight")
print("wrote fig_pseudogap_mechanism.png")
