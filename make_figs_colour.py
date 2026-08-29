"""Colour renders of Figures 5, 9, 11, 12, 14 and 15, from the archived
reduction outputs in this repository. Palette is fixed across the set:
blue #1f4e79 X-phase and crystalline, red #c0392b W-phase and quasiperiodic
motif, purple #6a51a3 the retargeted box, green #2e7d32 references,
light blue #c6dbef shaded windows. Each file is written under the name the
dissertation build includes.

Outputs
-------
    o2_md_supercell_vdos.png              Figure 5
    fig_participation.png                 Figure 9
    fig_crystal_vs_quasicrystal_vdos.png  Figure 11
    fig_composition_control.png           Figure 12
    fig_gr.png                            Figure 14
    fig_structure_search_validation.png   Figure 15

Usage: python make_figs_colour.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.frameon": False})

BLUE = "#1f4e79"; RED = "#c0392b"; PUR = "#6a51a3"
GRN = "#2e7d32"; BAND = "#c6dbef"; LBLUE = "#7bafd4"
tr = getattr(np, "trapezoid", None) or np.trapz


def norm(f, g):
    return g / tr(g, f)


# ---------------- Figure 5: the two routes and the element decomposition ----
fmd, smd = np.load("md_2x2x2_s11_vdos_total.npy"); smd = norm(fmd, smd)
dh = np.load("ph_x26_dos.npz")
eh, th = dh["energy"], norm(dh["energy"], dh["total"])
P = np.load("md_2x2x2_s11_vdos_partial.npz"); fp = P["freq_meV"]

fig = plt.figure(figsize=(11, 4.2))
gs = gridspec.GridSpec(1, 2, wspace=0.22)
ax = fig.add_subplot(gs[0])
ax.plot(fmd[fmd <= 62], smd[fmd <= 62], lw=0.8, color=BLUE,
        label="MD, 200 ps, 300 K")
ax.plot(eh, th, lw=1.1, color="k", label="harmonic, 0 K")
ax.set_xlim(0, 60); ax.set_ylim(0, None)
ax.set_xlabel("$E$ (meV)"); ax.set_ylabel("$g(E)$ (normalised)")
ax.legend(fontsize=8.5, loc="upper right")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top")
inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.08)
for i, (el, col, n) in enumerate([("Al", BLUE, 144), ("Co", RED, 40),
                                  ("Ni", GRN, 24)]):
    axp = fig.add_subplot(inner[i])
    g = norm(fp, P[el])
    axp.plot(fp[fp <= 62], g[fp <= 62], lw=0.7, color=col)
    axp.axvline(30, color="k", lw=0.6, ls=":")
    axp.set_xlim(0, 60); axp.set_ylim(0, None); axp.set_yticks([])
    axp.text(0.985, 0.88, f"{el} ({n})", transform=axp.transAxes,
             ha="right", va="top", fontsize=9, color=col)
    if i < 2:
        axp.set_xticklabels([])
    if i == 0:
        axp.text(0.015, 0.88, "(b)", transform=axp.transAxes, va="top")
    if i == 1:
        axp.set_ylabel("$g(E)$ (arb. units)")
axp.set_xlabel("$E$ (meV)")
plt.savefig("o2_md_supercell_vdos.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- Figure 9: participation ratio -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(7, 7.6), sharex=True,
                        gridspec_kw={"hspace": 0.12})
for axx, (f, N, lab, col) in zip(
        axs, [("ph_w265_band222_modes.npz", 265, "(a)  W-phase, $N=265$", RED),
              ("ph_al13co4_modes.npz", 102, "(b)  Al$_{13}$Co$_4$, $N=102$", BLUE)]):
    d = np.load(f)
    E = d["frequencies"].ravel(); Pr = d["participation"].ravel()
    m = (E > 0) & (E <= 60)
    axx.axvspan(6, 10, color=BAND, lw=0, zorder=0)
    axx.plot(E[m], Pr[m], ".", ms=2.6, color=col, alpha=0.75, zorder=3)
    axx.axhline(1.0 / N, color="k", lw=0.6, ls=":")
    axx.text(59.3, 1.0 / N + 0.01, "$1/N$", fontsize=8, va="bottom", ha="right")
    axx.set_ylim(0, 1); axx.set_xlim(0, 60); axx.set_ylabel("$P(s)$")
    axx.text(0.985, 0.96, lab, transform=axx.transAxes, va="top", ha="right")
    tw = axx.twinx(); tw.set_ylim(0, N); tw.set_ylabel("$N_P$")
    tw.tick_params(direction="in")
axs[1].set_xlabel("$E$ (meV)")
plt.savefig("fig_participation.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- Figure 11: the matched boxes ------------------------------
fx, sx = np.load("md_xphase_10x4x3_vdos_total.npy"); sx = norm(fx, sx)
fw, sw = np.load("md_wphase_vdos_total.npy"); sw = norm(fw, sw)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(fx[fx <= 60], sx[fx <= 60], lw=0.8, color=BLUE,
           label=r"X-phase 10$\times$4$\times$3")
ax[0].plot(fw[fw <= 60], sw[fw <= 60], lw=0.8, color=RED,
           label=r"W-phase 3$\times$2$\times$1")
ax[0].set_xlim(0, 60); ax[0].set_ylim(0, None)
ax[0].set_xlabel("$E$ (meV)"); ax[0].set_ylabel("$g(E)$ (normalised)")
ax[0].legend(fontsize=8.5, loc="upper right")
ax[0].text(0.02, 0.97, "(a)", transform=ax[0].transAxes, va="top")
ax[1].plot(fx[fx <= 6], sx[fx <= 6], lw=0.9, color=BLUE)
ax[1].plot(fw[fw <= 6], sw[fw <= 6], lw=0.9, color=RED)
ax[1].set_xlim(0, 6); ax[1].set_ylim(0, 0.011)
ax[1].set_xlabel("$E$ (meV)"); ax[1].set_ylabel("$g(E)$ (normalised)")
ax[1].text(0.02, 0.97, "(b)", transform=ax[1].transAxes, va="top")
plt.tight_layout()
plt.savefig("fig_crystal_vs_quasicrystal_vdos.png", dpi=300,
            bbox_inches="tight")
plt.close()

# ---------------- Figure 12: the composition control ------------------------
specs = [("X-phase, native", "md_xphase_10x4x3_vdos_total.npy", BLUE),
         ("X box, W composition", "md_xbox_wcomp_total.npy", PUR),
         ("W-phase, native", "md_wphase_vdos_total.npy", RED)]
fig, axs = plt.subplots(3, 2, figsize=(10.5, 6.6),
                        gridspec_kw={"hspace": 0.10, "wspace": 0.16})
for i, (lab, path, col) in enumerate(specs):
    f, s = np.load(path); s = norm(f, s)
    aL, aR = axs[i]
    aL.plot(f[f <= 60], s[f <= 60], lw=0.6, color=col)
    aL.set_xlim(0, 60); aL.set_ylim(0, None); aL.set_yticks([])
    aR.plot(f[f <= 6], s[f <= 6], lw=0.7, color=col)
    aR.set_xlim(0, 6); aR.set_ylim(0, 0.011); aR.set_yticks([])
    for a in (aL, aR):
        a.text(0.985, 0.88, lab, transform=a.transAxes, ha="right",
               va="top", fontsize=9, color=col)
    if i == 0:
        aL.text(0.015, 0.88, "(a)", transform=aL.transAxes, va="top")
        aR.text(0.03, 0.88, "(b)", transform=aR.transAxes, va="top")
    if i < 2:
        aL.set_xticklabels([]); aR.set_xticklabels([])
    if i == 1:
        aL.set_ylabel("$g(E)$ (arb. units)")
axs[2, 0].set_xlabel("$E$ (meV)"); axs[2, 1].set_xlabel("$E$ (meV)")
plt.savefig("fig_composition_control.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- Figure 14: pair distribution functions --------------------
d2 = np.load("md_2x2x2_s11_msd_gr.npz")
d3 = np.load("md_3x3x3_s12_msd_gr.npz")
fig, axs = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True,
                        gridspec_kw={"hspace": 0.10})
axs[0].plot(d2["r"], d2["g_total"], lw=1.0, color=LBLUE,
            label=r"$2\times2\times2$, 208 atoms")
axs[0].plot(d3["r"], d3["g_total"], lw=1.0, color=BLUE,
            label=r"$3\times3\times3$, 702 atoms")
axs[0].set_ylabel("$g(r)$"); axs[0].legend(fontsize=8.5)
axs[0].text(0.02, 0.94, "(a)  total", transform=axs[0].transAxes, va="top")
for k, lab, col, ls in [("g_Al-Al", "Al-Al", BLUE, "-"),
                        ("g_Al-Co", "Al-Co", RED, "--"),
                        ("g_Al-Ni", "Al-Ni", GRN, ":")]:
    axs[1].plot(d3["r"], d3[k], lw=1.0, color=col, ls=ls, label=lab)
axs[1].set_ylabel(r"$g_{\alpha\beta}(r)$"); axs[1].legend(fontsize=8.5)
axs[1].text(0.02, 0.94, "(b)  aluminium partials",
            transform=axs[1].transAxes, va="top")
for k, lab, col, ls in [("g_Co-Co", "Co-Co", RED, "-"),
                        ("g_Co-Ni", "Co-Ni", PUR, "--"),
                        ("g_Ni-Ni", "Ni-Ni", GRN, ":")]:
    axs[2].plot(d3["r"], d3[k], lw=1.0, color=col, ls=ls, label=lab)
axs[2].set_ylabel(r"$g_{\alpha\beta}(r)$"); axs[2].legend(fontsize=8.5)
axs[2].text(0.02, 0.94, "(c)  transition-metal partials",
            transform=axs[2].transAxes, va="top")
axs[2].set_xlim(1.5, 6.0); axs[2].set_xlabel(r"$r$ (\AA)")
plt.savefig("fig_gr.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- Figure 15: the structure searches -------------------------
REF = -5.104818
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
first = True
for f in ["mq_xphase_trace.npy", "mq_s2_trace.npy", "mq_s3_trace.npy",
          "mq_s4_trace.npy", "mq_s5_trace.npy"]:
    t = np.load(f); x = np.arange(1, len(t) + 1)
    ax[0].plot(x, (t[:, 1] - REF) * 1000, color=BLUE, lw=0.9, alpha=0.8,
               label="staged anneals (5)" if first else None)
    first = False
tre = np.load("mq_re_trace.npy")
xre = np.linspace(1, 20, len(tre))
ax[0].plot(xre, (tre[:, 1] - REF) * 1000, "--s", color=PUR, lw=1.2, ms=4,
           mfc="white", label="replica exchange")
ax[0].axhline(0, color=GRN, lw=1.2, label="reference")
ax[0].set_xlim(0.5, 20.5); ax[0].set_ylim(-6, 165)
ax[0].set_xlabel("annealing stage; replica-exchange checkpoint")
ax[0].set_ylabel(r"$E-E_{\rm ref}$ (meV/atom)")
ax[0].legend(fontsize=8)
ax[0].text(0.02, 0.05, "(a)", transform=ax[0].transAxes)
# Panel (b): the fixed-site validation, per-stage energies transcribed from
# the archived run log mq-2135709.out (26 atoms, species randomised on the
# reference site list; start -4.487405 eV/atom).
per = np.array([-5.101387, -5.076900, -5.093813, -5.101387, -5.093005,
                -5.093005, -5.088671, -5.104282, -5.088636, -5.094412,
                -5.104282, -5.091844, -5.091147, -5.089884, -5.104282,
                -5.104282, -5.101387, -5.094411, -5.101423, -5.101423])
xs = np.arange(0, 21)
vms = (np.concatenate([[-4.487405], per]) - REF) * 1000
best = np.minimum.accumulate(vms)
ax[1].plot(xs, vms, "-o", color=BLUE, lw=0.9, ms=3.5, label="per stage")
ax[1].step(xs, best, where="post", color=GRN, ls="--", lw=1.1,
           label="best so far")
ax[1].set_yscale("log"); ax[1].set_xlim(-0.5, 20.5); ax[1].set_ylim(0.4, 1000)
ax[1].set_xlabel("annealing stage")
ax[1].set_ylabel(r"$E-E_{\rm ref}$ (meV/atom)")
ax[1].legend(fontsize=8, loc="upper right")
ax[1].annotate("reference recovered exactly\nafter relaxation",
               xy=(20, best[-1]), xytext=(8.5, 150), fontsize=8,
               arrowprops=dict(arrowstyle="-", lw=0.6))
ax[1].text(0.02, 0.05, "(b)", transform=ax[1].transAxes)
plt.tight_layout()
plt.savefig("fig_structure_search_validation.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("wrote all six figures")
