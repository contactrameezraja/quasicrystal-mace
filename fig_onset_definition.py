"""
Figure for Definition 1: how the spectral floor is picked, and why the factor of
ten does not matter much.

Definition 1 states that the onset is the lowest energy at which g(E) exceeds ten
times a noise reference read at 0.8 meV. That factor is a choice and the text
concedes it. This figure answers the concession with evidence rather than argument.

Panel (a) draws one spectrum in its low-energy region with the noise reference, the
threshold and the crossing marked, so a reader can see that the region below the
crossing is genuinely empty rather than merely below a line, and that the curve
rises steeply enough there that the crossing barely moves when the factor changes.

Panel (b) makes the same point quantitatively across every box. The onset is
recomputed at factors of 5, 10 and 20, the relation onset = C / L_long is fitted at
each, and the three fitted prefactors are compared. If they agree, the calibration
reported in Chapter 4 does not depend on the choice, which is the substantive claim.

Usage
-----
    python fig_onset_definition.py                 # uses the repository .npy files
    python fig_onset_definition.py --synthetic     # self-test with known answers

The spectra are the *_vdos_total.npy and *_total.npy arrays written by
reduce_vdos.py, each holding a 2 x N array of energy in meV and normalised g(E).
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOISE_ENERGY = 0.8          # meV, where the noise reference is read
FACTORS = (5.0, 10.0, 20.0)  # 10 is Definition 1; the others test its sensitivity

# The six production boxes, with the longest box dimension in Angstrom.
BOXES = [
    ("X-phase 2x2x2",   "md_2x2x2_s11_vdos_total.npy",      24.3),
    ("X-phase 3x3x3",   "md_3x3x3_s12_vdos_total.npy",      36.4),
    ("X-phase 4x4x4",   "md_4x4x4_vdos_total.npy",          48.5),
    ("X-phase 8x4x4",   "md_8x4x4_s21_vdos_total.npy",      97.0),
    ("W-phase 3x2x1",   "md_wphase_vdos_total.npy",        118.0),
    ("X-phase 10x4x3",  "md_xphase_10x4x3_vdos_total.npy", 121.3),
]
HIGHLIGHT = "W-phase 3x2x1"   # the spectrum drawn in panel (a)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def noise_reference(E, g):
    """The amplitude at 0.8 meV, interpolated. This is the quantity Definition 1
    scales; it measures numerical baseline rather than physical weight, because no
    box here has a floor below 1.32 meV."""
    return float(np.interp(NOISE_ENERGY, E, g))


def onset(E, g, factor):
    """Lowest energy at which g exceeds factor x the noise reference. Searched
    above the reference energy itself, so the reference cannot trigger it."""
    thr = factor * noise_reference(E, g)
    mask = (E > NOISE_ENERGY) & (g > thr)
    if not mask.any():
        return np.nan
    return float(E[mask][0])


def synthetic(floor, n=24000, seed=0):
    """A spectrum with a known floor: a smooth numerical baseline below it, then
    discrete modes thickening above. Used to check that the estimator returns the
    floor it was given.

    The baseline is deliberately smooth rather than white. A first version used
    white per-bin noise, and the estimator then failed at the higher thresholds,
    because Definition 1 reads the reference at a single interpolated point and a
    white baseline makes that point fluctuate by a factor of several. That is a
    property of the estimator worth knowing, and Section 3.6 notes it, but it is not
    what this test is for.
    """
    E = np.linspace(0.001, 60, n)
    g = 2e-5 * (1.0 + 0.15 * np.sin(3.1 * E) + 0.08 * np.cos(11.0 * E))
    modes = floor * np.array([1.0, 1.18, 1.34, 1.55, 1.8, 2.1, 2.5, 3.0])
    for k, m in enumerate(modes):
        g += (0.004 / (1 + k)) * np.exp(-0.5 * ((E - m) / 0.02) ** 2)
    # The broad band must vanish below the floor, as a real spectrum does. A first
    # version used a bare Gaussian centred at 22 meV, whose tail at 0.8 meV was
    # thirty times the numerical baseline; the reference then measured that tail
    # rather than the baseline and the higher thresholds sailed past the modes.
    band = 0.02 * np.exp(-0.5 * ((E - 22) / 8) ** 2)
    g += band / (1.0 + np.exp(-(E - floor) / 0.05))
    return E, g / np.trapezoid(g, E)


def load(path):
    a = np.load(path)
    if a.ndim != 2 or a.shape[0] != 2:
        raise SystemExit(f"{path}: expected a 2 x N array, got shape {a.shape}")
    return a[0], a[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="self-test on spectra with known floors")
    ap.add_argument("--out", default="fig_onset_definition")
    args = ap.parse_args()

    data = []
    if args.synthetic:
        truths = [6.2, 4.8, 3.7, 1.85, 1.35, 1.40]
        for (name, _, L), t in zip(BOXES, truths):
            E, g = synthetic(t)
            data.append((name, L, E, g, t))
    else:
        for name, path, L in BOXES:
            if not os.path.exists(path):
                print(f"  missing, skipped: {path}")
                continue
            E, g = load(path)
            data.append((name, L, E, g, None))
        if not data:
            raise SystemExit("no spectra found; run in the repository directory "
                             "or use --synthetic")

    print(f"{'box':18s} {'L (A)':>7} " +
          " ".join(f"{'x'+str(int(f)):>8}" for f in FACTORS) +
          ("   truth" if args.synthetic else ""))
    table = {}
    for name, L, E, g, truth in data:
        ons = [onset(E, g, f) for f in FACTORS]
        table[name] = (L, ons)
        line = f"{name:18s} {L:7.1f} " + " ".join(f"{o:8.3f}" for o in ons)
        if truth is not None:
            line += f"   {truth:6.2f}"
        print(line)

    # Fit onset = C / L at each factor. C is the quantity Chapter 4 reports.
    print("\nfitted prefactor C in onset = C / L_long, by threshold factor:")
    Cs = []
    for i, f in enumerate(FACTORS):
        L = np.array([v[0] for v in table.values()])
        o = np.array([v[1][i] for v in table.values()])
        ok = np.isfinite(o)
        C = float(np.sum(o[ok] * (1 / L[ok])) / np.sum((1 / L[ok]) ** 2))
        Cs.append(C)
        resid = 100 * np.abs(o[ok] - C / L[ok]) / o[ok]
        print(f"  factor {f:4.0f}:  C = {C:6.1f} meV A   "
              f"mean |deviation| {resid.mean():.1f} per cent")
    spread = 100 * (max(Cs) - min(Cs)) / Cs[1]
    print(f"\n  spread in C across a factor of four in threshold: "
          f"{spread:.1f} per cent of the Definition 1 value")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.9),
                           gridspec_kw={"width_ratios": [1.0, 0.95]})

    # panel (a): the definition on one spectrum
    hl = [d for d in data if d[0] == HIGHLIGHT] or [data[-1]]
    name, L, E, g, _ = hl[0]
    ref = noise_reference(E, g)
    m = E <= 4.0
    ax[0].semilogy(E[m], np.maximum(g[m], 1e-7), "-", color="k", lw=0.7)
    ax[0].axvline(NOISE_ENERGY, color="k", lw=0.5, ls=":")
    ax[0].text(NOISE_ENERGY - 0.06, ax[0].get_ylim()[0] * 1.6,
               "reference\nread here", fontsize=6.3, va="bottom", ha="right")
    for f in FACTORS:
        thr = f * ref
        if f == 10.0:
            ax[0].axhline(thr, color="k", lw=0.55, ls="-")
        else:
            ax[0].axhline(thr, color="k", lw=0.45, ls="--", dashes=(3, 2))
        o = onset(E, g, f)
        ax[0].plot([o], [thr], "o", ms=3.4,
                   markerfacecolor="k" if f == 10 else "none",
                   markeredgecolor="k", mew=0.7, zorder=5)
        ax[0].text(4.0, thr, f"  ${{\\times}}{int(f)}$", fontsize=6.5,
                   va="center", ha="left")
    o10 = onset(E, g, 10.0)
    ax[0].annotate(f"onset\n{o10:.2f} meV", xy=(o10, 10 * ref),
                   xytext=(o10 + 0.22, 10 * ref * 0.28), fontsize=7,
                   ha="left", va="top",
                   arrowprops=dict(arrowstyle="->", lw=0.5,
                                   shrinkA=1, shrinkB=2))
    ax[0].set_xlim(0, 4.0)
    ax[0].set_xlabel(r"$E$ (meV)")
    ax[0].set_ylabel(r"$g(E)$ (meV$^{-1}$)")
    ax[0].text(0.02, 0.93, "(a)", transform=ax[0].transAxes, fontsize=8.5)

    # panel (b): the calibration at each threshold
    marks = {5.0: "^", 10.0: "o", 20.0: "s"}
    for i, f in enumerate(FACTORS):
        Lv = np.array([v[0] for v in table.values()])
        ov = np.array([v[1][i] for v in table.values()])
        ok = np.isfinite(ov)
        ax[1].plot(1 / Lv[ok], ov[ok], marks[f], color="k", ms=3.6,
                   markerfacecolor="k" if f == 10 else "none", mew=0.7,
                   linestyle="none", label=rf"${{\times}}{int(f)}$")
        xs = np.linspace(0, 1 / Lv[ok].min() * 1.05, 50)
        if f == 10.0:
            ax[1].plot(xs, Cs[i] * xs, "-", color="k", lw=0.7)
        else:
            ax[1].plot(xs, Cs[i] * xs, "--", color="k", lw=0.45, dashes=(3, 2))
    ax[1].set_xlim(0, None)
    ax[1].set_ylim(0, None)
    ax[1].set_xlabel(r"$1/L_{\mathrm{long}}$ (\AA$^{-1}$)" if False
                     else r"$1/L_{\mathrm{long}}$ (Å$^{-1}$)")
    ax[1].set_ylabel("measured onset (meV)")
    ax[1].legend(fontsize=7, loc="upper left", title="threshold",
                 title_fontsize=7)
    ax[1].text(0.93, 0.93, "(b)", transform=ax[1].transAxes, fontsize=8.5)

    plt.tight_layout(w_pad=1.6)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=600)
    print(f"\nwrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
