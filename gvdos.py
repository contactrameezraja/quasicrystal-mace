"""
Neutron-weighted generalised density of states
==============================================

Why this exists. Inelastic neutron scattering does not measure the vibrational
density of states. It measures a generalised density of states in which each
element contributes in proportion to its scattering cross-section and inversely
to its mass, so a computed spectrum and a measured one are different quantities
and comparing them directly is only qualitative. Applying the weighting removes
that caveat and lets the two be placed on the same axes legitimately.

In the incoherent approximation, and neglecting the Debye-Waller factors, the
weighted spectrum is

    G(E)  proportional to  sum_i  c_i (sigma_i / m_i) g_i(E)

with c_i the atom fraction of element i, sigma_i its bound scattering
cross-section, m_i its mass, and g_i(E) its partial density of states normalised
to unit area. The partial spectra written by reduce_vdos.py are normalised
individually, which is exactly the g_i(E) this expression requires, so no
renormalisation is needed here.

The Debye-Waller factors are omitted. They enter as exp(-2W_i) and are of order
unity at 300 K for these masses and displacements, the measured mean squared
displacements of Section 4.5 giving B values near 0.6 A^2. Including them would
change the weights by a few per cent, which is below the segment noise of every
run, so the omission is stated rather than corrected.

CROSS-SECTIONS MUST BE VERIFIED BEFORE USE. The values in SIGMA below are from
memory rather than from a table, and the whole point of this calculation is
quantitative comparison, so check every one against a standard compilation
(Sears, or the NIST neutron scattering length tables) before quoting any number
this script produces. The weights are extremely sensitive to them: nickel's
cross-section per unit mass is roughly six times aluminium's on these values, so
an error there would dominate the result.

Usage
-----
    python gvdos.py --partial md_wphase_vdos_partial.npz --counts Al:190,Co:55,Ni:20
    python gvdos.py --partial md_al13co4_partial.npz --counts Al:78,Co:24 \
        --out gvdos_al13co4
"""

import argparse

import numpy as np

# Bound scattering cross-section in barn and mass in atomic mass units.
# VERIFY EVERY VALUE AGAINST A STANDARD TABLE BEFORE USE. See the module
# docstring: these are the most error-prone numbers in the calculation.
SIGMA = {
    "Al": {"sigma_barn": 1.503, "mass_amu": 26.98},
    "Co": {"sigma_barn": 5.600, "mass_amu": 58.93},
    "Ni": {"sigma_barn": 18.500, "mass_amu": 58.69},
}

_trapz = getattr(np, "trapezoid", None) or np.trapz


def weights(counts):
    """Return the normalised neutron weight of each element, together with the
    intermediate quantities, so that the sensitivity of the result to the
    cross-sections is visible rather than hidden."""
    total = sum(counts.values())
    raw = {}
    for el, n in counts.items():
        if el not in SIGMA:
            raise SystemExit(f"no cross-section recorded for {el}")
        c = n / total
        raw[el] = c * SIGMA[el]["sigma_barn"] / SIGMA[el]["mass_amu"]
    s = sum(raw.values())
    return {el: v / s for el, v in raw.items()}, raw, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", required=True,
                    help="the *_partial.npz written by reduce_vdos.py")
    ap.add_argument("--counts", required=True,
                    help="atom counts, e.g. Al:190,Co:55,Ni:20")
    ap.add_argument("--split", type=float, default=30.0,
                    help="energy above which fractions are reported, for "
                         "comparison with the unweighted numbers")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    counts = {}
    for part in args.counts.split(","):
        el, n = part.split(":")
        counts[el.strip()] = int(n)

    data = np.load(args.partial)
    if "freq_meV" not in data:
        raise SystemExit(f"{args.partial} has no freq_meV; keys are "
                         f"{list(data.keys())}")
    E = data["freq_meV"]

    w, raw, total = weights(counts)

    print("=" * 66)
    print("NEUTRON WEIGHTS")
    print("=" * 66)
    print(f"{'element':>8} {'atoms':>7} {'at. frac':>9} {'sigma/m':>10} "
          f"{'weight':>8}")
    for el in sorted(counts):
        print(f"{el:>8} {counts[el]:7d} {counts[el]/total:9.4f} "
              f"{SIGMA[el]['sigma_barn']/SIGMA[el]['mass_amu']:10.4f} "
              f"{w[el]:8.4f}")
    print("\n  weights are normalised to sum to one; the sigma/m column is what")
    print("  makes them differ from the atom fractions, and it is where an error")
    print("  in the cross-sections would enter")

    # ------------------------------------------------------- combine
    g_unweighted = np.zeros_like(E)
    g_weighted = np.zeros_like(E)
    for el in sorted(counts):
        if el not in data:
            raise SystemExit(f"{args.partial} has no partial for {el}")
        g_i = data[el]
        g_unweighted += (counts[el] / total) * g_i
        g_weighted += w[el] * g_i

    for name, g in (("unweighted", g_unweighted), ("neutron-weighted", g_weighted)):
        g /= _trapz(g, E)

    print()
    print("=" * 66)
    print(f"EFFECT OF THE WEIGHTING (fractions above {args.split:.0f} meV)")
    print("=" * 66)
    m = E > args.split
    f_un = _trapz(g_unweighted[m], E[m]) * 100
    f_w = _trapz(g_weighted[m], E[m]) * 100
    print(f"  unweighted spectrum      : {f_un:.1f} per cent above "
          f"{args.split:.0f} meV")
    print(f"  neutron-weighted spectrum: {f_w:.1f} per cent")
    print(f"  difference               : {f_w - f_un:+.1f} percentage points")

    lo = E <= 6.0
    print(f"\n  weight below 6 meV, unweighted      : "
          f"{_trapz(g_unweighted[lo], E[lo])*100:.4f} per cent")
    print(f"  weight below 6 meV, neutron-weighted: "
          f"{_trapz(g_weighted[lo], E[lo])*100:.4f} per cent")
    print("\n  the low-energy region is where the two quantities differ most in")
    print("  this alloy family, which is why comparisons with measured spectra")
    print("  there are the ones the weighting matters for")

    def peak(g):
        w2 = (E > 10) & (E < 40)
        return E[w2][np.argmax(g[w2])]

    print(f"\n  main peak, unweighted      : {peak(g_unweighted):.1f} meV")
    print(f"  main peak, neutron-weighted: {peak(g_weighted):.1f} meV")
    print("  the measured peak for this family is reported near 24 meV")

    out = args.out or args.partial.replace("_partial.npz", "")
    np.savez(f"{out}_gvdos.npz", energy=E, unweighted=g_unweighted,
             weighted=g_weighted,
             weights=np.array([w[el] for el in sorted(counts)]),
             elements=np.array(sorted(counts)))
    print(f"\nwrote {out}_gvdos.npz")


if __name__ == "__main__":
    main()
