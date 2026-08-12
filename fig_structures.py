"""
Figure 2.1: the two experimentally solved approximants, viewed down the stacking axis.

Convention follows Mihalkovic, Elhor and Suck, Fig. 1: the structure is projected
onto the quasiperiodic plane, the out-of-plane coordinate is encoded in marker
radius so that the layer sequence is visible in projection, transition metals are
filled and aluminium is open, and the unit cell is outlined.

Two points of care, both of which would otherwise produce a wrong figure.

  The stacking axis differs between the two files. For the X-phase it is the
  second cell vector, 4.051 A, and every atom sits at fractional coordinate 0 or
  1/2 along it, so the cell holds exactly two layers. For the W-phase the
  deposited file is the primitive cell of a C-centred setting and NO primitive
  axis lies along the stacking direction, so the conventional cell must be
  recovered first with P = [[1,1,0],[-1,1,0],[0,0,1]]; in that setting the
  stacking axis is the second vector, 8.10 A.

  The panels carry their own scales rather than a shared one, because the in-plane
  extents differ by a factor of about three and the aspect ratios differ more than
  that, so a single scale would leave one panel unreadable. A 5 A bar is drawn in
  each panel instead, and the caption states the measured dimensions.

Usage
-----
    python fig_structures.py            # both panels, needs both structure files
    python fig_structures.py --xonly    # X-phase only, for testing
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TYPE_TO_ELEMENT = {13: "Al", 27: "Co", 28: "Ni"}
TM = ("Co", "Ni")

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
})


def parse_coords_txt(path):
    """The legacy CMU coordinate format. Same rules as the analysis codes: nine
    cell numbers first, then one row per site, type code zero dropped."""
    lines = open(path).read().splitlines()
    tok = open(path).read().split()
    cell = np.array([float(x) for x in tok[:9]]).reshape(3, 3)
    sym, frac = [], []
    for ln in lines:
        p = ln.split()
        if len(p) >= 6 and p[3].isdigit() and p[4].isdigit():
            t = int(p[3])
            if t == 0:
                continue
            if t in TYPE_TO_ELEMENT:
                sym.append(TYPE_TO_ELEMENT[t])
                frac.append([float(p[0]), float(p[1]), float(p[2])])
    return cell, np.array(frac), sym


def read_vasp(path):
    from ase.io import read
    a = read(path)
    return a.get_cell().array, a.get_scaled_positions(), a.get_chemical_symbols()


def to_conventional(cell, frac, sym, P):
    """Apply an integer transformation to the cell and re-index the sites, adding
    the periodic images the larger cell now contains."""
    P = np.asarray(P, dtype=float)
    new_cell = P @ cell
    inv = np.linalg.inv(P)
    n = int(round(abs(np.linalg.det(P))))
    out_frac, out_sym = [], []
    rng = range(-n, n + 1)
    seen = set()
    for i in rng:
        for j in rng:
            for k in rng:
                shifted = frac + np.array([i, j, k], dtype=float)
                cand = shifted @ inv
                inside = np.all((cand > -1e-9) & (cand < 1 - 1e-9), axis=1)
                for c, s in zip(cand[inside], np.array(sym)[inside]):
                    key = tuple(np.round(c, 5))
                    if key not in seen:
                        seen.add(key)
                        out_frac.append(c)
                        out_sym.append(s)
    return new_cell, np.array(out_frac), out_sym


def panel(ax, cell, frac, sym, stack_axis, label, repeats=(1, 1)):
    """Project down stack_axis and draw. In-plane axes are the other two."""
    plane = [i for i in range(3) if i != stack_axis]
    v1, v2 = cell[plane[0]], cell[plane[1]]
    # a 2D basis in the projection plane
    e1 = v1 / np.linalg.norm(v1)
    n = np.cross(v1, v2)
    e2 = np.cross(n, v1)
    e2 /= np.linalg.norm(e2)

    def project(cart):
        return np.column_stack([cart @ e1, cart @ e2])

    corners = project(np.array([[0, 0, 0], v1, v1 + v2, v2, [0, 0, 0]]))
    ax.plot(corners[:, 0], corners[:, 1], "-", color="k", lw=0.6, zorder=1)

    for ri in range(repeats[0]):
        for rj in range(repeats[1]):
            offset = ri * v1 + rj * v2
            cart = frac @ cell + offset
            xy = project(cart)
            h = frac[:, stack_axis] % 1.0
            size = 14 + 34 * h                      # radius encodes the layer
            for el, filled in (("Al", False), ("Co", True), ("Ni", True)):
                m = np.array(sym) == el
                if not m.any():
                    continue
                ax.scatter(xy[m, 0], xy[m, 1], s=size[m],
                           facecolors="k" if filled else "none",
                           edgecolors="k", linewidths=0.55,
                           marker="o" if el != "Ni" else "s", zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")
    span = corners[:, 0].max() - corners[:, 0].min()
    ax.text(0.0, 1.0, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5)
    # 5 A scale bar
    y0 = corners[:, 1].min() - 0.13 * span
    x0 = corners[:, 0].min()
    ax.plot([x0, x0 + 5], [y0, y0], "-", color="k", lw=1.4,
            solid_capstyle="butt")
    ax.text(x0 + 2.5, y0 - 0.045 * span, r"5 \AA" if False else "5 Å",
            ha="center", va="top", fontsize=7)
    ax.margins(0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xonly", action="store_true",
                    help="render the X-phase panel only, for testing without "
                         "the W-phase file present")
    ap.add_argument("--xfile", default="Al9Co2Ni2-coords.txt")
    ap.add_argument("--wfile", default="W-AlCoNi-265.vasp")
    ap.add_argument("--out", default="fig_structures")
    args = ap.parse_args()

    cell_x, frac_x, sym_x = parse_coords_txt(args.xfile)
    print(f"X-phase: {len(sym_x)} atoms, cell lengths "
          f"{np.round(np.linalg.norm(cell_x, axis=1), 3)}, "
          f"volume {abs(np.linalg.det(cell_x)):.1f} A^3")
    print(f"  stacking axis 1 = {np.linalg.norm(cell_x[1]):.3f} A, layers at "
          f"{sorted(set(np.round(frac_x[:, 1], 4)))}")

    if args.xonly:
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        panel(ax, cell_x, frac_x, sym_x, 1, "(a) X-phase, 26 atoms")
        fig.savefig(f"{args.out}_xonly.png", dpi=600)
        fig.savefig(f"{args.out}_xonly.pdf")
        print(f"wrote {args.out}_xonly.png and .pdf")
        return

    cell_w0, frac_w0, sym_w0 = read_vasp(args.wfile)
    P = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    cell_w, frac_w, sym_w = to_conventional(cell_w0, frac_w0, sym_w0, P)
    print(f"W-phase: {len(sym_w0)} atoms primitive -> {len(sym_w)} conventional, "
          f"cell lengths {np.round(np.linalg.norm(cell_w, axis=1), 3)}")
    layers = sorted(set(np.round(frac_w[:, 1], 3)))
    print(f"  stacking axis 1 = {np.linalg.norm(cell_w[1]):.3f} A, "
          f"{len(layers)} distinct layers at {layers}")
    print(f"  composition: "
          f"{ {e: sym_w.count(e) for e in sorted(set(sym_w))} }")
    if len(sym_w) != 2 * len(sym_w0):
        raise SystemExit("the conventional cell should hold twice the primitive "
                         "atom count; check the transformation")

    w_x = np.linalg.norm(cell_x[0]) + abs(cell_x[2] @ (cell_x[0] /
                                                       np.linalg.norm(cell_x[0])))
    w_w = np.linalg.norm(cell_w[0])
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 3.4),
                           gridspec_kw={"width_ratios": [1.0, 1.9]})
    panel(ax[0], cell_x, frac_x, sym_x, 1, "(a) X-phase, 26 atoms")
    panel(ax[1], cell_w, frac_w, sym_w, 1,
          f"(b) W-phase, {len(sym_w)} atoms in the conventional cell")
    plt.tight_layout(w_pad=1.5)
    fig.savefig(f"{args.out}.png", dpi=600)
    fig.savefig(f"{args.out}.pdf")
    print(f"wrote {args.out}.png and .pdf")


if __name__ == "__main__":
    main()
