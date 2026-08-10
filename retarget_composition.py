"""
Composition retargeting at fixed site geometry
==============================================

Builds the composition control of Section 3.2. Given a reference structure, a
supercell, and target species counts, this reassigns the minimum number of sites
required to reach those counts and leaves the rest of the ordering untouched, at
fixed cell and fixed atom count so that the density is unchanged.

Why this exists as a file rather than as an interactive session. The structure it
produces, xbox_wcomp.vasp, carries one of the three spectra in the composition
comparison, and a figure that cannot be regenerated from the repository is not
reproducible. This script reproduces that structure exactly, given the same seed.

The limitation the method carries, and which Section 4.5 examines: the sites to
be reassigned are drawn at random, so the product carries chemical disorder that
the reference structures, whose orderings were refined by Monte Carlo with
first-principles energies, do not. Nothing here orders the result. That is the
job of melt_quench.py --fixed-sites, which on this structure found no better
arrangement.

Reproducing the structure used in this work
-------------------------------------------
    python retarget_composition.py --reference Al9Co2Ni2-coords.txt \
        --supercell 10,4,3 --target Al:2237,Co:648,Ni:235 --seed 42 \
        --out xbox_wcomp.vasp

The reverse experiment, a W-phase box at the X-phase composition, would be
    python retarget_composition.py --reference W-AlCoNi-265.vasp \
        --supercell 3,3,0,-2,2,0,0,0,1 --target Al:2201,Co:611,Ni:368 --seed 43 \
        --out wbox_xcomp.vasp
with the counts recomputed for that site total.
"""

import argparse
from collections import Counter

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell

TYPE_TO_ELEMENT = {13: "Al", 27: "Co", 28: "Ni"}


def parse_structure(path):
    """Identical to the parser in md_run.py, reduce_vdos.py, phonon_run.py and
    msd_gr.py. The copies are kept byte-identical deliberately: each script is a
    standalone entry point run directly on the cluster with no package
    installation step, so a shared import would add a deployment dependency to
    every job. The 26-atom sanity check applies only to the legacy coords.txt
    format, where the expected contents are known."""
    if not path.lower().endswith(".txt"):
        atoms = read(path)
        atoms.pbc = True
        return atoms

    lines = open(path).read().splitlines()
    tokens = open(path).read().split()
    cell = np.array([float(x) for x in tokens[:9]]).reshape(3, 3)
    symbols, scaled = [], []
    for ln in lines:
        p = ln.split()
        if len(p) >= 6 and p[3].isdigit() and p[4].isdigit():
            t = int(p[3])
            if t == 0:
                continue
            if t in TYPE_TO_ELEMENT:
                symbols.append(TYPE_TO_ELEMENT[t])
                scaled.append([float(p[0]), float(p[1]), float(p[2])])
    atoms = Atoms(symbols=symbols, scaled_positions=np.array(scaled),
                  cell=cell, pbc=True)
    if len(atoms) != 26 or abs(atoms.get_volume() - 360.8) / 360.8 > 0.02:
        raise SystemExit(
            f"Parsed {len(atoms)} atoms, volume {atoms.get_volume():.1f} A^3. "
            f"Expected 26 atoms at 360.8 A^3. Check the file.")
    return atoms


def parse_matrix(spec):
    """'3' gives diag(3,3,3); '10,4,3' gives diag(10,4,3); nine numbers give the
    full matrix row by row. Same convention as every other script here."""
    parts = [int(x) for x in spec.split(",")]
    if len(parts) == 1:
        return np.diag(parts * 3)
    if len(parts) == 3:
        return np.diag(parts)
    if len(parts) == 9:
        return np.array(parts).reshape(3, 3)
    raise SystemExit(f"--supercell needs 1, 3 or 9 numbers, got {spec!r}")


def retarget(symbols, target, rng):
    """Reassign the minimum number of sites to reach the target counts.

    Species in surplus donate sites; species in deficit receive them. Donor
    sites are drawn at random without replacement, which is the source of the
    chemical disorder discussed in the module docstring. Returns the new symbol
    list and the number of sites moved.
    """
    symbols = np.array(symbols, dtype=object)
    current = Counter(symbols)
    if sum(target.values()) != len(symbols):
        raise SystemExit(f"target counts sum to {sum(target.values())} but the "
                         f"cell has {len(symbols)} sites")

    surplus, deficit = {}, {}
    for el in set(list(current) + list(target)):
        d = current.get(el, 0) - target.get(el, 0)
        if d > 0:
            surplus[el] = d
        elif d < 0:
            deficit[el] = -d

    donors = []
    for el, n in sorted(surplus.items()):
        idx = np.where(symbols == el)[0]
        donors.extend(rng.choice(idx, size=n, replace=False).tolist())
    rng.shuffle(donors)

    moved = 0
    for el, n in sorted(deficit.items()):
        for _ in range(n):
            symbols[donors.pop()] = el
            moved += 1
    if donors:
        raise SystemExit("internal error: donor sites left unassigned")
    return list(symbols), moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True,
                    help="structure to take the sites from; legacy coords.txt "
                         "or anything ASE reads")
    ap.add_argument("--supercell", default="1",
                    help="1, 3 or 9 numbers, as elsewhere in this project")
    ap.add_argument("--target", required=True,
                    help="target counts, e.g. Al:2237,Co:648,Ni:235. Must sum "
                         "to the number of sites in the supercell")
    ap.add_argument("--seed", type=int, required=True,
                    help="required, not optional: the reassignment is random and "
                         "the structure is only reproducible with the seed recorded")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target = {}
    for part in args.target.split(","):
        el, n = part.split(":")
        target[el.strip()] = int(n)

    base = parse_structure(args.reference)
    P = parse_matrix(args.supercell)
    sc = make_supercell(base, P)

    before = dict(Counter(sc.get_chemical_symbols()))
    cell_before = sc.cell.cellpar().copy()

    rng = np.random.default_rng(args.seed)
    new_symbols, moved = retarget(sc.get_chemical_symbols(), target, rng)
    sc.set_chemical_symbols(new_symbols)
    after = dict(Counter(sc.get_chemical_symbols()))

    n = len(sc)
    print(f"reference      : {args.reference}, {len(base)} atoms")
    print(f"supercell      : {np.asarray(P).tolist()}, det "
          f"{int(round(abs(np.linalg.det(P))))} -> {n} sites")
    print(f"native counts  : {before}")
    print(f"               : " + ", ".join(
        f"{el} {100*c/n:.1f} at.%" for el, c in sorted(before.items())))
    print(f"target counts  : {target}")
    print(f"achieved       : {after}")
    print(f"               : " + ", ".join(
        f"{el} {100*c/n:.1f} at.%" for el, c in sorted(after.items())))
    print(f"sites moved    : {moved} of {n} ({100*moved/n:.1f} per cent), seed "
          f"{args.seed}")
    print(f"cell unchanged : {np.allclose(cell_before, sc.cell.cellpar())}, "
          f"lengths {np.round(sc.cell.cellpar()[:3], 3)}")
    if after != target:
        raise SystemExit("target not reached; check the counts")

    write(args.out, sc, direct=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
