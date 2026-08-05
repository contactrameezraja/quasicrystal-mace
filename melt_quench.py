"""
Cell-constrained melt-quench with MACE: discovering AlCoNi approximants
=======================================================================

What this does, and how it differs from md_run.py
-------------------------------------------------
md_run.py measures the vibrations of a structure we already have. This script
does the opposite: it *discovers* a structure, given only a unit cell, a
composition and an atom count.

The method is "cell-constrained melt-quench", from

    M. Mihalkovic, M. Widom and C. L. Henley, "Cell-constrained melt-quench
    simulation of d-AlCoNi: Ni-rich versus Co-rich structures",
    Phil. Mag. 91 (2011) 2557-2566.  doi:10.1080/14786435.2010.515264

Their recipe: fix the cell, composition and atomic density to known values,
melt, then cool under MD while interleaving Monte Carlo swaps of atomic
species. The cell constraint is what makes it work - it limits the accessible
ensemble enough to shepherd the system toward the global minimum rather than
into a glass.

They used GPT pair potentials; we use MACE. Two consequences:

  - Every attempted MC swap costs a full MACE forward pass, where a pair
    potential could evaluate the energy change locally. So the swap count per
    loop is the main cost knob (--swaps), and --benchmark measures the real
    per-evaluation cost before committing to a schedule.

  - The paper notes the GPT potentials had "unrealistically deep Al-TM
    nearest-neighbour wells" which exaggerated the stability of the spurious B2
    phase, and that this - not compute time - was what limited how large a cell
    they could use. Whether MACE inherits that bias is an open question, and one
    of the reasons this is worth doing at all.

The validation gate
-------------------
Section 3 of the paper validates the method by reproducing known structures
from scratch: Al3Ni in 10 loops, m-Al13Co4 (51 atoms) in 119 loops, Al9Co2,
and the ternary X(AlCoNi) phase at Al:Co:Ni = 18:5:3 in 71 loops.

That last one is our 26-atom cell. So the first thing to do is reproduce it:

    python melt_quench.py --reference Al9Co2Ni2-coords.txt --loops-per-temp 5

Success criterion: the quenched structure lands at the same energy per atom as
the relaxed reference (printed side by side at the end). A higher energy means
the anneal got stuck in a metastable state - run longer, or raise --swaps.

The supercell species-swap experiment (as-tiled start)
------------------------------------------------------
The fixed-site diagnostic validated the chemistry machinery: from randomised
species on the reference site list it recovers the reference ordering exactly.
The supercell experiment asks the opposite question. A supercell built from a
periodic approximant repeats the SAME chemical ordering in every copy of the
cell - an artificial periodicity the tiling imposes, not one the chemistry
chose. Starting fixed-site MC from that as-tiled ordering (--start-from
as-tiled, i.e. do NOT randomise) lets the chemistry decide whether to break
the tile periodicity. If swaps are accepted and the energy drops, the true
ordering of the larger cell differs from tiled copies of the small one.

The ratio experiment (target counts)
------------------------------------
Instead of grand-canonical MC with chemical-potential reservoirs, composition
is changed MANUALLY: --target-counts reassigns the minimum number of sites
(chosen at random from the majority species) to hit the requested counts, then
the MC runs canonically at the new composition. Combined with --start-from
as-tiled this perturbs the tiled ordering as little as possible, so the swaps
explore the consequence of the composition change rather than recovering from
a fully scrambled start.

Protocol notes, and why each choice
-----------------------------------
  Timestep 0.2 fs, not the 2 fs used for the room-temperature VDOS runs. At
  2500 K atoms move an order of magnitude faster and 2 fs would not integrate
  stably. This is the paper's value.

  Langevin thermostat throughout, deliberately. This is annealing, so we *want*
  a thermostat holding each stage at temperature - the exact opposite of the
  VDOS runs, where NVE was essential because a thermostat modifies the
  equations of motion and would contaminate the spectrum.

  The cell never changes. That is the "cell constraint": positions relax and
  species swap, but the lattice is pinned at the target values. Even the
  quenches relax positions only.

  MC species swaps, not just MD. Solid-state diffusion is far too slow to
  sample chemical orderings in reachable simulation time, so the chemistry has
  to be moved by Monte Carlo rather than by atoms physically migrating.

  8 A stacking, not 4 A, for any new cell. The paper is blunt: "There is no
  hope to realistically model any decagonal without (at least local) 8 A period
  due to puckering of Al atoms out of their layers" - when they forced a 4 A
  cell they never obtained a decagonal at all.

  Random start with a minimum separation. Uniformly random placement will
  occasionally put two atoms ~0.5 A apart, which gives an enormous energy and
  can produce NaN forces on the very first MD step. We reject such placements.

Deviations from the paper, and why
---------------------------------
  NO REPLICA EXCHANGE. The paper runs 20 simulations at once, one per
  temperature, and exchanges whole configurations between temperatures by
  Metropolis. They identify this as the trick that makes the method work: their
  expectation was that a plain quench "should get caught in a glassy or highly
  defective metastable state rather than the true energy minimum", and it was
  "sufficient to apply the trick of replica exchange" to avoid that. This script
  instead does staged annealing - one trajectory cooling through the ladder -
  because it is far less code and ~20x cheaper. That is a real simplification,
  not an equivalent: if a validation run lands above the reference energy, this
  is the first thing to suspect, and larger cells will probably need the full
  replica-exchange treatment.

  500 swap attempts per loop instead of 7800, because each attempt is a full
  MACE forward pass rather than a local pair-potential update. See --swaps.

  MACE instead of GPT pair potentials, which is the point of the exercise.

  Atomic density is NOT swept. The paper treats density as a key parameter and
  finds the same composition gives a different structure type at low vs high
  density (Ni-type vs Co-type). With --reference the density comes from the
  known cell, so it is fixed correctly; when designing new cells it has to be
  varied deliberately. This script prints the density but does not explore it.

  Diagnostic: the paper reports swap acceptance of 0.03-0.05. If the acc column
  in the log reads far from that, something is off with the temperatures or the
  energetics and it is worth stopping to look.

Reference cells from the paper (Section 3-4), for later runs
-----------------------------------------------------------
  tilings use a_q = 2.44 A in-plane, c = 4.08 A minimum stacking period
  "2B+H" cell:  a = b = 19.78 A, gamma = 108 deg, c = 8.06 A
  "boat" cell:  a = b = 12.30 A, gamma = 108 deg, c = 8.20 A  (2B+H shrunk by
                tau; chosen because it matches the spurious B2 phase poorly)
  densities 0.066-0.070 atoms/A^3, i.e. 77-80 atoms in the boat cell
  compositions: Al56Co6Ni16 (Ni-rich), Al58Co14Ni9 (Co-rich)

Usage
-----
    python melt_quench.py --benchmark --reference Al9Co2Ni2-coords.txt
    python melt_quench.py --reference Al9Co2Ni2-coords.txt        # validation gate
    python melt_quench.py --cell 12.3,12.3,8.20 --angles 90,90,108 \
                          --counts Al:56,Co:6,Ni:16 --out boat_nirich

    # supercell species-swap MC: W-phase 3x2x1 conventional, as-tiled start
    python melt_quench.py --reference wphase.vasp \
        --supercell 3,3,0,-2,2,0,0,0,1 \
        --fixed-sites --start-from as-tiled --out w_astiled_mc

    # ratio experiment: same, with the composition nudged manually
    python melt_quench.py --reference wphase.vasp \
        --supercell 3,3,0,-2,2,0,0,0,1 \
        --fixed-sites --start-from as-tiled \
        --target-counts Al:2244,Co:696,Ni:240 --out w_ratio_coplus
"""

import argparse
import time
from collections import Counter

import numpy as np
from ase import Atoms, units
from ase.geometry import cellpar_to_cell, find_mic
from ase.io import write
from ase.md.langevin import Langevin
from ase.optimize import BFGS

TYPE_TO_ELEMENT = {13: "Al", 27: "Co", 28: "Ni"}


def log(msg, path="melt_quench_progress.txt"):
    print(msg, flush=True)
    with open(path, "a") as f:
        f.write(msg + "\n")


def parse_structure(path):
    """Same parser as md_run.py: cell vectors on one line, then x y z type per
    atom, with type 0 rows dropped as partial-occupancy placeholders. Duplicated
    here rather than imported so this script stands alone and can be tested
    without a MACE install."""
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
    return Atoms(symbols=symbols, scaled_positions=np.array(scaled),
                 cell=cell, pbc=True)


def load_structure(path):
    """Dispatch on file type: the legacy coords.txt format goes through
    parse_structure; anything ASE can read (.vasp/POSCAR, .cif, .xyz) goes
    through ase.io.read. Mirrors the reader added to md_run.py for the W-vs-X
    comparison, so both scripts accept the same inputs."""
    lower = path.lower()
    if lower.endswith((".vasp", ".poscar", ".cif", ".xyz")) or "poscar" in lower:
        from ase.io import read
        atoms = read(path)
        atoms.pbc = True
        return atoms
    return parse_structure(path)


def apply_supercell(atoms, spec):
    """Build a supercell from 3 diagonal repeats ('3,2,1') or a full 9-number
    transformation matrix row-by-row ('3,3,0,-2,2,0,0,0,1') - the latter is how
    the W-phase primitive cell becomes its 3x2x1 conventional box. Same
    convention as md_run.py's --supercell."""
    nums = [int(x) for x in spec.replace(",", " ").split()]
    if len(nums) == 3:
        P = np.diag(nums)
    elif len(nums) == 9:
        P = np.array(nums).reshape(3, 3)
    else:
        raise SystemExit("--supercell needs 3 or 9 integers")
    from ase.build import make_supercell
    sc = make_supercell(atoms, P)
    log(f"supercell matrix rows {P.tolist()}, det {int(round(np.linalg.det(P)))}: "
        f"{len(atoms)} -> {len(sc)} atoms, "
        f"box {np.round(sc.cell.cellpar(), 2)}")
    return sc


def retarget_counts(atoms, target, rng):
    """Reassign the MINIMUM number of sites needed to hit the target counts.
    Surplus species donate randomly-chosen sites; deficit species receive them
    in shuffled order. Everything else keeps its identity, so an as-tiled start
    is perturbed as little as the composition change allows. This is the manual
    alternative to grand-canonical MC: composition is set by hand, the MC then
    runs canonically at that composition."""
    syms = np.array(atoms.get_chemical_symbols())
    n = len(syms)
    total = sum(target.values())
    if total != n:
        raise SystemExit(f"--target-counts sums to {total}, but the cell has "
                         f"{n} sites - counts must match exactly (fixed cell, "
                         f"fixed density)")
    current = Counter(syms)
    species = sorted(set(current) | set(target))
    surplus = {s: current.get(s, 0) - target.get(s, 0) for s in species}

    give = []
    for s in species:
        if surplus[s] > 0:
            idx = np.flatnonzero(syms == s)
            chosen = rng.choice(idx, size=surplus[s], replace=False)
            give.extend(int(i) for i in chosen)
    need = []
    for s in species:
        if surplus[s] < 0:
            need += [s] * (-surplus[s])
    need = [str(s) for s in rng.permutation(need)]

    for i, s in zip(give, need):
        syms[i] = s
    atoms.set_chemical_symbols(list(syms))
    _refresh_masses(atoms)
    log(f"retarget: {({str(k): int(v) for k, v in current.items()})} -> "
        f"{({str(k): int(v) for k, v in Counter(syms).items()})} "
        f"({len(give)} of {n} sites reassigned)")
    return len(give)


def _refresh_masses(atoms):
    """ASE will cache a 'masses' array, and that array does NOT follow a species
    swap - so MD after a swap would silently use the wrong masses. Drop it and
    let ASE re-derive masses from atomic numbers."""
    if "masses" in atoms.arrays:
        del atoms.arrays["masses"]


def random_config(cell, counts, rng, min_dist=2.0, max_tries=300):
    """Scatter the given composition at random positions in a fixed cell,
    rejecting any position closer than min_dist to an already-placed atom
    (minimum-image). If a slot cannot be filled, min_dist is relaxed slightly
    rather than looping forever."""
    symbols = []
    for sym, n in counts.items():
        symbols += [sym] * n
    symbols = [str(s) for s in rng.permutation(symbols)]

    cell = np.asarray(cell, dtype=float)
    pos = np.zeros((len(symbols), 3))
    placed, tries = 0, 0
    while placed < len(symbols):
        cand = rng.random(3) @ cell
        if placed == 0:
            ok = True
        else:
            _, dlen = find_mic(cand - pos[:placed], cell, pbc=True)
            ok = float(dlen.min()) > min_dist
        if ok:
            pos[placed] = cand
            placed += 1
            tries = 0
        else:
            tries += 1
            if tries > max_tries:
                min_dist *= 0.95
                tries = 0
    return Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)


def md_burst(atoms, T, n_steps, timestep_fs, friction=0.02):
    """MD at fixed temperature. Friction is higher than the VDOS runs' 0.01/fs
    because at melt temperatures we want the thermostat to hold temperature
    firmly, not to perturb the dynamics as little as possible."""
    Langevin(atoms, timestep_fs * units.fs, temperature_K=T,
             friction=friction / units.fs).run(n_steps)


def try_swaps(atoms, T, n_attempts, rng, e_current):
    """Metropolis Monte Carlo on chemical identity. Returns the current energy
    and the number of accepted swaps. One MACE energy evaluation per attempt -
    this is the expensive part of the method."""
    kT = units.kB * T
    syms = np.array(atoms.get_chemical_symbols())
    n_acc = 0
    for _ in range(n_attempts):
        i = int(rng.integers(len(syms)))
        others = np.flatnonzero(syms != syms[i])
        if others.size == 0:
            break                                   # single-species cell
        j = int(others[rng.integers(others.size)])

        syms[i], syms[j] = syms[j], syms[i]
        atoms.set_chemical_symbols(list(syms))
        _refresh_masses(atoms)
        e_trial = atoms.get_potential_energy()

        dE = e_trial - e_current
        if dE <= 0.0 or rng.random() < np.exp(-dE / kT):
            e_current = e_trial
            n_acc += 1
        else:
            syms[i], syms[j] = syms[j], syms[i]      # revert
            atoms.set_chemical_symbols(list(syms))
            _refresh_masses(atoms)
    return e_current, n_acc


def quench(atoms, fmax):
    """Relax positions at fixed cell to read off the underlying zero-temperature
    energy. The paper does this periodically to monitor progress. The cell is
    NOT relaxed - that would break the constraint the method relies on."""
    snap = atoms.copy()
    snap.calc = atoms.calc
    BFGS(snap, logfile=None).run(fmax=fmax, steps=300)
    return snap, snap.get_potential_energy()


def fixed_site_mc(ref, args, rng):
    """Keep the ATOMIC POSITIONS fixed and let only the Monte Carlo swaps run.
    No MD, no position changes.

    Two uses, selected by --start-from:

    random (the original diagnostic): randomise the species first. This is what
    Mihalkovic and Widom did for the W-phase in Phil. Mag. 86 (2005):
    fixed-site Monte Carlo on a site list taken from diffraction. It decomposes
    the validation gap into its two causes. If this recovers the reference
    energy easily, the whole gap is the positional search and the chemistry
    machinery is fine. If it stalls well above, the swap schedule needs work
    too, and no amount of better annealing will fix that.

    as-tiled (the supercell experiment): keep the species exactly as read from
    the input, so a supercell starts from tiled copies of the small cell's
    ordering. The MC then decides whether chemistry wants to break that
    artificial tile periodicity. Accepted swaps that lower the energy are the
    signal; a dead run at acceptance ~0 with no energy drop says the tiled
    ordering is already locally optimal.
    """
    atoms = ref.copy()
    atoms.calc = ref.calc
    n = len(atoms)

    log(f"=== fixed-site MC: {n} atoms on the input site list ===")
    if args.start_from == "as-tiled":
        log(f"    species kept AS-TILED from the input file "
            f"({dict(Counter(atoms.get_chemical_symbols()))})")
    else:
        syms = [str(s) for s in rng.permutation(atoms.get_chemical_symbols())]
        atoms.set_chemical_symbols(syms)
        _refresh_masses(atoms)
        log("    species randomised")

    e = atoms.get_potential_energy()
    log(f"    start {e / n:+.6f} eV/atom")
    log(f"    {args.n_temps} stages {args.t_hi}->{args.t_lo} K, "
        f"{args.swaps} swap attempts each, positions frozen")

    t0 = time.time()
    best_e, best = e, atoms.copy()
    for T in np.linspace(args.t_hi, args.t_lo, args.n_temps):
        e, n_acc = try_swaps(atoms, T, args.swaps, rng, e)
        if e < best_e:
            best_e, best = e, atoms.copy()
        log(f"  T={T:7.0f} K | {e / n:+.6f} eV/atom | best {best_e / n:+.6f} "
            f"| acc {n_acc / max(1, args.swaps):.3f} | {(time.time() - t0) / 60:.0f} min")

    log(f"best chemistry on frozen sites: {best_e / n:+.6f} eV/atom")
    best.calc = ref.calc
    BFGS(best, logfile=None).run(fmax=args.fmax_final, steps=500)
    e_relaxed = best.get_potential_energy()
    log(f"after relaxing positions too:   {e_relaxed / n:+.6f} eV/atom")
    write(f"{args.out}_fixedsite.vasp", best, direct=True, sort=True)
    return best, e_relaxed


def replica_exchange(atoms, args, rng):
    """Replica exchange (parallel tempering), the piece the paper identifies as
    essential and that staged annealing lacks.

    Why it fixes what we saw: five independent staged anneals landed in five
    different basins 25-70 meV/atom above the reference, because once a single
    trajectory cools it has no way back out (swap acceptance had fallen to 0.01
    by the last stage - chemically frozen). Here several replicas run at once
    across a temperature ladder, and configurations are exchanged between
    adjacent temperatures by Metropolis. A cold replica stuck in a bad basin can
    hand its configuration up to a hotter replica that can escape, and inherit a
    better one in return.

    The exchange criterion for replicas i, j with inverse temperatures b_i, b_j:
        Delta = (b_i - b_j)(E_i - E_j),   accept if Delta >= 0 or rand < exp(Delta)
    which is always accepted when the hotter replica has found the lower energy -
    exactly the rescue we need.

    Temperatures are spaced geometrically, which keeps the exchange acceptance
    roughly uniform along the ladder rather than bunching at one end.
    """
    n = len(atoms)
    temps = np.geomspace(args.t_lo, args.t_hi, args.n_replicas)
    betas = 1.0 / (units.kB * temps)

    log(f"=== replica exchange: {n} atoms, {args.n_replicas} replicas ===")
    log(f"    ladder (K): {np.round(temps, 0)}")
    log(f"    {args.n_cycles} cycles x ({args.md_steps} MD + {args.swaps} swaps) "
        f"per replica = {args.n_cycles * args.n_replicas * (args.md_steps + args.swaps):,} "
        f"evaluations")

    # every replica starts from the same melted configuration
    log(f"melting at {args.t_hi} K ({args.melt_steps} steps)...")
    md_burst(atoms, args.t_hi, args.melt_steps, args.timestep)
    reps = []
    for _ in range(args.n_replicas):
        r = atoms.copy()
        r.calc = atoms.calc
        reps.append(r)
    energies = np.array([r.get_potential_energy() for r in reps])

    t0 = time.time()
    best_e, best = np.inf, None
    trace, n_exch_total, n_exch_acc = [], 0, 0

    for cycle in range(args.n_cycles):
        # --- each replica evolves at its own temperature ---
        for k, (r, T) in enumerate(zip(reps, temps)):
            md_burst(r, T, args.md_steps, args.timestep)
            energies[k] = r.get_potential_energy()
            energies[k], _ = try_swaps(r, T, args.swaps, rng, energies[k])

        # --- exchange attempts between adjacent pairs, alternating parity so
        #     every pair gets tried over successive cycles ---
        for i in range(cycle % 2, args.n_replicas - 1, 2):
            j = i + 1
            delta = (betas[i] - betas[j]) * (energies[i] - energies[j])
            n_exch_total += 1
            if delta >= 0.0 or rng.random() < np.exp(delta):
                reps[i], reps[j] = reps[j], reps[i]
                energies[i], energies[j] = energies[j], energies[i]
                n_exch_acc += 1

        # --- quench the coldest replica to see the underlying 0 K state ---
        if (cycle + 1) % args.quench_every == 0 or cycle == args.n_cycles - 1:
            snap, e_q = quench(reps[0], args.fmax_monitor)
            if e_q < best_e:
                best_e, best = e_q, snap.copy()
            exch_rate = n_exch_acc / max(1, n_exch_total)
            trace.append((cycle + 1, e_q / n, exch_rate))
            log(f"  cycle {cycle + 1:4d} | coldest quenched {e_q / n:+.6f} eV/atom "
                f"| best {best_e / n:+.6f} | exch {exch_rate:.3f} "
                f"| {(time.time() - t0) / 60:.0f} min")

    log("final tight relax of the best structure...")
    best.calc = atoms.calc
    BFGS(best, logfile=None).run(fmax=args.fmax_final, steps=500)
    e_final = best.get_potential_energy()
    log(f"FINISHED. best energy {e_final / n:+.6f} eV/atom "
        f"({(time.time() - t0) / 60:.0f} min, exchange acceptance "
        f"{n_exch_acc / max(1, n_exch_total):.3f})")

    write(f"{args.out}_best.vasp", best, direct=True, sort=True)
    write(f"{args.out}_best.xyz", best)
    np.save(f"{args.out}_trace.npy", np.array(trace))
    return best, e_final


def melt_quench(atoms, args, rng):
    n = len(atoms)
    log(f"=== melt-quench: {n} atoms, {dict(Counter(atoms.get_chemical_symbols()))} ===")
    log(f"    cell {np.round(atoms.cell.cellpar(), 3)}, "
        f"density {n / atoms.get_volume():.4f} atoms/A^3")
    log(f"    {args.n_temps} stages {args.t_hi}->{args.t_lo} K, "
        f"{args.loops_per_temp} loops each ({args.n_temps * args.loops_per_temp} total), "
        f"{args.md_steps} MD steps + {args.swaps} swap attempts per loop")

    t_start = time.time()
    log(f"melting at {args.t_hi} K ({args.melt_steps} steps)...")
    md_burst(atoms, args.t_hi, args.melt_steps, args.timestep)

    temps = np.linspace(args.t_hi, args.t_lo, args.n_temps)
    best_e, best = np.inf, None
    trace, loop = [], 0

    for T in temps:
        n_acc_stage = 0
        for _ in range(args.loops_per_temp):
            loop += 1
            md_burst(atoms, T, args.md_steps, args.timestep)
            e_current = atoms.get_potential_energy()
            e_current, n_acc = try_swaps(atoms, T, args.swaps, rng, e_current)
            n_acc_stage += n_acc

        snap, e_q = quench(atoms, args.fmax_monitor)
        if e_q < best_e:
            best_e, best = e_q, snap.copy()
        acc_rate = n_acc_stage / max(1, args.swaps * args.loops_per_temp)
        trace.append((T, e_q / n, acc_rate))
        el = (time.time() - t_start) / 60
        log(f"  T={T:7.0f} K | loop {loop:4d} | quenched {e_q / n:+.5f} eV/atom "
            f"| best {best_e / n:+.5f} | acc {acc_rate:.3f} | {el:.0f} min")

    log("final tight relax of the best structure...")
    best.calc = atoms.calc
    BFGS(best, logfile=None).run(fmax=args.fmax_final, steps=500)
    e_final = best.get_potential_energy()
    log(f"FINISHED. best energy {e_final / n:+.6f} eV/atom "
        f"({(time.time() - t_start) / 60:.0f} min)")

    write(f"{args.out}_best.vasp", best, direct=True, sort=True)
    write(f"{args.out}_best.xyz", best)
    np.save(f"{args.out}_trace.npy", np.array(trace))
    log(f"wrote {args.out}_best.vasp, {args.out}_best.xyz, {args.out}_trace.npy")
    return best, e_final


def benchmark(atoms, args):
    """Measure the real per-evaluation cost before committing to a schedule.

    Two traps this avoids, both learned the hard way in md_run.py's benchmark:
      - ASE caches results, so repeating an identical call times nothing. We
        perturb positions (MD-like) or swap species (MC-like) between calls.
      - CUDA calls are asynchronous, so without synchronize() we would time the
        queueing rather than the work.
    """
    try:
        import torch
        cuda = torch.cuda.is_available() and args.device == "cuda"
    except ImportError:
        torch, cuda = None, False

    atoms.get_potential_energy()                     # warm-up, not timed
    if cuda:
        torch.cuda.synchronize()

    reps = 20
    t0 = time.time()
    for _ in range(reps):
        atoms.rattle(1e-4)                           # invalidate the cache
        atoms.get_potential_energy()
    if cuda:
        torch.cuda.synchronize()
    t_md = (time.time() - t0) / reps

    syms = np.array(atoms.get_chemical_symbols())
    t0 = time.time()
    for k in range(reps):
        i, j = k % len(syms), (k + len(syms) // 2) % len(syms)
        syms[i], syms[j] = syms[j], syms[i]
        atoms.set_chemical_symbols(list(syms))
        _refresh_masses(atoms)
        atoms.get_potential_energy()
    if cuda:
        torch.cuda.synchronize()
    t_mc = (time.time() - t0) / reps

    loops = args.n_temps * args.loops_per_temp
    n_evals = loops * (args.md_steps + args.swaps)
    est_h = (loops * (args.md_steps * t_md + args.swaps * t_mc)) / 3600

    log("=== BENCHMARK ===")
    log(f"{len(atoms)} atoms, device {args.device}, dtype {args.dtype}")
    log(f"  energy after position change (MD-like): {t_md * 1000:8.2f} ms")
    log(f"  energy after species swap    (MC-like): {t_mc * 1000:8.2f} ms")
    log(f"  default schedule: {loops} loops x ({args.md_steps} MD + {args.swaps} swaps) "
        f"= {n_evals:,} evaluations")
    log(f"  estimated wall time: {est_h:.2f} h")
    log("  (the paper used 7800 swaps/loop with pair potentials, where a swap's")
    log("   energy change is local. Scale --swaps to what the estimate allows.)")
    if args.fixed_sites:
        est_fs_h = (args.n_temps * args.swaps * t_mc) / 3600
        log(f"  fixed-site schedule: {args.n_temps} stages x {args.swaps} swaps "
            f"= {args.n_temps * args.swaps:,} evaluations, ~{est_fs_h:.2f} h")


def parse_counts(spec):
    """'Al:18,Co:5,Ni:3' -> {'Al': 18, 'Co': 5, 'Ni': 3}"""
    counts = {}
    for part in spec.split(","):
        sym, n = part.split(":")
        counts[sym.strip()] = int(n)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None,
                    help="known structure to take the cell and composition from, "
                         "and to compare the result against. Accepts the legacy "
                         "coords.txt format or anything ASE reads (.vasp/POSCAR, "
                         ".cif, .xyz).")
    ap.add_argument("--supercell", default=None,
                    help="build a supercell of the reference before anything "
                         "else: 3 diagonal repeats ('3,2,1') or a 9-number "
                         "transformation matrix row-by-row "
                         "('3,3,0,-2,2,0,0,0,1' gives the W-phase 3x2x1 "
                         "conventional box). Same convention as md_run.py.")
    ap.add_argument("--cell", default=None, help="a,b,c in Angstrom")
    ap.add_argument("--angles", default="90,90,90", help="alpha,beta,gamma in degrees")
    ap.add_argument("--counts", default=None, help="e.g. Al:56,Co:6,Ni:16")
    ap.add_argument("--target-counts", default=None,
                    help="reassign the minimum number of sites of the loaded "
                         "(super)cell to hit these counts, e.g. "
                         "Al:2244,Co:696,Ni:240. Must sum to the site total. "
                         "This is the manual-composition route chosen instead "
                         "of grand-canonical MC.")
    ap.add_argument("--start-from", choices=["random", "as-tiled"], default="random",
                    help="random (default): previous behaviour - random "
                         "positions for anneals, randomised species for "
                         "--fixed-sites. as-tiled: keep positions AND species "
                         "exactly as loaded, so a supercell starts from tiled "
                         "copies of the small cell's ordering. Requires "
                         "--reference.")

    ap.add_argument("--t-hi", type=float, default=2500.0, help="melt temperature (K)")
    ap.add_argument("--t-lo", type=float, default=1000.0, help="final anneal temperature (K)")
    ap.add_argument("--n-temps", type=int, default=20,
                    help="annealing stages between t-hi and t-lo (paper used 20)")
    ap.add_argument("--loops-per-temp", type=int, default=5,
                    help="MD+MC loops per stage; 20 stages x 5 = 100 loops, "
                         "matching the paper's ~100 to first low-energy states")
    ap.add_argument("--md-steps", type=int, default=1000, help="MD steps per loop (paper: 1000)")
    ap.add_argument("--swaps", type=int, default=500,
                    help="MC swap attempts per loop. The paper used 7800, cheap "
                         "with pair potentials; each attempt is a full MACE "
                         "forward pass here, so this is reduced. Scale roughly "
                         "with atom count, and check --benchmark first.")
    ap.add_argument("--timestep", type=float, default=0.2,
                    help="fs; 0.2 as in the paper, not the 2 fs used at 300 K")
    ap.add_argument("--melt-steps", type=int, default=2000,
                    help="initial MD steps at t-hi to randomise the melt")
    ap.add_argument("--fmax-monitor", type=float, default=0.02,
                    help="eV/A for the periodic monitoring quenches (loose, for speed)")
    ap.add_argument("--fmax-final", type=float, default=0.001,
                    help="eV/A for the final relax of the best structure")

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fixed-sites", action="store_true",
                    help="keep the loaded positions, run species swaps only. "
                         "With --start-from random this is the validation-gap "
                         "diagnostic; with --start-from as-tiled it is the "
                         "supercell species-swap experiment. Requires "
                         "--reference.")
    ap.add_argument("--replica-exchange", action="store_true",
                    help="parallel tempering instead of staged annealing. This is "
                         "the paper's method and the fix for the trapping that "
                         "five independent staged anneals demonstrated.")
    ap.add_argument("--n-replicas", type=int, default=8,
                    help="replicas on the temperature ladder (paper used 20; 8 "
                         "keeps a full run inside one 24 h job)")
    ap.add_argument("--n-cycles", type=int, default=50,
                    help="exchange cycles; each is one MD+MC block per replica "
                         "followed by adjacent-pair exchange attempts")
    ap.add_argument("--quench-every", type=int, default=5,
                    help="quench the coldest replica every N cycles to monitor")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32",
                    help="float32 for the anneal; the final relax is where "
                         "precision matters most")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--out", default="mq")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.start_from == "as-tiled" and not args.reference:
        raise SystemExit("--start-from as-tiled needs --reference (it keeps "
                         "that file's positions and species)")

    # ---- work out the cell and composition ----
    ref = None
    if args.reference:
        ref = load_structure(args.reference)
        log(f"loaded {args.reference}: {len(ref)} atoms, "
            f"{dict(Counter(ref.get_chemical_symbols()))}, "
            f"volume {ref.get_volume():.1f} A^3")
        if args.supercell:
            ref = apply_supercell(ref, args.supercell)
        if args.target_counts:
            retarget_counts(ref, parse_counts(args.target_counts), rng)
        cell = np.array(ref.cell)
        counts = dict(Counter(ref.get_chemical_symbols()))
        log(f"working cell: {len(ref)} atoms, {counts}, "
            f"volume {ref.get_volume():.1f} A^3, "
            f"density {len(ref) / ref.get_volume():.4f} atoms/A^3")
        if args.counts:
            counts = parse_counts(args.counts)
            log(f"  overriding composition with {counts}")
    else:
        if not (args.cell and args.counts):
            raise SystemExit("give either --reference, or both --cell and --counts")
        a, b, c = [float(x) for x in args.cell.split(",")]
        al, be, ga = [float(x) for x in args.angles.split(",")]
        cell = cellpar_to_cell([a, b, c, al, be, ga])
        counts = parse_counts(args.counts)

    # ---- starting configuration for the annealing modes ----
    # (--fixed-sites works from ref directly and ignores this object, but the
    # benchmark uses it, so build it either way)
    if ref is not None and args.start_from == "as-tiled":
        atoms = ref.copy()
        log("as-tiled start: positions and species taken from the input")
    else:
        atoms = random_config(cell, counts, rng)
        log(f"random start: {len(atoms)} atoms in a fixed cell, "
            f"min separation {atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)].min():.2f} A")

    # ---- attach MACE ----
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium-mpa-0", device=args.device, default_dtype=args.dtype)
    atoms.calc = calc

    if args.benchmark:
        benchmark(atoms, args)
        return

    if args.fixed_sites:
        if ref is None:
            raise SystemExit("--fixed-sites needs --reference (it uses its site list)")
        ref.calc = calc
        best, e_final = fixed_site_mc(ref, args, rng)
    elif args.replica_exchange:
        best, e_final = replica_exchange(atoms, args, rng)
    else:
        best, e_final = melt_quench(atoms, args, rng)

    # ---- the validation comparison ----
    # With an as-tiled or retargeted start this compares the found chemistry
    # against the relaxed STARTING ordering - i.e. it answers "did the MC beat
    # the tiled arrangement", which is exactly the supercell experiment's
    # question. Note the comparison relaxes ref in place, so it runs on the
    # starting species, not the original file's, when --target-counts is used.
    if ref is not None:
        ref.calc = calc
        BFGS(ref, logfile=None).run(fmax=args.fmax_final, steps=500)
        e_ref = ref.get_potential_energy() / len(ref)
        e_got = e_final / len(best)
        label = ("as-tiled/starting ordering (relaxed)"
                 if args.start_from == "as-tiled" or args.target_counts
                 else "reference (relaxed)")
        log("=== validation against the reference structure ===")
        log(f"  {label}: {e_ref:+.6f} eV/atom")
        log(f"  search found:        {e_got:+.6f} eV/atom")
        log(f"  difference:          {(e_got - e_ref) * 1000:+.2f} meV/atom")
        if e_got - e_ref < 0.001:
            log("  -> matches or beats the starting ordering.")
        else:
            log("  -> higher than the starting ordering: the search is stuck. "
                "Run more loops, or raise --swaps.")


if __name__ == "__main__":
    main()
