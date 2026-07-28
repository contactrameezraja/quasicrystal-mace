"""
MACE molecular dynamics: VDOS of decagonal Al-Ni-Co approximants
================================================================

Why this needs a cluster
------------------------
The VDOS needs two things at once, and a laptop cannot give both:

  Mode density.  Under periodic boundary conditions a cell only supports modes
  commensurate with it. 208 atoms -> 624 modes; 1664 atoms -> ~5000. Below some
  size the spectrum is discrete spikes rather than a density. Running longer
  does not fix this; only a bigger cell adds modes.

  Frequency resolution.  Set by trajectory length, roughly 1/T. 30 ps gives
  0.138 meV.

A 4x4x4 (1664-atom) run projected to ~55 days on a MacBook Air.

The plan
--------
  1. --benchmark        time force evaluations across sizes and dtypes
  2. --supercell 2      reproduce the laptop result on the GPU (a check that
                        the cluster gives the same answer)
  3. --supercell 3      moderate
  4. --supercell 4      the target

Each run gives a VDOS. Comparing them across sizes answers whether the spectrum
has converged with system size, which is a result in itself rather than just a
stepping stone to the big run.

Protocol, and why
-----------------
  Tight relaxation, fmax = 0.001 eV/A.  Ten times tighter than the O1 stability
  check. Residual force contaminates the dynamics.

  NVT equilibration, Langevin, 300 K, 2 fs.  300 K is not arbitrary: it is the
  temperature Mihalkovic used for MD annealing of this system, close to the
  296 K of the experimental neutron GVDOS, and well below the ~2/3 T_melt
  threshold where Gahler & Hocker found Al starts to diffuse. Above that atoms
  migrate rather than vibrate and the VDOS interpretation breaks.

  NVE burn-in, thermostat off, discarded.  Removing the thermostat leaves a
  brief transient.

  NVE production, storing full velocities.  The thermostat must be off: it
  modifies the equations of motion, so the transform would include its action
  rather than just the phonons.

Sampling every 4 fs puts the Nyquist limit far above the ~53 meV cutoff.
The VACF -> VDOS analysis is done off the cluster.

Usage
-----
    python md_run.py --benchmark
    python md_run.py --supercell 2 --steps 15000    # 30 ps, matches the laptop run
    python md_run.py --supercell 4 --steps 100000   # 200 ps

    # option 2: independent runs from different seeds, to average later
    python md_run.py --supercell 4 --steps 100000 --seed 1 --out md_4x4x4_s1
    python md_run.py --supercell 4 --steps 100000 --seed 2 --out md_4x4x4_s2

    # size series with positions (for MSD and g(r), to answer whether atoms
    # vibrate about equilibrium or do something dramatic)
    python md_run.py --supercell 2 --steps 100000 --store-positions --out md_2x2x2
    python md_run.py --supercell 3 --steps 100000 --store-positions --out md_3x3x3

    # precision control
    python md_run.py --supercell 2 --steps 100000 --dtype float64 --out md_2x2x2_f64
"""

import argparse
import time
from collections import Counter

import numpy as np
from ase import Atoms
from ase.io import read
from ase.build import make_supercell
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase import units
from mace.calculators import mace_mp

TYPE_TO_ELEMENT = {13: "Al", 27: "Co", 28: "Ni"}


def log(msg, path="md_progress.txt"):
    print(msg, flush=True)
    with open(path, "a") as f:
        f.write(msg + "\n")


def parse_structure(path):
    """Read a structure. Two paths, deliberately:

    Anything ASE can read (.vasp, .cif, .xyz) goes through ase.io.read. This is
    how the 265-atom W-phase gets in, and the 26-atom sanity check below cannot
    apply to it.

    The custom coords.txt format (cell vectors on one line, then x y z type per
    atom, type 0 rows dropped as partial-occupancy placeholders) keeps the sanity
    check, because for that file we know exactly what it should contain and a
    silent misparse would waste hours of GPU time.
    """
    if not path.lower().endswith(".txt"):
        atoms = read(path)
        atoms.pbc = True
        log(f"read {path} via ASE: {len(atoms)} atoms, "
            f"volume {atoms.get_volume():.1f} A^3, "
            f"cellpar {np.round(atoms.cell.cellpar(), 3)}")
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

    # Sanity check against the values verified against the CMU database in O1.
    # If this fails the file has been read wrongly - stop rather than run for
    # hours on a garbage structure.
    if len(atoms) != 26 or abs(atoms.get_volume() - 360.8) / 360.8 > 0.02:
        raise SystemExit(
            f"Parsed {len(atoms)} atoms, volume {atoms.get_volume():.1f} A^3. "
            f"Expected 26 atoms at 360.8 A^3 (Al18Co5Ni3). Check the file.")
    return atoms


def benchmark(base, device):
    """Time force evaluations across supercell sizes and dtypes.

    On a MacBook Air, 208 atoms ran at 2.13 s/step but 1664 atoms at 317 s/step.
    That is 149x for 8x the atoms. MACE has a finite cutoff so cost should scale
    roughly linearly, which makes the laptop number suspect. Two candidates:

      - memory pressure (the Air shares unified memory; the GPU has 23 GB
        dedicated)
      - float64. It is recommended for geometry optimisation, but workstation
        GPUs rate-limit double precision heavily. MD normally runs float32.

    If either is the cause, the laptop number does not transfer.

    Two traps this avoids:
      - ASE caches force results. Calling get_forces() twice without moving the
        atoms returns the cached array instantly, so the timing is meaningless.
        rattle() perturbs positions to invalidate the cache.
      - CUDA calls are asynchronous. Without synchronize() you time the queueing
        rather than the work.
    """
    import torch

    log("=== BENCHMARK ===")
    log(f"{'dtype':>8} {'cell':>7} {'atoms':>7} {'s/step':>10} {'200ps (h)':>11}")
    for dtype in ("float32", "float64"):
        calc = mace_mp(model="medium-mpa-0", device=device, default_dtype=dtype)
        for n in (2, 3, 4):
            sc = make_supercell(base, np.diag([n, n, n]))
            sc.calc = calc
            try:
                sc.get_forces()                    # warm-up, not timed
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    sc.rattle(0.0001)              # invalidate ASE's force cache
                    sc.get_forces()
                if device == "cuda":
                    torch.cuda.synchronize()       # wait for the GPU to finish
                dt = (time.time() - t0) / 5
                log(f"{dtype:>8} {n}x{n}x{n:<3} {len(sc):7d} {dt:10.4f} "
                    f"{dt * 100000 / 3600:11.2f}")
            except Exception as e:
                log(f"{dtype:>8} {n}x{n}x{n:<3} {len(sc):7d}  FAILED: "
                    f"{type(e).__name__}: {e}")


def run_md(calc, base, P, n_steps, T, out, tag, seed=None, store_positions=False):
    sc = make_supercell(base, P)
    sc.calc = calc
    log(f"=== MD: {tag}, {len(sc)} atoms, {3*len(sc)} modes, {T} K ===")
    log(f"    supercell matrix rows {np.asarray(P).tolist()}, "
        f"det {int(round(abs(np.linalg.det(P))))}")
    log(f"    box lengths {np.round(sc.cell.cellpar()[:3], 2)}, "
        f"angles {np.round(sc.cell.cellpar()[3:], 2)}")
    log(f"    {dict(Counter(sc.get_chemical_symbols()))}")

    timestep = 2.0 * units.fs

    # A seed makes the initial velocities reproducible AND lets independent runs
    # differ (option 2: average several runs from different seeds to beat down
    # the long-lag VACF noise). Without a seed, ASE draws from global numpy state.
    if seed is not None:
        np.random.seed(seed)
        log(f"    seed {seed}")
    MaxwellBoltzmannDistribution(sc, temperature_K=T)
    Stationary(sc)                                 # zero net momentum before equilibration

    log("NVT equilibration (10 ps)...")
    t0 = time.time()
    Langevin(sc, timestep, temperature_K=T, friction=0.01 / units.fs).run(5000)
    log(f"  done in {time.time()-t0:.0f}s, instantaneous T = "
        f"{sc.get_kinetic_energy()/len(sc)/(1.5*units.kB):.0f} K")

    # Zero the net momentum AGAIN, after the thermostat. The Langevin random
    # forces can leave a small overall drift velocity, which NVE would then
    # preserve and which shows up as a constant offset in the VACF tail. The
    # 4x4x4 run showed this was negligible (tail mean -1e-5) but it costs
    # nothing to remove and rules the question out.
    Stationary(sc)

    dyn = VelocityVerlet(sc, timestep)
    log("NVE burn-in (2 ps, not recorded)...")
    dyn.run(1000)

    velocities, energies = [], []
    positions = [] if store_positions else None
    sample_every = 2                               # every 4 fs

    def record():
        velocities.append(sc.get_velocities().copy())
        energies.append(sc.get_total_energy())
        if store_positions:
            positions.append(sc.get_positions().copy())

    dyn.attach(record, interval=sample_every)

    n_blocks = 50
    res = 4.1357 / (n_steps * 2e-3)                # rough resolution, meV
    log(f"NVE production: {n_steps*2/1000:.0f} ps "
        f"(~{res:.3f} meV resolution), {n_blocks} blocks"
        f"{', storing positions' if store_positions else ''}...")
    t0 = time.time()
    for b in range(n_blocks):
        dyn.run(n_steps // n_blocks)
        np.save(f"{out}_velocities.npy", np.array(velocities))
        np.save(f"{out}_energies.npy", np.array(energies))
        if store_positions:
            np.save(f"{out}_positions.npy", np.array(positions))
        el = time.time() - t0
        log(f"  block {b+1}/{n_blocks} | {el/60:.0f} min | "
            f"~{el/(b+1)*(n_blocks-1-b)/60:.0f} min left | {len(velocities)} samples")

    e = np.array(energies)
    log(f"FINISHED. velocities {np.array(velocities).shape}")
    log(f"Energy drift: {(e.max()-e.min())/len(sc)*1000:.4f} meV/atom "
        f"(laptop 2x2x2 run gave 0.039)")
    if store_positions:
        log(f"positions saved: {np.array(positions).shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default="Al9Co2Ni2-coords.txt")
    ap.add_argument("--supercell", type=str, default="4",
                    help="cube size ('4' -> 4x4x4) or three comma-separated "
                         "dims ('8,4,4'). An elongated box lowers the floor "
                         "along its long axis: E_min is set by the LONGEST "
                         "dimension, so 8x4x4 reaches ~1.7 meV with fewer "
                         "atoms (3328) than a 6x6x6 (5616, OOM on 23 GB).")
    ap.add_argument("--steps", type=int, default=100000, help="production steps (x2 fs)")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32",
                    help="float32 for MD; float64 is more accurate but heavily "
                         "rate-limited on workstation GPUs")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for the initial velocities; use different "
                         "values for independent runs to average (option 2)")
    ap.add_argument("--store-positions", action="store_true",
                    help="also save positions, for MSD and g(r). Doubles the "
                         "output size - use only where you need the structure.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = parse_structure(args.structure)
    log(f"Loaded {len(base)} atoms: {dict(Counter(base.get_chemical_symbols()))}, "
        f"volume {base.get_volume():.1f} A^3")

    # Relax in float64 regardless. One-off cost, and the dynamics need a
    # well-converged starting point.
    calc64 = mace_mp(model="medium-mpa-0", device=args.device, default_dtype="float64")
    base.calc = calc64
    BFGS(FrechetCellFilter(base), logfile=None).run(fmax=0.001)
    fmax = np.sqrt((base.get_forces() ** 2).sum(axis=1)).max()
    log(f"Relaxed (float64): max residual force {fmax:.5f} eV/A "
        f"(laptop gave 0.00083)")

    if args.benchmark:
        benchmark(base, args.device)
        return

    # --supercell accepts:
    #   '4'                     -> diag(4,4,4)
    #   '8,4,4'                 -> diag(8,4,4)
    #   '3,3,0,-2,2,0,0,0,1'    -> the full 3x3 matrix, row by row
    # The matrix form is needed for the W-phase, whose file is the primitive
    # cell of a C-centred setting (gamma = 23.27 deg). P = [[1,1,0],[-1,1,0],
    # [0,0,1]] recovers the conventional 39.35 x 8.10 x 23.23 A axes, and
    # combining that with a 3x2x1 tiling gives [[3,3,0],[-2,2,0],[0,0,1]].
    parts = [int(x) for x in args.supercell.split(",")]
    if len(parts) == 1:
        P = np.diag(parts * 3)
        tag = f"{parts[0]}x{parts[0]}x{parts[0]}"
    elif len(parts) == 3:
        P = np.diag(parts)
        tag = "x".join(str(p) for p in parts)
    elif len(parts) == 9:
        P = np.array(parts).reshape(3, 3)
        tag = "mat" + "_".join(str(p) for p in parts)
    else:
        raise SystemExit(f"--supercell needs 1, 3 or 9 numbers, got {args.supercell!r}")

    det = np.linalg.det(P)
    if abs(det) < 0.5:
        raise SystemExit(f"supercell matrix is singular (det {det:.3f})")
    if det < 0:
        log(f"WARNING: det(P) = {det:.0f} is negative, which flips handedness. "
            f"Swap two rows if that matters.")

    out = args.out or f"md_{tag}"
    calc = mace_mp(model="medium-mpa-0", device=args.device, default_dtype=args.dtype)
    log(f"MD dtype: {args.dtype}")
    run_md(calc, base, P, args.steps, args.temperature, out, tag,
           seed=args.seed, store_positions=args.store_positions)


if __name__ == "__main__":
    main()
