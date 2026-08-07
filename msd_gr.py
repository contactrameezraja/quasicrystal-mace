"""
MSD and g(r) from stored MD positions
=====================================

What question this answers
--------------------------
Every spectrum in this project assumes the atoms VIBRATE ABOUT FIXED SITES. If
any species were diffusing at 300 K, the velocity autocorrelation would pick up
a diffusive contribution at low frequency and the VDOS interpretation would be
wrong in exactly the region the golden-ratio features live in. That assumption
has been argued from the temperature (300 K is well below the ~2/3 T_melt
threshold where Al is reported to start diffusing in these alloys) but never
measured on the trajectories themselves. This script measures it.

Three diagnostics, in increasing strength:

  MSD(t).  For a solid it rises and PLATEAUS at 2<u^2> (two independent
  fluctuations, one at each time); for a diffusing species
  it keeps rising linearly, MSD = 6Dt. The shape is the test, not the value.

  Lindemann ratio, L = sqrt(<u^2>) / d_nn.  The conventional melting criterion
  is L ~ 0.1. A vibrating crystal at room temperature sits well below that.
  Reported per element, because Al is the light, mobile species here and is the
  one that would move first.

  g(r).  Sharp, well separated peaks mean intact structure. Peak positions also
  cross-check the nearest-neighbour distances of the parsed structure, and a
  filled-in first minimum would be the signature of atoms hopping between sites.

Positions are recorded by md_run.py --store-positions, which are available for
the 2x2x2 (seed 11) and 3x3x3 (seed 12) runs.

Usage
-----
    python msd_gr.py --positions md_2x2x2_s11_positions.npy \
        --structure Al9Co2Ni2-coords.txt --supercell 2 --out msd_2x2x2

    python msd_gr.py --positions md_3x3x3_s12_positions.npy \
        --structure Al9Co2Ni2-coords.txt --supercell 3 --out msd_3x3x3

    # sanity check on synthetic data before trusting either
    python msd_gr.py --self-test
"""

import argparse

import numpy as np
from ase import Atoms
from ase.io import read
from ase.build import make_supercell

TYPE_TO_ELEMENT = {13: "Al", 27: "Co", 28: "Ni"}


def log(msg, path="msd_progress.txt"):
    print(msg, flush=True)
    with open(path, "a") as f:
        f.write(msg + "\n")


def parse_structure(path):
    """Same parser as md_run.py and reduce_vdos.py, so the atoms here are the
    atoms the MD ran on."""
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
        raise SystemExit(f"Parsed {len(atoms)} atoms, volume "
                         f"{atoms.get_volume():.1f} A^3; expected 26 at 360.8.")
    return atoms


def parse_matrix(spec):
    parts = [int(x) for x in spec.split(",")]
    if len(parts) == 1:
        return np.diag(parts * 3)
    if len(parts) == 3:
        return np.diag(parts)
    if len(parts) == 9:
        return np.array(parts).reshape(3, 3)
    raise SystemExit(f"--supercell needs 1, 3 or 9 numbers, got {spec!r}")


def msd_multi_origin(pos, n_origins=20, max_lag_frac=0.5):
    """MSD averaged over several time origins, which is what makes the plateau
    readable rather than noisy. pos: (n_frames, n_atoms, 3), unwrapped.
    Returns lags (frames) and MSD per atom-averaged, plus the per-atom mean
    squared displacement at the longest lag for element splitting."""
    n_frames = pos.shape[0]
    max_lag = int(n_frames * max_lag_frac)
    lags = np.unique(np.geomspace(1, max_lag, 40).astype(int))
    origins = np.linspace(0, n_frames - max_lag - 1, n_origins).astype(int)
    msd = np.zeros(len(lags))
    per_atom = None
    for k, lag in enumerate(lags):
        acc = np.zeros(pos.shape[1])
        for t0 in origins:
            d = pos[t0 + lag] - pos[t0]
            acc += (d ** 2).sum(axis=1)
        acc /= len(origins)
        msd[k] = acc.mean()
        if k == len(lags) - 1:
            per_atom = acc
    return lags, msd, per_atom


def rdf(pos_frames, cell, symbols, r_max=6.0, n_bins=300):
    """Pair distribution function with the minimum-image convention, averaged
    over the supplied frames. Returns r, total g(r), and a dict of partials."""
    inv = np.linalg.inv(cell)
    vol = abs(np.linalg.det(cell))
    n_at = pos_frames.shape[1]
    edges = np.linspace(0, r_max, n_bins + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    els = sorted(set(symbols))
    idx = {e: np.where(np.array(symbols) == e)[0] for e in els}
    hist_tot = np.zeros(n_bins)
    hist_part = {f"{a}-{b}": np.zeros(n_bins) for i, a in enumerate(els)
                 for b in els[i:]}

    for p in pos_frames:
        frac = p @ inv
        d = frac[:, None, :] - frac[None, :, :]
        d -= np.round(d)                     # minimum image
        dist = np.linalg.norm(d @ cell, axis=-1)
        iu = np.triu_indices(n_at, k=1)
        dd = dist[iu]
        hist_tot += np.histogram(dd, bins=edges)[0]
        for key in hist_part:
            a, b = key.split("-")
            m = np.isin(iu[0], idx[a]) & np.isin(iu[1], idx[b])
            if a != b:
                m |= np.isin(iu[0], idx[b]) & np.isin(iu[1], idx[a])
            hist_part[key] += np.histogram(dd[m], bins=edges)[0]

    n_frames = len(pos_frames)
    norm_tot = n_frames * 0.5 * n_at * (n_at / vol) * shell
    g_tot = hist_tot / norm_tot
    g_part = {}
    for key, h in hist_part.items():
        a, b = key.split("-")
        na, nb = len(idx[a]), len(idx[b])
        pairs = na * nb if a != b else 0.5 * na * (na - 1)
        g_part[key] = h / (n_frames * pairs * shell / vol) if pairs else h * 0.0
    return r, g_tot, g_part


def report(pos, cell, symbols, sample_fs, out, rdf_frames=50, r_min=2.0,
           d_nn_override=None):
    n_frames, n_at, _ = pos.shape
    log(f"{n_frames} frames, {n_at} atoms, sampled every {sample_fs} fs "
        f"({n_frames*sample_fs/1000:.0f} ps)")

    # Positions from a VelocityVerlet run are continuous, but a jump larger than
    # half the box between consecutive frames would mean they were wrapped, which
    # would corrupt the MSD. Check rather than assume.
    step = np.linalg.norm(np.diff(pos[:min(200, n_frames)], axis=0), axis=-1).max()
    half_box = 0.5 * min(np.linalg.norm(cell, axis=1))
    log(f"largest single-frame displacement {step:.3f} A "
        f"(half the shortest box vector is {half_box:.2f} A)")
    if step > half_box:
        log("  WARNING: looks wrapped; MSD would be corrupted. Unwrap first.")

    lags, msd, per_atom = msd_multi_origin(pos)
    t_ps = lags * sample_fs / 1000.0
    log("--- MSD(t), multi-origin average ---")
    log(f"{'t (ps)':>9} {'MSD (A^2)':>11}")
    for t, m in zip(t_ps, msd):
        if t in t_ps[::5] or t == t_ps[-1]:
            log(f"{t:9.2f} {m:11.4f}")

    # A plateau means vibration; a linear rise means diffusion. Compare the
    # second half of the curve against a straight line through it.
    half = len(msd) // 2
    slope = np.polyfit(t_ps[half:], msd[half:], 1)[0]
    plateau = msd[half:].mean()
    log(f"plateau (mean over second half) : {plateau:.4f} A^2")
    log(f"slope over second half          : {slope:.5f} A^2/ps")
    log(f"  -> implied D if diffusive     : {slope/6*1e-4:.3e} cm^2/s")
    log(f"  -> fractional rise per ps     : {slope/max(plateau,1e-12):.4f}")
    if abs(slope) / max(plateau, 1e-12) < 0.02:
        log("  VERDICT: flat within 2% per ps - vibration about fixed sites")
    else:
        log("  VERDICT: still rising - check for diffusion of some species")

    # Nearest-neighbour distance from the first g(r) peak, needed for Lindemann.
    stride = max(1, n_frames // rdf_frames)
    frames = pos[::stride][:rdf_frames]
    r, g_tot, g_part = rdf(frames, cell, symbols)

    # d_nn must be the FIRST coordination shell, not the tallest peak. Taking the
    # global maximum works only while the nearest-neighbour peak happens to be
    # dominant; if a further shell ever grew taller, every Lindemann ratio would
    # silently shift to the wrong shell. Find the first local maximum above g = 1
    # instead, and report the global maximum alongside it as a cross-check.
    i0 = int(np.searchsorted(r, r_min))
    # Smooth before peak-finding: with 0.02 A bins a raw g(r) carries noise
    # bumps that cross 1.0 and would be mistaken for a coordination shell.
    k = 5
    g_sm = np.convolve(g_tot, np.ones(k) / k, mode="same")
    thresh = max(1.5, 0.4 * g_sm[i0:].max())
    peaks = [i for i in range(i0 + 1, len(r) - 1)
             if g_sm[i] > thresh and g_sm[i] >= g_sm[i - 1] and g_sm[i] > g_sm[i + 1]]
    i_glob = i0 + int(np.argmax(g_sm[i0:]))
    if d_nn_override is not None:
        first = d_nn_override
        i_first = int(np.searchsorted(r, first))
        log(f"d_nn overridden on the command line: {first:.3f} A")
    elif peaks:
        i_first = peaks[0]
        first = r[i_first]
    else:
        i_first = i_glob
        first = r[i_glob]
        log("  no local maximum passed the significance threshold; falling back "
            "to the tallest peak")
    log("--- g(r) ---")
    log(f"averaged over {len(frames)} frames")
    log(f"first peak of total g(r)        : {first:.3f} A "
        f"(height {g_tot[i_first]:.2f}) <- d_nn used below")
    log(f"tallest peak of total g(r)      : {r[i_glob]:.3f} A "
        f"(height {g_tot[i_glob]:.2f})")
    if i_glob != i_first:
        log("  NOTE: the tallest peak is not the first shell; d_nn uses the "
            "first, which is the correct choice for the Lindemann ratio")
    log(f"significant peaks (g > {thresh:.2f}) at : "
        f"{[round(float(r[i]), 3) for i in peaks[:6]]} A")
    lo, hi = np.searchsorted(r, first + 0.2), np.searchsorted(r, first + 1.2)
    log(f"first minimum after it          : {r[lo:hi][np.argmin(g_tot[lo:hi])]:.3f} A "
        f"(g = {g_tot[lo:hi].min():.3f}; a filled-in minimum would mean hopping)")
    for key, g in g_part.items():
        if g.max() > 0:
            log(f"  {key:8s} first peak {r[np.argmax(g)]:.3f} A, height {g.max():.2f}")

    # <u^2> about each atom's own time-averaged position. NOT the two-time MSD
    # plateau, which equals 2<u^2> because r(t) and r(0) fluctuate independently
    # and would inflate L by sqrt(2).
    mean_pos = pos.mean(axis=0)
    u2 = ((pos - mean_pos[None, :, :]) ** 2).sum(axis=2).mean(axis=0)
    log("--- Lindemann ratio per element (L = sqrt(<u^2>)/d_nn) ---")
    log(f"d_nn taken as {first:.3f} A; the melting criterion is L ~ 0.1")
    log(f"<u^2> is about each atom's time-averaged position; the MSD plateau "
        f"above equals 2<u^2> ({plateau:.4f} vs {2*u2.mean():.4f})")
    for el in sorted(set(symbols)):
        m = np.array(symbols) == el
        u_rms = np.sqrt(u2[m].mean())
        log(f"  {el}: sqrt(<u^2>) = {u_rms:.4f} A, L = {u_rms/first:.4f} "
            f"({m.sum()} atoms)")
    u_all = np.sqrt(u2.mean())
    log(f"  all: sqrt(<u^2>) = {u_all:.4f} A, L = {u_all/first:.4f}")

    np.savez(f"{out}_msd_gr.npz", lags=lags, t_ps=t_ps, msd=msd,
             per_atom=per_atom, u2_per_atom=u2, r=r, g_total=g_tot,
             **{f"g_{k}": v for k, v in g_part.items()},
             symbols=np.array(symbols))
    log(f"wrote {out}_msd_gr.npz")


def self_test():
    """Synthetic trajectories with known answers: one vibrating, one diffusing.
    If the diagnostics cannot tell these apart they cannot be trusted on the
    real data either."""
    rng = np.random.default_rng(0)
    n_frames, n_at = 2000, 64
    cell = np.eye(3) * 16.2
    sites = rng.random((n_at, 3)) @ cell
    symbols = ["Al"] * 40 + ["Co"] * 14 + ["Ni"] * 10

    log("=== SELF TEST 1: vibrating solid (amplitude 0.12 A) ===")
    vib = sites[None, :, :] + rng.normal(0, 0.12, (n_frames, n_at, 3))
    report(vib, cell, symbols, sample_fs=4.0, out="selftest_vibrating")

    log("")
    log("=== SELF TEST 2: diffusing (random walk, step 0.05 A/frame) ===")
    walk = np.cumsum(rng.normal(0, 0.05, (n_frames, n_at, 3)), axis=0)
    dif = sites[None, :, :] + walk
    report(dif, cell, symbols, sample_fs=4.0, out="selftest_diffusing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions")
    ap.add_argument("--structure", default="Al9Co2Ni2-coords.txt")
    ap.add_argument("--supercell", default="2")
    ap.add_argument("--sample-fs", type=float, default=4.0)
    ap.add_argument("--rdf-frames", type=int, default=50,
                    help="frames used for g(r); the pair loop is O(N^2) per "
                         "frame so 50 is plenty and keeps this quick")
    ap.add_argument("--r-min", type=float, default=2.0,
                    help="ignore g(r) below this radius when locating the first "
                         "coordination shell; no Al-TM pair sits below ~2 A")
    ap.add_argument("--d-nn", type=float, default=None,
                    help="override the nearest-neighbour distance used for the "
                         "Lindemann ratio, if the automatic peak choice looks wrong")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.positions:
        raise SystemExit("--positions is required (or use --self-test)")

    base = parse_structure(args.structure)
    sc = make_supercell(base, parse_matrix(args.supercell))
    pos = np.load(args.positions)
    log(f"=== MSD and g(r): {args.positions} ===")
    if pos.shape[1] != len(sc):
        raise SystemExit(f"positions have {pos.shape[1]} atoms but the "
                         f"structure and supercell give {len(sc)} - mismatch")
    out = args.out or args.positions.replace("_positions.npy", "")
    report(pos, sc.get_cell().array, sc.get_chemical_symbols(),
           args.sample_fs, out, rdf_frames=args.rdf_frames,
           r_min=args.r_min, d_nn_override=args.d_nn)


if __name__ == "__main__":
    main()
