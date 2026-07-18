"""
VACF -> VDOS reduction
======================
Runs on the cluster, where the velocity data lives. Turns ~2 GB of velocities
into a few hundred KB of spectra, which you then bring back and plot locally.

Why FFT rather than an explicit loop
------------------------------------
The direct double loop over lags and time origins is O(n_lags x n_samples), and
for a 200 ps run on 1664 atoms that is roughly 350x the work of the 30 ps
208-atom run. Wiener-Khinchin gives the same autocorrelation via the power
spectrum in O(n log n): the ACF is the inverse transform of |FFT(v)|^2. The
series is zero-padded to at least 2N so the transform gives the linear
autocorrelation rather than the circular one.

The transform trap
------------------
The VDOS is the COSINE transform of the VACF, i.e. the real part. Taking |FFT|
folds the imaginary part in as a strictly positive contribution at every
frequency, which puts a floor across the whole spectrum. It inflates the
fractions for elements with little genuine high-frequency weight (Co, Ni) while
barely touching Al. Mirror the VACF to make it even, then take the real part.

Usage
-----
    python reduce_vdos.py --velocities md_4x4x4_velocities.npy --supercell 4
"""

import argparse

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.data import atomic_masses, atomic_numbers

THZ_TO_MEV = 4.13567
TYPE_TO_ELEMENT = {13: "Al", 27: "Co", 28: "Ni"}
_trapz = getattr(np, "trapezoid", None) or np.trapz


def parse_structure(path):
    lines = open(path).read().splitlines()
    tok = open(path).read().split()
    cell = np.array([float(x) for x in tok[:9]]).reshape(3, 3)
    sym, pos = [], []
    for ln in lines:
        p = ln.split()
        if len(p) >= 6 and p[3].isdigit() and p[4].isdigit():
            t = int(p[3])
            if t == 0:
                continue
            if t in TYPE_TO_ELEMENT:
                sym.append(TYPE_TO_ELEMENT[t])
                pos.append([float(p[0]), float(p[1]), float(p[2])])
    return Atoms(symbols=sym, scaled_positions=np.array(pos), cell=cell, pbc=True)


def vacf_fft(vel, masses, batch=64):
    """Mass-weighted VACF via Wiener-Khinchin, averaged over atoms and time
    origins. vel is (n_samples, n_atoms, 3). Atoms are processed in batches to
    keep the padded complex arrays a sensible size."""
    n_samples, n_atoms, _ = vel.shape
    n_fft = 1 << (2 * n_samples - 1).bit_length()      # >= 2N, power of two

    acc = np.zeros(n_samples)
    for start in range(0, n_atoms, batch):
        stop = min(start + batch, n_atoms)
        v = vel[:, start:stop, :].astype(np.float64)   # (n_samples, b, 3)
        f = np.fft.rfft(v, n=n_fft, axis=0)
        power = (f * np.conj(f)).real                  # |FFT|^2
        acf = np.fft.irfft(power, n=n_fft, axis=0)[:n_samples]
        acf = acf.sum(axis=2)                          # sum x,y,z -> (n_samples, b)
        # divide by the number of overlapping origins at each lag
        acf /= np.arange(n_samples, 0, -1)[:, None]
        acc += (acf * masses[start:stop][None, :]).sum(axis=1)

    acc /= acc[0]
    return acc


def vacf_to_vdos(vacf, dt):
    w = np.hanning(2 * len(vacf))[len(vacf):]          # half-Hann, decays at the tail
    vw = vacf * w
    full = np.concatenate([vw, vw[-1:0:-1]])           # mirror -> even function
    spec = np.maximum(np.real(np.fft.rfft(full)), 0)
    f = np.fft.rfftfreq(len(full), d=dt) / 1e12 * THZ_TO_MEV
    return f, spec


def normalise(f, s, fmax=60.0):
    m = (f >= 0) & (f <= fmax)
    fs, ss = f[m], s[m]
    return fs, ss / _trapz(ss, fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocities", required=True)
    ap.add_argument("--structure", default="Al9Co2Ni2-coords.txt")
    ap.add_argument("--supercell", type=int, required=True)
    ap.add_argument("--sample-fs", type=float, default=4.0)
    ap.add_argument("--chunks", type=int, default=8,
                    help="split the trajectory into blocks for an error bar, and "
                         "to check convergence with trajectory length")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or args.velocities.replace("_velocities.npy", "_vdos")

    vel = np.load(args.velocities, mmap_mode="r")      # do not pull 2 GB into RAM at once
    base = parse_structure(args.structure)
    sc = make_supercell(base, np.diag([args.supercell] * 3))
    syms = np.array(sc.get_chemical_symbols())
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in syms])

    if vel.shape[1] != len(syms):
        raise SystemExit(f"velocities have {vel.shape[1]} atoms, "
                         f"{args.supercell}x supercell has {len(syms)}")

    n_samples = vel.shape[0]
    dt = args.sample_fs * 1e-15
    print(f"{n_samples} samples, {len(syms)} atoms, "
          f"{n_samples * args.sample_fs / 1000:.0f} ps, {3*len(syms)} modes")

    # --- total ---
    vacf = vacf_fft(np.asarray(vel), masses)
    f, s = vacf_to_vdos(vacf, dt)
    fn, sn = normalise(f, s)
    print(f"Frequency resolution: {fn[1]-fn[0]:.4f} meV")
    print(f"VDOS at 60 meV: {sn[-1]:.5f}  (should be near zero)")
    print(f"|VACF| over last 25% of lags: "
          f"{np.abs(vacf[int(0.75*len(vacf)):]).mean():.4f}  (near zero = decayed)")
    np.save(f"{out}_total.npy", np.vstack([fn, sn]))
    np.save(f"{out}_vacf.npy", vacf)

    # --- element decomposition ---
    # Literature signature (Mihalkovic/Suck): the dominant low-energy peak is
    # from the transition metals, the high-energy band from Al.
    res = {"freq_meV": fn}
    print("\n--- element decomposition ---")
    for el in ("Al", "Co", "Ni"):
        idx = np.where(syms == el)[0]
        if len(idx) == 0:
            continue
        v_el = np.asarray(vel)[:, idx, :]
        fe, se = vacf_to_vdos(vacf_fft(v_el, masses[idx]), dt)
        fen, sen = normalise(fe, se)
        res[el] = sen
        above = _trapz(sen[fen > 30], fen[fen > 30]) * 100
        print(f"{el}: {above:.0f}% of weight above 30 meV ({len(idx)} atoms)")
    np.savez(f"{out}_partial.npz", **res)

    # --- convergence with trajectory length ---
    # Albert's point: "I need my blocks twice as long, then you know that you
    # have to run it twice as long." If the spectrum still moves when the block
    # length doubles, it is not converged.
    print("\n--- convergence with trajectory length ---")
    grid = np.linspace(0, 60, 600)
    lengths = {}
    for frac in (0.25, 0.5, 1.0):
        n = int(n_samples * frac)
        fv, sv = vacf_to_vdos(vacf_fft(np.asarray(vel[:n]), masses), dt)
        fvn, svn = normalise(fv, sv)
        lengths[f"{n*args.sample_fs/1000:.0f}ps"] = np.interp(grid, fvn, svn)
        print(f"  {n*args.sample_fs/1000:6.0f} ps done")
    keys = list(lengths)
    for a, b in zip(keys[:-1], keys[1:]):
        diff = _trapz(np.abs(lengths[b] - lengths[a]), grid) * 100
        print(f"  {a} -> {b}: spectra differ by {diff:.1f}% of total weight")
    np.savez(f"{out}_lengths.npz", freq_meV=grid, **lengths)

    # --- statistical uncertainty ---
    clen = n_samples // args.chunks
    specs = []
    for c in range(args.chunks):
        fc, sc_ = vacf_to_vdos(vacf_fft(np.asarray(vel[c*clen:(c+1)*clen]), masses), dt)
        fcn, scn = normalise(fc, sc_)
        specs.append(np.interp(grid, fcn, scn))
    specs = np.array(specs)
    mean, std = specs.mean(axis=0), specs.std(axis=0)
    np.save(f"{out}_uncertainty.npy", np.vstack([grid, mean, std]))
    print(f"\nStatistical uncertainty from {args.chunks} chunks "
          f"({clen*args.sample_fs/1000:.0f} ps each): "
          f"{(std/np.maximum(mean,1e-12)).mean()*100:.0f}% mean relative")

    print(f"\nWrote {out}_total.npy, {out}_partial.npz, {out}_lengths.npz, "
          f"{out}_uncertainty.npy, {out}_vacf.npy")
    print("Copy those back and plot locally - they are small.")


if __name__ == "__main__":
    main()
