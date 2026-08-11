"""
Harmonic lattice dynamics for Al-Ni-Co approximants
===================================================

What this adds over the MD route
--------------------------------
The MD/VACF spectrum gives spectral WEIGHT but not eigenvectors, so it cannot
say whether a mode is extended or localized. Diagonalising the dynamical matrix
gives both. That matters here for two reasons:

  Zone-boundary gap structure.  Earlier readings of a cross-approximant gap
  hierarchy were retracted: the reported sequence mixed gap EDGES for the two
  smaller cells with a gap CENTRE for the W-phase, and one of those cells turned
  out to be cubic rather than a decagonal approximant. Rebuilt on centres along
  the same direction the values are 9.68, 9.21 and 5.61 meV, so the two smaller
  cells are indistinguishable and no systematic ratio survives. Report gaps as a
  characterisation of each structure, not as a hierarchy across them.

  Participation ratio.  Mihalkovic, Elhor & Suck (PRB 63, 214301) report that in
  complex decagonal-related structures the low-energy modes localize sharply,
  with P falling to very low values already near 3-4 meV while Al3Ni stays
  extended to ~5 meV. P is computable from harmonic eigenvectors and is the
  literature-standard diagnostic, so it turns "the W-phase spreads weight into
  dense low-energy bands" into a statement about mode character.

Definition used (their Eq. 7 form):

    p_i(s) = sum_alpha |e_{i,alpha}(s)|^2        (site weight, sums to 1 over i)
    P(s)   = 1 / (N * sum_i p_i(s)^2)

with mass-weighted orthonormal eigenvectors e(s). P = 1 for a mode spread
evenly over all N atoms, P = 1/N for a mode on a single atom.

Everything is printed as text as well as saved, so the numbers can be read off
the log without opening the arrays.

Usage
-----
    # 265-atom W-phase on its conventional axes, gamma-point census + PR
    python phonon_run.py --structure W-AlCoNi-265.vasp \
        --supercell 1,1,0,-1,1,0,0,0,1 --out ph_w265

    # 26-atom X-phase reproduction of the earlier result (sanity check)
    python phonon_run.py --structure Al9Co2Ni2-coords.txt --fc-supercell 2,2,2 \
        --out ph_x26

    # force constants from a larger supercell, and a band path for gap centres
    python phonon_run.py --structure W-AlCoNi-265.vasp \
        --supercell 1,1,0,-1,1,0,0,0,1 --fc-supercell 2,1,1 --band --out ph_w265_fc211

    # dry run of the whole pipeline with a cheap classical calculator
    python phonon_run.py --structure Al9Co2Ni2-coords.txt --calculator emt --out ph_test
"""

import argparse
import time

import numpy as np
from ase import Atoms
from ase.io import read
from ase.build import make_supercell
from ase.optimize import BFGS

THZ_TO_MEV = 4.13567
TYPE_TO_ELEMENT = {13: "Al", 27: "Co", 28: "Ni"}


def log(msg, path="phonon_progress.txt"):
    print(msg, flush=True)
    with open(path, "a") as f:
        f.write(msg + "\n")


def parse_structure(path):
    """Mirrors md_run.py and reduce_vdos.py exactly, so all three routes start
    from the same atoms. The 26-atom sanity check applies only to the legacy
    coords.txt format, where we know what the file must contain."""
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
            f"Expected 26 atoms at 360.8 A^3 (Al18Co5Ni3). Check the file.")
    return atoms


def parse_matrix(spec):
    """'2' -> diag(2,2,2); '2,1,1' -> diag(2,1,1); nine numbers -> full matrix.
    Same convention as md_run.py so supercells are specified identically
    everywhere in the project."""
    parts = [int(x) for x in spec.split(",")]
    if len(parts) == 1:
        return np.diag(parts * 3)
    if len(parts) == 3:
        return np.diag(parts)
    if len(parts) == 9:
        return np.array(parts).reshape(3, 3)
    raise SystemExit(f"--supercell/--fc-supercell needs 1, 3 or 9 numbers, got {spec!r}")


def make_calc(name, device, dtype):
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium-mpa-0", device=device, default_dtype=dtype)
    if name == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device=device)
    if name == "emt":
        # For testing the pipeline only. EMT knows Al and Ni but not Co, so the
        # numbers are meaningless; it exists to prove the code path runs.
        from ase.calculators.emt import EMT
        return EMT()
    raise SystemExit(f"unknown --calculator {name!r}")


def participation_ratios(eigvecs, n_atoms):
    """eigvecs: (3N, n_modes) complex, columns are mass-weighted orthonormal
    eigenvectors as phonopy returns them. Returns P for each mode."""
    n_from_vec = eigvecs.shape[0] // 3
    if n_from_vec != n_atoms:
        raise SystemExit(f'eigenvector length implies {n_from_vec} atoms but '
                         f'{n_atoms} were passed: primitive-cell mismatch')
    e = eigvecs.reshape(n_atoms, 3, eigvecs.shape[-1])
    p_site = (np.abs(e) ** 2).sum(axis=1)            # (n_atoms, n_modes)
    p_site = p_site / p_site.sum(axis=0, keepdims=True)
    return 1.0 / (n_atoms * (p_site ** 2).sum(axis=0))


def find_gaps(freqs_mev, min_width=0.3, e_max=None):
    """Gaps in the mode spectrum: sort the frequencies and report intervals
    wider than min_width. Returns (lower, upper, centre, width) per gap."""
    f = np.sort(np.asarray(freqs_mev))
    f = f[f > 0.05]                                   # drop the acoustic zeros
    if e_max is not None:
        f = f[f <= e_max]
    gaps = []
    d = np.diff(f)
    for i in np.where(d > min_width)[0]:
        lo, hi = f[i], f[i + 1]
        gaps.append((lo, hi, 0.5 * (lo + hi), hi - lo))
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default="W-AlCoNi-265.vasp")
    ap.add_argument("--supercell", default=None,
                    help="transform the input cell before anything else, e.g. "
                         "'1,1,0,-1,1,0,0,0,1' to put the W-phase on its "
                         "conventional axes. Same convention as md_run.py.")
    ap.add_argument("--fc-supercell", default="1,1,1",
                    help="supercell used for the finite-displacement force "
                         "constants. 1,1,1 gives a gamma-point census of the "
                         "cell itself (cheap, enough for the DOS and the "
                         "participation ratio); 2,1,1 or larger is needed for "
                         "a meaningful band structure.")
    ap.add_argument("--displacement", type=float, default=0.01,
                    help="finite displacement in A. 0.005 was the halved "
                         "control in the 26-atom imaginary-mode check.")
    ap.add_argument("--mesh", default="1,1,1",
                    help="q-mesh for DOS and participation ratio")
    ap.add_argument("--band", action="store_true",
                    help="also compute band paths and report zone-boundary "
                         "gap centres. Four paths: the two oblique primitive "
                         "directions, the in-plane axis, and the true stacking "
                         "direction at (-0.5, 0.5, 0). Needs --fc-supercell "
                         "2,2,2 for the last of these to be commensurate.")
    ap.add_argument("--no-relax", action="store_true",
                    help="skip the relaxation (use if the input is already "
                         "relaxed to fmax 0.001)")
    ap.add_argument("--fmax", type=float, default=0.001)
    ap.add_argument("--calculator", default="mace",
                    choices=["mace", "mattersim", "emt"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64",
                    help="float64 throughout: force constants are second "
                         "derivatives and precision matters more here than in MD")
    ap.add_argument("--split", type=float, default=30.0,
                    help="energy in meV above which the element decomposition "
                         "is integrated; 30 matches Mihalkovic's TM/Al divide")
    ap.add_argument("--out", default="ph_run")
    args = ap.parse_args()

    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    t_start = time.time()
    atoms = parse_structure(args.structure)
    log(f"=== harmonic run: {args.out} ===")
    log(f"read {args.structure}: {len(atoms)} atoms, "
        f"volume {atoms.get_volume():.1f} A^3")

    if args.supercell:
        atoms = make_supercell(atoms, parse_matrix(args.supercell))
        log(f"transformed by {args.supercell}: {len(atoms)} atoms, "
            f"cellpar {np.round(atoms.cell.cellpar(), 3)}")

    calc = make_calc(args.calculator, args.device, args.dtype)
    atoms.calc = calc
    log(f"calculator: {args.calculator}")

    if not args.no_relax:
        log(f"relaxing positions at fixed cell to fmax {args.fmax}...")
        BFGS(atoms, logfile=None).run(fmax=args.fmax, steps=500)
        fmax_final = np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max()
        log(f"  E = {atoms.get_potential_energy()/len(atoms):.6f} eV/atom, "
            f"max residual force {fmax_final:.5f} eV/A")
    else:
        log("relaxation skipped (--no-relax)")

    # ---------------- force constants ----------------
    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                        cell=atoms.get_cell().array,
                        scaled_positions=atoms.get_scaled_positions())
    fc_sc = parse_matrix(args.fc_supercell)
    # phonopy 4.x defaults primitive_matrix='auto', which would silently reduce
    # the input cell (it turned a 4-atom fcc cell into a 1-atom primitive in
    # testing). These structures must be treated exactly as given.
    ph = Phonopy(unit, supercell_matrix=fc_sc, primitive_matrix=np.eye(3))
    ph.generate_displacements(distance=args.displacement)
    disp_supercells = ph.supercells_with_displacements
    n_disp = len(disp_supercells)
    log(f"force constants: fc-supercell {args.fc_supercell} "
        f"({len(ph.supercell)} atoms), displacement {args.displacement} A, "
        f"{n_disp} displaced configurations")

    forces = []
    t0 = time.time()
    for i, sc in enumerate(disp_supercells):
        a = Atoms(symbols=sc.symbols, cell=sc.cell,
                  scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
        if (i + 1) % max(1, n_disp // 20) == 0 or i == n_disp - 1:
            el = time.time() - t0
            log(f"  {i+1}/{n_disp} | {el/60:.1f} min | "
                f"{el/(i+1)*(n_disp-1-i)/60:.1f} min left")
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    log("force constants done")

    # ---------------- mode census at the mesh ----------------
    mesh = [int(x) for x in args.mesh.split(",")]
    is_diagonal = np.count_nonzero(fc_sc - np.diag(np.diag(fc_sc))) == 0
    fc_diag = np.abs(np.diag(fc_sc))
    if not is_diagonal:
        log(f"  NOTE: the force-constant supercell is non-diagonal, so the "
            f"aliasing check below compares against its diagonal only and is "
            f"approximate. det = {int(round(abs(np.linalg.det(fc_sc))))}; keep "
            f"--mesh at 1,1,1 unless you have checked commensurability by hand.")
    if any(m > d for m, d in zip(mesh, fc_diag)):
        log(f"  WARNING: mesh {mesh} is finer than the force-constant supercell "
            f"{fc_diag.tolist()}. Force constants only support q-points "
            f"commensurate with that supercell, so finer sampling aliases. "
            f"Use --mesh no larger than the fc-supercell, or enlarge --fc-supercell.")
    ph.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    md = ph.get_mesh_dict()
    freqs = md["frequencies"] * THZ_TO_MEV          # (nq, nband)
    eigvecs = md["eigenvectors"]                    # (nq, 3N, nband)
    n_at = len(ph.primitive)
    log(f"mesh {mesh}: {freqs.shape[0]} q-points x {freqs.shape[1]} bands")

    flat = freqs.ravel()
    n_imag = int((flat < -0.01).sum())
    log("--- mode census ---")
    log(f"  modes total          : {flat.size}")
    log(f"  imaginary (< -0.01)  : {n_imag}")
    if n_imag:
        log(f"  most negative        : {flat.min():.3f} meV")
        log(f"  all imaginary within : {flat[flat < -0.01].min():.3f} to "
            f"{flat[flat < -0.01].max():.3f} meV")
    log(f"  max frequency        : {flat.max():.3f} meV")
    pos = np.sort(flat[flat > 0.05])
    log(f"  lowest 10 real modes : {np.round(pos[:10], 3).tolist()}")

    # ---------------- participation ratio ----------------
    log("--- participation ratio (Mihalkovic Eq. 7 form) ---")
    all_f, all_p = [], []
    for iq in range(freqs.shape[0]):
        p = participation_ratios(eigvecs[iq], n_at)
        all_f.append(freqs[iq])
        all_p.append(p)
    all_f = np.concatenate(all_f)
    all_p = np.concatenate(all_p)
    keep = all_f > 0.05
    fk, pk = all_f[keep], all_p[keep]
    log(f"  P for a fully extended mode = 1.0, for a single-atom mode = "
        f"{1.0/n_at:.5f}")
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 30),
                   (30, 40), (40, 60)]:
        m = (fk >= lo) & (fk < hi)
        if m.sum():
            log(f"  {lo:2d}-{hi:2d} meV | {m.sum():5d} modes | "
                f"mean P {pk[m].mean():.4f} | min P {pk[m].min():.4f}")
    order = np.argsort(fk)
    log("  ten lowest modes, energy and P:")
    for i in order[:10]:
        log(f"    {fk[i]:7.3f} meV   P = {pk[i]:.4f}")

    # ---------------- gaps ----------------
    log("--- gaps in the mode spectrum (width > 0.3 meV, below 20 meV) ---")
    if list(np.abs(np.diag(fc_sc))) == [1, 1, 1] or freqs.shape[0] == 1:
        log("  NOTE: these are SPACINGS in the discrete mode list at the sampled")
        log("  q-point(s), which is NOT the same observable as the zone-boundary")
        log("  pseudo-gaps obtained from a band path. To compare with those, run")
        log("  with --band and --fc-supercell 2,1,1 or larger, which costs a")
        log("  supercell of that many times the atoms in force evaluations.")
    if n_imag:
        log(f"  ({n_imag} imaginary modes are excluded from the gap analysis; "
            f"see the census above before reading these gaps as clean)")
    gaps = find_gaps(fk, min_width=0.3, e_max=20.0)
    if not gaps:
        log("  none found")
    for lo, hi, ctr, w in gaps[:15]:
        log(f"  {lo:7.3f} - {hi:7.3f} meV | centre {ctr:7.3f} | width {w:.3f}")
    if len(gaps) >= 2:
        centres = [g[2] for g in gaps]
        ratios = [float(round(b / a, 3)) for a, b in zip(centres, centres[1:])]
        log(f"  successive gap-centre ratios: {ratios}")
        log(f"  (tau = 1.618. Do not read these against the retracted "
            f"cross-cell sequence; see the docstring)")

    # ---------------- element decomposition, from the eigenvectors ----------
    # Computed mode by mode rather than from phonopy's DOS routine, which needs
    # a proper q-mesh and returns nothing usable from a single sampled point.
    # For mass-weighted orthonormal eigenvectors the site weights sum to 1 per
    # mode, so the fraction of weight above the split energy carried by an
    # element is just its mean site weight over the modes above that energy.
    # This is the same quantity reduce_vdos.py reports from the MD spectra.
    syms = np.array(ph.primitive.symbols)
    log(f"--- element decomposition from eigenvectors (above {args.split} meV) ---")
    w_above = {el: 0.0 for el in sorted(set(syms))}   # weight above the split
    w_own = {el: 0.0 for el in sorted(set(syms))}     # same, for convention (b)
    w_tot = {el: 0.0 for el in sorted(set(syms))}     # this element's total
    n_above = 0
    for iq in range(freqs.shape[0]):
        e = eigvecs[iq].reshape(n_at, 3, eigvecs[iq].shape[-1])
        site = (np.abs(e) ** 2).sum(axis=1)                 # (n_atoms, n_modes)
        site = site / site.sum(axis=0, keepdims=True)
        sel = freqs[iq] > args.split
        real = freqs[iq] > 0.05
        n_above += int(sel.sum())
        for el in w_above:
            m = syms == el
            w_above[el] += site[m][:, sel].sum()
            w_own[el] += site[m][:, sel & real].sum()
            w_tot[el] += site[m][:, real].sum()
    if n_above:
        # TWO different quantities, printed together because confusing them is
        # easy and costly. (a) SHARE: of the weight above the split energy, what
        # fraction belongs to this element - these sum to 100% across elements.
        # (b) PER-ELEMENT FRACTION: of this element's OWN spectrum, what fraction
        # lies above the split - these do NOT sum to 100%, and this is the
        # quantity reduce_vdos.py reports, because it normalises each element's
        # partial spectrum to unit area before integrating.
        log("  (a) share of the weight above the split, sums to 100%:")
        for el in sorted(w_above):
            log(f"      {el}: {100*w_above[el]/n_above:.0f}%  "
                f"({int((syms == el).sum())} atoms, "
                f"{100*(syms == el).sum()/n_at:.0f}% of atoms)")
        log("  (b) fraction of each element's OWN spectrum above the split, "
            "which is what reduce_vdos.py prints:")
        for el in sorted(w_own):
            log(f"      {el}: {100*w_own[el]/max(w_tot[el], 1e-300):.0f}%")
        log(f"  ({n_above} of {freqs.size} modes lie above {args.split:.0f} meV)")
        log("  MD comparison, convention (b): Al 61 / Co 16 / Ni 15 for the "
            "X-phase at three sizes, Al 54 / Co 19 / Ni 16 for the W-phase, "
            "Al 54 / Co 16 for Al13Co4")
    else:
        log(f"  no modes above {args.split} meV (max {flat.max():.1f} meV)")

    # ---------------- DOS (secondary; needs a real mesh to be meaningful) ----
    ph.run_total_dos()
    tdos = ph.get_total_dos_dict()
    e_dos = tdos["frequency_points"] * THZ_TO_MEV
    g_dos = tdos["total_dos"]
    ph.run_projected_dos()
    pdos = ph.get_projected_dos_dict()["projected_dos"]
    log("--- DOS-based decomposition (secondary; needs a real mesh) ---")
    above = e_dos > args.split
    tot_above = np.trapezoid(g_dos[above], e_dos[above]) if hasattr(np, "trapezoid") \
        else np.trapz(g_dos[above], e_dos[above])
    if not np.isfinite(tot_above) or tot_above <= 0:
        log(f"  no spectral weight above {args.split} meV, decomposition skipped "
            f"(max frequency {flat.max():.1f} meV)")
        tot_above = np.nan
    part = {}
    for el in sorted(set(syms)):
        idx = np.where(syms == el)[0]
        gp = pdos[idx].sum(axis=0)
        a = np.trapezoid(gp[above], e_dos[above]) if hasattr(np, "trapezoid") \
            else np.trapz(gp[above], e_dos[above])
        part[el] = a
        log(f"  {el}: {100*a/tot_above:.0f}% of weight above {args.split:.0f} meV "
            f"({len(idx)} atoms)")
    log("  (MD gave Al 61 / Co 16 / Ni 15 for the X-phase at three sizes, "
        "and Al 54 / Co 19 / Ni 16 for the W-phase)")

    # ---------------- band structure ----------------
    if args.band:
        log("--- band path (gap centres at the zone boundary) ---")
        # Explicit paths. For the C-centred W-phase cell NO primitive axis
        # lies along the stacking direction: a and b are oblique, each mixing
        # the in-plane and stacking directions, and c is the 23.23 A in-plane
        # axis. Because a itself carries a stacking component of 4.051 A, the
        # lattice repeats every 4.051 A projected onto the stacking direction,
        # so the smallest purely-stacking reciprocal vector is (-1, 1, 0) and
        # the stacking zone boundary sits at half of it, (-0.5, 0.5, 0).
        # The fourth path is commensurate only with force constants carrying a
        # factor of two along BOTH a and b, so use --fc-supercell 2,2,2 or
        # larger for it to mean anything.
        paths = [[[0, 0, 0], [0.5, 0, 0]],
                 [[0, 0, 0], [0, 0.5, 0]],
                 [[0, 0, 0], [0, 0, 0.5]],
                 [[0, 0, 0], [-0.5, 0.5, 0]]]
        labels = ["a* (oblique)", "b* (oblique)",
                  "c* (in-plane, 23.23 A)", "stacking (4.051 A)"]
        ph.run_band_structure(paths, with_eigenvectors=False)
        bs = ph.get_band_structure_dict()
        for name, fq in zip(labels, bs["frequencies"]):
            zb = fq[-1] * THZ_TO_MEV                 # zone-boundary frequencies
            g = find_gaps(zb, min_width=0.3, e_max=20.0)
            log(f"  {name}: {len(g)} gaps below 20 meV")
            for lo, hi, ctr, w in g[:6]:
                log(f"      {lo:7.3f} - {hi:7.3f} | centre {ctr:7.3f} | "
                    f"width {w:.3f}")

    # ---------------- save ----------------
    np.savez(f"{args.out}_modes.npz", frequencies=freqs, participation=np.array(all_p),
             symbols=syms, n_atoms=n_at)
    np.savez(f"{args.out}_dos.npz", energy=e_dos, total=g_dos,
             projected=pdos, symbols=syms)
    ph.save(f"{args.out}_phonopy.yaml", settings={"force_constants": True})
    log(f"wrote {args.out}_modes.npz, {args.out}_dos.npz, {args.out}_phonopy.yaml")
    log(f"total wall time {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
