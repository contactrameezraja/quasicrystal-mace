# quasicrystal-mace

Simulation record for the MSc dissertation *Vibrations in Quasicrystals: A
machine-learned interatomic potential applied to decagonal Al-Co-Ni
approximants* (University of Warwick, 2026). The repository holds every
input structure, script, run log and reduced data file behind the numbers
and figures reported in the dissertation; the commit named in the
dissertation's Appendix A.4 is the citable state.

## What this is

Decagonal Al-Co-Ni approximants (the 26-atom X-phase and 265-atom W-phase),
a cubic 60-atom control and the periodic reference Al13Co4 are simulated
with the MACE-MPA-0 foundation potential (MatterSim and SevenNet for
comparison). Harmonic lattice dynamics (Phonopy) and 200 ps molecular
dynamics give the vibrational spectra; the analysis covers a finite-size
spectral floor law, participation ratios, a pre-registered ratio test for
golden-mean structure, neutron weighting, and fixed-site Monte Carlo
validation of the chemical ordering.

## Layout

Everything sits flat in the repository root.

- **Structures.** `Al9Co2Ni2-coords.txt`, `W-AlCoNi-265.vasp`,
  `W-AlCoNi-265-relaxed.vasp`, `al13co4.vasp`, `xbox_wcomp.vasp`, and the
  search outputs `mq_*_best.vasp`.
- **Pipeline scripts.** `md_run.py`, `reduce_vdos.py`, `gvdos.py`,
  `phonon_run.py`, `msd_gr.py`, `melt_quench.py`,
  `retarget_composition.py`, `verify_parsers.py`. Each is mapped to the
  equations it implements in the dissertation's Table 17.
- **Figure scripts.** `make_fig_floor.py`, `make_figs_colour.py`,
  `make_fig_sq.py`, `make_fig_stacking_centres.py`,
  `make_fig_pseudogap.py`, `make_fig_workflow.py`, `fig_structures.py`,
  `fig_onset_definition.py`, and the `O1_*`/`O2_*`/`O3_*` notebooks.
  Figure scripts regenerate the dissertation figures from the archived
  reductions alone; no trajectory is needed.
- **Reduced data.** `md_*_total.npy`, `md_*_partial.npz`,
  `md_*_lengths.npz`, `md_*_uncertainty.npy`, `md_*_msd_gr.npz`,
  `ph_*_dos.npz`, `ph_*_modes.npz`, `stacking_bandpath_gaps.txt`, and the
  `o1_*.json` relaxation records.
- **Run logs.** `md-*.out`, `mq-*.out`, `ph-*.out` and the progress files;
  box dimensions, drifts, residual forces and Monte Carlo stage energies
  quoted in the dissertation are read from these.
- **Raw data.** The ~24 GB of trajectories and force constants are archived
  off-repository under the sha256 manifest `archive_manifest.txt`.

## Reproducing a figure

```
pip install numpy matplotlib
python make_figs_colour.py     # Figures 5, 9, 11, 12, 14, 15
python make_fig_floor.py       # Figure 6
```

Each script prints the files it writes. The pipeline scripts additionally
need `mace`, `ase` and `phonopy` under the pinned environment named in
Appendix A.1.

## To come

- `crystal_lookup_app.py` — a small interactive lookup tool for the
  structures used here; documentation to follow.
- `dissertation_data.html` — a readable browser view of the key results
  tables; to be linked via GitHub Pages.

Both exist in the repository already and will be documented here
