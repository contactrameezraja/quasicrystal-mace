# Vibrations in Quasicrystals using MACE

MSc research project (Warwick ES98C). Studying golden-ratio vibrational features in decagonal Al-Ni-Co quasicrystal approximants using machine-learning interatomic potentials.

## Objective 1 — MACE stability and model choice

Confirming that foundation MLIPs produce stable, physically reasonable structures for Al-Ni-Co approximants across a size range, and comparing three models to settle which to carry forward into the vibrational work.

## Files

* `O1_Structure.ipynb` — main notebook: parses the CMU structures, relaxes them, runs MD stability tests, investigates the W-phase displacement, demonstrates NVE energy conservation, and justifies the collapse threshold (MACE).
* `O1_SevenNet.ipynb` — SevenNet-0 run across the three structures (separate environment).
* `O1_MatterSim.ipynb` — MatterSim-v1 run across the three structures (separate environment).
* `W-AlCoNi-coords`, `Al2CoNi-coords.txt`, `Al9Co2Ni2-coords.txt` — atomic coordinates for the 265, 60 and 26-atom structures from the CMU alloy database.
* `Structurefile.txt` — the CMU structure file (shows the partial-occupancy M1 site).
* `W-AlCoNi-265.vasp` / `W-AlCoNi-265-relaxed.vasp` — parsed and relaxed structures.
* `o1_final_table.json` — the final three-model, three-structure comparison table.
* `o1_table_v2.json`, `o1_sevennet_results.json`, `o1_mattersim_results.json` — per-model results.
* `o1_summary.png` — stability across the size series plus model comparison.
* `o1_energy_conservation.png` — NVE energy-conservation check.

## Structure parsing

The CMU database provides no CIF/POSCAR, so the notebook parses the coordinates files directly: reads the cell vectors, drops type-0 vacancy placeholders where present, and maps type codes to elements (13 Al, 27 Co, 28 Ni). The W-phase has a different file format from the smaller structures (cell vectors on one line), handled by a separate parser. Each structure is verified three ways against the database: atom count, composition, and cell volume.

## Workflow

Per model and structure: single-stage relaxation (positions and cell together) to a small residual force, then a short MD stability test (NVT equilibrate, remove drift, NVE burn-in) at 300 K. Recorded diagnostics are chosen to convey meaning: energy change on relaxation (not absolute energy), initial maximum force (not the convergence threshold), volume change, and maximum atomic displacement.

## Size series

Three structures spanning 26 to 265 atoms: Al9Co2Ni2 (26), Al2CoNi (60), W-AlCoNi (265). This establishes stability across the chemistry and size range.

## Models compared

Three architectures, chosen to span the phonon-relevant kappa_SRME metric on the Matbench Discovery leaderboard:

* MACE-MPA-0 (equivariant, primary, kappa_SRME 0.412)
* SevenNet-0 (NequIP-based)
* MatterSim-v1 (kappa_SRME 0.575, cross-check)

Each runs in its own conda environment to avoid dependency conflicts. All three relax every structure to a stable, intact configuration with consistent energetics and small relaxation energy changes. All three independently produce the largest displacement on the W-phase, at the partial-occupancy site, which confirms it is a real structural feature rather than a model artefact.

## Key findings

* MACE-MPA-0 is stable across the full size range and is carried into O2.
* The W-phase max displacement is the aluminium atom adjacent to a dropped partial-occupancy placeholder relaxing into the freed space, confirmed by all three models.
* NVE energy conservation is good (0.004 meV/atom drift), confirming well-behaved dynamics and an appropriate timestep.
* The 1.8 A collapse threshold is justified by a numerical experiment showing the energy penalty climbs steeply below it.

* ## Crystal Lookup and Verify (tool)

A small web app (`crystal_lookup_app.py`) that looks up structures on the
Materials Project and runs the same O1-style MLIP stability check on any of them.

Setup:

    conda activate mattersim-env      # needs mp-api, mattersim, ase, flask
    pip install flask
    export MP_API_KEY="your_materials_project_key"
    python crystal_lookup_app.py

Then open http://127.0.0.1:5000. Type a chemical system (Al-Co-Ni), formula
(Al2CoNi), or mp-id (mp-1229050), and click Verify on any result to relax it
and see the stability diagnostics. Get a free key at
https://next-gen.materialsproject.org. The key is read from the MP_API_KEY
environment variable, so it is not stored in the code.
