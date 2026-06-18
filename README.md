# Vibrations in Quasicrystals using MACE

MSc research project (Warwick ES98C). Studying golden-ratio vibrational
features in decagonal Al-Ni-Co quasicrystal approximants using machine-learning
interatomic potentials.

## Objective 1 — MACE stability and model choice

Confirming that foundation MLIPs produce stable, physically reasonable
structures for the W-AlCoNi approximant, and comparing models to settle which
to carry forward.

### Files

- `O1_Structure.ipynb` — working notebook: parses the CMU structure, relaxes it,
  runs an MD stability test, and compares models.
- `W-AlCoNi-coords` — atomic coordinates for the 265-atom W-AlCoNi (mC534)
  approximant from the CMU alloy database.
- `Structurefile.txt` — the CMU structure file (shows the partial-occupancy M1 site).
- `W-AlCoNi-265.vasp` / `W-AlCoNi-265-relaxed.vasp` — parsed and relaxed structures.
- `o1_results.json` — relaxation and stability results per model.
- `o1_model_comparison.png` — comparison figure.

### Structure parsing

The CMU database provides no CIF/POSCAR, so the notebook parses the coordinates
file directly: reads the three cell vectors, drops two type-0 vacancy
placeholders (267 entries to 265 atoms), and maps type codes to elements
(13 Al, 27 Co, 28 Ni). Verified three ways: atom count (265), composition
(Al 190, Co 55, Ni 20), and cell volume (3741.8 A^3), all matching the database.

### Workflow

Per model: single-stage relaxation (positions and cell together) to a small
residual force, then a short MD stability test (NVT equilibrate, remove drift,
NVE burn-in) at 300 K.

### Models compared

- MACE-MPA-0 (primary)
- MatterSim-v1 (different architecture, cross-check)

Both relax the W-phase to a stable, intact structure with consistent energetics;
they differ slightly in relaxed geometry, which feeds the later uncertainty
analysis.

## Status

O1 in progress. Size series across additional structures, and the final model
choice, to follow.
