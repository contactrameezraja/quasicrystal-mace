# O2 Progress Since Last Meeting

## Summary of actions from last week's call

| # | What Albert asked / we discussed | What I did | Result | Status |
|---|----------------------------------|------------|--------|--------|
| 1 | Re-plot VDOS showing the imaginary (negative) region, ~ -10 to 60 meV, to see if modes blend near zero or sit as distinct spikes | Extended x-axis (plotted out to -60 meV to be thorough) | Imaginary modes all sit between -4 and 0 meV and blend smoothly into zero. No distinct spikes, nothing far-negative. Consistent with the benign rigid-translation-noise picture you described | Done |
| 2 | Try a smaller finite-displacement (halve it), since finite differences get more accurate — but don't go so small it hits the residual force | Halved displacement 0.01 -> 0.005 A (residual force was 0.0008, so 0.005 is safe) | No change: still 19 imaginary modes, most negative -3.93 meV | Done |
| 3 | Try a bigger supercell as another check on the imaginary modes (suggested 4x4x4) | Ran 2x2x2 and 3x3x3. Started 4x4x4 (1664 atoms) but ETA was 8+ hours on CPU, so stopped it | 2x2x2 and 3x3x3 both give ~19 modes at ~ -3.9 meV, unchanged. Two supercell sizes + displacement check all agree | Done (2x2x2, 3x3x3); 4x4x4 not run (compute cost) |
| 4 | Repeat the harmonic VDOS for the other crystals so we can compare across the series | Completed the 60-atom Al2CoNi (harmonic VDOS + element decomposition) | Both 26-atom and 60-atom reproduce the Mihalkovic decomposition: Co and Ni confined below 30 meV, Al spans the full range | 2 of 3 done (26, 60); 265-atom W-phase pending |
| 5 | There is an experimental density of states we could compare against | Not started yet | - | Next step |

## Imaginary-mode convergence (26-atom Al9Co2Ni2)

| Supercell | Atoms | Displacement (A) | Imaginary modes | Most negative (meV) | Max freq (meV) |
|-----------|-------|------------------|-----------------|---------------------|----------------|
| 2x2x2 | 208 | 0.01 | 19 | -3.93 | 57.90 |
| 3x3x3 | 702 | 0.01 | 19 | -3.96 | 57.87 |
| 2x2x2 | 208 | 0.005 | 19 | -3.93 | 57.90 |

Stable under both supercell size and displacement -> not a finite-size or finite-difference artefact; a small, robust, benign feature that blends to zero.

## Element decomposition vs Mihalkovic (Co/Ni below 30 meV, Al across the range)

| Structure | Atoms | Al (% > 30 meV) | Co (% > 30 meV) | Ni (% > 30 meV) |
|-----------|-------|-----------------|-----------------|-----------------|
| Al9Co2Ni2 | 26 | 60% | 13% | 14% |
| Al2CoNi | 60 | 77% | 11% | 11% |

Both structures reproduce the published signature: transition metals (Co, Ni) confined to low frequency, Al spanning the whole spectrum.

## Main takeaway

MACE + Phonopy reproduces the Mihalkovic element-resolved VDOS across two structures independently (Co/Ni below 30 meV, Al everywhere). The low-frequency behaviour is clean, fixing the earlier Figure 3 problem. The small imaginary modes are stable across supercell size and displacement, so they are a minor benign feature rather than an artefact.

## What's left in O2

- 265-atom W-phase harmonic VDOS (large supercell, long run; ties to the M1 partial-occupancy supercell decision)
- Overlay the experimental density of states for quantitative validation
- The anharmonic side: MD (VACF) VDOS, with a large supercell, then overlay harmonic vs anharmonic

## Questions for you

1. 265-atom W-phase: worth the long harmonic run now, or focus on the MD? And what supercell size for the M1 partial occupancy?
2. For the MD, roughly what supercell size are you thinking (you flagged the mesh implies a large real-space cell)?
3. Should I prioritise digitising the experimental DOS for the two structures I have before moving to MD?