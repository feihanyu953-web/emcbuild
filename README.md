# emcbuild

**Template-driven builder for DFT/MD interface models.** ASE builds the surface
and places adsorbates, EMC packs water/ions/organic solvents to a target
density, and pymatgen handles lattice matching for heterojunctions — one JSON
config in, a complete VASP + LAMMPS model set out.

## What it builds

| Structure type | Description |
|---|---|
| `slab` | FCC/BCC metal slabs, any Miller index (element / miller / size / layers / vacuum) |
| `heterojunction` | Film + substrate with automatic lattice matching (pymatgen `CoherentInterfaceBuilder`) |
| `nanotube` | Carbon nanotubes (n, m, length) |
| *electrolyte box* | Standalone solvent/ion boxes with no surface — also supported |

Adsorbates are placed **between the metal surface and the solvent**, so water
layers pack outward from the surface and the adsorbate stays embedded in the
electrolyte — the physically correct ordering for electrochemistry models.

## Electrolyte packing (EMC)

- `water_count` + explicit ion/residue counts + target `density` / `temperature`
  — explicit counts avoid EMC rounding, mass fractions are computed for you
- Solvents by CHARMM/OPLS residue name, or arbitrary molecules via
  `CUSTOM:SMILES`
- Water molecules overlapping the adsorbate are deleted automatically; the
  lateral box is `max(surface cell, natural water box)` so solvent never
  overflows

## Force fields

| FF | Ions | Organic solvents | Water | Notes |
|---|---|---|---|---|
| **CHARMM** (default) | 15 species — Na⁺ K⁺ Cl⁻ OH⁻ Li⁺ Cs⁺ Mg²⁺ Ca²⁺ Ba²⁺ Rb⁺ Zn²⁺ Cd²⁺ + PO₄³⁻ HPO₄²⁻ H₂PO₄⁻ | **91 residues** (ETOH, MEOH, THF, DMSO, NMA, … + 84 carbohydrates: aldoses, ketoses, furanoses, sugar alcohols, inositols) | TIP3 | Dual-field loading (`water_ions` first, then `carb`) so wildcard rules can't clobber ion charges |
| **OPLS-AA** (2024) | 10 monatomic ions incl. full halides F⁻/Br⁻/I⁻ + Sr²⁺, plus OH⁻ (HYDR) | 105 residues (MEOH, DMSO, FORM, UREA, PHEN, …) | SPC/E | DMSO via patched PRM; no polyatomic anions (SO₄²⁻/NO₃⁻/CO₃²⁻ → use CHARMM) |
| **PCFF** | Na⁺ K⁺ Cl⁻ OH⁻ only | Ethanol only | — | Rule-matched; fragile for SMILES — prefer CHARMM when ions are present |

## Adsorbate placement (5 strategies)

1. ASE built-in molecule library
2. SMILES → ASE
3. EMC + PCFF builder for complex organics (>5 heavy atoms, many rotatable
   bonds, fluoroethers, ionic liquids; falls back to RDKit when parameters are
   missing)
4. RDKit ETKDGv3 + MMFF94 — 50 conformers, lowest energy kept
5. `Auto` — tries SMILES → RDKit → ASE in order

**Structure sources (7):** ASE built-in metals · Materials Project (`mp-xxx`,
API key) · COD (organic crystals) · PubChem (molecules) · procedural
`fullerene_c60` · manual `element:type:a` (e.g. `"Pt:fcc:3.92"`) · local file
(POSCAR / CIF / XYZ).

## Physical-correctness guards

- **Density rescale fix** — repairs EMC's `lzz` compression bug (observed 2.48×
  densification) and re-verifies the packed box (`_verify_box_density()` warns
  loudly beyond 10% deviation). Covered by a regression test that runs without
  the EMC binary: `python tests/test_density_rescale.py`
- Gap 1.5 Å / min clearance 1.5 Å between regions; 10 Å bottom / 15 Å top vacuum
- CHARMM `water_ions` loads before `carb` to protect ion charges; an auto-water
  guard adds one H₂O when solvents/ions are requested with `water_count = 0`

## Output files

| File | Purpose |
|---|---|
| `POSCAR` | VASP input, element-grouped |
| `structure.xyz` / `structure.cif` | viewer / archive formats |
| `structure.data` | LAMMPS data file **with bonds** |
| `model_report.txt` | formula, element counts, minimum distances, warnings |
| `water.data` | raw EMC output with CHARMM types kept (use to verify OH⁻ handling) |

## Installation

One-time (4 steps, handled by `setup.py`):

```powershell
python setup.py
# 1. detects OS
# 2. unpacks EMC v9.4.4 from emc_packages/ into ~/.emcbuild/emc
# 3. pip installs ase / numpy / pymatgen / mp-api   (into your conda env)
# 4. prompts for a Materials Project API key -> ~/.emcbuild/config.json
```

Requirements: conda environment with ASE / numpy / pymatgen / mp-api (example
env name below: `DFTbuild`); optional `rdkit`. EMC binaries for Windows/macOS
are bundled in `emc_packages/`; **Linux is not fully supported** — install EMC
manually from [montecarlo.sourceforge.net](https://montecarlo.sourceforge.net)
and point `EMC_ROOT` at it.

## Usage

```powershell
conda run -n DFTbuild python scripts\build.py examples\pt100_interface.json
```

`examples/` ships **26 ready-to-run configs**: pure solvent boxes (charmm /
opls / pcff), water + ions, water + ethanol, Pt(100) interfaces in every force
field, CeO₂/Pt and TiO₂/Pt heterojunctions with water or mixed solvents,
FeN-doped CNT, CNT interface, twisted graphene, Pt(100) + CO₂ + ethanol, …

## Known limitations

- OPLS does not support custom polyatomic anions (SO₄²⁻/NO₃⁻/CO₃²⁻) — use CHARMM
- PCFF ion support is weak; prefer CHARMM for ion-containing systems
- EMC templates must stay ISO-8859-1 encoded
- OH⁻ is merged with water O/H in `POSCAR` (verify via `water.data`)
- 4-layer Pt(100) 3×3 is the recommended minimal surface for electrolyte models

## Ecosystem

| Stage | Tool |
|---|---|
| **Build starting structures** | **this repo** |
| Generate & validate inputs | [input_build](https://github.com/feihanyu953-web/input_build) |
| Submit / transfer on HPC | [cluster-connect](https://github.com/feihanyu953-web/cluster-connect) |
| Remote VASP post-processing | [vaspkit-remote](https://github.com/feihanyu953-web/vaspkit-remote) |
| Local MD trajectory analysis | [lammps-analyze](https://github.com/feihanyu953-web/lammps-analyze) |

## License

MIT — see [LICENSE](LICENSE).
