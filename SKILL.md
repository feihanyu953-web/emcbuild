---
name: emcbuild
description: "Use when building DFT starting structures: metal slabs, heterojunctions, carbon nanotubes, adsorbates, and optionally electrolyte layers (water/ions/organic solvents) using EMC + ASE. Also use for standalone electrolyte box construction without a surface. Supports CHARMM dual-field loading for water+ions+organic-solvent mixtures."
---

# EMCbuild

## Overview

Template-driven DFT interface model builder — ASE builds surfaces and places adsorbates; EMC packs water, ions, and organic solvents at target density; pymatgen handles heterojunction lattice matching (CoherentInterfaceBuilder). For CHARMM, uses dual-field loading (water_ions + carb).

## Prerequisites

1. **EMC v9.4.4 installed** at `E:/emcwin32/v9.4.4` (set `EMC_ROOT` env var to override)
2. **DFTbuild conda environment** with ASE, numpy, pymatgen, mp-api
3. Set `MP_API_KEY` env var or `configs/databases.json` for Materials Project

## Skill Structure

```
emcbuild/
├── SKILL.md
├── scripts/
│   └── build.py                  # Main build script
├── templates/
│   ├── build_water_pcff.emc      # PCFF: water + ethanol (ion support limited)
│   ├── build_water_charmm.emc    # CHARMM: dynamic placeholder template
│   └── build_water_opls.emc      # OPLS: water + ions + organic
├── fields/
│   ├── pcff/                     # PCFF force field (bundled)
│   ├── charmm/                   # water_ions + carb (from EMC install, bundled)
│   ├── charmm_custom/            # Clean water_ions base for runtime patching
│   └── opls/                     # OPLS-AA force field (bundled)
├── examples/
└── configs/
```

## Quick Workflow

```powershell
# <skill-dir> = 本 skill 安装目录(全局 ~/.claude/skills/emcbuild/ 或项目 .claude/skills/emcbuild/)
cd '<skill-dir>'
conda run -n DFTbuild python scripts\build.py examples\pt100_interface.json
```

## Config Format

### Basic: water + ions

```json
{
  "surface": {
    "type": "slab",
    "element": "Pt",
    "miller": [1, 0, 0],
    "size": [3, 3],
    "layers": 4,
    "vacuum": 15.0
  },
  "adsorbate": {
    "list": [
      {"name": "NH3", "site": "ontop", "height": 2.0}
    ]
  },
  "electrolyte": {
    "water_count": 30,
    "ions": {"Na+": 1, "OH-": 1},
    "density": 1.0,
    "temperature": 300,
    "field": "charmm",
    "gap": 1.5,
    "min_clearance": 1.5
  },
  "output_dir": "outputs/pt100_charmm"
}
```

### Complex: water + organic solvents + ions

```json
{
  "electrolyte": {
    "water_count": 100,
    "solvents": {"ETOH": 20, "MEOH": 10},
    "ions": {"Na+": 2, "Cl-": 2},
    "density": 1.0,
    "temperature": 300,
    "field": "charmm"
  }
}
```

**`solvents` key format**: `{"RESIDUE_NAME": count}` where residue names are from the CHARMM carb force field. For backward compatibility, `ethanol_count` is still accepted and maps to `solvents: {"ETOH": N}`.

### Output Directory

**Default behavior**: If `output_dir` is not specified in the config, output goes to `EMCbuild/` folder in the same directory as the config file (user's project directory).

```json
// Config at: /my_project/models/pt100.json
{
  "surface": {...},
  "electrolyte": {...}
  // No output_dir specified
}
// Output: /my_project/models/EMCbuild/
```

**Custom output**: Specify `output_dir` explicitly (relative or absolute path):

```json
{
  "output_dir": "my_custom_output",  // Relative to current working directory
  ...
}
```

**Example output**: The `examples/` configs use `examples/outputs/xxx` for demonstration purposes.

### Custom SMILES for solvents

For solvents not in the built-in SMILES map, use the `name:smiles` format:

```json
"solvents": {"ETOH": 5, "CUSTOM:CCCCCOH": 3}
```

## Supported Structures

| type | surface config | 说明 |
|---|---|---|
| `slab` | `element`, `miller`, `size`, `layers` | FCC/BCC metal slabs, any miller index |
| `heterojunction` | `film` + `substrate` (pymatgen) | 通用异质结，自动晶格匹配 |
| `nanotube` | `n`, `m`, `length` | Carbon nanotubes |

## Force Fields

| Field | Template | Ions | Organic Solvents | Notes |
|---|---|---|---|---|
| `charmm` | `build_water_charmm.emc` | 12 monatomic + 3 phosphate | ✅ 91 residues | Dual-field: water_ions + carb |
| `pcff` | `build_water_pcff.emc` | Na+, K+, Cl-, OH- | ⚠️ Ethanol only | Ion support limited (rule-based matching) |
| `opls` | `build_water_opls.emc` | 10 monatomic (含全卤素) + OH⁻ | ✅ 105 residues | OPLS-AA 2024, 全卤素F⁻/Br⁻/I⁻ |

### CHARMM Dual-Field Loading

For `field: "charmm"`, EMC loads **two force field files sequentially**:

```emc
field = { mode -> charmm, name -> {water_ions.prm, water_ions.top} };
field = { mode -> charmm, name -> {carb.prm, carb.top} };
```

- `water_ions` provides: TIP3 water + **15 ions**: 12 monatomic (Na+/K+/Cl-/OH-/Li+/Cs+/Mg2+/Ca2+/Ba2+/Rb+/Zn2+/Cd2+) + 3 phosphate (PO4³⁻/HPO4²⁻/H₂PO₄⁻)
- `carb` provides: ETOH, MEOH, THF, EGLY, PRO2, CYHX, THP, and all other carbohydrate residue types

**Important**: water_ions must be loaded *first* so its ion charges are not overridden by carb's wildcard rule.

**Auto-water guard**: if `solvents` + `ions` are specified but `water_count=0`, the system automatically adds 1 H2O to ensure correct dual-field loading order. Pure organic solvent + ions without water is physically unrealistic (ions cannot dissociate in non-polar media).

### Supported Organic Solvents (CHARMM carb field)

All **91 residues** from EMC v9.4.4 `carb.top` TEMPLATES section have built-in SMILES. Use the 4-character residue name exactly as it appears in the force field.

#### Common solvents (frequently used)

| Residue | Name | Atoms |
|---|---|---|
| `ETOH` | Ethanol | 9 |
| `MEOH` | Methanol | 6 |
| `THF` | Tetrahydrofuran | 13 |
| `EGLY` | Ethylene glycol | 10 |
| `PRO2` | Isopropanol | 12 |
| `CYHX` | Cyclohexane | 18 |
| `THP` | Tetrahydropyran | 16 |
| `NMA` | N-methylacetamide | 12 |
| `IPAA` | N-isopropylacetamide | 18 |
| `TF2M` | 2-methyltetrahydrofuran | 16 |
| `PROA` | Propionic acid | 10 |
| `ACO` | Acetone | 10 |
| `AALD` | Acetaldehyde | 7 |
| `MGLYOL` | Glycerol | 14 |
| `PSICOS` | Psicose | 24 |

#### Carbohydrates (pyranoses, furanoses, alditols, inositols)

84 sugar/sugar-derivative residues fully covered:

**Aldopyranoses**: AALL, AALT, AGAL, AGLC, AGUL, AIDO, AMAN, ATAL, BALL, BALT, BGAL, BGLC, BGUL, BIDO, BMAN, BTAL — 16 D/L pyranose sugars

**Keto-pyranoses**: AFRU, BFRU, AFUC, BFUC

**Furanoses**: AARB, ADEO, ALYF, ARIB, AXYF, BARB, BDEO, BLYF, BRIB, BXYF, AXYL, BXYL

**Modified sugars**: AGALNA, AGLCA, AGLCNA, AIDOA, ALLOSE, AMOL, AMOP, ANE5AC, ARHM, ARHMOA, BGALNA, BGLCA, BGLCNA, BIDOA, BNE5AC, BRHM

**Alditols (linear)**: DALTOL, DARAOL, DGLUOL, DGULOL, DIDIOL, DMANOL, DTHROL, LALTOL, LARAOL, LGLUOL, LGULOL, LIDIOL, LMANOL, LTHROL, MALLOL, MERYOL, MGALOL, MRIBOL, MXYLOL

**Inositols**: INI1, INI2, INI3, INI4, INI5

**Other**: ABEQ, AB15, SUCR, SUF1B, SUF2B, SUF3B, SUF4B, SUF6, I3P1, I3P2, OMEB, AR15, SA23E, SA26E, SA28E, SA29E

### How to check if a solvent is supported

All 91 carb templates are pre-loaded in `build.py`. No need to check — all are usable directly:

```json
"solvents": {"ETOH": 20, "MEOH": 10, "THF": 15}
```

### OPLS-AA 2024 Force Field

OPLS uses **rule-based atom typing**: any SMILES that OPLS can type is supported. The EMC-bundled `opls-aa.prm/top` (2024) provides 105 pre-templated residues + 10 monatomic ions.

**Water model**: SPCE (`"O"` SMILES, EMC adds H automatically).

**Key advantage**: native support for **F⁻, Br⁻, I⁻** and NO₃⁻ (via literature parameters) — ions that CHARMM lacks.

#### OPLS Ion Support

All **10 monatomic ions** from OPLS-AA 2024 MAS section:

| Ion | OPLS Type | SMILES |
|---|---|---|
| Na+ | na+ | `[Na+]` |
| K+ | k+ | `[K+]` |
| Li+ | li+ | `[Li+]` |
| Cs+ | cs+ | `[Cs+]` |
| Mg2+ | mg+2 | `[Mg+2]` |
| Sr2+ | sr+2 | `[Sr+2]` |
| F- | f- | `[F-]` |
| Cl- | cl- | `[Cl-]` |
| Br- | br- | `[Br-]` |
| I- | i- | `[I-]` |
| OH- | HYDR | `[O-]` |

#### OPLS Organic Solvents (commonly used subset)

All 105 residues from EMC v9.4.4 `opls-aa.top` are supported. Notable solvents:

| Residue | Name | SMILES |
|---|---|---|
| `MEOH` | Methanol | `CO` |
| `ETDO` | Ethylene glycol | `C(O)CO` |
| `PRTO` | Glycerol | `C(O)C(O)CO` |
| `EACE` | Ethyl acetate | `CC(OCC)=O` |
| `DMSO` | Dimethyl sulfoxide | `CS(=O)C` |
| `FORM` | Formamide | `C(=O)N` |
| `UREA` | Urea | `NC(=O)N` |
| `NMAM` | N-methylacetamide | `CC(=O)NC` |
| `PHEN` | Phenol | `c1ccccc1O` |
| `MEAN` | Methylamine | `NC` |
| `DMET` | Dimethyl ether | `COC` |
| `CPRO` | Cyclopropane | `C1CC1` |

> **Note**: DMSO and sulfoxides are supported via a patched OPLS-AA PRM bundled with the skill. **OPLS does not support custom polyatomic anions** (SO₄²⁻, NO₃⁻, CO₃²⁻, etc.) — EMC's OPLS mode cannot assign partial charges to user-defined atom types. For these anions, use `field: "charmm"` instead.



## CHARMM Ion Support

All **15 ions** from EMC v9.4.4 `water_ions.prm/top` are supported. Use the full ion key format:

### Monatomic ions

| Ion | CHARMM Group | Charge | SMILES |
|---|---|---|---|
| Na+ | SOD | +1 | `[Na+]` |
| K+ | POT | +1 | `[K+]` |
| Cl- | CLA | −1 | `[Cl-]` |
| OH- | OH | −1 | `[O-](H)` |
| Li+ | LIT | +1 | `[Li+]` |
| Cs+ | CES | +1 | `[Cs+]` |
| Mg2+ | MG | +2 | `[Mg+2]` |
| Ca2+ | CAL | +2 | `[Ca+2]` |
| Ba2+ | BAR | +2 | `[Ba+2]` |
| Rb+ | RUB | +1 | `[Rb+]` |
| Zn2+ | ZN2 | +2 | `[Zn+2]` |
| Cd2+ | CD2 | +2 | `[Cd+2]` |

### Polyatomic ions (phosphate family)

| Ion | CHARMM Group | Charge | SMILES |
|---|---|---|---|
| PO4(3-) | PO4 | −3 | `P(:O)(:O)(:O):O` |
| HPO4(2-) | HPO4 | −2 | `P(:O)(:O)(:O)O` |
| H2PO4(-) | H2PO4 | −1 | `P(O)(:O)(:O)O` |

Config example:
```json
"ions": {"Na+": 2, "PO4(3-)": 1}
```

Mass fractions are automatically calculated. Explicit user counts are used in EMC clusters to avoid heavy/bulky ions being rounded to zero.

## Adsorbate Config (SMILES Support)

```json
{
  "adsorbate": {
    "list": [
      {"name": "NH3", "smiles": "[NH3]", "site": "ontop", "height": 2.0},
      {"name": "CO2", "smiles": "O=C=O", "site": "ontop", "height": 2.5},
      {"name": "TTE", "smiles": "FC(F)C(F)(F)COC(F)(F)C(F)F", "builder": "emc"}
    ]
  }
}
```

| Method | Example | Notes |
|---|---|---|
| SMILES (ASE) | `"smiles": "[NH3]"` | Simple molecules, ASE handles it |
| ASE built-in | `"name": "NH3"` | ASE molecule library |
| **EMC/PCFF** | `"builder": "emc"` | **Complex organics** (multi-rotatable bonds, fluoro-ethers, ionic liquids). Uses EMC+PCFF for high-quality 3D conformer generation. **Auto-falls back to RDKit MMFF94 when EMC cannot type the molecule** (missing force-field parameters) |
| **RDKit** | `"builder": "rdkit"` | ETKDGv3 + MMFF94 multi-conformer screening (50 conformers, lowest energy). Use when EMC force fields lack rules (amides, heterocycles, sugars, carboxylates). Requires `rdkit` (optional dependency) |
| Auto | `"name": "CO2"` | Tries SMILES first, falls back to RDKit, then ASE library |

> **When to use `builder: "emc"`**: For molecules with >5 heavy atoms, multiple rotatable bonds, or fluorine-substituted backbones (e.g., TTE/HFE, ionic liquid cations, long-chain solvents). Simple inorganics (CO₂, NH₃, COOH, N₂H₄) are better with ASE.
>
> **When to use `builder: "rdkit"`**: When EMC/PCFF fails with `Missing force field parameters` (e.g., pyrrolidone amides, glucopyranose rings, carboxymethyl ethers). RDKit preserves SMILES stereo-chemistry (`@` markers), which EMC's SMILES parser may ignore. Since `builder: "emc"` auto-falls back to RDKit, explicit `"builder": "rdkit"` is only needed to skip the EMC attempt.

### Heterojunction Config (pymatgen)

```json
{
  "surface": {
    "type": "heterojunction",
    "film": {"source": "mp-20194", "miller": [1, 1, 1], "thickness": 3},
    "substrate": {"source": "mp-126", "miller": [1, 0, 0], "thickness": 4}
  },
  "adsorbate": {
    "list": [
      {"name": "NH3", "element": "Pt", "site": "ontop", "height": 2.0}
    ]
  }
}
```

**`source` formats** (priority order):

| Source | Format | Use Case | Example |
|---|---|---|---|
| ASE built-in | `"element"` | Simple metals (FCC/BCC) | `"Pt"` |
| Materials Project | `mp-xxx` | Oxides, compounds | `"mp-20194"` |
| COD | `cod_xxxxx` | Organic crystals | `"cod_12345"` |
| PubChem | `pubchem_xxx` | Molecules | `"pubchem_123591"` |
| Programmatic | `fullerene_c60` | Fullerenes | `"fullerene_c60"` |
| Manual | `element:type:a` | Known structures | `"Pt:fcc:3.92"` |
| Local file | `/path/to/file` | Any structure | `"./POSCAR"` |

**Notes**:
- **ASE built-in**: Fast, no network, limited to simple metals (Pt, Pd, Au, Ag, Cu, Ni, Fe, W, Mo)
- **Materials Project**: Requires API key in `configs/databases.json`, best for oxides/compounds
- **COD**: Free, good for organic crystals, may need proxy for stable connection
- **PubChem**: Free, for molecules, uses curl internally (Python has SSL issues). Note: large molecules (C60, fullerenes) may only have 2D coordinates
- **Programmatic**: No network, for fullerenes only (C60 truncated icosahedron, verified 3D spherical structure)
- **Manual**: `element:lattice_type:lattice_constant` (e.g., `"Pt:fcc:3.92"`, `"Fe:bcc:2.87"`)
- **Local file**: Supports POSCAR, CIF, XYZ, and other ASE-readable formats

**Recommended for C60/fullerenes**: Use `fullerene_c60` (programmatic) instead of `pubchem_xxx` (may return 2D).

**`adsorbate.element`**: specifies which element's surface the adsorbate sits on. Automatically detects exposed side and positions water layer accordingly.

## Water Layer Positioning

Water layer starts from the metal surface (not adsorbate top), so adsorbates are embedded in the solvent — physically correct solvation.

| Parameter | Default | Description |
|---|---|---|
| `gap` | 1.5 Å | Metal surface → water layer bottom |
| `min_clearance` | 1.5 Å | Minimum water–adsorbate distance |

Water molecules overlapping adsorbates (distance < vdw_sum × 0.7) are automatically removed.

## Box and Cell Behavior

- **xy dimensions**: `max(surface_xy, natural_water_box_xy)` — prevents solvent spillage on small surfaces
- **Bottom vacuum**: always 10 Å below the lowest atom
- **Top vacuum**: 15 Å above the highest atom
- **4-layer Pt(100) 3×3 surface** is a recommended minimum for realistic electrolyte models

### Density pitfall (fixed 2026-08-27 — do not regress)

The EMC templates size the box as a **cube** derived from the requested density:

```text
vphase1 -> mphase1/nav/density1        (* target volume *)
lbox    -> (vphase1/fshape)^(1/3)      (* cube edge *)
lxx     -> lphase1                     (* = fshape*lbox *)
lyy     -> lbox
lzz     -> lbox
```

To tile a slab, `build.py` must override `lxx`/`lyy` with the surface dimensions.
**`lzz` must be rescaled in the same step**, otherwise it keeps the cube-edge value
and the real volume becomes `lx·ly·lbox` instead of `vphase1` — the electrolyte is
silently compressed.

The error scales with molecule count, so it is invisible on small test cases and
severe on production ones. Measured on Pt(100) 6×6 (16.63 × 16.63 Å) with
600 H₂O + Na⁺ + OH⁻ at `density: 1.0`:

| | box z | volume | density |
|---|---|---|---|
| Without `lzz` rescale | 26.21 Å | 7251 Å³ | **2.485 g/cm³** (2.48× off) |
| With `lzz` rescale | 65.13 Å | 18015 Å³ | **1.000 g/cm³** ✓ |

The fix expresses `lzz` in EMC's own variables so EMC's exact masses are used:

```text
lzz -> vphase1/(<lx>*<ly>)
```

This is deliberately **not** a Python-side mass calculation. A local mass table
would have to cover the 91-residue carb field plus all OPLS solvents, and any
missing entry would silently reintroduce a wrong density. `vphase1` is already
correct for every species EMC can type.

**Guard**: `_verify_box_density()` runs on every EMC output and prints a loud
warning if the measured density deviates >10% from the request. A `Density check:
X g/cm3 (target Y) OK` line in the build log is the confirmation. If you see
`!! DENSITY WARNING`, inspect `lzz` in the generated `build_water.emc` before
using the box for sampling.

Only the slab path overrides the box (`surface_cell_xy` set). Electrolyte-only
mode leaves the template cube untouched, so its density was always correct; the
guard covers both paths.

Regression test: `python tests/test_density_rescale.py` (no EMC binary needed).

## Output Files

| File | Format | Notes |
|---|---|---|
| `POSCAR` | VASP | Elements grouped |
| `structure.xyz` | XYZ | Generic coordinates |
| `structure.cif` | CIF | Crystallographic info |
| `structure.data` | LAMMPS data | Bonded + non-bonded |
| `model_report.txt` | Text | Formula, element counts, min distance, warnings |
| `water.data` | LAMMPS data | EMC raw output (preserves CHARMM types) |

## Known Limitations

- **Dual-field wildcard conflict**: carb's wildcard rule (`? ? UNK UNK 0 *`) can override water_ions' ion charges when no water is present. Auto-water guard mitigates this (see above).
- **PCFF ion support**: PCFF uses rule-based atom typing; Na+/K+/Cl- SMILES syntax is brittle. Use CHARMM for ion-containing systems.
- **EMC file encoding**: templates must be ISO-8859-1 (latin-1) encoded.
- **OH- display in POSCAR**: OH- atoms merge with water O/H (same element). Use `water.data` to verify OH- location.
- **`water_ions_carb.prm/top`**: the EMC-bundled merged file has duplicate ITEM sections and causes segfault. This skill uses separate dual-field loading instead of the broken merged file.
- **ASE vs complex organics**: ASE `molecule(smiles)` falls back to rough geometries for molecules with >5 heavy atoms or multiple rotatable bonds. Use `"builder": "emc"` for these cases (requires PCFF force field).

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Not an EMC format` | Template encoding wrong | Use ISO-8859-1 (latin-1) encoding |
| `Missing rules` | Group id doesn't match force field | For CHARMM, use residue names from carb field (e.g., `ETOH`, not `ethanol`) |
| `Total charge does not equal zero` | Ion charge mismatch or dual-field wildcard override | Ensure `Na+`/`Cl-` counts balance; add water_count ≥ 1 |
| `Type "BAR" not found` | Broken merged force field file | Don't use `water_ions_carb`; use dual-field loading |
| `Segfault` (exit 139) | Corrupt force field file with duplicate ITEM sections | Use clean `water_ions` + `carb` separately |
| `EMC output not found` (molecule mode) | SMILES not recognized by PCFF | Check SMILES string; add `field: "pcff"` to config if using solvent mode |
| `SMILES required with builder='emc'` | Missing SMILES | Add `"smiles": "..."` to the adsorbate entry |
| `Undefined variable` | Missing comma in EMC variables block | Check generated `build_water.emc` for syntax |
| `!! DENSITY WARNING: ... Nx off` | `lzz` not rescaled after overriding `lxx`/`lyy` — electrolyte compressed | See "Density pitfall" above; `lzz` must be `vphase1/(lx*ly)`, not `lbox` |
