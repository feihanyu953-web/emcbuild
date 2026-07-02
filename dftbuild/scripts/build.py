#!/usr/bin/env python
"""
DFTbuild - Structure builder for DFT calculations.

Builds crystal structures from various sources:
- Materials Project (mp-xxx)
- Crystallography Open Database (cod_xxxxx)
- PubChem (pubchem_xxx)
- Programmatic generation (fullerene_c60)
- Local files (POSCAR, CIF, etc.)
- Manual specification (Element:lattice_type:lattice_constant)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.build import (
    add_adsorbate as ase_add_adsorbate,
    bcc100,
    bcc110,
    bcc111,
    bulk,
    fcc100,
    fcc110,
    fcc111,
    hcp0001,
    molecule as ase_molecule,
    nanotube,
    surface,
)
from ase.geometry import get_distances
from ase.io import read as ase_read

# ─── Path configuration ──────────────────────────────────────────────
SKILL_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = SKILL_ROOT / "configs"


# ─── Config loading ──────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load config and resolve output_dir relative to config file location.

    If output_dir is not specified, defaults to 'dftbuild_output' in the
    same directory as the config file (user's project directory).

    Args:
        config_path: Path to JSON config file (absolute or relative)

    Returns:
        Config dict with output_dir resolved
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Resolve output_dir: default to dftbuild_output folder in config file's directory
    if 'output_dir' not in config:
        config_dir = config_path.parent
        config['output_dir'] = str(config_dir / 'dftbuild_output')
    else:
        # Make output_dir absolute if relative
        out = Path(config['output_dir'])
        if not out.is_absolute():
            config['output_dir'] = str(config_path.parent / out)

    return config


# ─── API key management ──────────────────────────────────────────────

_MP_API_KEY = None


def _load_mp_api_key() -> str:
    """Load Materials Project API key with three-level fallback.

    Priority order:
      1. MP_API_KEY environment variable
      2. ~/.emcbuild/config.json -> materials_project.api_key
      3. configs/databases.json -> materials_project.api_key

    Returns:
        API key string, or empty string if not found
    """
    global _MP_API_KEY
    if _MP_API_KEY:
        return _MP_API_KEY

    # Level 1: Environment variable
    env_key = os.environ.get("MP_API_KEY", "")
    if env_key:
        _MP_API_KEY = env_key
        return _MP_API_KEY

    # Level 2: ~/.emcbuild/config.json
    home_config = Path.home() / ".emcbuild" / "config.json"
    if home_config.exists():
        with open(home_config, 'r', encoding='utf-8') as f:
            home_cfg = json.load(f)
        _MP_API_KEY = home_cfg.get('materials_project', {}).get('api_key', '')
        if _MP_API_KEY:
            return _MP_API_KEY

    # Level 3: configs/databases.json (bundled with skill)
    db_config_path = CONFIGS_DIR / "databases.json"
    if db_config_path.exists():
        with open(db_config_path, 'r', encoding='utf-8') as f:
            db_cfg = json.load(f)
        _MP_API_KEY = db_cfg.get('materials_project', {}).get('api_key', '')
        # Skip placeholder values
        if _MP_API_KEY == 'YOUR_API_KEY_HERE':
            _MP_API_KEY = ''

    return _MP_API_KEY


def _get_pubchem_structure(cid: int):
    """Download 3D structure from PubChem using curl.

    Args:
        cid: PubChem Compound ID

    Returns:
        ASE Atoms object
    """
    import subprocess
    from ase.io import read as ase_read

    # Use curl to avoid Python SSL issues
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF"
    result = subprocess.run(['curl', '-s', '--max-time', '30', url],
                          capture_output=True, text=True)

    if result.returncode != 0 or 'Fault' in result.stdout:
        raise ValueError(f"PubChem download failed for CID {cid}: {result.stdout[:200]}")

    # Save to temp file and read with ASE
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
        f.write(result.stdout)
        temp_path = f.name

    try:
        atoms = ase_read(temp_path)
    finally:
        os.unlink(temp_path)

    return atoms


def _generate_c60():
    """Generate C60 truncated icosahedron structure.

    Returns:
        ASE Atoms object with 60 carbon atoms in spherical arrangement
    """
    import numpy as np
    phi = (1 + np.sqrt(5)) / 2

    # Generate 60 vertices of truncated icosahedron
    vertices = []

    # Even permutations of (0, +/-1, +/-3*phi)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            base = [0, s1, s2 * 3 * phi]
            vertices.extend([[base[0], base[1], base[2]],
                           [base[1], base[2], base[0]],
                           [base[2], base[0], base[1]]])

    # Even permutations of (+/-2, +/-(1+2*phi), +/-phi)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                base = [s1 * 2, s2 * (1 + 2 * phi), s3 * phi]
                vertices.extend([[base[0], base[1], base[2]],
                               [base[1], base[2], base[0]],
                               [base[2], base[0], base[1]]])

    # Even permutations of (+/-1, +/-(2+phi), +/-2*phi)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                base = [s1, s2 * (2 + phi), s3 * 2 * phi]
                vertices.extend([[base[0], base[1], base[2]],
                               [base[1], base[2], base[0]],
                               [base[2], base[0], base[1]]])

    coords = np.array(vertices)
    coords = np.unique(np.round(coords, 10), axis=0)

    # Center and scale to correct radius (C60 radius ~3.55 A)
    coords -= coords.mean(axis=0)
    dists = np.linalg.norm(coords, axis=1)
    scale = 3.55 / dists.mean()
    coords *= scale

    # Create ASE Atoms with a large unit cell (molecule in vacuum)
    from ase import Atoms
    atoms = Atoms('C60', positions=coords)
    atoms.center(vacuum=15.0)  # Add 15 A vacuum around the molecule
    atoms.pbc = True

    return atoms


# ─── Structure loading ───────────────────────────────────────────────

METAL_DEFAULTS = {
    "Ag": {"structure": "fcc", "a": 4.0862},
    "Au": {"structure": "fcc", "a": 4.0782},
    "Co": {"structure": "hcp", "a": 2.507, "c": 4.069},
    "Cu": {"structure": "fcc", "a": 3.615},
    "Fe": {"structure": "bcc", "a": 2.8665},
    "Mo": {"structure": "bcc", "a": 3.150},
    "Ni": {"structure": "fcc", "a": 3.524},
    "Pd": {"structure": "fcc", "a": 3.890},
    "Pt": {"structure": "fcc", "a": 3.9239},
    "W":  {"structure": "bcc", "a": 3.165},
}

# Simple {element: a} for _get_structure manual-spec fallback
LATTICE = {el: info["a"] for el, info in METAL_DEFAULTS.items()}


def _get_structure(source: str):
    """Load crystal structure from various sources.

    Supported sources (priority order):
      - "mp-xxx"        : Materials Project ID (oxides, compounds)
      - "cod_xxxxx"     : COD (Crystallography Open Database) ID
      - "pubchem_xxx"   : PubChem CID (molecules, uses curl)
      - "fullerene_c60" : Programmatic C60 generation
      - "/path/to/POSCAR" : Local POSCAR file
      - "/path/to/file.cif" : Local CIF file
      - "Pt:fcc:3.92"   : Manual specification (element:lattice_type:lattice_constant)

    Returns:
        pymatgen Structure object
    """
    from pymatgen.core import Structure, Lattice

    # Materials Project
    if source.startswith('mp-'):
        from mp_api.client import MPRester
        api_key = _load_mp_api_key()
        if not api_key:
            raise ValueError("Materials Project API key not found. "
                             "Set MP_API_KEY env var, ~/.emcbuild/config.json, "
                             "or configs/databases.json")
        with MPRester(api_key) as mpr:
            return mpr.get_structure_by_material_id(source)

    # COD
    if source.startswith('cod_'):
        from pymatgen.ext.cod import COD
        cod = COD()
        cod_id = int(source.replace('cod_', ''))
        return cod.get_structure_by_id(cod_id)

    # PubChem (molecules, uses curl to avoid SSL issues)
    if source.startswith('pubchem_'):
        cid = int(source.replace('pubchem_', ''))
        atoms = _get_pubchem_structure(cid)
        from pymatgen.io.ase import AseAtomsAdaptor
        return AseAtomsAdaptor().get_structure(atoms)

    # Programmatic C60 generation
    if source == 'fullerene_c60':
        atoms = _generate_c60()
        from pymatgen.io.ase import AseAtomsAdaptor
        return AseAtomsAdaptor().get_structure(atoms)

    # Local file
    if os.path.exists(source):
        if 'POSCAR' in source or 'CONTCAR' in source:
            from pymatgen.io.vasp import Poscar
            return Poscar.from_file(source).structure
        elif source.endswith('.cif'):
            from pymatgen.io.cif import CifParser
            return CifParser(source).get_structures()[0]
        else:
            from ase.io import read as ase_read
            ase_atoms = ase_read(source)
            from pymatgen.io.ase import AseAtomsAdaptor
            return AseAtomsAdaptor().get_structure(ase_atoms)

    # Manual: "Element:lattice_type:lattice_constant"
    if ':' in source:
        parts = source.split(':')
        element = parts[0]
        lattice_type = parts[1] if len(parts) > 1 else 'fcc'
        a = float(parts[2]) if len(parts) > 2 else LATTICE.get(element, 3.8)
        if lattice_type == 'fcc':
            lattice = Lattice.cubic(a)
            species = [element] * 4
            coords = [[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]
        elif lattice_type == 'bcc':
            lattice = Lattice.cubic(a)
            species = [element] * 2
            coords = [[0,0,0],[0.5,0.5,0.5]]
        elif lattice_type == 'hex':
            lattice = Lattice.hexagonal(a, a * 1.633)
            species = [element] * 2
            coords = [[0,0,0],[1/3,2/3,0.5]]
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
        return Structure(lattice, species, coords)

    raise ValueError(f"Cannot parse structure source: {source}")


# ─── Surface builders (ASE) ─────────────────────────────────────────

def _ase_surface(element: str, miller: tuple, size: tuple, layers: int,
                 vacuum: float) -> Atoms:
    """Build a surface slab for FCC/BCC/HCP metals using tabulated defaults.

    Supports FCC(100/110/111), BCC(100/110/111), HCP(0001).
    Falls back to ``ase.build.surface`` for other Miller indices.
    """
    defaults = METAL_DEFAULTS[element]
    structure = defaults["structure"]
    full_size = (size[0], size[1], layers)
    if structure == "fcc" and miller == (1, 0, 0):
        return fcc100(element, size=full_size, a=defaults["a"], vacuum=vacuum)
    if structure == "fcc" and miller == (1, 1, 0):
        return fcc110(element, size=full_size, a=defaults["a"], vacuum=vacuum)
    if structure == "fcc" and miller == (1, 1, 1):
        return fcc111(element, size=full_size, a=defaults["a"], vacuum=vacuum)
    if structure == "bcc" and miller == (1, 0, 0):
        return bcc100(element, size=full_size, a=defaults["a"], vacuum=vacuum)
    if structure == "bcc" and miller == (1, 1, 0):
        return bcc110(element, size=full_size, a=defaults["a"], vacuum=vacuum)
    if structure == "bcc" and miller == (1, 1, 1):
        return bcc111(element, size=full_size, a=defaults["a"], vacuum=vacuum)
    if structure == "hcp" and miller in {(0, 0, 1), (0, 0, 0, 1)}:
        return hcp0001(element, size=full_size, a=defaults["a"],
                       c=defaults["c"], vacuum=vacuum)

    # Generic fallback for non-standard Miller indices
    crystal = bulk(element, crystalstructure=structure,
                   a=defaults["a"], c=defaults.get("c"))
    return surface(crystal, miller, layers, vacuum=vacuum).repeat(
        (size[0], size[1], 1))


def _top_layer_indices(atoms: Atoms, layers: int,
                       tolerance: float = 0.35) -> list:
    """Return atom indices belonging to the top *layers* z-layers."""
    z_values = sorted(
        {round(float(z) / tolerance) * tolerance for z in atoms.positions[:, 2]},
        reverse=True,
    )
    threshold = min(z_values[:layers])
    return [i for i, pos in enumerate(atoms.positions)
            if pos[2] >= threshold - tolerance / 2]


def _even_counts(total: int, symbols: list) -> dict:
    """Split *total* as evenly as possible among *symbols*."""
    base = total // len(symbols)
    remainder = total % len(symbols)
    return {sym: base + (1 if idx < remainder else 0)
            for idx, sym in enumerate(symbols)}


def build_slab(element: str, miller: tuple, size: tuple, layers: int,
               vacuum: float = 15.0) -> Atoms:
    """Build a metal surface slab (FCC/BCC/HCP) with any Miller index.

    Uses ``METAL_DEFAULTS`` lattice constants and ``_ase_surface`` as
    the backend.  Returns an ASE Atoms object with PBC enabled.
    """
    if element not in METAL_DEFAULTS:
        raise ValueError(
            f"Unsupported built-in metal {element!r}; "
            "provide a CIF/POSCAR for custom bases.")
    slab = _ase_surface(element, miller, size, layers, vacuum)
    slab.pbc = (True, True, True)
    return slab


def build_nanotube(n: int, m: int, length: int = 3,
                   vacuum: float = 15.0) -> Atoms:
    """Build a carbon nanotube with chiral indices (n, m).

    *length* is the number of unit cells along the tube axis.
    *vacuum* adds vacuum padding in the non-periodic directions.
    """
    return nanotube(n, m, length=length, vacuum=vacuum)


def build_heterojunction_pymatgen(film_cfg: dict, substrate_cfg: dict,
                                   vacuum: float = 15.0) -> Atoms:
    """Build heterojunction using pymatgen's CoherentInterfaceBuilder.

    Args:
        film_cfg: {"source": "mp-xxx", "miller": [h,k,l], "thickness": N}
        substrate_cfg: same format
        vacuum: vacuum thickness above the structure (Angstrom)

    Returns:
        ASE Atoms object with the heterojunction
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.interfaces import CoherentInterfaceBuilder

    # Load bulk structures
    film_structure = _get_structure(film_cfg['source'])
    substrate_structure = _get_structure(substrate_cfg['source'])

    film_miller = tuple(film_cfg['miller'])
    substrate_miller = tuple(substrate_cfg['miller'])
    film_thickness = film_cfg.get('thickness', 3)
    substrate_thickness = substrate_cfg.get('thickness', 4)

    print(f"  Film: {film_structure.composition.reduced_formula}({film_miller}), "
          f"thickness={film_thickness} layers")
    print(f"  Substrate: {substrate_structure.composition.reduced_formula}({substrate_miller}), "
          f"thickness={substrate_thickness} layers")

    # Build interface
    cib = CoherentInterfaceBuilder(
        substrate_structure=substrate_structure,
        film_structure=film_structure,
        film_miller=film_miller,
        substrate_miller=substrate_miller,
    )

    if not cib.terminations:
        raise ValueError("No valid interface terminations found. "
                         "Check Miller indices and lattice matching.")

    print(f"  Found {len(cib.terminations)} termination(s): {cib.terminations}")

    # Use first termination (lowest strain)
    termination = cib.terminations[0]
    interfaces = list(cib.get_interfaces(
        termination=termination,
        gap=2.0,
        vacuum_over_film=vacuum,
        film_thickness=film_thickness,
        substrate_thickness=substrate_thickness,
    ))

    if not interfaces:
        raise ValueError("No interfaces generated. Check parameters.")

    # Pick the interface with lowest strain
    interface = interfaces[0]
    print(f"  Interface: {interface.composition.reduced_formula}, "
          f"{len(interface)} atoms, "
          f"cell={[round(x, 2) for x in interface.lattice.abc]}")

    # Convert to ASE Atoms
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(interface)
    atoms.pbc = True

    return atoms


def alloy_surface_layers(atoms: Atoms, symbols: Sequence[str],
                         top_layers: int = 1,
                         fractions: Mapping[str, float] | None = None,
                         ) -> tuple:
    """Replace atoms in the highest-z layers with alloy symbols.

    Args:
        atoms: Base slab (ASE Atoms)
        symbols: List of element symbols to substitute (e.g. ["Cu", "Co"])
        top_layers: Number of top z-layers to modify (default 1)
        fractions: Optional dict of {symbol: fraction}. If None, atoms are
            distributed evenly among *symbols*.

    Returns:
        (alloyed_atoms, changed_counts) where changed_counts is
        {symbol: n_replaced}.
    """
    if top_layers < 1:
        raise ValueError("top_layers must be >= 1")
    if not symbols:
        return atoms.copy(), {}

    alloyed = atoms.copy()
    candidate_indices = _top_layer_indices(alloyed, top_layers)

    if fractions is None:
        counts = _even_counts(len(candidate_indices), list(symbols))
    else:
        counts = {sym: int(round(len(candidate_indices) * fractions.get(sym, 0.0)))
                  for sym in symbols}

    selected = sorted(candidate_indices,
                      key=lambda i: (alloyed.positions[i][0],
                                     alloyed.positions[i][1]))
    cursor = 0
    changed: dict[str, int] = {}
    for sym in symbols:
        count = max(0, min(counts.get(sym, 0), len(selected) - cursor))
        for atom_index in selected[cursor:cursor + count]:
            alloyed[atom_index].symbol = sym
        changed[sym] = count
        cursor += count
    return alloyed, changed


# ─── Molecule building ─────────────────────────────────────────────

# Hardcoded fallback geometries for common adsorbates (symbols + positions)
FALLBACK_GEOMETRIES = {
    "H":  ("H",  [(0.0, 0.0, 0.0)]),
    "O":  ("O",  [(0.0, 0.0, 0.0)]),
    "N":  ("N",  [(0.0, 0.0, 0.0)]),
    "OH": ("OH", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.97)]),
    "NH": ("NH", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.02)]),
    "NH2": ("NH2", [(0.0, 0.0, 0.0), (0.92, 0.0, 0.55), (-0.92, 0.0, 0.55)]),
    "NH3": (
        "NH3",
        [(0.0, 0.0, 0.0), (0.94, 0.0, 0.45),
         (-0.47, 0.81, 0.45), (-0.47, -0.81, 0.45)],
    ),
    "H2O": ("OH2", [(0.0, 0.0, 0.0), (0.76, 0.58, 0.0), (-0.76, 0.58, 0.0)]),
}

ION_GEOMETRIES = {
    "K":   ("K",   [(0.0, 0.0, 0.0)]),
    "K+":  ("K",   [(0.0, 0.0, 0.0)]),
    "Na":  ("Na",  [(0.0, 0.0, 0.0)]),
    "Na+": ("Na",  [(0.0, 0.0, 0.0)]),
    "Cl":  ("Cl",  [(0.0, 0.0, 0.0)]),
    "Cl-": ("Cl",  [(0.0, 0.0, 0.0)]),
    "OH":  FALLBACK_GEOMETRIES["OH"],
    "OH-": FALLBACK_GEOMETRIES["OH"],
    "NH3": FALLBACK_GEOMETRIES["NH3"],
}

MOLECULE_ALIASES = {
    "K+":  "K",
    "Na+": "Na",
    "Cl-": "Cl",
    "OH-": "OH",
}

SURFACE_ANCHOR_SYMBOLS = {
    "CO": "C",
}


def build_molecule(name: str) -> tuple:
    """Build a molecule from file, SMILES/RDKit, ASE collection, or hardcoded fallback.

    Priority:
      1. ``file:/path/to/xyz`` — read user-supplied file
      2. ``smiles:CCO``       — RDKit 3D conformer
      3. ASE built-in molecule library
      4. Hardcoded fallback geometry

    Returns (Atoms, source_label) where source_label is one of
    'file', 'rdkit', 'ase', 'fallback'.
    """
    if name.startswith("file:"):
        from pathlib import Path as _P
        atoms = ase_read(_P(name.removeprefix("file:")))
        atoms.translate(-atoms.positions[0])
        return atoms, "file"

    if name.startswith("smiles:"):
        atoms = _rdkit_molecule_from_smiles(name.removeprefix("smiles:"))
        atoms.translate(-atoms.positions[0])
        return atoms, "rdkit"

    canonical_name = MOLECULE_ALIASES.get(name, name)
    try:
        atoms = ase_molecule(canonical_name)
        atoms = _surface_oriented_molecule(name, atoms)
        atoms.translate(-atoms.positions[0])
        return atoms, "ase"
    except Exception:
        pass

    geometry = ION_GEOMETRIES.get(name, FALLBACK_GEOMETRIES.get(name))
    if geometry is None:
        raise ValueError(f"Unsupported molecule {name!r}")
    formula, positions = geometry
    return Atoms(formula, positions=positions), "fallback"


def _surface_oriented_molecule(name: str, atoms: Atoms) -> Atoms:
    """Orient molecule so the anchor atom is at the bottom (closest to surface)."""
    anchor_symbol = SURFACE_ANCHOR_SYMBOLS.get(name)
    if anchor_symbol is None:
        return atoms

    symbols = atoms.get_chemical_symbols()
    try:
        anchor_index = symbols.index(anchor_symbol)
    except ValueError:
        return atoms

    order = [anchor_index] + [idx for idx in range(len(atoms)) if idx != anchor_index]
    oriented = atoms[order]
    oriented.translate(-oriented.positions[0])
    if len(oriented) > 1 and float(np.mean(oriented.positions[1:, 2])) < 0.0:
        oriented.positions[:, 2] *= -1.0
    return oriented


def _rdkit_molecule_from_smiles(smiles: str) -> Atoms:
    """Generate a 3D molecular structure from a SMILES string using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for smiles: molecule generation. "
            "Install with: conda install -c conda-forge rdkit"
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 20260520
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"RDKit could not embed a 3D conformer for SMILES {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
    ]
    return Atoms(symbols, positions=positions)


def _molecule_from_geometry(name: str, anchor: tuple) -> Atoms:
    """Build molecule and translate its first atom to *anchor* position."""
    atoms, _ = build_molecule(name)
    atoms.translate(anchor)
    return atoms


# ─── Adsorbate placement ──────────────────────────────────────────

def add_adsorbate(atoms: Atoms, adsorbate: str, site: str = "ontop",
                  height: float = 1.8, xy: tuple = None) -> Atoms:
    """Add a single adsorbate molecule above the top surface.

    Parameters
    ----------
    atoms : Atoms
        Base slab.
    adsorbate : str
        Molecule name (e.g. "NH3", "CO", "smiles:CCO").
    site : str
        ASE site name ("ontop", "bridge", "hollow", "fcc", "hcp")
        or "custom_xyz" when *xy* is provided.
    height : float
        Vertical distance (Angstrom) above the surface site.
    xy : tuple of float, optional
        Explicit (x, y) placement; overrides *site*.

    Returns
    -------
    Atoms
        Slab with adsorbate added.
    """
    base = atoms.copy()
    adsorbate_atoms, _ = build_molecule(adsorbate)
    position = xy if site == "custom_xyz" else _ase_site_name(site, base)
    ase_add_adsorbate(base, adsorbate_atoms, height=height, position=position)
    base.center(axis=2, vacuum=12.0)
    return base


def add_adsorbates(atoms: Atoms, specs: Sequence,
                   min_site_distance: float = 1.5) -> tuple:
    """Add multiple adsorbates with automatic site-spacing for co-adsorption.

    The first adsorbate is placed as specified.  Subsequent adsorbates whose
    ``site`` matches a known ASE site name automatically search candidate
    positions and pick the one that maximises the minimum distance to
    already-placed adsorbate atoms.

    Parameters
    ----------
    atoms : Atoms
        Base slab.
    specs : list of dict
        Each dict keys: ``name`` (str), ``site`` (str, default "ontop"),
        ``height`` (float, default 1.8), and optionally ``xy`` ([x, y]).
    min_site_distance : float
        Warning threshold (Angstrom) for inter-adsorbate distance.

    Returns
    -------
    model : Atoms
        Slab with all adsorbates placed.
    metadata : list of dict
        Per-adsorbate placement record.
    """
    if not specs:
        return atoms.copy(), []

    model = atoms.copy()
    result_meta: list = []
    adsorbate_indices: set = set()

    for idx, spec in enumerate(specs):
        name = str(spec["name"])
        site = str(spec.get("site", "ontop"))
        height = float(spec.get("height", 1.8))
        xy_spec = spec.get("xy")
        xy = (float(xy_spec[0]), float(xy_spec[1])) if xy_spec is not None else None

        if idx == 0:
            n_before = len(model)
            model = add_adsorbate(model, name, site=site, height=height, xy=xy)
            adsorbate_indices = set(range(n_before, len(model)))
            result_meta.append({
                "name": name, "site": site, "height": height,
                "placement": "custom_xy" if xy is not None else "direct",
            })
            continue

        # --- subsequent adsorbate: evaluate candidate sites ---
        mol_atoms, _ = build_molecule(name)
        mol_positions = mol_atoms.positions.copy()
        site_info = model.info.get("adsorbate_info", {}).get("sites", {})
        existing_pos = model.positions[sorted(adsorbate_indices)]

        if xy is not None:
            chosen_xy = xy
            placement_note = f"custom_xy ({xy[0]:.3f}, {xy[1]:.3f})"
        elif site in site_info:
            candidate_xy_list, surf_z = _expand_site_positions(model, site, height)
            best_xy = None
            best_min_dist = -1.0
            for cx, cy in candidate_xy_list:
                candidate_positions = mol_positions + np.array([cx, cy, surf_z + height])
                _, dists = get_distances(
                    candidate_positions, existing_pos,
                    cell=model.cell, pbc=model.pbc,
                )
                min_d = float(np.min(dists))
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_xy = (cx, cy)
            chosen_xy = best_xy
            placement_note = (
                f"auto ({len(candidate_xy_list)} {site} candidates, "
                f"best min dist {best_min_dist:.3f} A)"
            )
        else:
            chosen_xy = None
            placement_note = f"ase_default ({site})"

        n_before = len(model)
        if chosen_xy is not None:
            model = add_adsorbate(model, name, site="custom_xyz",
                                  height=height, xy=chosen_xy)
        else:
            model = add_adsorbate(model, name, site=site, height=height, xy=None)

        new_indices = set(range(n_before, len(model)))

        # --- distance check ---
        actual_min = float("inf")
        if adsorbate_indices and new_indices:
            for i in new_indices:
                for j in adsorbate_indices:
                    actual_min = min(actual_min,
                                     model.get_distance(i, j, mic=True))
        actual_min = float(actual_min)

        meta = {
            "name": name, "site": site, "height": height,
            "placement": placement_note,
            "min_inter_adsorbate_distance_A": float(round(actual_min, 3)),
        }
        if actual_min < min_site_distance:
            meta["warning"] = (
                f"inter-adsorbate distance {actual_min:.2f} A "
                f"below threshold {min_site_distance} A"
            )

        result_meta.append(meta)
        adsorbate_indices |= new_indices

    return model, result_meta


def _expand_site_positions(atoms: Atoms, site_name: str,
                           height: float) -> tuple:
    """Expand fractional site coordinates across the supercell.

    ASE stores site positions as fractional coordinates inside the
    primitive surface cell together with the primitive cell vectors.
    This helper enumerates every supercell image that fits within the
    slab's xy cell and returns the Cartesian (x, y) of each together
    with the surface z coordinate used for vertical placement.

    Returns
    -------
    positions : list of (x, y) tuples
    surf_z : float
    """
    info = atoms.info.get("adsorbate_info", {})
    site_frac = np.array(info["sites"][site_name], dtype=float)
    prim_cell = np.array(info["cell"], dtype=float)  # 2x2

    slab_cell_2d = atoms.cell[:2, :2]
    try:
        rep_matrix = slab_cell_2d @ np.linalg.inv(prim_cell)
    except np.linalg.LinAlgError:
        return [], 0.0
    nx = max(1, int(round(abs(rep_matrix[0, 0]))))
    ny = max(1, int(round(abs(rep_matrix[1, 1]))))

    positions = []
    for i in range(nx):
        for j in range(ny):
            spos = site_frac + np.array([float(i), float(j)])
            xy = np.dot(spos, prim_cell)
            positions.append((float(xy[0]), float(xy[1])))

    top_idx = info.get("top layer atom index", None)
    if top_idx is not None:
        surf_z = float(atoms.positions[int(top_idx), 2])
    else:
        surf_z = float(max(atoms.positions[:, 2]))

    return positions, surf_z


def _ase_site_name(site: str, atoms: Atoms) -> str:
    """Validate an ASE surface site name against what the slab knows."""
    valid = {"ontop", "bridge", "hollow", "fcc", "hcp", "longbridge", "shortbridge"}
    if site in valid:
        return site
    if site == "custom_xyz":
        raise ValueError("custom_xyz placement requires xy coordinates")
    available = sorted((atoms.info.get("adsorbate_info") or {}).get("sites", {}))
    raise ValueError(f"Unsupported site {site!r}. ASE-known sites for this slab: {available}")


# ─── LAMMPS data support ────────────────────────────────────────────

# CHARMM/OPLS atom type to element mapping (for LAMMPS data parsing)
CHARMM_TYPE_MAP = {
    # CHARMM water_ions types
    'HT': 'H', 'HX': 'H',  # Hydrogen in water, OH-
    'OT': 'O', 'OX': 'O',  # Oxygen in water, OH-
    'SOD': 'Na',
    'POT': 'K',
    'CLA': 'Cl',
    'LIT': 'Li',
    'CES': 'Cs',
    'MG': 'Mg',
    'CAL': 'Ca',
    'ZN': 'Zn',
    'BAR': 'Ba',
    'CAD': 'Cd',
    'RUB': 'Rb',
    # CHARMM carb/organic types
    'CC331': 'C', 'CC321': 'C', 'CC321C': 'C', 'CC322': 'C', 'CC322C': 'C',
    'CC311': 'C', 'CC312': 'C', 'CC312C': 'C', 'CC301': 'C',
    'CC3161': 'C', 'CC3162': 'C', 'CC3163': 'C', 'CC3161A': 'C', 'CC3161B': 'C', 'CC3161C': 'C',
    'CC3151': 'C', 'CC3152': 'C', 'CC3153': 'C', 'CC3051': 'C',
    'CC3062': 'C', 'CC3062A': 'C', 'CC3062B': 'C', 'CC3251': 'C', 'CC3261': 'C', 'CC3263': 'C',
    'CC331A': 'C', 'CC2O1': 'C', 'CC2O2': 'C', 'CC2O2A': 'C', 'CC2O2B': 'C',
    'CC2O3': 'C', 'CC2O4': 'C', 'CC2O4A': 'C', 'CC2O5': 'C',
    'HCP1': 'H', 'HCP1M': 'H', 'HCA1': 'H', 'HCA1C2': 'H',
    'HCA2': 'H', 'HCA2C2': 'H', 'HCA3': 'H', 'HCR1': 'H',
    'OC311M': 'O', 'OC311': 'O', 'OC312': 'O', 'OC301': 'O', 'OC302': 'O', 'OC30P': 'O',
    'OC3C61': 'O', 'OC3C51': 'O', 'OC3C5M': 'O',
    'OC2D1': 'O', 'OC2D2': 'O', 'OC2D3': 'O', 'OC2D4': 'O', 'OC2DP': 'O',
    'NC2D1': 'N', 'PCA': 'P', 'PCB': 'P', 'SC': 'S',
    # CHARMM phosphate types (water_ions)
    'PP': 'P', 'PPA': 'P', 'PPB': 'P',
    'OP': 'O', 'OPH': 'O', 'OPC': 'O',
    'HP': 'H', 'HPC': 'H',
    # OPLS types
    'hw': 'H', 'ho': 'H',
    'ow4': 'O', 'oh3': 'O',
    'na+': 'Na', 'k+': 'K', 'cl-': 'Cl',
    'SODIUM': 'Na', 'POTASSIUM': 'K', 'CHLORIDE': 'Cl', 'HYDROXIDE': 'O',
    # OPLS-AA 2024 atom types
    'br-': 'Br', 'br1': 'Br', 'c2na': 'C', 'c3=': 'C', 'c3a': 'C', 'c3a+': 'C',
    'c3a51': 'C', 'c3a53': 'C', 'c3ab': 'C', 'c3am': 'C', 'c3as': 'C', 'c3aso': 'C',
    'c3im': 'C', 'c3o': 'C', 'c3oa': 'C', 'c3oe': 'C', 'c3oh': 'C', 'c3ou': 'C',
    'c4': 'C', 'c4br': 'C', 'c4cl': 'C', 'c4f': 'C', 'c4i': 'C', 'c4n': 'C',
    'c4ns': 'C', 'c4o': 'C', 'c4o2': 'C', 'c4oe': 'C', 'c4p': 'C', 'c4pm': 'C',
    'c4s': 'C', 'c4sa': 'C', 'cl1': 'Cl', 'cs+': 'Cs', 'f-': 'F', 'f1': 'F',
    'h1': 'H', 'h1=': 'H', 'h1a': 'H', 'h1cn': 'H', 'h1co': 'H', 'h1cp': 'H',
    'h1cs': 'H', 'h1n': 'H', 'h1n3': 'H', 'h1n4': 'H', 'h1o': 'H', 'h1oh': 'H',
    'h1r': 'H', 'h1s': 'H', 'h1si': 'H', 'h1w': 'H', 'i-': 'I', 'i1': 'I',
    'k+': 'K', 'li+': 'Li', 'mg+2': 'Mg', 'sr+2': 'Sr',
    'n1': 'N', 'n3': 'N', 'n3a': 'N', 'n3am0': 'N', 'n3am1': 'N', 'n3am2': 'N',
    'n3ch': 'N', 'n3ch2': 'N', 'n3h': 'N', 'n3h2': 'N', 'n3im1': 'N',
    'n3sa1': 'N', 'n3sa2': 'N', 'n3ur2': 'N', 'n4': 'N',
    'n4h': 'N', 'n4h2': 'N', 'n4h3': 'N', 'n4h4': 'N',
    'o1=': 'O', 'o1a': 'O', 'o1h-': 'O', 'o1p': 'O', 'o1p=': 'O',
    'o1s': 'O', 'o1s=': 'O', 'o2e': 'O', 'o2h': 'O', 'o2ha': 'O',
    'o2hh': 'O', 'o2ht': 'O', 'o2p': 'O', 'o2s': 'O', 'o2w': 'O',
    'p41': 'P', 'p42': 'P', 'p43': 'P',
    's2': 'S', 's2h': 'S', 's2h2': 'S', 's3': 'S', 's4': 'S',
    's4am': 'S', 's4c': 'S', 'si4': 'Si',
    # PCFF types
    'o*': 'O', 'ow': 'O', 'oh3': 'O', 'oh2': 'O', 'oh1': 'O',
    'c': 'C', 'c_1': 'C', 'c_2': 'C', 'c_3': 'C', 'c_4': 'C',
    'cp': 'C', 'c=': 'C', 'c#': 'C', 'ca': 'C', 'cb': 'C',
    'h': 'H', 'h1': 'H', 'h2': 'H', 'hw': 'H', 'ho': 'H', 'hn': 'H',
    'n': 'N', 'n_1': 'N', 'n_2': 'N', 'n_3': 'N', 'n_4': 'N',
    'na': 'N', 'nb': 'N', 'nc': 'N', 'nd': 'N', 'ne': 'N', 'nh': 'N', 'no': 'N',
    's': 'S', 's_1': 'S', 's_2': 'S', 's_o': 'S', 's_o2': 'S',
    'p': 'P', 'p_1': 'P', 'p_2': 'P', 'p_3': 'P',
    'f': 'F', 'cl': 'Cl', 'br': 'Br', 'i': 'I',
    'na+': 'Na', 'k+': 'K', 'cl-': 'Cl', 'li+': 'Li', 'cs+': 'Cs',
    'mg+2': 'Mg', 'ca+2': 'Ca', 'zn+2': 'Zn',
}

# Van der Waals radii (A) for overlap detection
VDW_RADIUS = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
    'S': 1.80, 'Cl': 1.75, 'K': 2.75, 'Ca': 2.31, 'Li': 1.82,
    'Cs': 3.43, 'Fe': 2.04, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39,
    'Pt': 1.75, 'Pd': 1.63, 'Au': 1.66, 'Ag': 1.72, 'W': 2.10,
}


# ─── Packmol electrolyte builder ─────────────────────────────────────

IONS_CONFIG_PATH = SKILL_ROOT / "configs" / "ions.json"

# TIP3P water geometry (O-H bond 0.9572 A, angle 104.52 deg)
WATER_POSITIONS = [
    (0.0000, 0.0000, 0.0000),    # O
    (0.9572, 0.0000, 0.0000),    # H1
    (-0.2400, 0.9266, 0.0000),   # H2
]


def _load_ions_config() -> dict:
    """Load ion SMILES definitions from configs/ions.json."""
    with open(IONS_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _generate_molecule_xyz(name: str, smiles: str = None, tmp_dir: Path = None) -> Path:
    """Generate XYZ file for a molecule/ion using RDKit or hardcoded geometry.

    Returns path to the XYZ file.
    """
    if name == "H2O":
        return _write_xyz_file("H2O", ["O", "H", "H"], WATER_POSITIONS, tmp_dir)

    ions_config = _load_ions_config()
    smiles_map = ions_config.get("smiles_map", {})

    # Check monatomic ions with direct symbol (no RDKit needed)
    monatomic = ions_config.get("monatomic", {})
    if name in monatomic and "symbol" in monatomic[name]:
        symbol = monatomic[name]["symbol"]
        return _write_xyz_file(name, [symbol], [(0.0, 0.0, 0.0)], tmp_dir)

    # Check polyatomic ions with hardcoded positions first
    polyatomic = ions_config.get("polyatomic", {})
    if name in polyatomic and "positions" in polyatomic[name]:
        ion = polyatomic[name]
        return _write_xyz_file(name, ion["symbols"], ion["positions"], tmp_dir)

    # Use SMILES (from config or provided) + RDKit
    if smiles is None:
        smiles = smiles_map.get(name)
    if smiles:
        return _rdkit_xyz_from_smiles(name, smiles, tmp_dir)

    # Single atom fallback
    symbol = name.replace("+", "").replace("-", "").replace("2", "").replace("3", "")
    return _write_xyz_file(name, [symbol], [(0.0, 0.0, 0.0)], tmp_dir)


def _write_xyz_file(name: str, symbols: list, positions: list, tmp_dir: Path) -> Path:
    """Write an XYZ file for Packmol."""
    safe_name = name.replace("+", "plus").replace("-", "minus").replace("(", "").replace(")", "")
    path = tmp_dir / f"{safe_name}.xyz"
    lines = [str(len(symbols)), f"{name}"]
    for sym, pos in zip(symbols, positions):
        lines.append(f"{sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rdkit_xyz_from_smiles(name: str, smiles: str, tmp_dir: Path) -> Path:
    """Generate 3D XYZ from SMILES using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise RuntimeError(
            f"RDKit required for SMILES '{smiles}'. "
            "Install with: conda install -c conda-forge rdkit"
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"RDKit could not embed 3D conformer for: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
    ]
    return _write_xyz_file(name, symbols, positions, tmp_dir)


def _ion_packmol_box(x_length, y_length, z_min, z_max, margin, z_fraction):
    """Compute ion placement box with lateral margins."""
    x_margin = min(max(margin, 0.5), max(0.5, x_length / 2.0 - 0.25))
    y_margin = min(max(margin, 0.5), max(0.5, y_length / 2.0 - 0.25))
    z_span = z_max - z_min
    return (
        x_margin, y_margin, z_min + z_fraction[0] * z_span,
        max(x_margin + 0.25, x_length - x_margin),
        max(y_margin + 0.25, y_length - y_margin),
        z_min + z_fraction[1] * z_span,
    )


def _format_box(box):
    return " ".join(f"{v:.4f}" for v in box)


def build_electrolyte_packmol(
    atoms: Atoms,
    electrolyte_cfg: dict,
    output_dir: str,
) -> Atoms:
    """Build water + ions + solvents using Packmol with layered constraints.

    Strategy:
    [1] Place ions inside box (with lateral margin + z_fraction constraints)
    [2] Place water inside box (covers full region, surrounds ions)
    [3] Place organic solvents inside box (optional)

    Parameters
    ----------
    atoms : Atoms
        Surface slab with possible adsorbates.
    electrolyte_cfg : dict
        Config keys: water_count, ions, solvents, tolerance, seed,
        ion_lateral_margin, ion_z_fraction, z_offset, layer_height.
    output_dir : str
        Directory for Packmol working files and output.

    Returns
    -------
    Atoms
        Slab with packed electrolyte atoms added.
    """
    water_count = electrolyte_cfg.get('water_count', 0)
    ions = electrolyte_cfg.get('ions', {})
    solvents = electrolyte_cfg.get('solvents', {})
    # Handle legacy ethanol_count
    ethanol_count = electrolyte_cfg.get('ethanol_count', 0)
    if ethanol_count > 0 and 'ETOH' not in solvents:
        solvents['ETOH'] = ethanol_count

    tolerance = electrolyte_cfg.get('tolerance', 2.0)
    seed = electrolyte_cfg.get('seed', 12345)
    ion_lateral_margin = electrolyte_cfg.get('ion_lateral_margin', 2.5)
    ion_z_fraction = electrolyte_cfg.get('ion_z_fraction', (0.2, 0.8))
    z_offset = electrolyte_cfg.get('z_offset', 3.0)
    layer_height = electrolyte_cfg.get('layer_height', 10.0)

    molecule_counts = {}
    if water_count > 0:
        molecule_counts['H2O'] = water_count
    for ion_name, count in ions.items():
        if count > 0:
            molecule_counts[ion_name] = count
    for solv_name, count in solvents.items():
        if count > 0:
            molecule_counts[solv_name] = count

    if not molecule_counts:
        return atoms.copy()

    # Find Packmol executable
    packmol_bin = shutil.which("packmol")
    if packmol_bin is None:
        # Try python-packmol Scripts location
        scripts_candidate = Path(sys.executable).parent / "Scripts" / "packmol.exe"
        if scripts_candidate.exists():
            packmol_bin = str(scripts_candidate)
        else:
            raise RuntimeError(
                "Packmol not found. Install with: pip install packmol"
            )

    model = atoms.copy()
    x_length, y_length, _ = model.cell.lengths()
    top_z = float(max(model.positions[:, 2]))
    z_min = top_z + z_offset
    z_max = z_min + layer_height

    # Ion box: constrained with margins
    ion_box = _ion_packmol_box(x_length, y_length, z_min, z_max,
                               ion_lateral_margin, ion_z_fraction)
    # Water/solvent box: full region
    water_box = (0.5, 0.5, z_min, max(0.6, x_length - 0.5),
                 max(0.6, y_length - 0.5), z_max)

    work_dir = Path(output_dir) / "packmol_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Generate molecule XYZ files
    molecule_files = {}
    for name in molecule_counts:
        xyz_path = _generate_molecule_xyz(name, tmp_dir=work_dir)
        molecule_files[name] = xyz_path

    # Generate Packmol input
    input_lines = [
        f"tolerance {tolerance:.4f}",
        "filetype xyz",
        "output packed.xyz",
        f"seed {seed}",
        "",
    ]

    # Water first (covers full region)
    if water_count > 0:
        input_lines.extend([
            f"structure {molecule_files['H2O'].name}",
            f"  number {water_count}",
            f"  inside box {_format_box(water_box)}",
            "end structure",
            "",
        ])

    # Solvents (full region)
    for solv_name, count in solvents.items():
        if count > 0 and solv_name in molecule_files:
            input_lines.extend([
                f"structure {molecule_files[solv_name].name}",
                f"  number {count}",
                f"  inside box {_format_box(water_box)}",
                "end structure",
                "",
            ])

    # Ions last (constrained box)
    for ion_name, count in ions.items():
        if count > 0 and ion_name in molecule_files:
            input_lines.extend([
                f"structure {molecule_files[ion_name].name}",
                f"  number {count}",
                f"  inside box {_format_box(ion_box)}",
                "end structure",
                "",
            ])

    input_path = work_dir / "packmol.inp"
    output_path = work_dir / "packed.xyz"
    input_path.write_text("\n".join(input_lines), encoding="utf-8")

    # Run Packmol
    print(f"  Running Packmol: {sum(molecule_counts.values())} molecules "
          f"({', '.join(f'{v} {k}' for k, v in molecule_counts.items())})...")
    result = subprocess.run(
        [packmol_bin, "-i", str(input_path)],
        cwd=str(work_dir),
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"Packmol failed.\nreturncode: {result.returncode}\n"
            f"stdout:\n{result.stdout[-500:]}\nstderr:\n{result.stderr[-500:]}"
        )

    packed = ase_read(str(output_path), format="xyz")
    model += packed
    model.cell = atoms.cell
    model.pbc = atoms.pbc
    print(f"  Packmol completed: {len(packed)} atoms added, total {len(model)}")
    return model


# ─── ASE LJ relaxation ─────────────────────────────────────────────

def relax_structure(atoms: Atoms, steps: int = 1000, fmax: float = 0.5) -> Atoms:
    """Run short LJ relaxation to remove close contacts from Packmol.

    Uses ASE's built-in Lennard-Jones calculator (no force field file needed).
    Fixes the bottom slab region and only relaxes electrolyte atoms.
    """
    if steps <= 0:
        return atoms.copy()

    from ase.calculators.lj import LennardJones
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS

    relaxed = atoms.copy()
    relaxed.calc = LennardJones(epsilon=0.01, sigma=2.5)

    # Identify slab atoms: the bottom portion by z-coordinate
    # Heuristic: atoms below the median z are "slab" and get fixed
    z_values = relaxed.positions[:, 2]
    z_median = np.median(z_values)
    slab_indices = [i for i, z in enumerate(z_values) if z < z_median]

    if slab_indices:
        relaxed.constraints = [FixAtoms(indices=slab_indices)]

    print(f"  Relaxing {len(relaxed) - len(slab_indices)} electrolyte atoms "
          f"({len(slab_indices)} slab atoms fixed)...")
    dyn = BFGS(relaxed, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    final_fmax = float(np.max(np.abs(relaxed.get_forces())))
    print(f"  Relaxation done in {dyn.nsteps} steps, final max force: {final_fmax:.4f} eV/A")
    return relaxed


def parse_lammps_data(data_path: str) -> tuple:
    """Parse LAMMPS data file and return (symbols, positions, cell, mol_ids).

    Unlike ASE's read(), this preserves CHARMM atom type names (SOD, OX, HX)
    and maps them to correct element symbols. Also returns molecule IDs for
    proper grouping of multi-atom molecules and single-atom ions.

    Args:
        data_path: Path to LAMMPS data file

    Returns:
        Tuple of (symbols, positions, cell, mol_ids)
    """
    with open(data_path, 'r') as f:
        lines = f.readlines()

    # Parse header
    natoms = 0
    xlo, xhi = 0.0, 0.0
    ylo, yhi = 0.0, 0.0
    zlo, zhi = 0.0, 0.0
    atom_types = {}  # type_id -> element symbol

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if 'atoms' in line and not line.startswith('#'):
            natoms = int(line.split()[0])
        elif 'atom types' in line:
            ntypes = int(line.split()[0])
        elif 'xlo' in line and 'xhi' in line:
            parts = line.split()
            xlo, xhi = float(parts[0]), float(parts[1])
        elif 'ylo' in line and 'yhi' in line:
            parts = line.split()
            ylo, yhi = float(parts[0]), float(parts[1])
        elif 'zlo' in line and 'zhi' in line:
            parts = line.split()
            zlo, zhi = float(parts[0]), float(parts[1])
        elif line.strip() == 'Masses':
            # Parse mass section to get atom types
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('Atoms') or line.startswith('Bonds'):
                    break
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            type_id = int(parts[0])
                            type_name = parts[-1].replace('#', '').strip()
                            # Map to element
                            element = CHARMM_TYPE_MAP.get(type_name, type_name)
                            atom_types[type_id] = element
                        except (ValueError, IndexError):
                            pass  # Skip malformed lines
                i += 1
            continue
        elif line.strip() == 'Atoms':
            # Parse atom section
            i += 1
            symbols = []
            positions = []
            mol_ids = []
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('Bonds') or line.startswith('Velocities') or line.startswith('Angles'):
                    break
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            atom_id = int(parts[0])
                            mol_id = int(parts[1])
                            type_id = int(parts[2])
                            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                            element = atom_types.get(type_id, 'X')
                            symbols.append(element)
                            positions.append([x, y, z])
                            mol_ids.append(mol_id)
                        except (ValueError, IndexError):
                            pass  # Skip malformed lines
                i += 1
            break
        i += 1

    cell = [[xhi - xlo, 0, 0], [0, yhi - ylo, 0], [0, 0, zhi - zlo]]
    return symbols, positions, cell, mol_ids


# ─── Element sorting & distance helpers ──────────────────────────────

# Preferred order for POSCAR grouping: metals, then main-group, then H
_ELEMENT_PRIORITY = {
    'Pt': 0, 'Pd': 1, 'Au': 2, 'Ag': 3, 'Cu': 4, 'Ni': 5, 'Co': 6,
    'Fe': 7, 'Mo': 8, 'W': 9, 'Ti': 10, 'Zr': 11, 'Al': 12, 'Si': 13,
    'P': 14, 'S': 15, 'Cl': 16, 'Na': 17, 'K': 18, 'Ca': 19, 'Mg': 20,
    'Li': 21, 'Cs': 22, 'Br': 23, 'F': 24, 'I': 25, 'C': 26, 'N': 27,
    'O': 28, 'H': 29,
}


def _group_by_symbol(atoms: Atoms) -> Atoms:
    """Sort atoms by element for proper POSCAR grouping.

    Metals first, then main-group, then H.  Within each group the original
    order is preserved.
    """
    symbols = atoms.get_chemical_symbols()
    sort_indices = sorted(range(len(symbols)),
                         key=lambda i: _ELEMENT_PRIORITY.get(symbols[i], 100))
    return atoms[sort_indices]


def min_distance(atoms: Atoms) -> float:
    """Return the minimum inter-atomic pair distance (Angstrom, MIC)."""
    if len(atoms) < 2:
        return 0.0
    distances = atoms.get_all_distances(mic=True)
    # Zero out diagonal
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


# ─── Report & metadata ──────────────────────────────────────────────

def _plain_metadata(value):
    """Convert numpy types to plain Python for JSON-safe reports."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain_metadata(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_metadata(v) for v in value)
    if isinstance(value, list):
        return [_plain_metadata(v) for v in value]
    return value


def _report(atoms: Atoms, metadata: dict) -> str:
    """Generate a human-readable model_report.txt."""
    from collections import Counter
    lines = ["DFTbuild model report"]
    for key, value in (metadata or {}).items():
        lines.append(f"{key}: {_plain_metadata(value)}")
    counts = Counter(atoms.get_chemical_symbols())
    lines.append(f"formula: {atoms.get_chemical_formula()}")
    lines.append("element_counts: " + ", ".join(
        f"{k}={counts[k]}" for k in sorted(counts)))
    lengths = atoms.cell.lengths()
    lines.append(f"cell_lengths_angstrom: {lengths[0]:.4f}, "
                 f"{lengths[1]:.4f}, {lengths[2]:.4f}")
    md = min_distance(atoms)
    lines.append(f"min_distance_angstrom: {md:.4f}")
    lines.append("warnings: " +
                 ("close_contacts_below_0.7A" if md < 0.7 else "none"))
    return "\n".join(lines) + "\n"


# ─── Model output ───────────────────────────────────────────────────

def write_model_package(atoms: Atoms, output_dir: str,
                        metadata: dict | None = None) -> dict:
    """Write POSCAR, XYZ, CIF, LAMMPS data, and model_report.txt.

    Returns a dict of paths for logging.
    """
    from ase.io import write as ase_write
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped = _group_by_symbol(atoms)

    poscar = out / "POSCAR"
    xyz = out / "structure.xyz"
    cif = out / "structure.cif"
    data = out / "structure.data"
    report = out / "model_report.txt"

    ase_write(str(poscar), grouped, format='vasp', vasp5=True)
    ase_write(str(xyz), grouped, format='xyz')
    ase_write(str(cif), grouped, format='cif')
    ase_write(str(data), grouped, format='lammps-data')
    report.write_text(_report(grouped, metadata or {}), encoding='utf-8')

    paths = {
        'POSCAR': str(poscar),
        'XYZ': str(xyz),
        'CIF': str(cif),
        'LAMMPS': str(data),
        'report': str(report),
    }
    for label, p in paths.items():
        print(f"  {label:8s}: {p}")
    return paths


# ─── Main build pipeline ────────────────────────────────────────────

def _build_surface(surface_cfg: dict) -> Atoms:
    """Dispatch surface construction based on config type."""
    stype = surface_cfg.get('type', 'slab')

    if stype == 'nanotube':
        atoms = build_nanotube(
            surface_cfg['n'], surface_cfg['m'],
            surface_cfg.get('length', 3),
            surface_cfg.get('vacuum', 15.0))
    elif stype == 'heterojunction':
        film_cfg = surface_cfg.get('film')
        sub_cfg = surface_cfg.get('substrate')
        if film_cfg and sub_cfg:
            atoms = build_heterojunction_pymatgen(
                film_cfg, sub_cfg, surface_cfg.get('vacuum', 15.0))
        else:
            raise ValueError(
                "Heterojunction config requires 'film' and 'substrate' keys")
    elif stype == 'slab':
        atoms = build_slab(
            surface_cfg['element'],
            tuple(surface_cfg['miller']),
            tuple(surface_cfg['size']),
            surface_cfg['layers'],
            surface_cfg.get('vacuum', 15.0))
    else:
        raise ValueError(f"Unknown surface type: {stype!r}")

    atoms.pbc = True
    return atoms


def build(config_path: str) -> dict:
    """Main build pipeline.

    Steps:
      1. Build surface (slab / nanotube / heterojunction)
      2. Alloy substitution (optional)
      3. Add adsorbates (optional)
      4. Build electrolyte with Packmol (optional)
      5. LJ relaxation to remove close contacts (optional)
      6. Write output package (POSCAR / XYZ / CIF / LAMMPS data / report)

    Returns dict of output file paths.
    """
    config = load_config(config_path)
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    metadata = {'config': os.path.basename(config_path)}

    # ── Step 1: Surface ──────────────────────────────────────────────
    surface_cfg = config.get('surface', {})
    if not surface_cfg:
        raise ValueError("Config must contain a 'surface' section")
    print("[1/6] Building surface...")
    atoms = _build_surface(surface_cfg)
    print(f"  {len(atoms)} atoms, formula {atoms.get_chemical_formula()}")

    # ── Step 2: Alloy ────────────────────────────────────────────────
    alloy_cfg = config.get('alloy')
    if alloy_cfg:
        print("[2/6] Applying alloy substitution...")
        symbols = alloy_cfg.get('symbols', [])
        top_layers = alloy_cfg.get('top_layers', 1)
        fractions = alloy_cfg.get('fractions')
        if fractions:
            fractions = {k: float(v) for k, v in fractions.items()}
        atoms, changed = alloy_surface_layers(
            atoms, symbols, top_layers=top_layers, fractions=fractions)
        print(f"  Alloyed: {changed}")
    else:
        print("[2/6] No alloy specified, skipping...")

    # ── Step 3: Adsorbates ───────────────────────────────────────────
    ads_cfg = config.get('adsorbate', {})
    ads_list = ads_cfg.get('list', [])
    if ads_list:
        print("[3/6] Adding adsorbates...")
        atoms, ads_meta = add_adsorbates(atoms, ads_list)
        metadata['adsorbates'] = ads_meta
        print(f"  {len(atoms)} atoms after adsorbate placement")
    else:
        print("[3/6] No adsorbates specified, skipping...")

    # ── Step 4: Electrolyte (Packmol) ────────────────────────────────
    elec_cfg = config.get('electrolyte', {})
    has_electrolyte = bool(
        elec_cfg.get('water_count', 0) > 0 or
        elec_cfg.get('ions') or
        elec_cfg.get('solvents') or
        elec_cfg.get('ethanol_count', 0) > 0)

    if has_electrolyte:
        print("[4/6] Building electrolyte with Packmol...")
        atoms = build_electrolyte_packmol(atoms, elec_cfg, output_dir)
        metadata['electrolyte'] = {
            k: v for k, v in elec_cfg.items()
            if k not in ('relax_steps',)
        }
    else:
        print("[4/6] No electrolyte specified, skipping...")

    # ── Step 5: LJ relaxation ────────────────────────────────────────
    relax_steps = elec_cfg.get('relax_steps', 1000)
    if has_electrolyte and relax_steps > 0:
        print("[5/6] Running LJ relaxation...")
        atoms = relax_structure(atoms, steps=relax_steps)
        metadata['relax_steps'] = relax_steps
    else:
        print("[5/6] No relaxation needed, skipping...")

    # ── Step 6: Output ───────────────────────────────────────────────
    print("[6/6] Writing output files...")
    paths = write_model_package(atoms, output_dir, metadata=metadata)
    print("\nDone!")
    return paths


# ─── CLI entry point ────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/build.py <config.json>")
        sys.exit(1)
    build(sys.argv[1])
