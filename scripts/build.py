#!/usr/bin/env python
"""
EMCbuild v3 - Template-based approach.

Strategy:
1. ASE builds surface + adsorbates (good at site control)
2. Copy a verified EMC build.emc template, replace parameters
3. Run EMC directly to build water layer
4. Merge surface + water into final structure

This matches the user's workflow: pick an EMC example template, edit parameters, run.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc100, fcc111, nanotube, molecule
from ase.io import read, write


def _detect_emc():
    """Auto-detect EMC installation.

    Priority: EMC_ROOT env var → ~/.emcbuild/emc/ → OS defaults.
    """
    if "EMC_ROOT" in os.environ:
        emc_root = Path(os.environ["EMC_ROOT"])
    else:
        home_emc = Path.home() / ".emcbuild" / "emc"
        if home_emc.exists():
            emc_root = home_emc
        else:
            system = platform.system()
            if system == "Windows":
                emc_root = Path("E:/emcwin32/v9.4.4")
            elif system == "Darwin":
                emc_root = Path("/Applications/emc")
            else:
                emc_root = Path("/usr/local/emc")

    system = platform.system()
    if system == "Windows":
        bin_name = "emc_win32.exe"
    elif system == "Darwin":
        bin_name = "emc_macos"
    else:
        bin_name = "emc_linux"

    emc_bin = emc_root / "bin" / bin_name
    if not emc_bin.exists():
        # Try any emc* binary in the bin directory
        bin_dir = emc_root / "bin"
        if bin_dir.exists():
            candidates = sorted(bin_dir.glob("emc*"))
            if candidates:
                emc_bin = candidates[0]
                print(f"  EMC binary auto-detected: {emc_bin}")

    return emc_root, emc_bin


EMC_ROOT, EMC_BIN = _detect_emc()

# Templates - all bundled with the skill for portability
SKILL_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = SKILL_ROOT / "templates"
FIELDS_DIR = SKILL_ROOT / "fields"

# Template selection map
TEMPLATE_MAP = {
    'pcff': TEMPLATES_DIR / "build_solvent_pcff.emc",
    'charmm': TEMPLATES_DIR / "build_water_charmm.emc",
    'opls': TEMPLATES_DIR / "build_water_opls.emc",
}

MOLECULE_TEMPLATE = TEMPLATES_DIR / "build_molecule_pcff.emc"


def _find_project_root(config_path: str) -> str:
    """Find project root by walking up to git root, falling back to CWD."""
    try:
        import subprocess as _sp
        result = _sp.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=os.path.dirname(os.path.abspath(config_path)),
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return os.getcwd()


def load_config(config_path: str) -> dict:
    """Load config and resolve output_dir.

    Default output: <project_root>/EMCbuild/<config_folder_name>/
    Project root = git root (if in a repo) or CWD.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = _find_project_root(config_path)

    # Use config's parent folder name as the project subfolder
    config_name = os.path.basename(config_dir)

    if 'output_dir' not in config:
        config['output_dir'] = os.path.join(project_root, 'EMCbuild', config_name)
    else:
        if not os.path.isabs(config['output_dir']):
            # Relative path: resolve relative to project root
            config['output_dir'] = os.path.join(project_root, config['output_dir'])

    return config


# ─── Structure loading (pymatgen) ────────────────────────────────────

# Materials Project API key (loaded from config or environment)
_MP_API_KEY = os.environ.get("MP_API_KEY", None)
_DB_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "databases.json"


def _load_mp_api_key() -> str:
    """Load Materials Project API key from config or environment."""
    global _MP_API_KEY
    if _MP_API_KEY:
        return _MP_API_KEY
    if _DB_CONFIG_PATH.exists():
        with open(_DB_CONFIG_PATH, 'r', encoding='utf-8') as f:
            db_cfg = json.load(f)
        _MP_API_KEY = db_cfg.get('materials_project', {}).get('api_key', '')
    return _MP_API_KEY


def _get_pubchem_structure(cid: int):
    """Download 3D structure from PubChem using curl.

    Args:
        cid: PubChem Compound ID

    Returns:
        ASE Atoms object
    """
    import subprocess
    import tempfile
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
    from ase import Atoms

    phi = (1 + np.sqrt(5)) / 2

    # Generate 60 vertices of truncated icosahedron
    vertices = []

    # Even permutations of (0, ±1, ±3phi)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            base = [0, s1, s2 * 3 * phi]
            vertices.extend([[base[0], base[1], base[2]],
                           [base[1], base[2], base[0]],
                           [base[2], base[0], base[1]]])

    # Even permutations of (±2, ±(1+2phi), ±phi)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                base = [s1 * 2, s2 * (1 + 2 * phi), s3 * phi]
                vertices.extend([[base[0], base[1], base[2]],
                               [base[1], base[2], base[0]],
                               [base[2], base[0], base[1]]])

    # Even permutations of (±1, ±(2+phi), ±2phi)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                base = [s1, s2 * (2 + phi), s3 * 2 * phi]
                vertices.extend([[base[0], base[1], base[2]],
                               [base[1], base[2], base[0]],
                               [base[2], base[0], base[1]]])

    coords = np.array(vertices)
    coords = np.unique(np.round(coords, 10), axis=0)

    # Center and scale to correct radius (C60 radius ~3.55 Å)
    coords -= coords.mean(axis=0)
    dists = np.linalg.norm(coords, axis=1)
    scale = 3.55 / dists.mean()
    coords *= scale

    # Create ASE Atoms with a large unit cell (molecule in vacuum)
    atoms = Atoms('C60', positions=coords)
    atoms.center(vacuum=15.0)  # Add 15 Å vacuum around the molecule
    atoms.pbc = True

    return atoms


def _get_structure(source: str):
    """Load crystal structure from various sources.

    Supported sources (priority order):
      - "mp-xxx"        : Materials Project ID (oxides, compounds)
      - "cod_xxxxx"     : COD (Crystallography Open Database) ID (organic crystals)
      - "pubchem_xxx"   : PubChem CID (molecules, uses curl)
      - "fullerene_c60" : Programmatic C60 generation
      - "/path/to/POSCAR" : Local POSCAR file
      - "/path/to/file.cif" : Local CIF file
      - "Pt:fcc:3.92"   : Manual specification (element:lattice_type:lattice_constant)
    """
    from pymatgen.core import Structure, Lattice

    # Materials Project
    if source.startswith('mp-'):
        from mp_api.client import MPRester
        api_key = _load_mp_api_key()
        if not api_key:
            raise ValueError("Materials Project API key not found. "
                             "Set MP_API_KEY env var or configs/databases.json")
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


# ─── Surface builders (ASE) ──────────────────────────────────────────

LATTICE = {
    'Pt': 3.92, 'Pd': 3.89, 'Au': 4.08, 'Ag': 4.09,
    'Cu': 3.61, 'Ni': 3.52, 'Fe': 2.87, 'W': 3.16, 'Mo': 3.15
}


def build_slab(element, miller, size, layers, vacuum):
    a = LATTICE.get(element, 3.8)
    m = tuple(miller)
    if m == (1,0,0):
        return fcc100(element, size=(size[0], size[1], layers), a=a, vacuum=vacuum)
    if m == (1,1,1):
        return fcc111(element, size=(size[0], size[1], layers), a=a, vacuum=vacuum)
    from ase.build import surface, bulk
    b = bulk(element, crystalstructure='fcc', a=a)
    s = surface(b, m, layers, vacuum=vacuum)
    return s.repeat((size[0], size[1], 1))


def build_nanotube(n, m, length, vacuum):
    return nanotube(n, m, length=length, vacuum=vacuum)


def build_heterojunction(bottom, top_cfg, distance, vacuum):
    """Build bottom slab + CeO2 (fluorite) on top."""
    bl = bottom
    b_slab = build_slab(bl['element'], bl['miller'], bl['size'], bl['layers'], 0)

    # CeO2 fluorite unit cell
    a = 5.41
    ce = [[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]]
    ox = [[.25,.25,.25],[.75,.75,.25],[.75,.25,.75],[.25,.75,.75],
          [.25,.25,.75],[.75,.75,.75],[.75,.25,.25],[.25,.75,.25]]
    pos = [[p[0]*a, p[1]*a, p[2]*a] for p in ce+ox]
    topo = Atoms('Ce4O8', positions=pos)
    topo.cell = [a, a, a]
    topo.pbc = True
    ts = top_cfg.get('size', [3,3])
    tl = top_cfg.get('layers', 2)
    topo = topo.repeat((ts[0], ts[1], tl))

    offset = max(b_slab.positions[:,2]) - min(topo.positions[:,2]) + distance
    topo.positions[:,2] += offset

    hetero = b_slab + topo
    cell = hetero.cell.copy()
    cell[2,2] = max(hetero.positions[:,2]) + vacuum
    hetero.cell = cell
    hetero.pbc = True
    return hetero


def build_heterojunction_pymatgen(film_cfg: dict, substrate_cfg: dict,
                                   vacuum: float = 15.0):
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
          f"cell={[round(x,2) for x in interface.lattice.abc]}")

    # Convert to ASE Atoms
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(interface)
    atoms.pbc = True

    return atoms


def build_molecule(name, smiles=None, builder=None, output_dir=None):
    """Build a molecule from SMILES or name.

    Priority:
    1. File path: file:xxx
    2. EMC/PCFF builder (builder="emc") — for complex organics
    3. SMILES string via ASE
    4. ASE built-in molecule library
    5. Hardcoded fallback for common molecules
    """
    # 1. File path
    if name.startswith('file:'):
        filepath = name[5:]
        mol = read(filepath)
        print(f"  Read '{name}' from file: {filepath}")
        return mol

    # 2. EMC/PCFF builder for complex organic molecules
    if builder == 'emc':
        if not smiles:
            raise ValueError(
                f"SMILES required with builder='emc' for molecule '{name}'"
            )
        if not output_dir:
            import tempfile
            output_dir = tempfile.mkdtemp(prefix='emcbuild_molecule_')
        return build_molecule_emc(name, smiles, output_dir)

    # Method 3: Use SMILES if provided
    if smiles:
        try:
            mol = molecule(smiles)
            print(f"  Built '{name}' from SMILES: {smiles}")
            return mol
        except Exception as e:
            print(f"  Warning: SMILES '{smiles}' failed: {e}, falling back...")

    # Method 2: Try ASE built-in molecule library
    try:
        mol = molecule(name)
        print(f"  Built '{name}' from ASE built-in library")
        return mol
    except Exception:
        pass

    # Method 3: Hardcoded fallback for common molecules
    molecules = {
        'NH3': Atoms('NH3', positions=[[0,0,0],[0.94,0,0.45],[-0.47,0.81,0.45],[-0.47,-0.81,0.45]]),
        'H2O': Atoms('OH2', positions=[[0,0,0],[0.76,0.58,0],[-0.76,0.58,0]]),
        'CO':  Atoms('CO',  positions=[[0,0,0],[0,0,1.13]]),
        'OH':  Atoms('OH',  positions=[[0,0,0],[0,0,0.97]]),
        'NH2': Atoms('NH2', positions=[[0,0,0],[0.92,0,0.55],[-0.92,0,0.55]]),
    }
    if name in molecules:
        return molecules[name]

    print(f"  Error: Cannot build molecule '{name}'")
    return Atoms(name, positions=[[0,0,0]])


def build_molecule_emc(name: str, smiles: str, output_dir: str,
                        field: str = 'pcff') -> Atoms:
    """Build a single molecule conformation using EMC/PCFF.

    Uses build_molecule_pcff.emc template with ultra-low density to create
    an isolated-molecule box that avoids PBC self-interaction.

    Args:
        name: Molecule name (used for EMC group id)
        smiles: SMILES string
        output_dir: Directory for EMC working files
        field: Force field (default 'pcff')

    Returns:
        ASE Atoms with optimized 3D geometry
    """
    import time

    work_dir = Path(output_dir) / "molecule_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    build_emc = work_dir / "build_molecule.emc"
    shutil.copy2(str(MOLECULE_TEMPLATE), build_emc)

    # Fix force field paths
    fields_abs = str(FIELDS_DIR).replace('\\', '/')
    with open(build_emc, 'r', encoding='latin-1') as f:
        content = f.read()
    content = content.replace('"fields/', f'"{fields_abs}/')
    # Shrink box: density=0.05 → box ~20Å (molecule ~8Å + 2×cutoff 9.5Å)
    content = content.replace('density1\t-> 0.001', 'density1\t-> 0.05')

    # Generate a safe group id from name
    group_id = name.replace('+', 'plus').replace('-', 'minus').replace('(', '').replace(')', '')
    group_id = ''.join(c if c.isalnum() or c == '_' else '_' for c in group_id)
    var_prefix = 'mol'

    # __OUTPUT__
    content = content.replace('__OUTPUT__', 'molecule')

    # __VARS__
    vars_block = f'  f_{var_prefix}\t-> 1,\n  chem_{var_prefix}\t-> "{smiles}"'
    content = content.replace('__VARS__', vars_block)

    # __GROUPS__
    groups_block = (
        f'  group\t\t-> {{\n'
        f'    id\t\t-> {group_id},\n'
        f'    depth\t-> 8,\n'
        f'    chemistry\t-> chem_{var_prefix}\n'
        f'  }}'
    )
    content = content.replace('__GROUPS__', groups_block)

    # __SIZING__
    sizing_lines = [
        f'  lg_{var_prefix}\t-> nsites({group_id}),',
        f'  l_{var_prefix}\t-> lg_{var_prefix},',
        f'  mg_{var_prefix}\t-> mass({group_id}),',
        f'  m_{var_prefix}\t-> mg_{var_prefix},',
        f'  f_{var_prefix}\t-> f_{var_prefix}*l_{var_prefix}/m_{var_prefix},',
        f'  norm\t\t-> f_{var_prefix},',
        f'  f_{var_prefix}\t-> f_{var_prefix}/norm,',
        f'  n_{var_prefix}\t-> int(f_{var_prefix}*ntotal/l_{var_prefix}+0.5),',
        f'  ntotal\t-> 0,',
        f'  mtotal\t-> 0',
    ]
    content = content.replace('__SIZING__', '\n'.join(sizing_lines) + '\n')

    # __CLUSTERS__
    clusters_block = (
        f'  cluster\t-> {{\n'
        f'    id\t\t-> {group_id}, system -> main, group -> {group_id}, n -> 1\n'
        f'  }}'
    )
    content = content.replace('__CLUSTERS__', clusters_block)

    # __CLUSTER_NAMES__
    content = content.replace('__CLUSTER_NAMES__', group_id)

    with open(build_emc, 'w', encoding='latin-1') as f:
        f.write(content)

    # Run EMC
    print(f"  Building '{name}' with EMC/PCFF (SMILES: {smiles})...")
    result = subprocess.run(
        [str(EMC_BIN), str(build_emc.resolve())],
        cwd=str(work_dir),
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"EMC failed for '{name}' (returncode={result.returncode})\n"
            f"stderr: {result.stderr[:500] if result.stderr else 'none'}"
        )

    # Find and read output
    pdb_path = None
    for attempt in range(15):
        time.sleep(0.5)
        for candidate in [
            work_dir / 'molecule.pdb.gz',
            work_dir / 'molecule.pdb',
            work_dir / 'name.pdb.gz',
            work_dir / 'name.pdb',
        ]:
            if candidate.exists():
                pdb_path = candidate
                break
        if pdb_path:
            break

    if not pdb_path:
        found = list(work_dir.glob('*'))
        raise FileNotFoundError(
            f"EMC output not found for '{name}' in {work_dir}. "
            f"Files present: {[f.name for f in found]}"
        )

    mol = read(str(pdb_path))

    # Center molecule in the box to avoid periodic image artifacts
    # EMC with unwrap=true may place atoms outside the cell
    cell = mol.cell
    box_center = np.array([cell[0, 0] / 2.0, cell[1, 1] / 2.0, cell[2, 2] / 2.0])
    mol_center = mol.positions.mean(axis=0)
    shift = box_center - mol_center
    mol.positions += shift
    mol.pbc = False

    print(f"  Built '{name}': {len(mol)} atoms, "
          f"formula {mol.get_chemical_formula()}")
    return mol


def add_adsorbates(atoms, ads_list, output_dir=None):
    """Place adsorbates on the surface at ontop sites.

    Placement logic (generic, works for any molecule):
    - If 'element' is specified, only consider atoms of that element
    - Find top-layer surface atoms
    - Prefer interior atoms over boundary atoms (avoid corners/edges)
    - For each adsorbate, pick an ontop site (directly above a surface atom)
    - z position: surface_atom_z + height
    """
    cell = atoms.cell
    cx = cell[0, 0] / 2.0
    cy = cell[1, 1] / 2.0

    # Sort by distance to cell center using minimum image convention (PBC)
    def dist_to_center_pbc(idx):
        pos = atoms.positions[idx]
        dx = (pos[0] - cx) % cell[0, 0]
        dy = (pos[1] - cy) % cell[1, 1]
        # minimum image
        if dx > cell[0, 0] / 2:
            dx -= cell[0, 0]
        if dy > cell[1, 1] / 2:
            dy -= cell[1, 1]
        return dx**2 + dy**2

    symbols = atoms.get_chemical_symbols()

    for i, ads in enumerate(ads_list):
        smiles = ads.get('smiles', None)
        builder = ads.get('builder', None)
        mol = build_molecule(ads['name'], smiles=smiles,
                            builder=builder, output_dir=output_dir)
        height = ads.get('height', 1.8)
        target_element = ads.get('element', None)

        # Determine candidate atoms
        if target_element:
            # Filter by element (for heterojunctions: only exposed side)
            candidate_indices = [j for j, s in enumerate(symbols)
                                if s == target_element]
            if not candidate_indices:
                print(f"  Warning: no {target_element} atoms found, skipping {ads['name']}")
                continue
        else:
            candidate_indices = list(range(len(atoms)))

        # Find surface-layer among candidates
        # For heterojunctions: detect if element is substrate (bottom) or film (top)
        z_coords = atoms.positions[candidate_indices, 2]
        all_z = atoms.positions[:, 2]
        mid_z = (all_z.min() + all_z.max()) / 2.0
        avg_candidate_z = np.mean(z_coords)

        if target_element and avg_candidate_z < mid_z:
            # Substrate element: exposed surface is at the BOTTOM
            surf_z = min(z_coords)
            # Place adsorbate BELOW the surface (in the vacuum direction)
            z_sign = -1
        else:
            # Film or simple slab: exposed surface is at the TOP
            surf_z = max(z_coords)
            z_sign = 1

        surface_indices = [candidate_indices[j] for j, z in enumerate(z_coords)
                          if abs(z - surf_z) < 0.5]

        # Sort by distance to cell center (prefer interior, avoid boundaries)
        surface_indices.sort(key=dist_to_center_pbc)

        # Pick surface atom closest to center
        site_idx = surface_indices[i % len(surface_indices)]
        surface_pos = atoms.positions[site_idx]

        # Place adsorbate: above surface (film/slab) or below surface (substrate)
        mol.translate([surface_pos[0], surface_pos[1], surf_z + z_sign * height])
        atoms += mol

    return atoms


# ─── EMC water layer (template-based) ────────────────────────────────

def _sed(build_emc, old, new):
    """Replace string in EMC build file (cross-platform)."""
    with open(build_emc, 'r', encoding='latin-1') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(build_emc, 'w', encoding='latin-1') as f:
        f.write(content)



def _sed_append(build_emc, pattern, lines):
    """Append lines after a line containing the pattern in the EMC build file.

    Uses Python file I/O with latin-1 encoding to preserve EMC compatibility.
    """
    with open(build_emc, 'r', encoding='latin-1') as f:
        file_lines = f.readlines()

    new_lines_text = '\n'.join(lines) + '\n'
    result = []
    for line in file_lines:
        result.append(line)
        if pattern in line:
            result.append(new_lines_text)

    with open(build_emc, 'w', encoding='latin-1') as f:
        f.writelines(result)


def build_water_layer(surface_xyz: str, electrolyte_cfg: dict,
                      output_dir: str, surface_cell_xy: tuple = None) -> str:
    """Build water+ion+solvent layer using EMC with template placeholder approach.

    Template uses __PLACEHOLDER__ markers that are replaced here.
    Supports any solvent in the CHARMM carb force field.
    """
    water_count = electrolyte_cfg.get('water_count', 0)
    # Backward compat: accept 'ethanol_count', prefer 'solvents' dict
    solvents = electrolyte_cfg.get('solvents', {})
    ethanol_count = electrolyte_cfg.get('ethanol_count', 0)
    if ethanol_count > 0 and 'ETOH' not in solvents:
        solvents['ETOH'] = ethanol_count
    density = electrolyte_cfg.get('density', 1.0)
    temperature = electrolyte_cfg.get('temperature', 300)
    ions = electrolyte_cfg.get('ions', {})
    field = electrolyte_cfg.get('field', None)
    is_charmm = field and field.startswith('charmm')
    is_opls = field and field.startswith('opls')
    is_pcff = field and field.startswith('pcff')
    total_solvent = sum(solvents.values())
    total_ions = sum(ions.values())

    # Carb field's wildcard can override ion charges when no water is present.
    # Adding a single water molecule ensures correct dual-field loading order.
    if water_count == 0 and total_solvent > 0 and total_ions > 0 and is_charmm:
        water_count = 1
        print(f"  Auto-added 1 H2O to balance dual-field loading")

    if water_count == 0 and total_solvent == 0 and total_ions == 0:
        return surface_xyz

    # Solvent SMILES reference — full carb field TEMPLATES from EMC v9.4.4 carb.top
    # 91 residues: simple solvents + sugars + sugar derivatives + inositols + alditols
    # SMILES are EMC-native (tolerant parser: explicit H, : for aromatic bonds)
    SOLVENT_SMILES = {
        'AALD': 'HC(=O)C', 'AALL': 'C1(O)C(O)C(O)C(O)C(O1)CO',
        'AALT': 'C1(O)C(O)C(O)C(O)C(O1)CO', 'AARB': 'O1C(O)C(O)C(O)C1CO',
        'ABEQ': 'C1(O)C(O)CC(O)C(O1)C', 'ACO': 'O=C(C)C',
        'ADEO': 'O1C(O)CC(O)C1CO', 'AFRU': 'O1C(C(O)C(O)C1CO)(O)CO',
        'AFUC': 'C1(O)C(O)C(O)C(O)C(O1)C', 'AGAL': 'C1(O)C(O)C(O)C(O)C(O1)CO',
        'AGALNA': 'C1(O)C(NC(=O)C)C(O)C(O)C(O1)CO',
        'AGLC': 'C1(O)C(O)C(O)C(O)C(O1)CO', 'AGLCA': 'C1(O)C(O)C(O)C(O)C(O1)C(:O):O',
        'AGLCNA': 'C1(O)C(NC(=O)C)C(O)C(O)C(O1)CO',
        'AGUL': 'C1(O)C(O)C(O)C(O)C(O1)CO', 'AIDO': 'C1(O)C(O)C(O)C(O)C(O1)CO',
        'AIDOA': 'C1(O)C(O)C(O)C(O)C(O1)C(:O):O',
        'ALLOSE': 'C(=O)C(O)C(O)C(O)C(O)CO', 'ALYF': 'O1C(O)C(O)C(O)C1CO',
        'AMAN': 'C1(O)C(O)C(O)C(O)C(O1)CO', 'AMOL': 'C(C(:O):O)(O)(OC)C',
        'AMOP': 'C(C(:O):O)(OC)C',
        'ANE5AC': 'C(:O)(:O)C1(O)CC(O)C(NC(=O)C)C(O1)C(O)C(O)CO',
        'ARHM': 'C1(O)C(O)C(O)C(O)C(O1)C', 'ARHMOA': 'C1(O)C(OC(=O)C)C(O)C(O)C(O1)C',
        'ARIB': 'O1C(O)C(O)C(O)C1CO', 'ATAL': 'C1(O)C(O)C(O)C(O)C(O1)CO',
        'AXYF': 'O1C(O)C(O)C(O)C1CO', 'AXYL': 'C1(O)C(O)C(O)C(O)CO1',
        'BALL': 'C1(O)C(O)C(O)C(O)C(O1)CO', 'BALT': 'C1(O)C(O)C(O)C(O)C(O1)CO',
        'BARB': 'O1C(O)C(O)C(O)C1CO', 'BDEO': 'O1C(O)CC(O)C1CO',
        'BFRU': 'O1C(C(O)C(O)C1CO)(O)CO', 'BFUC': 'C1(O)C(O)C(O)C(O)C(O1)C',
        'BGAL': 'C1(O)C(O)C(O)C(O)C(O1)CO',
        'BGALNA': 'C1(O)C(NC(=O)C)C(O)C(O)C(O1)CO',
        'BGLC': 'C1(O)C(O)C(O)C(O)C(O1)CO', 'BGLCA': 'C1(O)C(O)C(O)C(O)C(O1)C(:O):O',
        'BGLCNA': 'C1(O)C(NC(=O)C)C(O)C(O)C(O1)CO',
        'BGUL': 'C1(O)C(O)C(O)C(O)C(O1)CO', 'BIDO': 'C1(O)C(O)C(O)C(O)C(O1)CO',
        'BIDOA': 'C1(O)C(O)C(O)C(O)C(O1)C(:O):O', 'BLYF': 'O1C(O)C(O)C(O)C1CO',
        'BMAN': 'C1(O)C(O)C(O)C(O)C(O1)CO',
        'BNE5AC': 'C(:O)(:O)C1(O)CC(O)C(NC(=O)C)C(O1)C(O)C(O)CO',
        'BRHM': 'C1(O)C(O)C(O)C(O)C(O1)C', 'BRIB': 'O1C(O)C(O)C(O)C1CO',
        'BTAL': 'C1(O)C(O)C(O)C(O)C(O1)CO', 'BXYF': 'O1C(O)C(O)C(O)C1CO',
        'BXYL': 'C1(O)C(O)C(O)C(O)CO1', 'CYHX': 'C1CCCCC1', 'CYHX2': 'C1CCCCC1',
        'DALTOL': 'C(O)C(O)C(O)C(O)C(O)CO', 'DARAOL': 'C(O)C(O)C(O)C(O)CO',
        'DGLUOL': 'C(O)C(O)C(O)C(O)C(O)CO', 'DGULOL': 'C(O)C(O)C(O)C(O)C(O)CO',
        'DIDIOL': 'C(O)C(O)C(O)C(O)C(O)CO', 'DMANOL': 'C(O)C(O)C(O)C(O)C(O)CO',
        'DTHROL': 'C(O)C(O)C(O)CO', 'EGLY': 'C(O)CO', 'ETOH': 'C(O)C',
        'INI1': 'C1(O)C(O)C(O)C(O)C(O)C1O', 'INI2': 'C1(O)C(O)C(O)C(O)C(O)C1O',
        'INI3': 'C1(O)C(O)C(O)C(O)C(O)C1O', 'INI4': 'C1(O)C(O)C(O)C(O)C(O)C1O',
        'INI5': 'C1(O)C(O)C(O)C(O)C(O)C1O', 'IPAA': 'CC(=O)NC(C)C',
        'LALTOL': 'C(O)C(O)C(O)C(O)C(O)CO', 'LARAOL': 'C(O)C(O)C(O)C(O)CO',
        'LGLUOL': 'C(O)C(O)C(O)C(O)C(O)CO', 'LGULOL': 'C(O)C(O)C(O)C(O)C(O)CO',
        'LIDIOL': 'C(O)C(O)C(O)C(O)C(O)CO', 'LMANOL': 'C(O)C(O)C(O)C(O)C(O)CO',
        'LTHROL': 'C(O)C(O)C(O)CO', 'MALLOL': 'C(O)C(O)C(O)C(O)C(O)CO',
        'MEOH': 'CO', 'MERYOL': 'C(O)C(O)C(O)CO', 'MGALOL': 'C(O)C(O)C(O)C(O)C(O)CO',
        'MGLYOL': 'C(O)C(O)CO', 'MRIBOL': 'C(O)C(O)C(O)C(O)CO',
        'MXYLOL': 'C(O)C(O)C(O)C(O)CO', 'NMA': 'CC(=O)NC', 'PRO2': 'C(C)(O)C',
        'PROA': 'C(C(:O):O)C', 'PSICOS': 'C(O)C(=O)C(O)C(O)C(O)CO',
        'TF2M': 'O1C(C)CCC1', 'THF': 'O1CCCC1', 'THF2': 'O1CCCC1',
        'THP': 'O1CCCCC1', 'THP2': 'C1CCCCO1',
        # OPLS-AA 2024 specific (also accessible when field='opls')
        'SPCE': 'O', 'ETDO': 'C(O)CO', 'PRTO': 'C(O)C(O)CO',
        'BUDO': 'CC(O)C(O)C', 'BUTO': 'C(O)C(O)C(O)C', 'PRDO': 'C(O)C(O)C',
        'DMSO': 'CS(=O)C', 'EACE': 'CC(OCC)=O', 'FORM': 'C(=O)N',
        'UREA': 'NC(=O)N', 'NMAM': 'CC(=O)NC', 'IMID': 'CC(=O)NC(=O)C',
        'PHEN': 'c1ccccc1O', 'TOLU': 'c1ccccc1C', 'ANIL': 'c1ccccc1N',
        'THPH': 'c1ccccc1S', 'BZCL': 'c1ccccc1Cl', 'BZTF': 'c1ccccc1C(F)(F)F',
        'ACAC': 'C(:O)(:O)C', 'ACAH': 'C(=O)(O)C', 'ACHY': 'CC=O',
        'MEAN': 'NC', 'DMAN': 'N(C)C', 'TMAN': 'N(C)(C)C', 'ETAN': 'NCC',
        'DMET': 'COC', 'EMET': 'COCC', 'DMME': 'COCOC', 'MMAE': 'COCO',
        'CPRO': 'C1CC1', 'EOXI': 'O1CC1', 'PROE': 'C=CC', 'METH': 'C',
        'H2S': 'S', 'DMDS': 'CSSC', 'MESH': 'CS', 'EMES': 'CSCC',
        'ETCL': 'CCCl', 'ETBR': 'CCBr', 'ETIO': 'CCI', 'EECL': 'C=CCl',
        'ETOF': 'C(F)(F)(F)CO', 'MACE': 'CC(OC)=O', 'MEAO': 'COC(O)C',
        'FOAC': 'C(:O):O', 'FOAH': 'C(O)=O',
    }

    ION_MAP = {
        'charmm': {
            'Na+': ('SOD', '[Na+]'), 'K+': ('POT', '[K+]'),
            'Cl-': ('CLA', '[Cl-]'), 'OH-': ('OH', '[O-](H)'),
            'Li+': ('LIT', '[Li+]'), 'Cs+': ('CES', '[Cs+]'),
            'Mg2+': ('MG', '[Mg+2]'), 'Ca2+': ('CAL', '[Ca+2]'),
            'Ba2+': ('BAR', '[Ba+2]'), 'Rb+': ('RUB', '[Rb+]'),
            'Zn2+': ('ZN2', '[Zn+2]'), 'Cd2+': ('CD2', '[Cd+2]'),
            'PO4(3-)': ('PO4', 'P(:O)(:O)(:O):O'),
            'HPO4(2-)': ('HPO4', 'P(:O)(:O)(:O)O'),
            'H2PO4(-)': ('H2PO4', 'P(O)(:O)(:O)O'),
        },
        'opls': {
            'Na+': ('na+', '[Na+]'), 'K+': ('k+', '[K+]'),
            'Cl-': ('cl-', '[Cl-]'), 'OH-': ('HYDR', '[O-]'),
            'Li+': ('li+', '[Li+]'), 'Cs+': ('cs+', '[Cs+]'),
            'Mg2+': ('mg+2', '[Mg+2]'), 'Sr2+': ('sr+2', '[Sr+2]'),
            'F-': ('f-', '[F-]'), 'Br-': ('br-', '[Br-]'), 'I-': ('i-', '[I-]'),
        },
        'pcff': {
            'Na+': ('na+', '[>Na]'), 'K+': ('K', '[>K]'),
            'Cl-': ('Cl', '[>Cl]'), 'OH-': ('OH', '[O-](H)'),
        },
        'default': {
            'Na+': ('Na', '[Na]'), 'K+': ('K', '[K]'),
            'Cl-': ('Cl', '[Cl]'), 'OH-': ('OH', '[O-](H)'),
        }
    }
    ion_map = ION_MAP.get(field, ION_MAP['default']) if field else ION_MAP['default']

    # ---- Build component list: (var_prefix, group_id, smiles, count) ----
    component_list = []

    if is_charmm:
        if water_count > 0:
            component_list.append(('tip3', 'TIP3', 'O(H)(H)', water_count))
        for solv_name, count in solvents.items():
            smiles = solv_name if solv_name not in SOLVENT_SMILES else SOLVENT_SMILES[solv_name]
            if ':' in solv_name:
                solv_name, smiles = solv_name.split(':', 1)
            component_list.append((solv_name, solv_name, smiles, count))
        for ion_name, count in ions.items():
            grp_name, smiles = ion_map.get(ion_name, (ion_name, ion_name))
            component_list.append((grp_name, grp_name, smiles, count))
    elif is_opls:
        if water_count > 0:
            component_list.append(('spce', 'SPCE', 'O', water_count))
        for solv_name, count in solvents.items():
            smiles = solv_name if solv_name not in SOLVENT_SMILES else SOLVENT_SMILES[solv_name]
            if ':' in solv_name:
                solv_name, smiles = solv_name.split(':', 1)
            component_list.append((solv_name, solv_name, smiles, count))
        for ion_name, count in ions.items():
            grp_name, smiles = ion_map.get(ion_name, (ion_name, ion_name))
            # Sanitize var_prefix: OPLS atom types contain +/-
            var_prefix = grp_name.replace('+', 'p').replace('-', 'm')
            component_list.append((var_prefix, grp_name, smiles, count))
    else:
        # PCFF or unspecified — uses placeholder template
        if water_count > 0:
            component_list.append(('water', 'water', 'O', water_count))
        for solv_name, count in solvents.items():
            smiles = solv_name if solv_name not in SOLVENT_SMILES else SOLVENT_SMILES[solv_name]
            if ':' in solv_name:
                solv_name, smiles = solv_name.split(':', 1)
            component_list.append((solv_name, solv_name, smiles, count))
        for ion_name, count in ions.items():
            grp_name, smiles = ion_map.get(ion_name, (ion_name, ion_name))
            component_list.append((grp_name.lower(), grp_name.lower(), smiles, count))

    if not component_list:
        return surface_xyz

    # Select template
    default_template = TEMPLATES_DIR / "build_solvent_pcff.emc"
    template_path = TEMPLATE_MAP.get(field, default_template) if field else default_template
    build_emc = os.path.join(output_dir, "build_water.emc")
    shutil.copy2(str(template_path), build_emc)

    # Fix relative field paths for PCFF template
    fields_abs = str(FIELDS_DIR).replace('\\', '/')
    with open(build_emc, 'r', encoding='latin-1') as f:
        content = f.read()
    content = content.replace('"fields/', f'"{fields_abs}/')
    with open(build_emc, 'w', encoding='latin-1') as f:
        f.write(content)

    # Calculate total sites for ntotal
    sites_per = {
        # Water
        'water': 3, 'TIP3': 3, 'TP3M': 3, 'SPCE': 3,
        # Ions (monatomic)
        'SOD': 1, 'POT': 1, 'CLA': 1, 'OH': 2,
        'CES': 1, 'LIT': 1, 'MG': 1, 'CAL': 1,
        'BAR': 1, 'RUB': 1, 'ZN2': 1, 'CD2': 1,
        # Polyatomic ions
        'PO4': 5, 'HPO4': 6, 'H2PO4': 7,
        # Common organic solvents
        'ETOH': 9, 'MEOH': 6, 'THF': 13, 'EGLY': 10,
        'PRO2': 12, 'CYHX': 18, 'THP': 16, 'NMA': 12, 'IPAA': 18,
        'THF2': 13, 'CYHX2': 18, 'THP2': 16, 'TF2M': 16,
        # Sugars & sugar derivatives (aldopyranoses, ketopyranoses, furanoses)
        'AALL': 24, 'AALT': 24, 'AARB': 20, 'ABEQ': 22,
        'ADEO': 19, 'AFRU': 24, 'AFUC': 23, 'AGAL': 24,
        'AGALNA': 30, 'AGLC': 24, 'AGLCA': 22, 'AGLCNA': 30,
        'AGUL': 24, 'AIDO': 24, 'AIDOA': 22, 'ALLOSE': 24,
        'ALYF': 20, 'AMAN': 24, 'ANE5AC': 39, 'ARHM': 23,
        'ARHMOA': 28, 'ARIB': 20, 'ATAL': 24, 'AXYF': 20,
        'AXYL': 20, 'BALL': 24, 'BALT': 24, 'BARB': 20,
        'BDEO': 19, 'BFRU': 24, 'BFUC': 23, 'BGAL': 24,
        'BGALNA': 30, 'BGLC': 24, 'BGLCA': 22, 'BGLCNA': 30,
        'BGUL': 24, 'BIDO': 24, 'BIDOA': 22, 'BLYF': 20,
        'BMAN': 24, 'BNE5AC': 39, 'BRHM': 23, 'BRIB': 20,
        'BTAL': 24, 'BXYF': 20, 'BXYL': 20,
        # Alditols (linear polyols)
        'DALTOL': 26, 'DARAOL': 22, 'DGLUOL': 26, 'DGULOL': 26,
        'DIDIOL': 26, 'DMANOL': 26, 'DTHROL': 18,
        'LALTOL': 26, 'LARAOL': 22, 'LGLUOL': 26, 'LGULOL': 26,
        'LIDIOL': 26, 'LMANOL': 26, 'LTHROL': 18,
        'MALLOL': 26, 'MERYOL': 18, 'MGALOL': 26, 'MGLYOL': 14,
        'MRIBOL': 22, 'MXYLOL': 22,
        # Inositols & acids
        'INI1': 24, 'INI2': 24, 'INI3': 24, 'INI4': 24, 'INI5': 24,
        # Other
        'AALD': 7, 'ACO': 10, 'AMOL': 15, 'AMOP': 14,
        'PROA': 10, 'PSICOS': 24,
        # OPLS-AA 2024 specific
        'SPCE': 3, 'HYDR': 2, 'ETDO': 10, 'PRTO': 14, 'BUTO': 17, 'BUDO': 16, 'PRDO': 13,
        'DMSO': 10, 'EACE': 14, 'FORM': 6, 'UREA': 8, 'NMAM': 12, 'IMID': 14,
        'PHEN': 13, 'TOLU': 15, 'ANIL': 14, 'THPH': 13, 'BZCL': 12, 'BZTF': 15,
        'ACAC': 10, 'ACAH': 8, 'ACHY': 7, 'ACPH': 17, 'ACAN': 19, 'ACAM': 9,
        'MEAN': 7, 'DMAN': 10, 'TMAN': 13, 'ETAN': 10, 'DMAM': 15,
        'DMET': 9, 'EMET': 12, 'DMME': 13, 'MMAE': 10, 'DMEE': 16,
        'CPRO': 9, 'EOXI': 7, 'PROE': 9, 'METH': 5, 'H2S': 3,
        'DMDS': 10, 'MESH': 6, 'EMES': 12, 'MESI': 7,
        'ETCL': 8, 'ETBR': 8, 'ETIO': 8, 'EECL': 6, 'ETOF': 9,
        'MACE': 11, 'MEAO': 13, 'FOAC': 7, 'FOAH': 5,
        'BZHY': 14, 'BZAM': 16, 'BZUR': 20, 'BZNI': 13, 'BZPH': 32, 'BZPO': 20, 'BZSA': 17,
        'BNAH': 15, 'BNMA': 16, 'BNEA': 19, 'BNME': 19, 'BNMS': 19, 'BNAM': 17,
        'MBEN': 18, 'PACE': 18, 'DMAL': 20, 'MEAL': 17, 'THAN': 16, 'ANIS': 16,
        'DIPH': 22, 'NAPH': 18, 'DPHE': 23, 'DPHS': 25, 'DPSO': 24,
        'GUAN': 11, 'TMGU': 20, 'DMMP': 16, 'DMPO': 16, 'DMSO': 10,
        'BSAT': 18, 'BNAC': 25, 'BZSU': 22, 'BNSU': 19,
        'ESNA': 14, 'ESUL': 15, 'MSUL': 12, 'EPHO': 16, 'MPHO': 13, 'EMPO': 19,
        'ESAM': 13, 'MSAM': 10, 'NESA': 16, 'NMSA': 13,
        'RMIM': 22, 'EMCA': 16, 'ETCA': 13, 'MSME': 11,
    }
    ntotal = 0
    for var_prefix, group_id, smiles, count in component_list:
        ntotal += count * sites_per.get(group_id, (9 if len(smiles) > 3 else 1))

    # ---- Generate all template sections ----
    # Variables (2 lines per component: f_X and chem_X)
    var_lines = []
    total_var_lines = len(component_list) * 2
    line_idx = 0
    for var_prefix, group_id, smiles, count in component_list:
        for content_line in [f'  f_{var_prefix}\t-> {count}', f'  chem_{var_prefix}\t-> "{smiles}"']:
            line_idx += 1
            comma = ',' if line_idx < total_var_lines else ''
            var_lines.append(content_line + comma)

    # Groups
    grp_lines = []
    for i, (var_prefix, group_id, smiles, count) in enumerate(component_list):
        comma = ',' if i < len(component_list) - 1 else ''
        grp_lines.append(f'  group\t\t-> {{')
        grp_lines.append(f'    id\t\t-> {group_id},')
        grp_lines.append(f'    depth\t-> 8,')
        grp_lines.append(f'    chemistry\t-> chem_{var_prefix}')
        grp_lines.append(f'  }}{comma}')

    # Sizing
    siz_lines = []
    siz_lines.append('  (* lengths *)')
    for var_prefix, group_id, smiles, count in component_list:
        siz_lines.append(f'  lg_{var_prefix}\t-> nsites({group_id}),')
    for var_prefix, group_id, smiles, count in component_list:
        siz_lines.append(f'  l_{var_prefix}\t-> lg_{var_prefix},')
    siz_lines.append('  (* masses *)')
    for var_prefix, group_id, smiles, count in component_list:
        siz_lines.append(f'  mg_{var_prefix}\t-> mass({group_id}),')
    for var_prefix, group_id, smiles, count in component_list:
        siz_lines.append(f'  m_{var_prefix}\t-> mg_{var_prefix},')
    siz_lines.append('  (* mass fractions *)')
    for var_prefix, group_id, smiles, count in component_list:
        siz_lines.append(f'  f_{var_prefix}\t-> f_{var_prefix}*l_{var_prefix}/m_{var_prefix},')
    siz_lines.append('  (* normalization *)')
    norm_sum = 'f_' + '+f_'.join(vp for vp, _, _, _ in component_list)
    siz_lines.append(f'  norm\t\t-> {norm_sum},')
    for var_prefix, group_id, smiles, count in component_list:
        siz_lines.append(f'  f_{var_prefix}\t-> f_{var_prefix}/norm,')
    siz_lines.append('  (* sizing *)')
    for var_prefix, group_id, smiles, count in component_list:
        siz_lines.append(f'  n_{var_prefix}\t-> int(f_{var_prefix}*ntotal/l_{var_prefix}+0.5),')

    # Clusters — use explicit user-specified counts directly
    # (bypasses mass-fraction sizing to avoid heavy ions rounding to zero)
    clu_lines = []
    for i, (var_prefix, group_id, smiles, count) in enumerate(component_list):
        clu_lines.append(f'  cluster\t-> {{')
        clu_lines.append(f'    id\t\t-> {group_id}, system -> main, group -> {group_id}, n -> {count}')
        clu_lines.append(f'  }}{"," if i < len(component_list) - 1 else ""}')

    # Cluster names for build section
    clu_names = ', '.join(gid for _, gid, _, _ in component_list)

    # ---- Replace template placeholders ----
    content = content.replace('__OUTPUT__', 'water')
    if is_charmm:
        # water_ions first for water cases, carb first for solvent-only
        if True:  # always water_ions first (auto-water ensures >=1)
            fields_block = 'field\t\t= { mode -> charmm, name -> {wi_parameters, wi_topology}, compress -> false };\nfield\t\t= { mode -> charmm, name -> {carb_parameters, carb_topology}, compress -> false };'
        else:
            fields_block = 'field\t\t= { mode -> charmm, name -> {carb_parameters, carb_topology}, compress -> false };\nfield\t\t= { mode -> charmm, name -> {wi_parameters, wi_topology}, compress -> false };'
        content = content.replace('__FIELDS__', fields_block)
    content = content.replace('  __VARS__', '\n'.join(var_lines))
    content = content.replace('  __GROUPS__', '\n'.join(grp_lines))
    content = content.replace('  __SIZING__', '\n'.join(siz_lines) + '\n')
    content = content.replace('  __CLUSTERS__', '\n'.join(clu_lines))
    content = content.replace('__CLUSTER_NAMES__', clu_names)
    content = content.replace('ntotal\t-> 5000', f'ntotal\t-> {ntotal}')
    content = content.replace('density1\t-> 1.0', f'density1\t-> {density}')
    content = content.replace('temperature\t-> 300', f'temperature\t-> {temperature}')

    with open(build_emc, 'w', encoding='latin-1') as f:
        f.write(content)

    # ---- Set water box xy ----
    if surface_cell_xy is not None:
        lx, ly = surface_cell_xy
        # Water box xy should match surface dimensions
        # The z dimension will be determined by EMC based on density and molecule count
        box_lx = lx
        box_ly = ly
        _sed(build_emc, 'lxx\t\t-> lphase1', f'lxx\t\t-> {box_lx:.4f}')
        _sed(build_emc, 'lyy\t\t-> lbox', f'lyy\t\t-> {box_ly:.4f}')
        print(f"  Water box xy: {box_lx:.2f} x {box_ly:.2f} A (matches surface)")

    # ---- Run EMC ----
    total = water_count + total_solvent + total_ions
    parts = []
    if water_count > 0: parts.append(f"{water_count} H2O")
    for solv_name, count in solvents.items(): parts.append(f"{count} {solv_name}")
    if ions:
        parts.append(f"{sum(ions.values())} ions ({', '.join(ions.keys())})")
    print(f"  Running EMC to build {total} molecules ({', '.join(parts)})...")
    env = os.environ.copy()
    emc_bin = str(EMC_BIN.parent)
    if 'PATH' in env:
        env['PATH'] = emc_bin + ';' + env['PATH']
    else:
        env['PATH'] = emc_bin

    result = subprocess.run(
        [str(EMC_BIN), os.path.abspath(build_emc)],
        cwd=os.path.abspath(output_dir),
        capture_output=True, text=True, env=env
    )

    if result.returncode != 0:
        print(f"  EMC stderr: {result.stderr[:200] if result.stderr else 'none'}")
    else:
        print(f"  EMC completed successfully")

    # Wait for file system to sync and check for output
    import time
    for attempt in range(10):
        time.sleep(1.0)
        for name in ['water.data', 'name.data', 'water.lammps', 'name.lammps']:
            path = os.path.join(output_dir, name)
            if os.path.exists(path):
                print(f"  EMC water layer: {path}")
                return path

    print(f"  Warning: EMC output not found, using surface only")
    return surface_xyz


# ─── LAMMPS data parser ──────────────────────────────────────────────

# CHARMM/OPLS atom type to element mapping
CHARMM_TYPE_MAP = {
    # CHARMM types
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
    # CHARMM phosphate types (water_ions; HPP = Aug 2025 conversion naming)
    'PP': 'P', 'PPA': 'P', 'PPB': 'P',
    'OP': 'O', 'OPH': 'O', 'OPC': 'O',
    'HP': 'H', 'HPC': 'H', 'HPP': 'H',
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

# OPLS type map (subset of CHARMM_TYPE_MAP)
OPLS_TYPE_MAP = {}

# PCFF type map (subset of CHARMM_TYPE_MAP)
PCFF_TYPE_MAP = {}

# Van der Waals radii (Å) for overlap detection
VDW_RADIUS = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
    'S': 1.80, 'Cl': 1.75, 'K': 2.75, 'Ca': 2.31, 'Li': 1.82,
    'Cs': 3.43, 'Fe': 2.04, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39,
    'Pt': 1.75, 'Pd': 1.63, 'Au': 1.66, 'Ag': 1.72, 'W': 2.10,
}


def _remove_overlapping_species(combined, mol_groups, adsorbate_indices,
                                min_clearance=1.5):
    """Remove molecules/ions that overlap with adsorbate atoms.

    Uses mol_id-based grouping from LAMMPS data, so it works for any solvent
    (water, ethanol, etc.) and any ion (Na+, OH-, etc.) without hardcoding
    molecule sizes.

    Args:
        combined: Combined ASE Atoms object (surface + water + ions)
        mol_groups: List of (mol_id, atom_indices) tuples from LAMMPS mol_ids
        adsorbate_indices: Indices of adsorbate atoms in the combined system
        min_clearance: Minimum distance (Å) between species and adsorbate atoms

    Returns:
        keep_mask: Boolean mask per molecule/ion group (True = keep)
    """
    if not adsorbate_indices or not mol_groups:
        return [True] * len(mol_groups)

    ads_symbols = [combined[i].symbol for i in adsorbate_indices]

    keep_mask = []
    for mol_id, atom_indices in mol_groups:
        conflict = False
        for idx in atom_indices:
            w_sym = combined[idx].symbol
            for a_idx, a_sym in zip(adsorbate_indices, ads_symbols):
                r_sum = VDW_RADIUS.get(w_sym, 1.5) + VDW_RADIUS.get(a_sym, 1.5)
                dist = combined.get_distance(idx, a_idx, mic=True)
                if dist < r_sum * 0.7:
                    conflict = True
                    break
            if conflict:
                break
        keep_mask.append(not conflict)
    return keep_mask


def parse_lammps_data(data_path: str) -> tuple:
    """Parse LAMMPS data file and return (symbols, positions, cell, mol_ids).

    Unlike ASE's read(), this preserves CHARMM atom type names (SOD, OX, HX)
    and maps them to correct element symbols. Also returns molecule IDs for
    proper grouping of multi-atom molecules and single-atom ions.
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


def _write_lammps_data(filename: str, atoms: 'Atoms', mol_ids: list = None):
    """Write LAMMPS data file in atom_style atomic format.

    MACE and most ML potentials don't need mol_id or charge —
    atomic format (atom-ID atom-type x y z) is the safe default.

    Args:
        filename: Output file path
        atoms: ASE Atoms object
        mol_ids: List of molecule IDs (ignored in atomic format, kept for API compat)
    """
    from ase.data import atomic_numbers, atomic_masses, chemical_symbols

    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    cell = atoms.cell

    # Get unique elements and create type mapping
    unique_elements = sorted(set(symbols), key=lambda s: atomic_numbers.get(s, 0))
    type_map = {elem: i+1 for i, elem in enumerate(unique_elements)}

    # Write file
    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file (atom_style atomic) - written by EMCbuild\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(unique_elements)} atom types\n\n")
        f.write(f"0.0 {cell[0, 0]:.6f} xlo xhi\n")
        f.write(f"0.0 {cell[1, 1]:.6f} ylo yhi\n")
        f.write(f"0.0 {cell[2, 2]:.6f} zlo zhi\n\n")

        # Masses section
        f.write("Masses\n\n")
        for elem in unique_elements:
            z = atomic_numbers.get(elem, 1)
            mass = atomic_masses[z] if z < len(atomic_masses) else z
            f.write(f"{type_map[elem]} {mass:.4f}  # {elem}\n")
        f.write("\n")

        # Atoms section (atom_style atomic format)
        f.write("Atoms  # atomic\n\n")
        for i, (symbol, pos) in enumerate(zip(symbols, positions)):
            atom_type = type_map[symbol]
            # Format: atom-ID atom-type x y z
            f.write(f"{i+1} {atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        f.write("\n")


def merge_and_output(surface_xyz: str, water_data: str, output_dir: str,
                     bare_surf_top_z: float = None, gap: float = 1.5,
                     min_clearance: float = 1.5, water_side: str = 'top'):
    """Merge surface + water layer and write all output formats.

    Positioning logic:
    - Surface stays where it is
    - Water layer starts from bare_surf_top_z + gap (default 1.5 A above
      the metal surface), so the adsorbate is embedded inside the water layer
    - Water molecules overlapping with adsorbate atoms are removed
    - Water xy is centered on the surface cell
    - Final cell z = max atom z + 15.0 A vacuum

    Args:
        surface_xyz: Path to surface+adsorbate XYZ file
        water_data: Path to EMC water LAMMPS data file
        output_dir: Output directory
        bare_surf_top_z: Z-coordinate of bare metal surface (before adsorbates).
            If provided, water layer starts from this + gap.
        gap: Distance (A) from metal surface to water layer bottom (default 1.5)
        min_clearance: Minimum distance (A) between water and adsorbate atoms (default 1.5)

    Uses custom LAMMPS data parser to preserve CHARMM atom types (SOD, OX, HX).
    """
    surface = read(surface_xyz)

    if water_data != surface_xyz and os.path.exists(water_data):
        try:
            # Use custom parser to preserve CHARMM atom types
            w_symbols, w_positions, w_cell, w_mol_ids = parse_lammps_data(water_data)
            water = Atoms(symbols=w_symbols, positions=w_positions, cell=w_cell, pbc=True)

            # Surface boundaries
            surf_top_z = max(surface.positions[:, 2])
            surf_cell = surface.cell

            # Water boundaries (before translation)
            water_min_z = min(water.positions[:, 2])
            water_max_z = max(water.positions[:, 2])
            water_height = water_max_z - water_min_z
            water_cell = water.cell

            # Determine water position based on which side is exposed
            water_bottom_target = None
            water_top_target = None
            if water_side == 'bottom':
                # Substrate exposed at bottom: water layer goes below
                if bare_surf_top_z is not None:
                    water_top_target = bare_surf_top_z - gap
                else:
                    water_top_target = min(surface.positions[:, 2]) - gap
                z_shift = water_top_target - water_max_z
            else:
                # Film/slab exposed at top: water layer goes above (default)
                if bare_surf_top_z is not None:
                    water_bottom_target = bare_surf_top_z + gap
                else:
                    water_bottom_target = surf_top_z + gap
                z_shift = water_bottom_target - water_min_z

            # Center water xy on surface cell
            surf_cx = surf_cell[0, 0] / 2.0
            surf_cy = surf_cell[1, 1] / 2.0
            water_cx = water_cell[0, 0] / 2.0
            water_cy = water_cell[1, 1] / 2.0
            x_shift = surf_cx - water_cx
            y_shift = surf_cy - water_cy

            water.positions[:, 0] += x_shift
            water.positions[:, 1] += y_shift
            water.positions[:, 2] += z_shift

            # Combine (no wrap needed - water box xy matches surface cell)
            n_surf = len(surface)
            combined = surface + water
            # Create mol_ids: surface atoms get mol_id=1, water atoms keep their original mol_ids
            max_surf_mol_id = 1
            combined_mol_ids = [1] * n_surf + [mol_id + max_surf_mol_id for mol_id in w_mol_ids]

            # Add vacuum padding to avoid PBC artifacts
            bottom_vacuum = 10.0
            top_vacuum = 15.0

            # Shift all atoms so min_z = bottom_vacuum
            min_z = combined.positions[:, 2].min()
            combined.positions[:, 2] += (bottom_vacuum - min_z)

            # Set cell xy: use max(surface xy, water xy) to avoid spilling
            cell = surf_cell.copy()
            cell[0, 0] = max(surf_cell[0, 0], water_cell[0, 0])
            cell[1, 1] = max(surf_cell[1, 1], water_cell[1, 1])
            cell[2, 2] = max(combined.positions[:, 2]) + top_vacuum
            combined.cell = cell
            combined.pbc = True

            # Identify adsorbate atoms (not metals, near the exposed surface)
            if bare_surf_top_z is not None:
                METALS = {'Pt', 'Pd', 'Au', 'Ag', 'Cu', 'Ni', 'Fe', 'W',
                          'Mo', 'Al', 'Ti', 'Cr', 'Co', 'Zn', 'Sn'}
                if water_side == 'bottom':
                    # Substrate exposed: adsorbate is below the surface
                    adsorbate_indices = [
                        i for i in range(n_surf)
                        if (combined.positions[i, 2] < bare_surf_top_z - 0.5
                            and combined[i].symbol not in METALS)
                    ]
                else:
                    # Film/slab exposed: adsorbate is above the surface
                    adsorbate_indices = [
                        i for i in range(n_surf)
                        if (combined.positions[i, 2] > bare_surf_top_z + 0.5
                            and combined[i].symbol not in METALS)
                    ]

                # Group water+ion atoms by mol_id for proper molecule detection
                # Works for any solvent (water, ethanol, etc.) and any ion
                from collections import defaultdict
                mol_dict = defaultdict(list)
                for local_idx, mol_id in enumerate(w_mol_ids):
                    combined_idx = n_surf + local_idx
                    mol_dict[mol_id].append(combined_idx)
                mol_groups = list(mol_dict.items())

                # Remove molecules/ions overlapping with adsorbate
                keep_mask = _remove_overlapping_species(
                    combined, mol_groups, adsorbate_indices, min_clearance)

                # Rebuild combined system without overlapping species
                removed_count = keep_mask.count(False)
                if removed_count > 0:
                    keep_indices = []
                    for ki, (mol_id, atom_indices) in enumerate(mol_groups):
                        if keep_mask[ki]:
                            keep_indices.extend(atom_indices)
                    combined = surface + combined[keep_indices]
                    combined.cell = cell
                    combined.pbc = True
                    # Update mol_ids
                    combined_mol_ids = [1] * n_surf + [combined_mol_ids[i] for i in keep_indices]

                if water_side == 'bottom':
                    print(f"  Water positioned: z={water_top_target-water_height:.1f} to {water_top_target:.1f} A")
                else:
                    print(f"  Water positioned: z={water_bottom_target:.1f} to {water_bottom_target+water_height:.1f} A")
                print(f"  Bare surface: z={bare_surf_top_z:.1f} A, "
                      f"adsorbate top: z={surf_top_z:.1f} A, gap: {gap} A")
                if removed_count > 0:
                    print(f"  Removed {removed_count} molecules/ions "
                          f"overlapping with adsorbate")
            else:
                if water_side == 'bottom':
                    print(f"  Water positioned: z={water_top_target-water_height:.1f} to {water_top_target:.1f} A")
                else:
                    print(f"  Water positioned: z={water_bottom_target:.1f} to {water_bottom_target+water_height:.1f} A")
                print(f"  Surface top: z={surf_top_z:.1f} A, gap: {gap} A")

        except Exception as e:
            print(f"  Warning: Could not merge water layer: {e}")
            import traceback
            traceback.print_exc()
            combined = surface
            combined_mol_ids = [1] * len(combined)
    else:
        combined = surface
        combined_mol_ids = [1] * len(combined)

    # Sort atoms by element for proper POSCAR grouping
    # Order: metals first, then C, H, O, N, S, etc.
    element_priority = {
        'Pt': 0, 'Au': 1, 'Ag': 2, 'Cu': 3, 'Ni': 4, 'Co': 5, 'Fe': 6,
        'Ce': 7, 'Ti': 8, 'Zr': 9, 'Al': 10, 'Si': 11, 'P': 12, 'S': 13,
        'Cl': 14, 'Na': 15, 'K': 16, 'Ca': 17, 'Mg': 18, 'Li': 19, 'Cs': 20,
        'Br': 21, 'F': 22, 'I': 23, 'C': 24, 'N': 25, 'O': 26, 'H': 27,
    }
    symbols = combined.get_chemical_symbols()
    sort_indices = sorted(range(len(symbols)),
                         key=lambda i: element_priority.get(symbols[i], 100))
    combined = combined[sort_indices]
    combined_mol_ids = [combined_mol_ids[i] for i in sort_indices]

    # Write all formats
    xyz = os.path.join(output_dir, "structure.xyz")
    write(xyz, combined)

    poscar = os.path.join(output_dir, "POSCAR")
    write(poscar, combined, format='vasp', vasp5=True)

    cif = os.path.join(output_dir, "structure.cif")
    write(cif, combined, format='cif')

    data = os.path.join(output_dir, "structure.data")
    _write_lammps_data(data, combined, combined_mol_ids)

    # Report
    from collections import Counter
    report = os.path.join(output_dir, "model_report.txt")
    with open(report, 'w', encoding='utf-8') as f:
        f.write("EMCbuild model report\n")
        f.write(f"formula: {combined.get_chemical_formula()}\n")
        counts = Counter(combined.get_chemical_symbols())
        f.write(f"element_counts: {', '.join(f'{k}={v}' for k,v in sorted(counts.items()))}\n")
        lengths = combined.cell.lengths()
        f.write(f"cell: {lengths[0]:.2f} x {lengths[1]:.2f} x {lengths[2]:.2f} A\n")
        dists = combined.get_all_distances(mic=True)
        min_d = dists[dists>0].min() if len(combined) > 1 else 0
        f.write(f"min_distance: {min_d:.3f} A\n")
        f.write(f"warnings: {'close_contacts' if min_d < 0.7 else 'none'}\n")

    return xyz, poscar, cif, data, report


# ─── Main build ──────────────────────────────────────────────────────

def build(config_path: str):
    """Main build pipeline: surface + adsorbates + water layer → output files.

    Steps:
    1. Build metal surface slab (ASE)
    2. Add adsorbates at specified sites (ASE)
    3. Build water+ion layer (EMC)
    4. Merge and write POSCAR/XYZ/CIF/LAMMPS data
    """
    config = load_config(config_path)
    output_dir = config['output_dir']  # Already resolved by load_config
    os.makedirs(output_dir, exist_ok=True)

    surface_cfg = config.get('surface', {})
    elec_cfg = config.get('electrolyte', {})

    # Electrolyte-only mode: no surface, just build water/solvent box
    if not surface_cfg or surface_cfg.get('type') == 'electrolyte_only':
        print("[1/1] Building electrolyte box with EMC...")
        water_data = build_water_layer(None, elec_cfg, output_dir, surface_cell_xy=None)
        print(f"  LAMMPS data: {water_data}")
        # Convert LAMMPS data to POSCAR/XYZ/CIF
        if water_data and os.path.exists(water_data):
            from ase.io import read as ase_read
            from ase.io.lammpsdata import read_lammps_data
            # Build combined type map: atom type NAME -> element symbol
            combined_type_map = {}
            combined_type_map.update(CHARMM_TYPE_MAP)
            # Read LAMMPS data to get atom type numbers and names
            # The data file has lines like: "1    1.00797  # hw"
            # We need to map type numbers to atomic numbers
            type_name_to_Z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
                              'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                              'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                              'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                              'Fe': 26, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Br': 35,
                              'Sr': 38, 'Cs': 55, 'Ba': 56, 'Pt': 78, 'Au': 79}
            Z_of_type = {}
            in_masses = False
            with open(water_data, 'r') as f:
                for line in f:
                    if line.strip().startswith('Masses'):
                        in_masses = True
                        continue
                    if in_masses and line.strip().startswith('Atoms'):
                        break
                    if in_masses and '#' in line and line.strip()[0].isdigit():
                        parts = line.split('#')
                        if len(parts) == 2:
                            type_num = int(parts[0].split()[0])
                            type_name = parts[1].strip()
                            # Look up element in combined map
                            elem = combined_type_map.get(type_name, None)
                            if elem and elem in type_name_to_Z:
                                Z_of_type[type_num] = type_name_to_Z[elem]
                            elif type_name in type_name_to_Z:
                                Z_of_type[type_num] = type_name_to_Z[type_name]
            if not Z_of_type:
                print(f"  Warning: Could not determine atom types from LAMMPS data")
                print("\nDone!")
                return
            try:
                atoms = read_lammps_data(water_data, Z_of_type=Z_of_type)
                atoms.pbc = True
            except Exception as e:
                print(f"  Warning: Could not parse LAMMPS data file: {e}")
                print("\nDone!")
                return
            # Sort atoms by element for proper POSCAR grouping
            element_priority = {
                'Pt': 0, 'Au': 1, 'Ag': 2, 'Cu': 3, 'Ni': 4, 'Co': 5, 'Fe': 6,
                'Ce': 7, 'Ti': 8, 'Zr': 9, 'Al': 10, 'Si': 11, 'P': 12, 'S': 13,
                'Cl': 14, 'Na': 15, 'K': 16, 'Ca': 17, 'Mg': 18, 'Li': 19, 'Cs': 20,
                'Br': 21, 'F': 22, 'I': 23, 'C': 24, 'N': 25, 'O': 26, 'H': 27,
            }
            symbols = atoms.get_chemical_symbols()
            sort_indices = sorted(range(len(symbols)),
                                 key=lambda i: element_priority.get(symbols[i], 100))
            atoms = atoms[sort_indices]
            write(os.path.join(output_dir, "structure.xyz"), atoms)
            write(os.path.join(output_dir, "POSCAR"), atoms, format='vasp', vasp5=True)
            write(os.path.join(output_dir, "structure.cif"), atoms)
            # Copy water.data as structure.data
            import shutil
            shutil.copy2(water_data, os.path.join(output_dir, "structure.data"))
            print(f"  POSCAR:  {os.path.join(output_dir, 'POSCAR')}")
            print(f"  XYZ:     {os.path.join(output_dir, 'structure.xyz')}")
            print(f"  CIF:     {os.path.join(output_dir, 'structure.cif')}")
            print(f"  LAMMPS:  {os.path.join(output_dir, 'structure.data')}")
        print("\nDone!")
        return

    stype = surface_cfg.get('type', 'slab')

    # Step 1: Build surface
    print("[1/4] Building surface...")
    if stype == 'nanotube':
        atoms = build_nanotube(surface_cfg['n'], surface_cfg['m'],
                               surface_cfg.get('length', 3), surface_cfg.get('vacuum', 15.0))
    elif stype == 'heterojunction':
        # Check for new pymatgen format (film/substrate) vs old format (bottom/top)
        film_cfg = surface_cfg.get('film')
        sub_cfg = surface_cfg.get('substrate')
        if film_cfg and sub_cfg:
            # New format: pymatgen-based heterojunction
            atoms = build_heterojunction_pymatgen(
                film_cfg, sub_cfg, surface_cfg.get('vacuum', 15.0))
        else:
            # Legacy format: hardcoded heterojunction
            atoms = build_heterojunction(surface_cfg['bottom'], surface_cfg['top'],
                                         surface_cfg.get('distance', 3.0),
                                         surface_cfg.get('vacuum', 15.0))
    else:
        atoms = build_slab(surface_cfg['element'], surface_cfg['miller'],
                           surface_cfg['size'], surface_cfg['layers'], surface_cfg.get('vacuum', 15.0))
    atoms.pbc = True

    # Clean up extra arrays from pymatgen Interface (causes XYZ read issues)
    for key in list(atoms.arrays.keys()):
        if key not in ('positions', 'numbers', 'cell', 'pbc'):
            del atoms.arrays[key]

    # Determine exposed surface z for water layer positioning
    # For heterojunctions with adsorbate.element, use that element's top z
    ads_cfg = config.get('adsorbate', {})
    ads_list = ads_cfg.get('list', [])
    exposed_element = None
    for ads in ads_list:
        if 'element' in ads:
            exposed_element = ads['element']
            break

    if exposed_element:
        symbols = atoms.get_chemical_symbols()
        exposed_indices = [i for i, s in enumerate(symbols) if s == exposed_element]
        exposed_z = atoms.positions[exposed_indices, 2]
        # Determine if element is substrate (bottom) or film (top)
        all_z = atoms.positions[:, 2]
        mid_z = (all_z.min() + all_z.max()) / 2.0
        if np.mean(exposed_z) < mid_z:
            # Substrate: exposed surface is at the bottom
            bare_surf_top_z = min(exposed_z)
        else:
            # Film: exposed surface is at the top
            bare_surf_top_z = max(exposed_z)
    else:
        bare_surf_top_z = max(atoms.positions[:, 2])

    surface_xyz = os.path.join(output_dir, "surface.xyz")
    write(surface_xyz, atoms)
    print(f"  {len(atoms)} atoms, formula {atoms.get_chemical_formula()}")

    # Step 2: Add adsorbates
    print("[2/4] Adding adsorbates...")
    if ads_list:
        atoms = add_adsorbates(atoms, ads_list, output_dir=output_dir)
        ads_xyz = os.path.join(output_dir, "surface_ads.xyz")
        write(ads_xyz, atoms)
        surface_xyz = ads_xyz
        print(f"  {len(atoms)} atoms after adsorbates")

    # Step 3: Build water layer with EMC
    print("[3/4] Building water layer with EMC...")
    elec_cfg = config.get('electrolyte', {})
    surf_cell = atoms.cell
    surface_cell_xy = (surf_cell[0, 0], surf_cell[1, 1])
    water_data = build_water_layer(surface_xyz, elec_cfg, output_dir, surface_cell_xy)

    # Step 4: Merge and output
    # Water layer starts from bare surface + gap, embedding the adsorbate
    print("[4/4] Writing output files...")
    # Determine water side: 'bottom' if substrate exposed, 'top' if film/slab exposed
    water_side = 'top'
    if exposed_element:
        symbols = atoms.get_chemical_symbols()
        exposed_z = atoms.positions[[i for i, s in enumerate(symbols)
                                     if s == exposed_element], 2]
        all_z = atoms.positions[:, 2]
        mid_z = (all_z.min() + all_z.max()) / 2.0
        if np.mean(exposed_z) < mid_z:
            water_side = 'bottom'
    print(f"  Water layer side: {water_side}")
    xyz, poscar, cif, data, report = merge_and_output(
        surface_xyz, water_data, output_dir,
        bare_surf_top_z=bare_surf_top_z,
        gap=elec_cfg.get('gap', 1.5),
        min_clearance=elec_cfg.get('min_clearance', 1.5),
        water_side=water_side
    )
    print(f"  POSCAR:    {poscar}")
    print(f"  XYZ:       {xyz}")
    print(f"  CIF:       {cif}")
    print(f"  LAMMPS:    {data}")
    print(f"  Report:    {report}")
    print("\nDone!")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python build_v3.py <config.json>")
        sys.exit(1)
    build(sys.argv[1])
