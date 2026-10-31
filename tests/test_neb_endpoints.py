#!/usr/bin/env python
"""NEB endpoint-pair tests for scripts/neb_endpoints.py.

Pins the 2026-10-31 classroom-case contract without needing EMC or a real
slab build: a tiny synthetic 2x2 Cu(100) data file must yield (a) two
endpoint files with identical atom order/IDs and one appended adatom,
(b) an IS/FS hop distance equal to the surface nearest-neighbour spacing,
(c) the trailing-blank-line hazard handled (adatom lands on the LAST atom
line, not after the blank line that terminates the Atoms section), and
(d) pick_endpoints refusing a site type the net does not have.

Run:
    python tests/test_neb_endpoints.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import neb_endpoints as ne  # noqa: E402

# Synthetic 2x2 Cu(100) slab, 2 atomic planes, a_surf = 2.556 A between
# surface rows (primitive (100) mesh), 16 atoms, EMCbuild-style layout:
# header / counts / box / Masses / Atoms # atomic / 16 entries / trailing
# blank line (the hazard this script must handle).
A = 2.556
NCELL = 4
Z0, DZ = 10.0, 1.28
_surf = [(i * A, j * A) for i in range(NCELL) for j in range(NCELL)]
_atoms = ([(x, y, Z0) for x, y in _surf]            # plane 1
          + [(x + A / 2, y + A / 2, Z0 + DZ) for x, y in _surf]  # plane 2
          + [(x, y, Z0 + 2 * DZ) for x, y in _surf]  # plane 3
          + [(x + A / 2, y + A / 2, Z0 + 3 * DZ) for x, y in _surf])
BOX = NCELL * A

DATA = "\n".join(
    ["LAMMPS data file (atom_style atomic) - written by EMCbuild", "",
     "%d atoms" % (4 * len(_surf)), "1 atom types", "",
     "0.0 %.6f xlo xhi" % BOX,
     "0.0 %.6f ylo yhi" % BOX,
     "0.0 30.0 zlo zhi", "",
     "Masses", "", "1 63.546  # Cu", "",
     "Atoms  # atomic", ""]
    + ["%d 1 %.6f %.6f %.6f" % (i + 1, x, y, z)
       for i, (x, y, z) in enumerate(_atoms)]
    + ["", ""])  # trailing blank line: the Atoms-section terminator hazard


def test_pick_endpoints_hop_equals_nn():
    slab = read(_write_data(), format="lammps-data", atom_style="atomic")
    surf_idx, surf_z = ne.surface_top(slab)
    is_site, fs_site, hop = ne.pick_endpoints(slab, surf_idx, surf_z)
    assert abs(hop - A) < 1e-3 * A, hop          # adjacent hollow = a_surf
    xy_is, xy_fs = np.array(is_site["xy"]), np.array(fs_site["xy"])
    assert abs(xy_is[0] - xy_fs[0]) + abs(xy_is[1] - xy_fs[1]) > 1e-6


def test_splice_preserves_ids_and_handles_blank_line():
    slab = read(_write_data(), format="lammps-data", atom_style="atomic")
    surf_idx, surf_z = ne.surface_top(slab)
    is_site, fs_site, _ = ne.pick_endpoints(slab, surf_idx, surf_z)
    with tempfile.TemporaryDirectory() as td:
        paths = ne.splice_endpoints(_write_data(), td, len(slab),
                                    np.array(is_site["xy"]),
                                    np.array(fs_site["xy"]),
                                    surf_z, 1.8, "Cu")
        assert len(paths) == 2
        for p in paths:
            lines = Path(p).read_text().splitlines()
            assert "%d atoms" % (len(_atoms) + 1) in lines          # 16 slab + 1 adatom
            atom_lines = [l for l in lines
                          if l.strip() and l.split()[0].isdigit()
                          and len(l.split()) == 5]
            assert len(atom_lines) == len(_atoms) + 1
            last = atom_lines[-1].split()
            assert last[0] == str(len(_atoms) + 1) and last[1] == "1"
            # the adatom must sit on the LAST atom line, not after a blank
            assert lines.index(atom_lines[-1]) > lines.index("Atoms  # atomic")
        # identical content except the final adatom line (order/IDs fixed)
        a = Path(paths[0]).read_text().splitlines()
        b = Path(paths[1]).read_text().splitlines()
        assert a[:-1] == b[:-1]
        assert a[-1] != b[-1]


def test_pick_endpoints_rejects_missing_site():
    slab = read(_write_data(), format="lammps-data", atom_style="atomic")
    surf_idx, surf_z = ne.surface_top(slab)
    try:
        ne.pick_endpoints(slab, surf_idx, surf_z, site="fcc")
        raise AssertionError("expected ValueError for fcc on a (100) net")
    except ValueError as exc:
        assert "available" in str(exc)


def test_read_back_is_lammps_clean():
    # the spliced file must parse again with ASE (atom count + no type-0)
    slab = read(_write_data(), format="lammps-data", atom_style="atomic")
    surf_idx, surf_z = ne.surface_top(slab)
    is_site, fs_site, _ = ne.pick_endpoints(slab, surf_idx, surf_z)
    with tempfile.TemporaryDirectory() as td:
        paths = ne.splice_endpoints(_write_data(), td, len(slab),
                                    np.array(is_site["xy"]),
                                    np.array(fs_site["xy"]),
                                    surf_z, 1.8, "Cu")
        for p in paths:
            atoms = read(p, format="lammps-data", atom_style="atomic")
            assert len(atoms) == len(_atoms) + 1
            assert set(atoms.get_chemical_symbols()) == {"Cu"}


def _write_data():
    fd = tempfile.NamedTemporaryFile("w", suffix=".data", delete=False)
    fd.write(DATA)
    fd.close()
    return fd.name


if __name__ == "__main__":
    test_pick_endpoints_hop_equals_nn()
    test_splice_preserves_ids_and_handles_blank_line()
    test_pick_endpoints_rejects_missing_site()
    test_read_back_is_lammps_clean()
    print("4 tests OK")
