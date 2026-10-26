#!/usr/bin/env python
"""Slab geometry tests across the whole ASE element database.

Background (2026-10-26): build_slab() used to hardcode a 9-entry lattice
table and always pass crystalstructure='fcc'. Anything missing silently
became a=3.8 A fcc, so Ti/Ru/Co/Cr built a completely wrong cell and
Fe/W/Mo -- present in the table -- were still forced into FCC, giving a
nearest-neighbour distance 18% short of the real bcc value. bcc(110) and
other non-(100)/(111) cuts were also wrong because ase.build.surface()
needs the CONVENTIONAL cell, not the primitive one.

These tests pin the fix down physically rather than by a golden atom
count: for every element ASE knows a reference structure for, the slab's
nearest-neighbour distance must equal that structure's bulk nn, the
requested number of layers must appear, and the requested vacuum must be
reproduced. Nothing here needs EMC.

Run:
    python tests/test_slab_elements.py
"""
import sys
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, chemical_symbols, reference_states

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import build  # noqa: E402

# Low-index face used per structure. These are the faces that exist for
# every element of that structure, so the sweep never needs a per-element
# exception list.
PRIMARY_FACE = {
    'fcc': (1, 1, 1),
    'bcc': (1, 1, 0),
    'hcp': (0, 0, 1),
    'diamond': (1, 1, 1),
    'sc': (1, 0, 0),
    'bct': (1, 0, 0),
    'orthorhombic': (1, 0, 0),
    'rhombohedral': (1, 0, 0),
    'tetragonal': (1, 0, 0),
    'monoclinic': (1, 0, 0),
    'cubic': (1, 0, 0),     # P, Mn -- expected to be refused, see below
}

# Structures ASE's reference_states marks 'cubic' but cannot build: P and
# Mn are not simple-cubic cells (a = 7.17 and 8.89 A). Pu has no lattice
# constant at all. B and Pa are tetragonal with no atomic basis, which
# ase.build.bulk() refuses. These must fail loudly, never silently.
UNBUILDABLE = {'B', 'Pa', 'P', 'Mn', 'Pu'}
# The noble gases are in the database as solids; building them is valid
# for tests but pointless for modelling, so keep them in the sweep (they
# are exactly the kind of entry the old hardcoded table dropped).
NATURAL_VACUUM = 12.0
LAYERS = 3
SIZE = (2, 2)


def elements_with_structure():
    """(symbol, structure) for every element ASE has a solid-state entry for."""
    out = []
    for sym in chemical_symbols[1:]:
        ref = reference_states[atomic_numbers[sym]]
        if not ref or 'symmetry' not in ref:
            continue
        if ref['symmetry'] in ('atom', 'diatom'):
            continue
        out.append((sym, ref['symmetry']))
    return out


def bulk_nn(sym, structure):
    """True nearest-neighbour distance of the reference crystal, or None."""
    ref = reference_states[atomic_numbers[sym]]
    a = ref.get('a')
    if a is None:
        return None
    if structure == 'fcc':
        return a / np.sqrt(2.0)
    if structure == 'bcc':
        return a * np.sqrt(3.0) / 2.0
    if structure == 'diamond':
        return a * np.sqrt(3.0) / 4.0
    if structure == 'sc':
        return a
    if structure == 'hcp':
        c = ref.get('c/a', 1.633) * a
        # in-plane nn is a; the out-of-plane one is only shorter for
        # non-ideal c/a (Ru 1.584, Zn 1.856) and then it is the real nn
        return min(a, np.hypot(a / np.sqrt(3.0), c / 2.0))
    return None


def _nn_of(atoms):
    d = atoms.get_all_distances(mic=False)
    np.fill_diagonal(d, 1e9)
    return float(d.min())


def test_sweep_every_element_builds():
    """Every buildable element builds; every unbuildable one raises."""
    built, raised, wrong = [], [], []
    for sym, st in elements_with_structure():
        face = PRIMARY_FACE.get(st)
        if face is None:
            continue
        try:
            atoms = build.build_slab(sym, face, SIZE, LAYERS, NATURAL_VACUUM)
        except ValueError as exc:
            raised.append((sym, st, str(exc)))
            continue
        if len(atoms) == 0:
            wrong.append((sym, st, 'empty slab'))
        elif set(atoms.get_chemical_symbols()) != {sym}:
            wrong.append((sym, st, f'symbols {set(atoms.get_chemical_symbols())}'))
        else:
            built.append((sym, st))

    assert not wrong, wrong
    built_syms = {s for s, _ in built}
    # No element may silently disappear: anything ASE lists as a solid is
    # either built or explicitly refused.
    assert not (built_syms & UNBUILDABLE), built_syms & UNBUILDABLE
    # 79 of ASE's 84 solid-state entries build on their primary face; the
    # remaining 5 are exactly the UNBUILDABLE set.
    assert len(built) >= 79, f'only {len(built)} elements built'
    for sym, st, msg in raised:
        assert sym in UNBUILDABLE, f'{sym} ({st}) unexpectedly refused: {msg}'
        assert len(msg) > 40, f'{sym}: error message not actionable: {msg}'
    # each unbuildable element is reported, not skipped
    assert {s for s, _, _ in raised} == UNBUILDABLE, \
        sorted({s for s, _, _ in raised})


def test_nearest_neighbour_matches_bulk():
    """The slab's nn distance equals the true bulk nn for its structure.

    This is the test that catches the original bug: Fe/W/Mo were labelled
    fcc and came out 18% short, Ti/Ru/Co/Cr were given a=3.8 fcc.
    """
    bad = []
    checked = 0
    for sym, st in elements_with_structure():
        if sym in UNBUILDABLE:
            continue
        expected = bulk_nn(sym, st)
        if expected is None:
            continue
        face = PRIMARY_FACE.get(st)
        if face is None:
            continue
        atoms = build.build_slab(sym, face, SIZE, LAYERS, NATURAL_VACUUM)
        got = _nn_of(atoms)
        checked += 1
        if abs(got - expected) > 1e-3 * expected:
            bad.append((sym, st, round(got, 4), round(expected, 4)))
    assert not bad, bad
    # fcc/bcc/hcp/diamond/sc all have an unambiguous bulk nn: 65 of them.
    assert checked >= 65, checked


def test_layer_count_and_vacuum():
    """`layers` distinct z planes appear and the vacuum is 2 x the request."""
    bad = []
    for sym, st in elements_with_structure():
        if sym in UNBUILDABLE:
            continue
        face = PRIMARY_FACE.get(st)
        if face is None:
            continue
        atoms = build.build_slab(sym, face, SIZE, LAYERS, NATURAL_VACUUM)
        z = np.unique(np.round(atoms.positions[:, 2], 3))
        # Low-index faces: one z plane per requested layer for the
        # single-atom-basis structures (fcc/bcc/hcp/diamond). Structures
        # with a basis (bct Sn, rhombohedral As/Sb/Bi/Sm) put several
        # atoms in each layer, so the plane count is a multiple of it.
        # Stepped faces such as (211) legitimately show more planes and
        # are covered by test_stepped_face_is_built instead.
        if len(z) < LAYERS or len(z) % LAYERS:
            bad.append((sym, st, 'z planes', len(z), LAYERS))
            continue
        gap = atoms.cell[2, 2] - (z[-1] - z[0])
        if abs(gap - 2 * NATURAL_VACUUM) > 0.05:
            bad.append((sym, st, 'vacuum', round(gap, 3)))
    assert not bad, bad


def test_no_overlapping_atoms():
    """No two atoms collapse onto each other in any face we advertise."""
    bad = []
    faces = {
        'fcc': [(1, 1, 1), (1, 0, 0), (1, 1, 0), (2, 1, 1)],
        'bcc': [(1, 1, 0), (1, 0, 0), (1, 1, 1)],
        'hcp': [(0, 0, 1), (1, 0, 0)],
        'diamond': [(1, 1, 1), (1, 0, 0)],
    }
    probes = {
        'fcc': ['Pt', 'Cu', 'Al', 'Ni', 'Au', 'Ag', 'Pd', 'Rh', 'Ir', 'Pb'],
        'bcc': ['Fe', 'W', 'Mo', 'Cr', 'V', 'Nb', 'Ta', 'Na', 'K'],
        'hcp': ['Ru', 'Ti', 'Co', 'Zn', 'Mg', 'Zr', 'Cd', 'Re', 'Os'],
        'diamond': ['C', 'Si', 'Ge'],
    }
    for st, syms in probes.items():
        for sym in syms:
            expected = bulk_nn(sym, st)
            for face in faces[st]:
                atoms = build.build_slab(sym, face, SIZE, LAYERS,
                                         NATURAL_VACUUM)
                got = _nn_of(atoms)
                if abs(got - expected) > 2e-3 * expected:
                    bad.append((sym, st, face, round(got, 4),
                                round(expected, 4)))
    assert not bad, bad


def test_bcc_not_forced_to_fcc():
    """Fe/W/Mo used to be built on an fcc lattice despite being in the table."""
    for sym, a in (('Fe', 2.87), ('W', 3.16), ('Mo', 3.15)):
        atoms = build.build_slab(sym, (1, 1, 0), SIZE, LAYERS,
                                 NATURAL_VACUUM)
        got = _nn_of(atoms)
        bcc_nn = a * np.sqrt(3.0) / 2.0
        fcc_nn = a / np.sqrt(2.0)
        assert abs(got - bcc_nn) < 1e-2, (sym, got, bcc_nn)
        assert abs(got - fcc_nn) > 0.1, (sym, 'still fcc-like', got)


def test_stepped_face_is_built():
    """(211) via the generic path: correct nn, no collapsed atoms."""
    atoms = build.build_slab('Pt', (2, 1, 1), SIZE, LAYERS, NATURAL_VACUUM)
    got = _nn_of(atoms)
    assert abs(got - 3.9231 / np.sqrt(2.0)) < 1e-2, got
    assert len(np.unique(np.round(atoms.positions[:, 2], 3))) > LAYERS


def test_site_enumeration_per_structure():
    """ontop == surface atoms; hollows are genuinely equidistant rings."""
    cases = [
        ('Pt', (1, 1, 1), 3),      # close-packed triangle
        ('Pt', (1, 0, 0), 4),      # square net
        ('Pt', (1, 1, 0), 4),      # rectangular net
        ('Fe', (1, 1, 0), 3),      # bcc(110): nearly triangular
        ('Fe', (1, 0, 0), 4),
        ('Ru', (0, 0, 1), 3),
        ('C', (1, 1, 1), 3),
        ('C', (1, 0, 0), 4),
    ]
    bad = []
    for sym, face, k_expected in cases:
        atoms = build.build_slab(sym, face, (3, 3), 4, 15.0)
        pos = atoms.positions
        zmax = pos[:, 2].max()
        top = [i for i in range(len(atoms)) if abs(pos[i, 2] - zmax) < 0.5]
        sites = build._enumerate_surface_sites(atoms, top, len(atoms), zmax, 1)
        if len(sites['ontop']) != len(top):
            bad.append((sym, face, 'ontop', len(sites['ontop']), len(top)))
        hollows = sites[f'hollow{k_expected}']
        if not hollows:
            bad.append((sym, face, f'no hollow{k_expected}'))
            continue
        c2 = np.array(atoms.cell)[:2, :2]
        frac_top = atoms.get_scaled_positions(wrap=False)[top][:, :2]
        for h in hollows:
            d = frac_top - h['frac'][:2]
            d -= np.round(d)
            dist = np.linalg.norm(d @ c2, axis=1)
            ring = np.sort(dist)[:k_expected]
            if ring[-1] - ring[0] > 2e-2:
                bad.append((sym, face, 'not equidistant', ring))
                break
            if h.get('k') != k_expected:
                bad.append((sym, face, 'k mismatch', h.get('k')))
                break
    assert not bad, bad


def test_fcc_hcp_labels_only_on_close_packed():
    """fcc/hcp names only where the net is close-packed; bcc(111) is neither."""
    def labels(sym, face):
        atoms = build.build_slab(sym, face, (3, 3), 4, 15.0)
        pos = atoms.positions
        zmax = pos[:, 2].max()
        top = [i for i in range(len(atoms)) if abs(pos[i, 2] - zmax) < 0.5]
        s = build._enumerate_surface_sites(atoms, top, len(atoms), zmax, 1)
        return len(s['fcc']), len(s['hcp']), len(s['hollow3'])

    for sym, face in (('Pt', (1, 1, 1)), ('Al', (1, 1, 1)),
                      ('Ru', (0, 0, 1)), ('Ti', (0, 0, 1)),
                      ('Co', (0, 0, 1)), ('Zn', (0, 0, 1))):
        fcc, hcp, h3 = labels(sym, face)
        assert fcc > 0 and hcp > 0, (sym, face, fcc, hcp)
        assert fcc + hcp == h3, (sym, face, fcc, hcp, h3)

    # bcc(111): same 3-fold geometry, but not an ABC close packing, so the
    # fcc/hcp names must not be applied.
    for sym in ('Fe', 'W', 'Mo'):
        fcc, hcp, h3 = labels(sym, (1, 1, 1))
        assert h3 > 0, (sym, 'no hollows on bcc(111)')
        assert fcc == 0 and hcp == 0, (sym, 'mislabeled bcc(111)', fcc, hcp)

    # non-triangular nets have no 3-fold sites at all
    for sym, face in (('Pt', (1, 0, 0)), ('Pt', (1, 1, 0))):
        fcc, hcp, h3 = labels(sym, face)
        assert (fcc, hcp, h3) == (0, 0, 0), (sym, face, fcc, hcp, h3)


def test_explicit_overrides():
    """structure/a/c overrides let caller fix anything ASE gets wrong."""
    custom = build.build_slab('Pt', (1, 1, 1), (2, 2), 3, 12.0,
                              structure='fcc', a=4.0)
    assert abs(_nn_of(custom) - 4.0 / np.sqrt(2.0)) < 1e-3
    default = build.build_slab('Pt', (1, 1, 1), (2, 2), 3, 12.0)
    assert abs(_nn_of(default) - 3.92 / np.sqrt(2.0)) < 1e-3

    # a 'cubic' element can be forced into a real structure
    forced = build.build_slab('Mn', (1, 1, 0), (2, 2), 3, 12.0,
                              structure='bcc', a=2.89)
    assert abs(_nn_of(forced) - 2.89 * np.sqrt(3.0) / 2.0) < 1e-2

    # unknown symbol still fails loudly
    try:
        build.build_slab('Xx', (1, 0, 0), (1, 1), 2, 10.0)
    except (ValueError, KeyError) as exc:
        assert 'Xx' in str(exc)
    else:
        raise AssertionError('Xx did not raise')


TESTS = [
    test_sweep_every_element_builds,
    test_nearest_neighbour_matches_bulk,
    test_layer_count_and_vacuum,
    test_no_overlapping_atoms,
    test_bcc_not_forced_to_fcc,
    test_stepped_face_is_built,
    test_site_enumeration_per_structure,
    test_fcc_hcp_labels_only_on_close_packed,
    test_explicit_overrides,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {str(e)[:600]}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
