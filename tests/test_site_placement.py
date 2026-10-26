#!/usr/bin/env python
"""Regression tests for adsorbate site selection (the 'site' key).

Background: before 2026-10-22 add_adsorbates() documented a 'site' config
key it never read -- placement was hardcoded to ontop of the surface atom
nearest the cell center. That silently ignored fcc/bridge/hcp requests
(hit the same day on the Pt(111)+H HER course case: H belongs on the fcc
hollow, the builder put it ontop and needed a post-processing fixer).

Also covered: the center-distance ranking must use fractional-coordinate
minimum image -- the (111) cell is hexagonal and an orthogonal dx % lx
wrap mis-ranks atoms there (observed: a spurious 2.40 A nearest neighbor
where the true nn is 2.77 A).

Fast, needs no EMC binary. Run:
    python tests/test_site_placement.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from ase.build import fcc100, fcc110, fcc111

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import build  # noqa: E402

A_PT = 3.9231


def _pt111():
    atoms = fcc111('Pt', size=(3, 3, 4), a=A_PT, vacuum=15.0)
    atoms.pbc = True
    pos = atoms.positions
    zmax = pos[:, 2].max()
    top = [i for i in range(len(atoms)) if abs(pos[i, 2] - zmax) < 0.5]
    return atoms, top, zmax


def test_site_counts():
    """3x3 (111): 9 ontop, 27 nn-pair bridges, 9 fcc + 9 hcp hollows."""
    atoms, top, zmax = _pt111()
    sites = build._enumerate_surface_sites(atoms, top, len(atoms), zmax, 1)
    assert len(sites['ontop']) == 9, len(sites['ontop'])
    assert len(sites['bridge']) == 27, len(sites['bridge'])
    assert len(sites['fcc']) == 9, len(sites['fcc'])
    assert len(sites['hcp']) == 9, len(sites['hcp'])


def test_fcc_hcp_classification():
    """fcc hollow has its subsurface atom two layer spacings down, hcp one."""
    atoms, top, zmax = _pt111()
    sites = build._enumerate_surface_sites(atoms, top, len(atoms), zmax, 1)
    d111 = A_PT / math.sqrt(3)
    pos = atoms.positions
    cell2d = np.array(atoms.cell)[:2, :2]
    frac = atoms.get_scaled_positions(wrap=False)

    def depth(fxy):
        ds = []
        for k in range(len(atoms)):
            d = frac[k][:2] - np.asarray(fxy)
            d -= np.round(d)
            if np.linalg.norm(d @ cell2d) < 0.1 and pos[k, 2] < zmax - 0.5:
                ds.append(zmax - pos[k, 2])
        return min(ds) if ds else None

    assert abs(depth(sites['fcc'][0]['frac']) - 2 * d111) < 0.2, \
        depth(sites['fcc'][0]['frac'])
    assert abs(depth(sites['hcp'][0]['frac']) - d111) < 0.2, \
        depth(sites['hcp'][0]['frac'])


def test_interior_first():
    """Sites are ranked center-out; the first one is never near 0/1."""
    atoms, top, zmax = _pt111()
    sites = build._enumerate_surface_sites(atoms, top, len(atoms), zmax, 1)
    for kind, lst in sites.items():
        if not lst:
            continue
        scores = [s['score'] for s in lst]
        assert scores == sorted(scores), f'{kind} not sorted'
        f = lst[0]['frac'] % 1.0
        assert 0.05 < f[0] < 0.95 and 0.05 < f[1] < 0.95, (kind, f)


def test_h_end_to_end_fcc():
    """H with site=fcc: equilateral hollow, 3x Pt-H, frac (4/9, 4/9)."""
    atoms, top, zmax = _pt111()
    n0 = len(atoms)
    build.add_adsorbates(atoms, [{'name': 'H', 'site': 'fcc', 'height': 1.022}])
    assert len(atoms) == n0 + 1
    h = len(atoms) - 1
    assert atoms[h].symbol == 'H'
    d = sorted(atoms.get_distance(h, k, mic=True) for k in range(n0))[:3]
    expected = math.sqrt(1.022 ** 2 + (A_PT / math.sqrt(2) / math.sqrt(3)) ** 2)
    assert max(d) - min(d) < 0.02, d
    assert abs(d[0] - expected) < 0.01, (d, expected)
    f = atoms.get_scaled_positions(wrap=True)[h]
    assert abs(f[0] - 4 / 9) < 0.01 and abs(f[1] - 4 / 9) < 0.01, f


def test_bridge_end_to_end():
    """H with site=bridge: two equidistant nearest Pt at the nn midpoint."""
    atoms, top, zmax = _pt111()
    n0 = len(atoms)
    build.add_adsorbates(atoms, [{'name': 'H', 'site': 'bridge', 'height': 1.2}])
    h = len(atoms) - 1
    d = sorted(atoms.get_distance(h, k, mic=True) for k in range(n0))[:2]
    expected = math.sqrt(1.2 ** 2 + (A_PT / math.sqrt(2) / 2) ** 2)
    assert abs(d[0] - d[1]) < 0.02, d
    assert abs(d[0] - expected) < 0.01, (d, expected)


def test_default_is_ontop():
    """No site key: behavior unchanged -- directly above a surface atom."""
    atoms, top, zmax = _pt111()
    n0 = len(atoms)
    build.add_adsorbates(atoms, [{'name': 'H'}])
    h = len(atoms) - 1
    d = min(atoms.get_distance(h, k, mic=True) for k in range(n0))
    assert abs(d - 1.8) < 1e-6, d  # default height, atom directly beneath
    cell2d = np.array(atoms.cell)[:2, :2]
    frac = atoms.get_scaled_positions(wrap=True)
    delta = frac[list(top)][:, :2] - frac[h][:2]
    delta -= np.round(delta)
    assert np.linalg.norm(delta @ cell2d, axis=1).min() < 1e-6


def test_invalid_site_raises():
    """Unsupported site values raise instead of being silently ignored."""
    atoms, top, zmax = _pt111()
    try:
        build.add_adsorbates(atoms, [{'name': 'H', 'site': 'pentafold'}])
    except ValueError as e:
        assert 'pentafold' in str(e)
        return
    raise AssertionError("site='pentafold' did not raise")


def test_non111_hollow_rejected():
    """(100) has 4-fold hollows, not fcc/hcp: fcc/hcp empty, request raises."""
    atoms = fcc100('Pt', size=(3, 3, 4), a=A_PT, vacuum=15.0)
    atoms.pbc = True
    pos = atoms.positions
    zmax = pos[:, 2].max()
    top = [i for i in range(len(atoms)) if abs(pos[i, 2] - zmax) < 0.5]
    sites = build._enumerate_surface_sites(atoms, top, len(atoms), zmax, 1)
    assert len(sites['fcc']) == 0 and len(sites['hcp']) == 0
    assert len(sites['bridge']) > 0   # (100) does have nn pairs
    assert len(sites['hollow4']) == 9  # and four 4-fold hollows per 3x3
    try:
        build.add_adsorbates(atoms, [{'name': 'H', 'site': 'fcc'}])
    except ValueError as e:
        assert 'hollow4' in str(e)   # the error names what IS available
        return
    raise AssertionError("site='fcc' on (100) did not raise")


def test_hollow4_placement_on_100():
    """4-fold hollow on Pt(100): 4 equidistant Pt, centre of the square."""
    atoms = fcc100('Pt', size=(3, 3, 4), a=A_PT, vacuum=15.0)
    atoms.pbc = True
    n0 = len(atoms)
    build.add_adsorbates(atoms, [{'name': 'H', 'site': 'hollow4',
                                  'height': 1.0}])
    h = len(atoms) - 1
    d = sorted(atoms.get_distance(h, k, mic=True) for k in range(n0))[:5]
    assert max(d[:4]) - min(d[:4]) < 0.02, d[:4]
    assert d[4] > d[3] + 0.3, d
    expected = math.hypot(1.0, A_PT / 2.0)
    assert abs(d[0] - expected) < 0.02, (d[0], expected)


def test_longbridge_on_110():
    """(110) trough: longbridge between atoms a/2 apart along the row."""
    atoms = fcc110('Pt', size=(3, 3, 4), a=A_PT, vacuum=15.0)
    atoms.pbc = True
    pos = atoms.positions
    zmax = pos[:, 2].max()
    top = [i for i in range(len(atoms)) if abs(pos[i, 2] - zmax) < 0.5]
    sites = build._enumerate_surface_sites(atoms, top, len(atoms), zmax, 1)
    assert len(sites['longbridge']) == 9, len(sites['longbridge'])
    # longbridge separation is a/sqrt(2) -> a/2*sqrt(2) = 2.77/1.414*... check
    seps = [s['separ'] for s in sites['longbridge'] if 'separ' in s]
    n0 = len(atoms)
    build.add_adsorbates(atoms, [{'name': 'H', 'site': 'longbridge',
                                  'height': 1.0}])
    h = len(atoms) - 1
    d = sorted(atoms.get_distance(h, k, mic=True) for k in range(n0))[:3]
    assert max(d[:2]) - min(d[:2]) < 0.02, d[:2]


TESTS = [test_site_counts, test_fcc_hcp_classification, test_interior_first,
         test_h_end_to_end_fcc, test_bridge_end_to_end, test_default_is_ontop,
         test_invalid_site_raises, test_non111_hollow_rejected,
         test_hollow4_placement_on_100, test_longbridge_on_110]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
