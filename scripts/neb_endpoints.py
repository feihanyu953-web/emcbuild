#!/usr/bin/env python
"""NEB endpoint-pair preparation from an EMCbuild slab data file.

Produces two LAMMPS data files (initial.data / final.data) holding the SAME
slab with ONE adatom placed at the rank-0 hollow of the requested type (IS)
and at its nearest adjacent hollow of the same type (FS). Atom order and IDs
are identical between the two files - the `neb` command interpolates
intermediate replicas by atom ID, so any order drift silently corrupts the
path.

The slab data file is EMCbuild's own `structure.data` (atom_style atomic),
spliced at the text level: the adatom line is appended as the LAST entry of
the Atoms section. A trailing blank line (which EMCbuild files carry)
terminates the Atoms section for LAMMPS, so it is stripped before appending -
appending after it yields 'Did not assign all atoms correctly' at read_data
(observed 2026-10-31, Cu(100) classroom case).

Usage:
    python neb_endpoints.py --data model/structure.data --out-dir ENDPOINTS
        [--height 1.8] [--site hollow4] [--element Cu]

Nothing here needs the EMC binary; the surface-site enumeration is imported
from build.py so both paths share one verified implementation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from build import _enumerate_surface_sites  # noqa: E402


def surface_top(slab):
    """Return (surface_indices, surf_z) of the exposed top plane."""
    z = slab.positions[:, 2]
    surf_z = float(z.max())
    indices = [i for i in range(len(slab)) if abs(z[i] - surf_z) < 0.5]
    return indices, surf_z


def pick_endpoints(slab, surface_indices, surf_z, site="hollow4"):
    """Rank-0 site of `site` type (IS) + nearest same-type site (FS).

    Returns (is_site, fs_site, hop_distance). The hop target is the nearest
    OTHER site of the same type - for a (100) fourfold net this is the
    adjacent hollow a/sqrt(2) away.
    """
    sites = _enumerate_surface_sites(slab, surface_indices, len(slab),
                                     surf_z, 1)
    if site not in sites or not sites[site]:
        raise ValueError(f"no '{site}' sites enumerated; available: "
                         f"{', '.join(k for k, v in sites.items() if v)}")
    hollows = sites[site]
    is_site = hollows[0]  # ranked by distance to cell center -> interior
    xy_is = np.array(is_site["xy"])
    cell2d = slab.cell[:2, :2]
    inv = np.linalg.inv(cell2d)

    def mic_distance(xy):
        frac = (np.array(xy) - xy_is) @ inv
        frac -= np.round(frac)
        return float(np.linalg.norm(frac @ cell2d))

    cands = sorted(((mic_distance(s["xy"]), s) for s in hollows[1:]),
                   key=lambda t: t[0])
    hop_d, fs_site = cands[0]
    return is_site, fs_site, hop_d


def splice_endpoints(src, out_dir, n_slab, xy_is, xy_fs, surf_z, height,
                     element="Cu"):
    """Text-level splice of the adatom line into EMCbuild's own data file.

    The Atoms section is the last one in an atomic-style file; its trailing
    blank line is stripped first (a blank line inside the section ends it),
    then the adatom becomes the final entry with id = n_slab + 1, type 1.
    """
    n_plus = n_slab + 1
    txt = open(src, encoding="utf-8").read()
    marker = txt.index("Atoms")
    head, atoms_part = txt[:marker], txt[marker:]
    head = head.replace("%d atoms" % n_slab, "%d atoms" % n_plus)
    atoms_part = atoms_part.rstrip("\n")
    paths = []
    for name, xy in (("initial", xy_is), ("final", xy_fs)):
        body = head + atoms_part + "\n"
        body += "%d 1 %.6f %.6f %.6f\n" % (n_plus, xy[0], xy[1],
                                           surf_z + height)
        out = Path(out_dir) / ("%s.data" % name)
        out.write_text(body, encoding="utf-8", newline="\n")
        paths.append(out)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True,
                    help="EMCbuild structure.data (atom_style atomic)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--site", default="hollow4",
                    help="site type for both endpoints (default hollow4)")
    ap.add_argument("--height", type=float, default=1.8,
                    help="adatom start height above the surface plane (A)")
    ap.add_argument("--element", default=None,
                    help="adatom element (default: the top-plane element)")
    args = ap.parse_args()

    slab = read(args.data, format="lammps-data", atom_style="atomic")
    n_slab = len(slab)
    element = args.element
    if element is None:
        symbols = slab.get_chemical_symbols()
        element = symbols[surface_top(slab)[0][0]]

    surface_indices, surf_z = surface_top(slab)
    is_site, fs_site, hop_d = pick_endpoints(slab, surface_indices, surf_z,
                                             site=args.site)
    from ase.geometry import get_distances
    pos = slab.positions[surface_indices]
    _, dmat = get_distances(pos, cell=slab.cell, pbc=True)
    nn = float(dmat[dmat > 1e-6].min())  # surface nearest-neighbour spacing

    print("slab atoms: %d; adatom element: %s (id = %d)"
          % (n_slab, element, n_slab + 1))
    print("IS: %s frac=(%.3f, %.3f)"
          % (is_site["info"], *is_site["frac"][:2]))
    print("FS: %s frac=(%.3f, %.3f)"
          % (fs_site["info"], *fs_site["frac"][:2]))
    print("hop distance: %.3f A" % hop_d)
    if abs(hop_d - nn) > 0.05 * nn:
        print("WARNING: hop distance deviates from the surface nearest-"
              "neighbour spacing (%.3f A) - check the enumerated sites" % nn)

    xy_is, xy_fs = np.array(is_site["xy"]), np.array(fs_site["xy"])
    paths = splice_endpoints(args.data, args.out_dir, n_slab, xy_is, xy_fs,
                             surf_z, args.height, element)
    for p in paths:
        print("wrote %s (%d atoms, adatom id = %d)"
              % (p, n_slab + 1, n_slab + 1))


if __name__ == "__main__":
    main()
