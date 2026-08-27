#!/usr/bin/env python
"""Regression tests for the electrolyte box density fix (2026-08-27).

Background: the EMC templates size the box as a cube derived from the requested
density. build_water_layer() overrides lxx/lyy with the surface dimensions to
tile a slab, and MUST rescale lzz in the same step. Forgetting lzz leaves the
cube edge in place, so the real volume is lx*ly*lbox instead of vphase1 and the
electrolyte is silently compressed (measured 2.485 vs 1.000 g/cm3 on a 600-H2O
Pt(100) 6x6 case).

Live templates are read from build.TEMPLATE_MAP so this file cannot drift when
templates are added or renamed.

Fast, needs no EMC binary. Run:
    python tests/test_density_rescale.py
"""

import re
import sys
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import build  # noqa: E402

# The exact strings build_water_layer() feeds to _sed(). If a template is
# reformatted (tabs -> spaces, vars renamed) these stop matching and the
# override silently no-ops -- which is precisely the original bug.
PAT_LXX = "lxx\t\t-> lphase1"
PAT_LYY = "lyy\t\t-> lbox"
PAT_LZZ = "lzz\t\t-> lbox"

# Minimal electrolyte configs, one per field routed by TEMPLATE_MAP
FIELD_CFG = {
    "charmm": {"water_count": 600, "ions": {"Na+": 1, "OH-": 1},
               "density": 1.0, "temperature": 300, "field": "charmm"},
    "pcff": {"water_count": 100, "density": 1.0, "field": "pcff"},
    "opls": {"water_count": 100, "density": 1.0, "field": "opls"},
}

_failures = []


def check(cond, label, detail=""):
    if cond:
        print("  ok    %s" % label)
    else:
        print("  FAIL  %s %s" % (label, detail))
        _failures.append(label)


def test_live_templates_carry_sed_patterns():
    """Every template reachable via TEMPLATE_MAP must expose the sed anchors."""
    print("\n[1] live templates expose the lxx/lyy/lzz sed patterns")
    for field, path in sorted(build.TEMPLATE_MAP.items()):
        name = Path(path).name
        if not Path(path).exists():
            check(False, "%s -> %s" % (field, name), "(missing template)")
            continue
        text = Path(path).read_text(encoding="latin-1")
        check(PAT_LXX in text, "%s/%s: lxx pattern" % (field, name))
        check(PAT_LYY in text, "%s/%s: lyy pattern" % (field, name))
        check(PAT_LZZ in text, "%s/%s: lzz pattern" % (field, name))
        # vphase1 must be defined BEFORE lzz or the EMC expression
        # `lzz -> vphase1/(lx*ly)` cannot resolve at that point.
        i_v = text.find("vphase1\t-> mphase1")
        i_z = text.find(PAT_LZZ)
        check(i_v != -1 and i_z != -1 and i_v < i_z,
              "%s/%s: vphase1 defined before lzz" % (field, name),
              "(vphase1@%d lzz@%d)" % (i_v, i_z))


def test_all_templates_ordering_invariant():
    """Any other *.emc must also satisfy vphase1-before-lzz.

    Informational/future-proofing: templates not yet in TEMPLATE_MAP (e.g. an
    unused build_water_pcff.emc, or the molecule builders) should already
    satisfy the ordering so wiring one in later cannot silently break density.
    """
    print("\n[2] non-live templates satisfy the same ordering invariant")
    live = {Path(p).resolve() for p in build.TEMPLATE_MAP.values()}
    others = sorted(p for p in (SKILL_ROOT / "templates").glob("*.emc")
                    if p.resolve() not in live)
    if not others:
        print("  (none)")
        return
    for p in others:
        text = p.read_text(encoding="latin-1")
        if PAT_LZZ not in text:
            print("  skip  %s (no 'lzz -> lbox')" % p.name)
            continue
        i_v = text.find("vphase1\t-> mphase1")
        i_z = text.find(PAT_LZZ)
        check(i_v != -1 and i_z != -1 and i_v < i_z,
              "%s: vphase1 before lzz" % p.name,
              "(vphase1@%d lzz@%d)" % (i_v, i_z))


def test_build_water_layer_rescales_lzz():
    """build_water_layer() itself must emit an lzz bound to vphase1.

    This calls the real function so that deleting the rescale from build.py
    fails the test. An earlier version applied the _sed calls locally and
    therefore passed even with the fix removed (confirmed by mutation testing)
    -- do not weaken it back to that form.

    EMC_BIN is pointed at a nonexistent path so the subprocess call fails
    immediately; build_water.emc is fully written before EMC is invoked, so the
    generated file is still inspectable and no EMC install is required.
    """
    print("\n[3] build_water_layer() rescales lzz (integration)")
    lx, ly = 16.6312, 16.6312
    saved_bin = build.EMC_BIN
    build.EMC_BIN = Path(tempfile.gettempdir()) / "__emc_does_not_exist__.exe"
    try:
        for field in sorted(build.TEMPLATE_MAP):
            cfg = FIELD_CFG.get(field)
            if cfg is None:
                print("  skip  %s (no test config)" % field)
                continue
            with tempfile.TemporaryDirectory() as td:
                try:
                    build.build_water_layer(None, cfg, td, (lx, ly))
                except Exception:
                    pass  # EMC launch failure expected and irrelevant
                emc = Path(td) / "build_water.emc"
                if not emc.exists():
                    check(False, "%s: build_water.emc generated" % field)
                    continue
                out = emc.read_text(encoding="latin-1")
                check(PAT_LZZ not in out,
                      "%s: no bare 'lzz -> lbox' left" % field)
                check("lzz\t\t-> vphase1/(" in out,
                      "%s: lzz bound to vphase1" % field)
                check(re.search(r"lxx\t\t-> 16\.6312", out) is not None,
                      "%s: lxx overridden to surface x" % field)
                check(re.search(r"lyy\t\t-> 16\.6312", out) is not None,
                      "%s: lyy overridden to surface y" % field)
    finally:
        build.EMC_BIN = saved_bin


def test_electrolyte_only_keeps_template_cube():
    """With surface_cell_xy=None the template cube must stay untouched.

    Electrolyte-only mode never had the bug: EMC's own cube already satisfies
    the requested density. Rewriting lzz there would be wrong.
    """
    print("\n[4] electrolyte-only mode leaves the cube alone")
    cfg = {"water_count": 100, "density": 1.0, "field": "charmm"}
    saved_bin = build.EMC_BIN
    build.EMC_BIN = Path(tempfile.gettempdir()) / "__emc_does_not_exist__.exe"
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                build.build_water_layer(None, cfg, td, None)
            except Exception:
                pass
            emc = Path(td) / "build_water.emc"
            if not emc.exists():
                check(False, "build_water.emc generated")
                return
            out = emc.read_text(encoding="latin-1")
            check(PAT_LZZ in out, "lzz still template default (lbox)")
            check(PAT_LXX in out, "lxx still template default (lphase1)")
    finally:
        build.EMC_BIN = saved_bin


def _write_data(path, lx, ly, lz, n_h2o):
    """Minimal LAMMPS data file: n_h2o TIP3 waters in an lx*ly*lz box."""
    lines = [
        "LAMMPS test fixture", "",
        "%d  atoms" % (3 * n_h2o), "",
        "2  atom types", "",
        "0 %.6f xlo xhi" % lx,
        "0 %.6f ylo yhi" % ly,
        "0 %.6f zlo zhi" % lz, "",
        "Masses", "",
        "1    1.00790  # HT",
        "2   15.99940  # OT", "",
        "Atoms", "",
    ]
    aid = 0
    for m in range(n_h2o):
        for t in (2, 1, 1):
            aid += 1
            lines.append("%d %d %d 0.0 0.0 0.0 0.0" % (aid, m + 1, t))
    Path(path).write_text("\n".join(lines) + "\n")


def test_verify_box_density_measures_correctly():
    """The guard must reproduce a known density and flag a known bad box."""
    print("\n[5] _verify_box_density measures and flags")
    lx = ly = 16.6312
    n = 600
    mass = n * (15.9994 + 2 * 1.0079)
    lz_good = mass / (1.0 * lx * ly * 0.6022140857)

    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "good.data"
        _write_data(good, lx, ly, lz_good, n)
        d = build._verify_box_density(str(good), 1.0)
        check(abs(d - 1.0) < 0.005, "measures 1.0 g/cm3", "(got %.4f)" % d)

        # The original bug: lzz left at the cube edge (vphase1^(1/3))
        lz_bad = (lx * ly * lz_good) ** (1.0 / 3.0)
        bad = Path(td) / "bad.data"
        _write_data(bad, lx, ly, lz_bad, n)
        d_bad = build._verify_box_density(str(bad), 1.0)
        check(d_bad > 2.0, "flags the squeezed box", "(got %.4f)" % d_bad)
        check(abs(d_bad - 2.48) < 0.1,
              "reproduces the historical 2.48x error", "(got %.4f)" % d_bad)

        junk = Path(td) / "junk.xyz"
        junk.write_text("3\ncomment\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
        d_junk = build._verify_box_density(str(junk), 1.0)
        check(d_junk == 0.0, "returns 0.0 on unparseable input",
              "(got %.4f)" % d_junk)


def main():
    print("EMCbuild density-rescale regression tests")
    print("skill root: %s" % SKILL_ROOT)
    print("live templates: %s"
          % ", ".join("%s=%s" % (f, Path(p).name)
                      for f, p in sorted(build.TEMPLATE_MAP.items())))
    test_live_templates_carry_sed_patterns()
    test_all_templates_ordering_invariant()
    test_build_water_layer_rescales_lzz()
    test_electrolyte_only_keeps_template_cube()
    test_verify_box_density_measures_correctly()
    print("\n%s (%d failure%s)"
          % ("PASS" if not _failures else "FAIL",
             len(_failures), "" if len(_failures) == 1 else "s"))
    if _failures:
        for f in _failures:
            print("  - %s" % f)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
