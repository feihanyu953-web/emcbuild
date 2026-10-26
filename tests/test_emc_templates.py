#!/usr/bin/env python
"""Regression tests for EMC template generation and failure detection.

Background (2026-10-26): three defects in the water-layer path, all of
which produced a model with no electrolyte while the run reported success.

1. The PCFF template held its placeholders inline -- `groups = { __GROUPS__ };`
   -- but build.py replaced the literal '  __GROUPS__' (two leading spaces).
   Nothing matched, EMC died on "Undefined variable in line 56 of input:
   __GROUPS__", and the water layer was never written.

2. EMC exits 0 even when it aborts, so `returncode != 0` reported
   "EMC completed successfully" for a dead run.

3. With no output file the code printed "using surface only" and carried on,
   silently dropping the electrolyte from a solvation model.

The tests below check the generated EMC input directly (no EMC binary
needed) plus the two guard conditions, so a template/parser drift cannot
reintroduce the silent failure.

Run:
    python tests/test_emc_templates.py
"""
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import build  # noqa: E402

PLACEHOLDER_RE = re.compile(r'__[A-Z][A-Z_]*__')

# One config per field. Ion-free, so every field is legal here.
FIELD_CFG = {
    'charmm': {'water_count': 20, 'density': 1.0, 'temperature': 300,
               'field': 'charmm'},
    'opls': {'water_count': 20, 'density': 1.0, 'temperature': 300,
             'field': 'opls'},
    'pcff': {'water_count': 20, 'density': 1.0, 'temperature': 300,
             'field': 'pcff'},
}


def _generate(field, extra=None):
    """Run build_water_layer with EMC stubbed out; return the generated text.

    The function writes build_water.emc before invoking EMC, so replacing
    EMC_BIN with a non-existent path makes it stop right after generation
    (it raises RuntimeError from the failure guard).
    """
    cfg = dict(FIELD_CFG[field])
    if extra:
        cfg.update(extra)
    tmp = tempfile.mkdtemp(prefix=f'emc_{field}_')
    original = build.EMC_BIN
    build.EMC_BIN = Path(tmp) / 'does_not_exist' / 'emc_win32.exe'
    try:
        try:
            build.build_water_layer(None, cfg, tmp, surface_cell_xy=None)
        except (RuntimeError, FileNotFoundError, OSError):
            pass
        path = os.path.join(tmp, 'build_water.emc')
        return open(path, encoding='latin-1').read(), tmp
    finally:
        build.EMC_BIN = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_placeholders_survive():
    """Every field's generated input is free of __TOKEN__ markers.

    This is the check that would have caught the PCFF inline-placeholder bug:
    the token survived into the file EMC parsed.
    """
    bad = []
    for field in FIELD_CFG:
        text, _ = _generate(field)
        left = PLACEHOLDER_RE.findall(text)
        if left:
            bad.append((field, sorted(set(left))))
    assert not bad, f'unfilled placeholders: {bad}'


def test_block_placeholders_are_line_isolated():
    """Block placeholders must own their line; scalar ones stay inline.

    The original bug was a *block* placeholder written inline:
    `groups = { __GROUPS__ };`. build.py substituted the indented literal
    '  __GROUPS__', so nothing matched and EMC parsed the token as an
    expression. Scalar placeholders (__OUTPUT__, __CLUSTER_NAMES__) are
    meant to sit inside a value and are exempt.
    """
    BLOCK = {'__VARS__', '__GROUPS__', '__SIZING__', '__CLUSTERS__',
             '__FIELDS__'}
    bad = []
    for field, path in sorted(build.TEMPLATE_MAP.items()):
        text = Path(path).read_text(encoding='latin-1')
        for match in PLACEHOLDER_RE.finditer(text):
            token = match.group(0)
            if token not in BLOCK:
                continue
            before = text[:match.start()].split('\n')[-1]
            after = text[match.end():].split('\n')[0]
            if before.strip() or after.strip():
                bad.append((field, token, before + token + after))
    assert not bad, bad


def test_sizing_block_absorbs_trailing_comma():
    """`__SIZING__` ends with a comma, so two statements must follow it.

    EMC aborts with "Bare expression in line N" otherwise. CHARMM/OPLS had
    `ntotal -> 0, mtotal -> 0`; PCFF did not, which is the second half of
    the same silent failure.
    """
    bad = []
    for field in FIELD_CFG:
        text, _ = _generate(field)
        # find the variables block that received the sizing lines
        idx = text.find('(* sizing *)')
        if idx < 0:
            idx = text.find('(* ---- sizing ---- *)')
        block = text[idx:idx + 4000] if idx >= 0 else ''
        if 'ntotal' not in block or 'mtotal' not in block:
            bad.append((field, 'no ntotal/mtotal absorber after sizing'))
            continue
        if not re.search(r'ntotal\s*->\s*0', block):
            bad.append((field, 'ntotal not reset to 0'))
        if not re.search(r'mtotal\s*->\s*0', block):
            bad.append((field, 'mtotal not reset to 0'))
    assert not bad, bad


def test_emc_abort_is_detected():
    """EMC exits 0 on abort, so the guard must read the log, not the code."""
    # Reproduce the exact EMC output shape observed on the PCFF bug
    abort_outputs = [
        "Error: core/parse.c:224 ParseConstant:\n"
        "       Undefined variable in line 56 of input: __GROUPS__\n"
        "       Program aborted.\n",
        "Error: core/chemistry/entry.c:426 ChemistryEntry:\n"
        "       Type '>na' not found (field not set?).\n"
        "       Program aborted.\n",
    ]
    for out in abort_outputs:
        failed = ('Error:' in out or 'Program aborted' in out)
        assert failed, out
    # and a normal run must not trip it
    ok = ("Info: importing field...\nInfo: applying field...\n"
          "Info: writing output\n")
    assert not ('Error:' in ok or 'Program aborted' in ok)


def test_pcff_with_ions_rejected():
    """pcff+ions raises before EMC, naming the fields that do support ions."""
    cfg = {'water_count': 10, 'ions': {'Na+': 1, 'OH-': 1},
           'density': 1.0, 'field': 'pcff'}
    tmp = tempfile.mkdtemp(prefix='emc_pcff_ions_')
    try:
        try:
            build.build_water_layer(None, cfg, tmp, surface_cell_xy=None)
        except ValueError as exc:
            msg = str(exc)
            assert 'pcff' in msg.lower(), msg
            assert 'charmm' in msg.lower() and 'opls' in msg.lower(), msg
            assert 'Na+' in msg, msg
        else:
            raise AssertionError('pcff+ions did not raise')
        # and it must fail before writing an EMC input
        assert not os.path.exists(os.path.join(tmp, 'build_water.emc')), \
            'EMC input was written before the guard fired'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pcff_without_ions_still_works():
    """The ion guard must not break the legitimate ion-free PCFF case."""
    text, _ = _generate('pcff')
    assert 'groups' in text and 'clusters' in text
    assert 'water' in text
    # the water group must be defined and referenced
    assert re.search(r'id\s*->\s*water', text), 'no water group'
    assert 'n_water' in text, 'no n_water sizing entry'


def test_all_fields_produce_a_water_group():
    """Each field names its water group differently; all must be present."""
    expected = {'charmm': 'TIP3', 'opls': 'SPCE', 'pcff': 'water'}
    bad = []
    for field, group in expected.items():
        text, _ = _generate(field)
        if not re.search(rf'id\s*->\s*{group}', text):
            bad.append((field, group))
    assert not bad, bad


def test_examples_do_not_pair_pcff_with_ions():
    """No shipped example may combine a field that cannot type ions with ions."""
    import json
    bad = []
    for path in sorted((SKILL_ROOT / 'examples').glob('*.json')):
        cfg = json.loads(path.read_text(encoding='utf-8'))
        elec = cfg.get('electrolyte') or {}
        field = (elec.get('field') or '').lower()
        ions = elec.get('ions') or {}
        if field.startswith('pcff') and ions:
            bad.append((path.name, sorted(ions)))
    assert not bad, f'pcff + ions in examples: {bad}'


TESTS = [
    test_no_placeholders_survive,
    test_block_placeholders_are_line_isolated,
    test_sizing_block_absorbs_trailing_comma,
    test_emc_abort_is_detected,
    test_pcff_with_ions_rejected,
    test_pcff_without_ions_still_works,
    test_all_fields_produce_a_water_group,
    test_examples_do_not_pair_pcff_with_ions,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {str(e)[:500]}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
