#!/usr/bin/env python
"""
EMCbuild Setup - One-shot silent installer.

Detects OS, extracts EMC, installs Python dependencies, prompts for MP API key.

Usage:
    python setup.py
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


EMC_HOME = Path.home() / ".emcbuild" / "emc"
CONFIG_PATH = Path.home() / ".emcbuild" / "config.json"
SKILL_ROOT = Path(__file__).parent
PACKAGES_DIR = SKILL_ROOT / "emc_packages"


def step(msg):
    print(f"  {msg}")


def detect_os():
    """Detect OS and return package filename."""
    system = platform.system()
    if system == "Windows":
        return "emc_win32_v9.4.4.tgz", "emc_win32.exe"
    elif system == "Darwin":
        return "emc_macos_v9.4.4.tgz", "emc_macos"
    elif system == "Linux":
        print("  Warning: Linux not fully supported in this release.")
        print("  EMC for Linux must be installed manually from https://montecarlo.sourceforge.net/")
        return None, None
    else:
        print(f"  Error: Unsupported OS: {system}")
        sys.exit(1)


def extract_emc(package_name: str, binary_name: str):
    """Extract EMC from bundled tgz to ~/.emcbuild/emc/."""
    tgz_path = PACKAGES_DIR / package_name
    if not tgz_path.exists():
        print(f"  Error: EMC package not found: {tgz_path}")
        print(f"  Expected at: {tgz_path}")
        sys.exit(1)

    if EMC_HOME.exists():
        existing_bin = EMC_HOME / "bin" / binary_name
        if existing_bin.exists():
            step(f"EMC already installed at {EMC_HOME}, skipping extraction")
            return

    EMC_HOME.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {tgz_path.name} ({tgz_path.stat().st_size // 1024 // 1024} MB)...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=EMC_HOME)

    # Verify
    binary = EMC_HOME / "bin" / binary_name
    if binary.exists():
        step(f"EMC installed: {binary}")
    else:
        # Search for the binary if not at expected path
        found = list(EMC_HOME.rglob(binary_name))
        if found:
            step(f"EMC found at: {found[0]}")
        else:
            step(f"Warning: EMC binary '{binary_name}' not found after extraction")
            step(f"Contents of {EMC_HOME}:")
            for p in sorted(EMC_HOME.rglob("*"))[:20]:
                print(f"    {p.relative_to(EMC_HOME)}")
            print(f"  You may need to adjust the binary path manually.")


def install_python_deps():
    """Install required Python packages."""
    deps = ["ase", "numpy", "pymatgen", "mp-api"]
    print(f"  Installing: {' '.join(deps)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + deps)
    step("Python dependencies installed")


def prompt_api_key():
    """Prompt user for Materials Project API key and save to config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            existing = json.load(f)
        key = existing.get("materials_project", {}).get("api_key", "")
        if key and key != "YOUR_API_KEY_HERE":
            step("MP API key already configured, skipping")
            return

    print()
    print("  ─────────────────────────────────────────────")
    print("  Materials Project API Key")
    print("  ─────────────────────────────────────────────")
    print("  The Materials Project provides 150K+ computed crystal structures.")
    print("  Get a free API key at: https://materialsproject.org/api")
    print("  (Press Enter to skip — you can set it later via MP_API_KEY env var)")
    print()
    key = input("  Enter your MP API key: ").strip()

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            cfg = json.load(f)
    cfg["materials_project"] = cfg.get("materials_project", {})
    if key:
        cfg["materials_project"]["api_key"] = key
    else:
        cfg["materials_project"]["api_key"] = "YOUR_API_KEY_HERE"

    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    step(f"Config saved to {CONFIG_PATH}")


def main():
    print("=" * 60)
    print("  EMCbuild Setup")
    print("=" * 60)
    print()

    # [1/4] Detect OS
    print("[1/4] Detecting platform...")
    package, binary = detect_os()
    if package is None:
        return
    step(f"{platform.system()} — {binary}")
    print()

    # [2/4] Extract EMC
    print("[2/4] Installing EMC...")
    extract_emc(package, binary)
    print()

    # [3/4] Install Python dependencies
    print("[3/4] Installing Python packages...")
    install_python_deps()
    print()

    # [4/4] API key
    print("[4/4] Configuring Materials Project API key...")
    prompt_api_key()

    print()
    print("=" * 60)
    print("  Setup complete! Usage:")
    print(f"    python {SKILL_ROOT / 'scripts' / 'build.py'} <config.json>")
    print("=" * 60)


if __name__ == "__main__":
    main()
