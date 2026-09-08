#!/usr/bin/env python3
"""Cross-platform installer for the ``bs_blender_addon``.

Creates a symlink (or directory junction on Windows) so Blender
discovers the addon.  Works on **Linux**, **macOS**, and **Windows**.

Once the link exists and the addon is enabled in Blender, everything
else is automatic — the addon's own Python bootstrap handles:

- installing pip (if missing from Blender's bundled Python)
- pip-installing ``freemocap_blender_addon`` from GitHub
- pip-installing all PyPI dependencies (polars, pydantic, etc.)
- git-cloning source repos to ``~/.cache/freemocap/git_sources/``
- keeping everything up-to-date on every Blender launch

Usage
-----
.. code-block:: bash

    python install_blender_addon.py
    python install_blender_addon.py --blender-version 5.1

Platform notes
--------------
- **Linux**: symlink → ``~/.config/blender/<ver>/scripts/addons/``
- **macOS**: symlink → ``~/Library/Application Support/Blender/<ver>/scripts/addons/``
- **Windows**: directory junction → ``%APPDATA%\\Blender Foundation\\Blender\\<ver>\\scripts\\addons\\``
  (Junctions don't require admin — regular symlinks on Windows do.)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ADDON_NAME = "bs_blender_addon"
DEFAULT_BLENDER_VERSION = "5.1"


# ── Platform detection ───────────────────────────────────────────────

def _blender_addons_dir(version: str) -> Path:
    """Return the platform-specific Blender addons directory."""
    if sys.platform == "win32":
        appdata = Path(_require_env("APPDATA"))
        return appdata / "Blender Foundation" / "Blender" / version / "scripts" / "addons"
    elif sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Blender"
            / version
            / "scripts"
            / "addons"
        )
    else:  # Linux / BSD
        return Path.home() / ".config" / "blender" / version / "scripts" / "addons"


# ── Link creation ─────────────────────────────────────────────────────

def _create_link(target: Path, link_path: Path) -> None:
    """Create *link_path* → *target*.  Uses a directory junction on Windows
    so admin privileges are NOT required."""
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        # ``mklink /J`` creates a directory junction — works without admin
        _run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
            description=f"mklink /J {link_path}",
        )
    else:
        link_path.symlink_to(target, target_is_directory=True)

    print(f"  {link_path} → {target}")


def _read_link_target(link_path: Path) -> str | None:
    """Return the target of *link_path* or ``None`` if it isn't a link."""
    try:
        return str(link_path.readlink())
    except OSError:
        return None


def install_addon(
    addon_src: Path,
    addons_dir: Path,
    *,
    force: bool = False,
) -> bool:
    """Create (or re-create) the addon link.  Returns ``True`` if a change
    was made."""
    link_path = addons_dir / ADDON_NAME
    target_str = str(addon_src.resolve())

    # ── Already linked correctly? ──
    if _read_link_target(link_path) == target_str:
        print(f"[OK] Already linked: {link_path}")
        return False

    # ── Link exists but points elsewhere ──
    if link_path.is_symlink() or link_path.exists():
        current = _read_link_target(link_path) or "<not a link>"
        if not force:
            print(
                f"[WARN] {link_path} already exists\n"
                f"       Current target: {current}\n"
                f"       Desired target: {target_str}\n"
                f"       Re-run with --force to replace."
            )
            return False
        print(f"[!] Replacing: {current} → {target_str}")
        _remove(link_path)

    # ── Create the link ──
    _create_link(addon_src.resolve(), link_path)
    return True


# ── Utilities ─────────────────────────────────────────────────────────

def _remove(path: Path) -> None:
    """Remove a file, directory, or link."""
    if path.is_symlink():
        path.unlink()
    elif path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _require_env(name: str) -> str:
    """Return an env var or exit with a clear error."""
    value = os.environ.get(name)
    if not value:
        print(f"[ERROR] Environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return value


def _run(cmd: list[str], *, description: str = "") -> None:
    """Run a subprocess command, printing what we're doing."""
    label = description or " ".join(cmd)
    print(f"  $ {label}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {label}\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the BS Blender addon (cross-platform).",
    )
    parser.add_argument(
        "--blender-version",
        default=DEFAULT_BLENDER_VERSION,
        help=f"Blender version (default: {DEFAULT_BLENDER_VERSION})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing symlink/link that points elsewhere",
    )
    args = parser.parse_args()

    # Locate the addon source (lives next to this script)
    script_dir = Path(__file__).resolve().parent
    addon_src = script_dir / "python_code" / "viz" / "blender" / ADDON_NAME

    if not addon_src.is_dir():
        print(
            f"[ERROR] Addon source not found:\n"
            f"  {addon_src}\n\n"
            f"  Make sure this script lives in the root of the 'bs' repository.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Platform:     {sys.platform}")
    print(f"Addon source: {addon_src}")
    print(f"Blender ver:  {args.blender_version}")

    addons_dir = _blender_addons_dir(args.blender_version)
    print(f"Addons dir:   {addons_dir}")
    print()

    changed = install_addon(addon_src, addons_dir, force=args.force)
    if changed or args.force:
        print()
        print("Next steps:")
        print("  1. Open Blender")
        print("  2. Edit → Preferences → Add-ons")
        print(f"  3. Search '{ADDON_NAME}' and enable the checkbox")
        print("  4. The addon auto-installs all dependencies on first run")
    print()


import os

if __name__ == "__main__":
    main()
