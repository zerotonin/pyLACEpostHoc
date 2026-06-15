# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — config                                          ║
# ║  « machine-specific paths, resolved not hardcoded »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Resolves data roots in three steps: environment variable, then  ║
# ║  the gitignored local_paths.json, then an in-repo data/ symlink. ║
# ║  Copy local_paths.template.json to local_paths.json to begin.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Resolve machine-specific paths without hardcoding them in tracked code."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PATHS_FILENAME: str = "local_paths.json"
TEMPLATE_FILENAME: str = "local_paths.template.json"
PROFILE_ENV: str = "PYLACE_POSTHOC_PROFILE"
ENV_PREFIX: str = "PYLACE_POSTHOC_"
DEFAULT_PROFILE: str = "local"


class PathConfigError(RuntimeError):
    """Raised when machine-specific paths cannot be resolved."""


def repo_root() -> Path:
    """Return the repository root (the directory holding this module)."""
    return Path(__file__).resolve().parent


def _paths_file(paths_file: Path | None = None) -> Path:
    return Path(paths_file) if paths_file is not None else repo_root() / PATHS_FILENAME


def load_paths(paths_file: Path | None = None) -> dict[str, dict[str, str]]:
    """Load the profile→key→path mapping from ``local_paths.json``.

    Raises:
        PathConfigError: If the file is absent, naming the template to copy.
    """
    path = _paths_file(paths_file)
    if not path.exists():
        raise PathConfigError(
            f"{PATHS_FILENAME} not found at {path}.\n"
            f"Copy {TEMPLATE_FILENAME} to {PATHS_FILENAME} and fill in the "
            f"real paths for your machine before running anything."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def active_profile(profile: str | None = None) -> str:
    """Return the requested profile, the env override, or the default."""
    return profile or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE


def get_path(
    key: str,
    profile: str | None = None,
    paths_file: Path | None = None,
) -> Path:
    """Resolve a single named path.

    Resolution order: ``PYLACE_POSTHOC_<KEY>`` environment variable wins,
    then the ``key`` under the active profile in ``local_paths.json``.

    Args:
        key:        Path name, e.g. ``"database_path"`` or ``"data_root"``.
        profile:    Profile name; defaults to the env override or "local".
        paths_file: Override the JSON location (used in tests).

    Raises:
        PathConfigError: If the key is defined for neither source.
    """
    env_value = os.environ.get(ENV_PREFIX + key.upper())
    if env_value:
        return Path(env_value)

    prof = active_profile(profile)
    paths = load_paths(paths_file)
    if prof not in paths:
        raise PathConfigError(
            f"Profile {prof!r} not in {PATHS_FILENAME}; "
            f"available: {sorted(paths)}."
        )
    if key not in paths[prof]:
        raise PathConfigError(
            f"Key {key!r} not defined for profile {prof!r} in {PATHS_FILENAME}."
        )
    return Path(paths[prof][key])


def export_env(profile: str | None = None, paths_file: Path | None = None) -> str:
    """Render the active profile as ``KEY=VALUE`` lines for shell/SLURM use."""
    prof = active_profile(profile)
    paths = load_paths(paths_file)
    if prof not in paths:
        raise PathConfigError(f"Profile {prof!r} not in {PATHS_FILENAME}.")
    return "\n".join(f"{ENV_PREFIX}{k.upper()}={v}" for k, v in paths[prof].items())


def _main() -> None:
    parser = argparse.ArgumentParser(description="Resolve pyLACEpostHoc paths.")
    parser.add_argument("--export", action="store_true", help="print KEY=VALUE lines")
    parser.add_argument("--profile", default=None, help="profile name (default: local)")
    parser.add_argument("--get", metavar="KEY", default=None, help="print one resolved path")
    args = parser.parse_args()

    if args.get:
        print(get_path(args.get, profile=args.profile))
    elif args.export:
        print(export_env(profile=args.profile))
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
