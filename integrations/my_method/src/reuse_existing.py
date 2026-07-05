from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from .utils import resolve_path


def resolved_paths(config: Dict[str, Any], paths: Iterable[str | Path]) -> list[Path]:
    out = []
    for path in paths:
        p = Path(path)
        out.append(p if p.is_absolute() else resolve_path(config, str(p)))
    return out


def all_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def reuse_if_exists(
    config: Dict[str, Any],
    paths: Iterable[str | Path],
    *,
    label: str,
    reuse_existing: bool = False,
    force: bool = False,
) -> bool:
    """Return True when a caller should skip because all products exist."""
    if force or not reuse_existing:
        return False
    resolved = resolved_paths(config, paths)
    if all_exist(resolved):
        print(f"[reuse_existing] {label}: found existing product(s), reusing.")
        for path in resolved:
            print(f"  - {path}")
        return True
    missing = [path for path in resolved if not path.exists()]
    print(f"[reuse_existing] {label}: missing {len(missing)} product(s), recomputing.")
    for path in missing:
        print(f"  - missing: {path}")
    return False
