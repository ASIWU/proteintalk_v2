#!/usr/bin/env python3
"""Rewrite explicit repository output references using an organization manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
HKT = ZoneInfo("Asia/Hong_Kong")
DEFAULT_SCAN_ROOTS = (
    REPO_ROOT / "docs",
    REPO_ROOT / "data/review_summary",
    REPO_ROOT / "scripts",
    REPO_ROOT / "utils",
    REPO_ROOT / "new_version",
)
TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".txt"}
DATED_OUTPUT_REFERENCE = re.compile(r"outputs/(20\d{2})(\d{2})(\d{2})(?!\d)")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_files(scan_roots: list[Path]) -> list[Path]:
    result: set[Path] = set()
    for root in scan_roots:
        if root.is_file() and root.suffix.lower() in TEXT_SUFFIXES:
            result.add(root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                result.add(path)
    for path in REPO_ROOT.iterdir():
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            result.add(path)
    return sorted(result)


def load_replacements(manifest: Path) -> list[tuple[str, str]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    moves = payload.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError(f"manifest has no moves: {manifest}")
    replacements = [(str(row["old_path"]), str(row["new_path"])) for row in moves]
    return sorted(replacements, key=lambda pair: len(pair[0]), reverse=True)


def rewrite_text(text: str, replacements: list[tuple[str, str]]) -> tuple[str, int]:
    count = 0
    updated = text
    for old, new in replacements:
        occurrences = updated.count(old)
        if occurrences:
            updated = updated.replace(old, new)
            count += occurrences
    updated, dated_count = DATED_OUTPUT_REFERENCE.subn(
        lambda match: (
            f"outputs/{match.group(1)}-{match.group(2)}/"
            f"{match.group(1)}-{match.group(2)}-{match.group(3)}/"
            f"{match.group(1)}{match.group(2)}{match.group(3)}"
        ),
        updated,
    )
    count += dated_count
    return updated, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan-root", action="append", type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    replacements = load_replacements(manifest)
    scan_roots = [path.expanduser().resolve() for path in (args.scan_root or list(DEFAULT_SCAN_ROOTS))]
    changes: list[dict[str, Any]] = []
    for path in candidate_files(scan_roots):
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated, count = rewrite_text(original, replacements)
        if not count:
            continue
        changes.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "replacement_count": count,
                "sha256_before": sha256_text(original),
                "sha256_after": sha256_text(updated),
            }
        )
        if args.execute:
            path.write_text(updated, encoding="utf-8")

    payload = {
        "generated_at_hkt": datetime.now(HKT).replace(microsecond=0).isoformat(),
        "status": "executed" if args.execute else "dry_run",
        "manifest": str(manifest),
        "scanned_files": len(candidate_files(scan_roots)),
        "changed_files": len(changes),
        "replacement_count": sum(row["replacement_count"] for row in changes),
        "changes": changes,
    }
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "changed_files", "replacement_count")}, indent=2))


if __name__ == "__main__":
    main()
