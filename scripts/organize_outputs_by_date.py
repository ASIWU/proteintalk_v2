#!/usr/bin/env python3
"""Move top-level output artifacts into month/day buckets with an audit manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
HKT = ZoneInfo("Asia/Hong_Kong")
FULL_DATE_PREFIX = re.compile(r"^(20\d{2})(\d{2})(\d{2})(?:_|$)")
SHORT_DATE_PREFIX = re.compile(r"^(\d{2})(\d{2})(?:[^0-9]|$)")
EMBEDDED_FULL_DATE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
MONTH_BUCKET = re.compile(r"^20\d{2}-\d{2}$")
ADMIN_NAMES = {"README.md", "_organization"}


@dataclass(frozen=True)
class Move:
    source: Path
    target: Path
    output_date: date
    date_source: str
    entry_type: str
    inode: int


def parse_date(groups: tuple[str, str, str]) -> date | None:
    try:
        return date(*(int(item) for item in groups))
    except ValueError:
        return None


def infer_output_date(path: Path, *, short_year: int) -> tuple[date, str]:
    match = FULL_DATE_PREFIX.match(path.name)
    if match:
        parsed = parse_date(match.groups())
        if parsed is not None:
            return parsed, "yyyymmdd_prefix"

    match = SHORT_DATE_PREFIX.match(path.name)
    if match:
        parsed = parse_date((str(short_year), *match.groups()))
        if parsed is not None:
            return parsed, "mmdd_prefix"

    match = EMBEDDED_FULL_DATE.search(path.name)
    if match:
        parsed = parse_date(match.groups())
        if parsed is not None:
            return parsed, "embedded_yyyymmdd"

    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=HKT).date()
    return modified, "mtime_hkt_fallback"


def build_moves(output_root: Path, *, short_year: int) -> list[Move]:
    moves: list[Move] = []
    for source in sorted(output_root.iterdir(), key=lambda item: item.name):
        if source.name in ADMIN_NAMES or MONTH_BUCKET.fullmatch(source.name):
            continue
        output_date, date_source = infer_output_date(source, short_year=short_year)
        target = output_root / output_date.strftime("%Y-%m") / output_date.isoformat() / source.name
        entry_type = "symlink" if source.is_symlink() else "directory" if source.is_dir() else "file"
        moves.append(
            Move(
                source=source,
                target=target,
                output_date=output_date,
                date_source=date_source,
                entry_type=entry_type,
                inode=source.stat().st_ino,
            )
        )
    return moves


def validate_moves(output_root: Path, moves: list[Move]) -> None:
    resolved_root = output_root.resolve()
    targets: set[Path] = set()
    for move in moves:
        if move.source.parent.resolve() != resolved_root:
            raise ValueError(f"source is not a direct child of output root: {move.source}")
        if move.target in targets:
            raise ValueError(f"duplicate target in migration plan: {move.target}")
        if move.target.exists() or move.target.is_symlink():
            raise FileExistsError(f"migration target already exists: {move.target}")
        targets.add(move.target)


def recursive_inventory(output_root: Path) -> dict[str, int]:
    files = 0
    directories = 0
    symlinks = 0
    file_bytes = 0
    for path in output_root.rglob("*"):
        if path.is_symlink():
            symlinks += 1
        elif path.is_dir():
            directories += 1
        elif path.is_file():
            files += 1
            file_bytes += path.stat().st_size
    return {
        "files": files,
        "directories": directories,
        "symlinks": symlinks,
        "file_bytes": file_bytes,
    }


def manifest_payload(output_root: Path, moves: list[Move], *, executed: bool) -> dict[str, Any]:
    by_date = Counter(move.output_date.isoformat() for move in moves)
    by_source = Counter(move.date_source for move in moves)
    return {
        "generated_at_hkt": datetime.now(HKT).replace(microsecond=0).isoformat(),
        "status": "executed" if executed else "dry_run",
        "output_root": str(output_root),
        "layout": "outputs/YYYY-MM/YYYY-MM-DD/<original-top-level-name>",
        "move_count": len(moves),
        "counts_by_date": dict(sorted(by_date.items())),
        "counts_by_date_source": dict(sorted(by_source.items())),
        "inventory": recursive_inventory(output_root),
        "moves": [
            {
                "old_path": str(move.source.relative_to(REPO_ROOT)),
                "new_path": str(move.target.relative_to(REPO_ROOT)),
                "date": move.output_date.isoformat(),
                "date_source": move.date_source,
                "entry_type": move.entry_type,
                "inode": move.inode,
            }
            for move in moves
        ],
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute_moves(moves: list[Move]) -> None:
    completed: list[Move] = []
    try:
        for move in moves:
            move.target.parent.mkdir(parents=True, exist_ok=True)
            move.source.rename(move.target)
            completed.append(move)
    except Exception:
        for move in reversed(completed):
            move.source.parent.mkdir(parents=True, exist_ok=True)
            move.target.rename(move.source)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--short-year", type=int, default=2026)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.short_year < 2000 or args.short_year > 2099:
        parser.error("--short-year must be between 2000 and 2099")
    return args


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    moves = build_moves(output_root, short_year=args.short_year)
    validate_moves(output_root, moves)
    before = manifest_payload(output_root, moves, executed=False)
    if not args.execute:
        if args.manifest:
            write_manifest(args.manifest, before)
        print(json.dumps({key: before[key] for key in ("status", "move_count", "counts_by_date_source")}, indent=2))
        return

    execute_moves(moves)
    for move in moves:
        if not move.target.exists() or move.target.stat().st_ino != move.inode:
            raise RuntimeError(f"post-move inode validation failed: {move.target}")
    payload = before
    payload["status"] = "executed"
    payload["completed_at_hkt"] = datetime.now(HKT).replace(microsecond=0).isoformat()
    payload["inventory_after"] = recursive_inventory(output_root)
    manifest = args.manifest or output_root / "_organization" / "latest_output_path_map.json"
    write_manifest(manifest, payload)
    print(json.dumps({"status": "executed", "move_count": len(moves), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
