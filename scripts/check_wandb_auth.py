#!/usr/bin/env python3
"""Fail early when online W&B logging is enabled without credentials."""

from __future__ import annotations

import argparse
import netrc
import os
from pathlib import Path
from urllib.parse import urlparse


def wandb_enabled(logger_backend: str, log_to_wandb: str, wandb_mode: str) -> bool:
    if wandb_mode.lower() in {"disabled", "offline"}:
        return False
    if str(log_to_wandb).strip() == "1":
        return True
    return logger_backend.lower() in {"wandb", "both"}


def netrc_has_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    try:
        auth = netrc.netrc().authenticators(hostname)
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    return auth is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check W&B credentials before non-interactive training")
    parser.add_argument("--logger-backend", default="wandb")
    parser.add_argument("--log-to-wandb", default="1")
    parser.add_argument("--wandb-mode", default="")
    parser.add_argument("--wandb-env-file", default="scripts/wandb_env.local")
    args = parser.parse_args()

    if not wandb_enabled(args.logger_backend, args.log_to_wandb, args.wandb_mode):
        print("[wandb] online logging disabled or offline; credential check skipped")
        return 0
    if os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY is configured")
        return 0

    base_url = os.environ.get("WANDB_BASE_URL") or "https://api.wandb.ai"
    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    hostname = parsed.hostname
    if netrc_has_host(hostname):
        print(f"[wandb] credentials found in ~/.netrc for host={hostname}")
        return 0

    env_path = Path(args.wandb_env_file)
    print(
        "[wandb] ERROR: online W&B logging is enabled, but no API key was found in "
        "WANDB_API_KEY or ~/.netrc.\n"
        "This will fail in non-interactive 8-GPU jobs with: api_key not configured (no-tty).\n"
        f"Fix one of these before launching:\n"
        f"  1. export WANDB_API_KEY=... and optionally WANDB_BASE_URL=...\n"
        f"  2. create {env_path} from scripts/wandb_env.local.example\n"
        f"  3. run with WANDB_MODE=offline, or LOGGER_BACKEND=none LOG_TO_WANDB=0\n",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
