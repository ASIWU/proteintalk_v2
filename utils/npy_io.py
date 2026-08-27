#!/usr/bin/env python3
"""Small numpy I/O helpers shared by training and inference code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def safe_np_load(path: str | Path, *args: Any, mmap_mode: str | None = None, **kwargs: Any) -> Any:
    """Load a numpy artifact, falling back when filesystem mmap is unavailable."""

    path = Path(path)
    if mmap_mode is None:
        return np.load(path, *args, **kwargs)
    try:
        return np.load(path, *args, mmap_mode=mmap_mode, **kwargs)
    except OSError as exc:
        print(f"[warn] mmap load failed for {path}: {exc}; falling back to normal np.load")
        return np.load(path, *args, **kwargs)
