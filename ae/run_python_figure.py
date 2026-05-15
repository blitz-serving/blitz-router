#!/usr/bin/env python3
"""Run an xmetric-plots figure script with Linux-friendly plotting defaults."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def patch_matplotlib_fonts() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    import matplotlib.font_manager as fm

    original = fm.FontProperties

    class SafeFontProperties(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            fname = kwargs.get("fname")
            if fname and not os.path.exists(fname):
                kwargs.pop("fname")
                kwargs.setdefault("family", "DejaVu Sans")
            super().__init__(*args, **kwargs)

    fm.FontProperties = SafeFontProperties  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Path to a figure .py script")
    args = parser.parse_args()

    script = Path(args.script).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    patch_matplotlib_fonts()
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))
    sys.path.insert(0, str(script.parent.parent))
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
