#!/usr/bin/env python3
"""Print dependency JSON for CI / GUI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from instasplat.utils.deps import report_dict  # noqa: E402


def main() -> None:
    print(json.dumps(report_dict(), indent=2))


if __name__ == "__main__":
    main()
