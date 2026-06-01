#!/usr/bin/env python3
"""Print the canonical SHA-256 hash for a freeze config JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_json_bytes(path: Path) -> bytes:
    data = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return canonical.encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    digest = hashlib.sha256(canonical_json_bytes(args.config)).hexdigest()
    print(digest)


if __name__ == "__main__":
    main()
