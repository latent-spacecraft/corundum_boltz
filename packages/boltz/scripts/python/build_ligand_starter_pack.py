#!/usr/bin/env python3
"""
Build the curated ligand starter pack.

Reads CCD codes from ligand_starter_pack.txt (one per line, # comments OK)
and runs extract_ligand_blob on each. Writes each blob to public/ccd/<CODE>.json
and a summary of what landed.

Usage:
    /path/to/boltz-dev/.venv/bin/python scripts/python/build_ligand_starter_pack.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the single-ligand extractor's helpers.
sys.path.insert(0, str(Path(__file__).parent))
from extract_ligand_blob import extract_blob, write_blob, summarise, MOL_DIR


def read_pack_list(path: Path) -> list[str]:
    codes: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        codes.append(line.upper())
    return codes


def main() -> None:
    here = Path(__file__).parent
    pack_list = here / "ligand_starter_pack.txt"
    codes = read_pack_list(pack_list)
    print(f"Starter pack: {len(codes)} ligands\n")

    ok: list[str] = []
    missing: list[str] = []
    failed: list[tuple[str, str]] = []

    for code in codes:
        pkl = MOL_DIR / f"{code}.pkl"
        if not pkl.exists():
            missing.append(code)
            print(f"  ? {code}: cache miss")
            continue
        try:
            blob = extract_blob(code)
            write_blob(code, blob)
            print(summarise(code, blob))
            ok.append(code)
        except Exception as e:
            failed.append((code, str(e)))
            print(f"  ! {code}: {e}")

    print()
    print(f"Built {len(ok)} / {len(codes)} blobs.")
    if missing:
        print(f"  Missing from cache ({len(missing)}): {' '.join(missing)}")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for code, err in failed:
            print(f"    {code}: {err}")


if __name__ == "__main__":
    main()
