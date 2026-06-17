#!/usr/bin/env python3
"""
Build a Boltz-2 ligand blob from a user SMILES (or a 3D mol file: .pdb/.sdf/
.mol), for ligands NOT in the CCD. Thin CLI over `ligand_blob.py` (the shared
RDKit-only core used by both this CLI and the deploy endpoint, server/app.py).

The core mirrors Boltz's exact SMILES branch (MolFromSmiles → AddHs →
CanonicalRank names → ETKDGv3 conformer) and reuses its constraint extractors,
so the emitted blob is bit-compatible with the CCD blobs the TS featurizer
consumes. Hydrogens are dropped (Boltz ligands are heavy-atom only). Output
JSON is shaped like the CCD blobs, so the TS side needs no changes.

Usage:
    .../python smiles_to_blob.py --code ZFB --smiles "Cc1[nH]c2c(C#N)..."
    .../python smiles_to_blob.py --code ZFB --molfile ../../zfBxCS.pdb
    .../python smiles_to_blob.py --smiles "c1ccccc1" --stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ligand_blob

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "public" / "ccd"


def write_blob(code: str, blob: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{code}.json"
    wrapped = {
        "_doc": (
            f"Custom (non-CCD) ligand blob for {code}, built from user input via "
            f"smiles_to_blob.py. Atoms/bonds plus the six geometry constraint "
            f"groups Boltz-2 expects on the trunk's small-molecule input tensors. "
            f"Indices are 0-based within this ligand; the TS featurizer adds the "
            f"global atom offset when concatenating into the prediction."
        ),
        "data": blob,
    }
    path.write_text(json.dumps(wrapped, separators=(",", ":")))
    return path


def summarise(code: str, blob: dict) -> str:
    return (
        f"  {code}: {blob['num_atoms']} atoms, {len(blob['bonds'])} bonds, "
        f"{len(blob['rdkit_bounds'])} bounds, "
        f"{len(blob['chiral_atoms'])} chir, "
        f"{len(blob['stereo_bonds'])} stereo, "
        f"{len(blob['planar_bonds'])} planar-bonds, "
        f"{len(blob['planar_rings_5'])}/{len(blob['planar_rings_6'])} ring5/ring6"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", default=None, help="ligand code; default = hash of canonical SMILES")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="SMILES string")
    src.add_argument("--molfile", help="3D ligand file (.pdb/.sdf/.mol)")
    ap.add_argument("--use-file-conformer", action="store_true",
                    help="keep the input file's coordinates instead of regenerating via ETKDG")
    ap.add_argument("--stdout", action="store_true",
                    help="emit blob JSON on stdout (for the endpoint) instead of writing a file")
    args = ap.parse_args()

    if args.smiles:
        result = ligand_blob.build_from_smiles(args.smiles, args.code)
    else:
        result = ligand_blob.build_from_molfile(args.molfile, args.code, args.use_file_conformer)

    code, blob = result["code"], result["data"]

    if args.stdout:
        json.dump({"ok": True, **result}, sys.stdout, separators=(",", ":"))
        return

    path = write_blob(code, blob)
    print(summarise(code, blob))
    print(f"  → wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — endpoint needs a clean JSON error
        if "--stdout" in sys.argv:
            json.dump({"ok": False, "error": str(e)}, sys.stdout, separators=(",", ":"))
            sys.exit(0)
        raise
