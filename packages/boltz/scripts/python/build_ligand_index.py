#!/usr/bin/env python3
"""
Build the public/ccd/index.json registry consumed by the in-browser ligand
drawer. One row per CCD entry in the starter pack:

    { ccd, name, synonyms, formula, n_atoms }

`formula` is computed from atom counts in the blob; `name` and `synonyms`
come from a curated map below — kept hand-written rather than fetched from
PDBe so the build stays offline (BYOD). Add new entries to NAMES as the
starter pack grows.

Run AFTER build_ligand_starter_pack.py — reads each emitted blob to count
atoms by element symbol.

Usage:
    /path/to/boltz-dev/.venv/bin/python scripts/python/build_ligand_index.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CCD_DIR = REPO_ROOT / "public" / "ccd"

# Curated names + synonyms. The drawer's search runs over these plus the CCD
# code itself, so the synonym list is the user-facing layer ("heme", "vitamin
# b12", "atp"). Lowercase only; comparison is case-insensitive in JS.
NAMES: dict[str, tuple[str, list[str]]] = {
    # ── Heme variants ─────────────────────────────────────────────────────
    "HEM": ("Heme B", ["heme", "hematin", "protoporphyrin IX iron", "ppix-fe"]),
    "HEC": ("Heme C", ["heme c", "cytochrome c heme"]),
    "HEA": ("Heme A", ["heme a"]),

    # ── Nicotinamide cofactors ────────────────────────────────────────────
    "NAD": ("NAD+", ["nicotinamide adenine dinucleotide", "nad"]),
    "NAP": ("NADP+", ["nicotinamide adenine dinucleotide phosphate"]),
    "NDP": ("NADPH", ["nicotinamide adenine dinucleotide phosphate reduced"]),

    # ── Flavins ───────────────────────────────────────────────────────────
    "FAD": ("FAD", ["flavin adenine dinucleotide"]),
    "FMN": ("FMN", ["flavin mononucleotide", "riboflavin phosphate"]),

    # ── Methyl / acetyl carriers ──────────────────────────────────────────
    "COA": ("Coenzyme A", ["coa", "coenzyme a"]),
    "SAM": ("S-adenosyl-L-methionine", ["sam", "ado-met", "adomet"]),
    "SAH": ("S-adenosyl-L-homocysteine", ["sah", "ado-hcy"]),

    # ── Pyridoxal / biotin / thiamine ─────────────────────────────────────
    "PLP": ("Pyridoxal 5'-phosphate", ["plp", "vitamin b6", "pyridoxal phosphate"]),
    "BTN": ("Biotin", ["vitamin b7", "vitamin h"]),
    "TPP": ("Thiamine pyrophosphate", ["tpp", "thiamine diphosphate", "vitamin b1"]),
    "PQQ": ("Pyrroloquinoline quinone", ["pqq", "methoxatin"]),
    "B12": ("Cyanocobalamin (vitamin B12)", ["vitamin b12", "cobalamin", "b12"]),

    # ── Nucleotides ───────────────────────────────────────────────────────
    "ATP": ("ATP", ["adenosine triphosphate"]),
    "ADP": ("ADP", ["adenosine diphosphate"]),
    "AMP": ("AMP", ["adenosine monophosphate"]),
    "GTP": ("GTP", ["guanosine triphosphate"]),
    "GDP": ("GDP", ["guanosine diphosphate"]),
    "CTP": ("CTP", ["cytidine triphosphate"]),
    "UTP": ("UTP", ["uridine triphosphate"]),

    # ── Sugars ────────────────────────────────────────────────────────────
    "GLC": ("α-D-glucose", ["alpha glucose", "glucose"]),
    "BGC": ("β-D-glucose", ["beta glucose", "glucose"]),
    "NAG": ("N-acetylglucosamine", ["glcnac", "nag"]),

    # ── Vitamins / pigments / lipids ──────────────────────────────────────
    "CLR": ("Cholesterol", []),
    "RET": ("Retinal", ["vitamin a aldehyde", "all-trans-retinal"]),

    # ── Metal ions ────────────────────────────────────────────────────────
    "ZN": ("Zinc ion", ["zn", "zn2+", "zinc"]),
    "MG": ("Magnesium ion", ["mg", "mg2+", "magnesium"]),
    "FE": ("Iron ion", ["fe", "iron"]),
    "CA": ("Calcium ion", ["ca", "ca2+", "calcium"]),
    "MN": ("Manganese ion", ["mn", "manganese"]),
    "CU": ("Copper ion", ["cu", "cu2+", "copper"]),
}


def hill_formula(symbols: list[str]) -> str:
    """Standard Hill notation: C first, then H, then others alphabetically."""
    counts = Counter(symbols)
    parts: list[str] = []
    for sym in ("C", "H"):
        if sym in counts:
            n = counts.pop(sym)
            parts.append(f"{sym}{n if n > 1 else ''}")
    for sym in sorted(counts):
        n = counts[sym]
        parts.append(f"{sym}{n if n > 1 else ''}")
    return "".join(parts)


def main() -> None:
    entries: list[dict] = []
    for code, (name, synonyms) in NAMES.items():
        blob_path = CCD_DIR / f"{code}.json"
        if not blob_path.exists():
            print(f"  ! {code}: blob not built (run build_ligand_starter_pack.py first)")
            continue
        blob = json.loads(blob_path.read_text())["data"]
        symbols = [a["element_sym"] for a in blob["atoms"]]
        entry = {
            "ccd": code,
            "name": name,
            "synonyms": synonyms,
            "formula": hill_formula(symbols),
            "n_atoms": int(blob["num_atoms"]),
        }
        entries.append(entry)
        print(f"  {code}: {name}  [{entry['formula']}]")

    out_path = CCD_DIR / "index.json"
    payload = {
        "_doc": (
            "Searchable index for the ligand drawer. One row per CCD entry "
            "shipped under public/ccd/. The drawer matches user input "
            "(case-insensitive substring) against ccd | name | synonyms[]. "
            "Atom-level / bond / constraint data lives in the per-CCD blob "
            "loaded on demand."
        ),
        "version": "1",
        "entries": entries,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} ({len(entries)} entries, "
          f"{out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
