#!/usr/bin/env python3
"""
Extract a single CCD ligand's full topology + geometry constraints into a
JSON blob consumable by the TypeScript featurizer.

Reuses Boltz upstream's own constraint-extraction routines so the output is
bit-compatible with what the trunk was trained against:

    schema.compute_geometry_constraints     → rdkit_bounds_* tensors
    schema.compute_chiral_atom_constraints  → chiral_* tensors
    schema.compute_stereo_bond_constraints  → stereo_bond_* tensors
    schema.compute_flatness_constraints     → planar_* tensors

Reads the cached RDKit Mol from ~/.boltz/mols/<CCD>.pkl (populated by Boltz
on first use of that CCD; can be primed by running Boltz once on any input
that references the ligand).

Usage:
    /path/to/boltz-dev/.venv/bin/python scripts/python/extract_ligand_blob.py HEM
    /path/to/boltz-dev/.venv/bin/python scripts/python/extract_ligand_blob.py HEM NAD ATP

Outputs into packages/boltz/public/ccd/<CCD>.json (one blob per ligand).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from boltz.data import const
from boltz.data.parse import schema as boltz_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "public" / "ccd"
MOL_DIR = Path.home() / ".boltz" / "mols"

BOND_NAME = {1: "SINGLE", 1.5: "AROMATIC", 2: "DOUBLE", 3: "TRIPLE"}


def extract_atoms_and_bonds(mol) -> tuple[list[dict], list[dict]]:
    """Same shape as residue_topology_protein.json."""
    conf = mol.GetConformer()
    chir_ids = const.chirality_type_ids
    bond_ids = const.bond_type_ids

    atoms = []
    for i, a in enumerate(mol.GetAtoms()):
        props = a.GetPropsAsDict()
        pos = conf.GetAtomPosition(i)
        chir_str = str(a.GetChiralTag())
        atoms.append({
            "name":          props.get("name", f"atom_{i}"),
            "alt_name":      props.get("alt_name", ""),
            "element":       a.GetAtomicNum(),
            "element_sym":   a.GetSymbol(),
            "charge":        a.GetFormalCharge(),
            "chirality":     chir_str,
            "chirality_id":  chir_ids.get(chir_str, chir_ids[const.unk_chirality_type]),
            "leaving":       bool(props.get("leaving_atom", False)),
            "ref_pos":       [float(pos.x), float(pos.y), float(pos.z)],
        })

    bonds = []
    for b in mol.GetBonds():
        order = b.GetBondTypeAsDouble()
        bname = BOND_NAME.get(order, "OTHER")
        bonds.append({
            "i":         b.GetBeginAtomIdx(),
            "j":         b.GetEndAtomIdx(),
            "order":     order,
            "type":      bname,
            "type_id":   bond_ids.get(bname, bond_ids[const.unk_bond_type]),
            "aromatic":  bool(b.GetIsAromatic()),
        })
    return atoms, bonds


def extract_constraints(mol) -> dict:
    """Run the four boltz schema extractors and serialise their outputs.

    The Parsed*Constraint dataclasses carry atom_idxs that index into the
    `idx_map` we hand them. Using an identity map keeps the indices aligned
    with our own atoms list above (no atom subsetting on the polymer-side
    path — Boltz applies idx_map elsewhere to skip leaving atoms etc., but
    ligands ship every atom).
    """
    idx_map = {i: i for i in range(mol.GetNumAtoms())}

    rdkit_bounds = boltz_schema.compute_geometry_constraints(mol, idx_map)
    chiral = boltz_schema.compute_chiral_atom_constraints(mol, idx_map)
    stereo = boltz_schema.compute_stereo_bond_constraints(mol, idx_map)
    planar_bonds, planar_5, planar_6 = boltz_schema.compute_flatness_constraints(
        mol, idx_map,
    )

    return {
        "rdkit_bounds": [
            {
                "i":         int(c.atom_idxs[0]),
                "j":         int(c.atom_idxs[1]),
                "is_bond":   bool(c.is_bond),
                "is_angle":  bool(c.is_angle),
                "upper":     float(c.upper_bound),
                "lower":     float(c.lower_bound),
            }
            for c in rdkit_bounds
        ],
        "chiral_atoms": [
            {
                "atoms":         [int(x) for x in c.atom_idxs],   # 4 indices
                "is_reference":  bool(c.is_reference),
                "is_r":          bool(c.is_r),
            }
            for c in chiral
        ],
        "stereo_bonds": [
            {
                "atoms":     [int(x) for x in c.atom_idxs],       # 4 indices
                "is_check":  bool(c.is_check),
                "is_e":      bool(c.is_e),
            }
            for c in stereo
        ],
        "planar_bonds":   [{"atoms": [int(x) for x in c.atom_idxs]} for c in planar_bonds],   # 6 idx
        "planar_rings_5": [{"atoms": [int(x) for x in c.atom_idxs]} for c in planar_5],       # 5 idx
        "planar_rings_6": [{"atoms": [int(x) for x in c.atom_idxs]} for c in planar_6],       # 6 idx
    }


def extract_blob(ccd: str) -> dict:
    pkl = MOL_DIR / f"{ccd}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"{ccd}.pkl missing under {MOL_DIR}. Prime the cache by running "
            f"Boltz once on any input that references {ccd}."
        )
    mol = pickle.load(open(pkl, "rb"))
    atoms, bonds = extract_atoms_and_bonds(mol)
    constraints = extract_constraints(mol)
    return {
        "ccd":        ccd,
        "num_atoms":  len(atoms),
        "atoms":      atoms,
        "bonds":      bonds,
        **constraints,
    }


def write_blob(ccd: str, blob: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{ccd}.json"
    wrapped = {
        "_doc": (
            f"CCD ligand blob for {ccd}. Atoms/bonds plus the six geometry "
            f"constraint groups Boltz-2 expects on the trunk's small-molecule "
            f"input tensors (rdkit_bounds_*, chiral_*, stereo_bond_*, "
            f"planar_bond_index, planar_ring_5_index, planar_ring_6_index). "
            f"Indices are 0-based within this ligand; the TS featurizer adds "
            f"the global atom offset when concatenating into the prediction."
        ),
        "data": blob,
    }
    path.write_text(json.dumps(wrapped, separators=(",", ":")))
    return path


def summarise(ccd: str, blob: dict) -> str:
    return (
        f"  {ccd}: {blob['num_atoms']} atoms, {len(blob['bonds'])} bonds, "
        f"{len(blob['rdkit_bounds'])} bounds, "
        f"{len(blob['chiral_atoms'])} chir, "
        f"{len(blob['stereo_bonds'])} stereo, "
        f"{len(blob['planar_bonds'])} planar-bonds, "
        f"{len(blob['planar_rings_5'])}/{len(blob['planar_rings_6'])} ring5/ring6"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for ccd in sys.argv[1:]:
        ccd = ccd.upper()
        blob = extract_blob(ccd)
        path = write_blob(ccd, blob)
        print(summarise(ccd, blob))
        print(f"  → wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
