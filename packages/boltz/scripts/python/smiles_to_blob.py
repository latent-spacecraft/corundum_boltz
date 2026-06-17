#!/usr/bin/env python3
"""
Build a Boltz-2 ligand blob from a user-supplied SMILES string (or a 3D mol
file: .pdb / .sdf / .mol), for ligands that are NOT in the CCD.

This mirrors `extract_ligand_blob.py`, but instead of loading a cached CCD
RDKit Mol it constructs the Mol the exact way Boltz does for SMILES ligands
(boltz/data/parse/schema.py, the `entity_type == "ligand" and "smiles" in ...`
branch):

    mol = AllChem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    # name atoms via canonical rank: <ELEMENT><canon_idx+1>
    AllChem.AssignStereochemistry(mol, force=True, cleanIt=True)
    compute_3d_conformer(mol)          # ETKDGv3 + UFF, same as upstream

It then reuses Boltz's OWN constraint extractors (compute_geometry_constraints
/ compute_chiral_atom_constraints / compute_stereo_bond_constraints /
compute_flatness_constraints) over a HEAVY-ATOM-ONLY idx_map — exactly as
`parse_ccd_residue` does — so the emitted blob is bit-compatible with the CCD
blobs the TS featurizer already consumes. Hydrogens are dropped (Boltz ligands
are heavy-atom only).

The output JSON is byte-shaped identically to extract_ligand_blob.py's, so the
TS side (featurizer/ligand.ts, loadLigandBlob) needs no changes — just drop the
file in public/ccd/<CODE>.json and reference <CODE> as a ligand chain.

Usage:
    .../python smiles_to_blob.py --code ZFB --smiles "Cc1[nH]c2c(C#N)..."
    .../python smiles_to_blob.py --code ZFB --molfile ../../zfBxCS.pdb
    .../python smiles_to_blob.py --code ZFB --molfile lig.sdf --use-file-conformer

By default a fresh ETKDG conformer is generated (matches Boltz's SMILES
training distribution). Pass --use-file-conformer to keep the coordinates from
an input 3D file instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from boltz.data import const
from boltz.data.parse import schema as boltz_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "public" / "ccd"

BOND_NAME = {1: "SINGLE", 1.5: "AROMATIC", 2: "DOUBLE", 3: "TRIPLE"}


def derive_code(canonical_smiles: str) -> str:
    """Stable short ligand code from canonical SMILES.

    Used as the blob's identity (entity-dedup key) and the residue name in
    exported structures. Deterministic, so the same molecule always resolves
    to the same code (two identical SMILES dedup into one entity), and
    collision-resistant enough that distinct molecules stay distinct.
    """
    h = hashlib.sha1(canonical_smiles.encode()).hexdigest()[:5].upper()
    return "L" + h


def name_atoms(mol) -> None:
    """Assign CCD-style atom names exactly as Boltz's SMILES branch does."""
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"atom name longer than 4 chars: {atom_name} "
                f"(canonical rank {can_idx} too large for this molecule)"
            )
        atom.SetProp("name", atom_name)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = AllChem.AddHs(mol)
    name_atoms(mol)
    if not boltz_schema.compute_3d_conformer(mol):
        raise ValueError(f"Failed to compute 3D conformer for {smiles!r}")
    return mol


def mol_from_file(path: Path, use_file_conformer: bool) -> Chem.Mol:
    """Load a 3D ligand file (.pdb / .sdf / .mol).

    By default we round-trip through canonical SMILES so the Mol and conformer
    match what Boltz would produce for the equivalent SMILES input. With
    --use-file-conformer we keep the file's coordinates (the Mol still gets
    Boltz-style names + a 'Computed' conformer tag so get_conformer picks it up).
    """
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        raw = Chem.MolFromPDBFile(str(path), removeHs=False, proximityBonding=False)
    elif suffix in (".sdf", ".mol"):
        raw = Chem.MolFromMolFile(str(path), removeHs=False)
    else:
        raise ValueError(f"Unsupported mol file type: {suffix} (use .pdb/.sdf/.mol)")
    if raw is None:
        raise ValueError(f"RDKit could not parse mol file: {path}")

    if not use_file_conformer:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(raw))
        print(f"  derived SMILES from {path.name}: {smiles}")
        return mol_from_smiles(smiles)

    # Keep file coordinates. Rebuild Hs + names but preserve the conformer.
    mol = AllChem.AddHs(raw, addCoords=True)
    name_atoms(mol)
    conf = mol.GetConformer()
    conf.SetProp("name", "Computed")
    return mol


def build_blob(mol, code: str) -> dict:
    """Heavy-atom blob + 6 constraint groups, identical shape to CCD blobs.

    Mirrors parse_ccd_residue: skip hydrogens, build a heavy-atom idx_map, and
    hand that map to Boltz's constraint extractors so their atom indices land
    in the same heavy-atom space the TS featurizer enumerates.
    """
    conformer = boltz_schema.get_conformer(mol)
    chir_ids = const.chirality_type_ids
    bond_ids = const.bond_type_ids
    unk_chir = chir_ids[const.unk_chirality_type]
    unk_bond = bond_ids[const.unk_bond_type]

    atoms = []
    idx_map: dict[int, int] = {}  # orig mol atom idx -> heavy-atom blob idx
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() == 1:  # drop hydrogens
            continue
        heavy_idx = len(atoms)
        idx_map[i] = heavy_idx
        pos = conformer.GetAtomPosition(i)
        chir_str = str(a.GetChiralTag())
        atoms.append({
            "name":         a.GetProp("name"),
            "alt_name":     "",
            "element":      a.GetAtomicNum(),
            "element_sym":  a.GetSymbol(),
            "charge":       a.GetFormalCharge(),
            "chirality":    chir_str,
            "chirality_id": chir_ids.get(chir_str, unk_chir),
            "leaving":      False,
            "ref_pos":      [float(pos.x), float(pos.y), float(pos.z)],
        })

    bonds = []
    for b in mol.GetBonds():
        bi, bj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if bi not in idx_map or bj not in idx_map:
            continue  # skip bonds to hydrogens
        order = b.GetBondTypeAsDouble()
        bname = b.GetBondType().name  # SINGLE / DOUBLE / TRIPLE / AROMATIC
        bonds.append({
            "i":        idx_map[bi],
            "j":        idx_map[bj],
            "order":    order,
            "type":     bname,
            "type_id":  bond_ids.get(bname, unk_bond),
            "aromatic": bool(b.GetIsAromatic()),
        })

    # Boltz's own extractors. They apply idx_map internally, so their outputs
    # are already in heavy-atom (blob) index space.
    rdkit_bounds = boltz_schema.compute_geometry_constraints(mol, idx_map)
    chiral = boltz_schema.compute_chiral_atom_constraints(mol, idx_map)
    stereo = boltz_schema.compute_stereo_bond_constraints(mol, idx_map)
    planar_bonds, planar_5, planar_6 = boltz_schema.compute_flatness_constraints(
        mol, idx_map,
    )

    return {
        "ccd":        code,
        "num_atoms":  len(atoms),
        "atoms":      atoms,
        "bonds":      bonds,
        "rdkit_bounds": [
            {
                "i":        int(c.atom_idxs[0]),
                "j":        int(c.atom_idxs[1]),
                "is_bond":  bool(c.is_bond),
                "is_angle": bool(c.is_angle),
                "upper":    float(c.upper_bound),
                "lower":    float(c.lower_bound),
            }
            for c in rdkit_bounds
        ],
        "chiral_atoms": [
            {
                "atoms":        [int(x) for x in c.atom_idxs],
                "is_reference": bool(c.is_reference),
                "is_r":         bool(c.is_r),
            }
            for c in chiral
        ],
        "stereo_bonds": [
            {
                "atoms":    [int(x) for x in c.atom_idxs],
                "is_check": bool(c.is_check),
                "is_e":     bool(c.is_e),
            }
            for c in stereo
        ],
        "planar_bonds":   [{"atoms": [int(x) for x in c.atom_idxs]} for c in planar_bonds],
        "planar_rings_5": [{"atoms": [int(x) for x in c.atom_idxs]} for c in planar_5],
        "planar_rings_6": [{"atoms": [int(x) for x in c.atom_idxs]} for c in planar_6],
    }


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
    ap.add_argument(
        "--code", default=None,
        help="ligand code / residue name. Defaults to a stable hash of the "
             "canonical SMILES (e.g. LAB12C).",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="SMILES string")
    src.add_argument("--molfile", help="3D ligand file (.pdb/.sdf/.mol)")
    ap.add_argument(
        "--use-file-conformer", action="store_true",
        help="keep the input file's coordinates instead of regenerating via ETKDG",
    )
    ap.add_argument(
        "--stdout", action="store_true",
        help="emit the blob as JSON on stdout (for the preprocessing endpoint) "
             "instead of writing public/ccd/<CODE>.json",
    )
    args = ap.parse_args()

    if args.smiles:
        mol = mol_from_smiles(args.smiles)
    else:
        mol = mol_from_file(Path(args.molfile), args.use_file_conformer)

    # Canonical SMILES (heavy atoms) — identity + default code source.
    from rdkit import Chem as _Chem
    canonical = _Chem.MolToSmiles(_Chem.RemoveHs(mol))
    code = (args.code or derive_code(canonical)).upper()

    blob = build_blob(mol, code)

    if args.stdout:
        json.dump(
            {"ok": True, "code": code, "canonical_smiles": canonical, "data": blob},
            sys.stdout, separators=(",", ":"),
        )
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
