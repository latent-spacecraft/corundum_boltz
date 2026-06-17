#!/usr/bin/env python3
"""
Prove the standalone `ligand_blob.py` (rdkit+numpy only) produces blobs
byte-identical to the boltz-based reference path — so the minimal deploy image
stays compatible with the trained trunk.

Everything is compared EXCEPT atoms[].ref_pos: ETKDG conformer coordinates are
stochastic per run, so they legitimately differ between two constructions. Bond
graph, atom names/elements/charges/chirality ids, and all six constraint groups
(including the graph-derived rdkit distance bounds) ARE deterministic and must
match exactly.

Run with the boltz venv:
    .../boltz-dev/.venv/bin/python verify_blob_equiv.py
"""
from __future__ import annotations

import sys

import ligand_blob
from boltz.data import const
from boltz.data.parse import schema as bs
from rdkit import Chem
from rdkit.Chem import AllChem

TESTS = [
    "c1ccccc1",                                            # benzene
    "Cc1[nH]c2c(C#N)c(C#N)c([C@H](F)CCO)c(-c3ccccc3)c2c1C",  # zfBxCS
    "CC(=O)Oc1ccccc1C(=O)O",                              # aspirin
    "C/C=C/C=C\\C",                                        # E/Z stereo bonds
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",        # glucose (chiral)
]


def boltz_reference(smiles: str) -> dict:
    """Reconstruct the blob using boltz's own functions (the old code path)."""
    mol = AllChem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    order = AllChem.CanonicalRankAtoms(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    for atom, can_idx in zip(mol.GetAtoms(), order):
        atom.SetProp("name", atom.GetSymbol().upper() + str(can_idx + 1))
    bs.compute_3d_conformer(mol)

    conf = bs.get_conformer(mol)
    chir_ids = const.chirality_type_ids
    bond_ids = const.bond_type_ids
    idx_map = {}
    atoms = []
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() == 1:
            continue
        idx_map[i] = len(atoms)
        chir = str(a.GetChiralTag())
        atoms.append({
            "name": a.GetProp("name"), "element": a.GetAtomicNum(),
            "element_sym": a.GetSymbol(), "charge": a.GetFormalCharge(),
            "chirality_id": chir_ids.get(chir, chir_ids[const.unk_chirality_type]),
        })
    bonds = []
    for b in mol.GetBonds():
        bi, bj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if bi not in idx_map or bj not in idx_map:
            continue
        bname = b.GetBondType().name
        bonds.append({
            "i": idx_map[bi], "j": idx_map[bj], "type_id": bond_ids.get(bname, bond_ids[const.unk_bond_type]),
        })
    rb = bs.compute_geometry_constraints(mol, idx_map)
    ch = bs.compute_chiral_atom_constraints(mol, idx_map)
    st = bs.compute_stereo_bond_constraints(mol, idx_map)
    pb, r5, r6 = bs.compute_flatness_constraints(mol, idx_map)
    return {
        "atoms": atoms, "bonds": bonds,
        "rdkit_bounds": [(int(c.atom_idxs[0]), int(c.atom_idxs[1]), bool(c.is_bond), bool(c.is_angle), round(float(c.upper_bound), 6), round(float(c.lower_bound), 6)) for c in rb],
        "chiral": sorted([tuple(int(x) for x in c.atom_idxs) + (bool(c.is_reference), bool(c.is_r)) for c in ch]),
        "stereo": sorted([tuple(int(x) for x in c.atom_idxs) + (bool(c.is_check), bool(c.is_e)) for c in st]),
        "pb": sorted([tuple(int(x) for x in c.atom_idxs) for c in pb]),
        "r5": sorted([tuple(int(x) for x in c.atom_idxs) for c in r5]),
        "r6": sorted([tuple(int(x) for x in c.atom_idxs) for c in r6]),
    }


def normalise_vendored(blob: dict) -> dict:
    return {
        "atoms": [{"name": a["name"], "element": a["element"], "element_sym": a["element_sym"],
                   "charge": a["charge"], "chirality_id": a["chirality_id"]} for a in blob["atoms"]],
        "bonds": [{"i": b["i"], "j": b["j"], "type_id": b["type_id"]} for b in blob["bonds"]],
        "rdkit_bounds": [(c["i"], c["j"], c["is_bond"], c["is_angle"], round(c["upper"], 6), round(c["lower"], 6)) for c in blob["rdkit_bounds"]],
        "chiral": sorted([tuple(c["atoms"]) + (c["is_reference"], c["is_r"]) for c in blob["chiral_atoms"]]),
        "stereo": sorted([tuple(c["atoms"]) + (c["is_check"], c["is_e"]) for c in blob["stereo_bonds"]]),
        "pb": sorted([tuple(c["atoms"]) for c in blob["planar_bonds"]]),
        "r5": sorted([tuple(c["atoms"]) for c in blob["planar_rings_5"]]),
        "r6": sorted([tuple(c["atoms"]) for c in blob["planar_rings_6"]]),
    }


def main() -> None:
    failures = 0
    for smi in TESTS:
        ref = boltz_reference(smi)
        got = normalise_vendored(ligand_blob.build_from_smiles(smi)["data"])
        if ref == got:
            print(f"  ✓ {smi}")
        else:
            failures += 1
            print(f"  ✗ {smi}")
            for k in ref:
                if ref[k] != got[k]:
                    print(f"      mismatch in {k}: ref={ref[k]!r:.120} got={got[k]!r:.120}")
    print("OK" if failures == 0 else f"{failures} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
