#!/usr/bin/env python3
"""
Standalone ligand-blob core — RDKit + numpy only, NO `boltz` import.

This vendors the small molecule preprocessing Boltz-2 needs so it can run in a
minimal deploy image (rdkit + numpy, ~no torch/biopython). The four constraint
extractors and the conformer routine are copied VERBATIM from upstream
`boltz/data/parse/schema.py`; the two const maps below are the exact integer
ids from `boltz.data.const`. An equivalence check (scripts/python/
verify_blob_equiv.py) confirms the output is byte-identical to the boltz-based
path, so blobs stay compatible with the trained trunk.

`smiles_to_blob.py` (CLI) and `server/app.py` (deploy endpoint) both import
`build_from_smiles` / `build_from_molfile` from here — one source of truth.
"""
from __future__ import annotations

import hashlib

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, HybridizationType
from rdkit.Chem.rdchem import BondStereo, Mol
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms

# ── const (exact ids from boltz.data.const) ──────────────────────────────────
CHIRALITY_TYPE_IDS = {
    "CHI_UNSPECIFIED": 0,
    "CHI_TETRAHEDRAL_CW": 1,
    "CHI_TETRAHEDRAL_CCW": 2,
    "CHI_SQUAREPLANAR": 3,
    "CHI_OCTAHEDRAL": 4,
    "CHI_TRIGONALBIPYRAMIDAL": 5,
    "CHI_OTHER": 6,
}
UNK_CHIRALITY = "CHI_OTHER"
BOND_TYPE_IDS = {
    "OTHER": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
    "COVALENT": 5,
}
UNK_BOND = "OTHER"


# ── conformer (verbatim from boltz schema.compute_3d_conformer / get_conformer)
def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coords via ETKDG, then UFF-optimise. Taken from pdbeccdutils."""
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1
    try:
        conf_id = AllChem.EmbedMolecule(mol, options)
        if conf_id == -1:
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)
    except RuntimeError:
        pass  # force field issue
    except ValueError:
        pass  # sanitization issue

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")
        return True
    return False


def get_conformer(mol: Mol):
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:
            pass
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:
            pass
    conf_ids = [int(conf.GetId()) for conf in mol.GetConformers()]
    if len(conf_ids) > 0:
        return mol.GetConformer(conf_ids[0])
    raise ValueError("No conformer found")


# ── constraint extractors (verbatim from boltz schema) ───────────────────────
def compute_geometry_constraints(mol: Mol, idx_map):
    if mol.GetNumAtoms() <= 1:
        return []
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    bounds = GetMoleculeBoundsMatrix(
        mol, set15bounds=True, scaleVDW=True,
        doTriangleSmoothing=True, useMacrocycle14config=False,
    )
    bonds = set(
        tuple(sorted(b)) for b in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))
    )
    angles = set(
        tuple(sorted([a[0], a[2]]))
        for a in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    )
    out = []
    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1)):
        if i in idx_map and j in idx_map:
            out.append({
                "i": int(idx_map[i]),
                "j": int(idx_map[j]),
                "is_bond": tuple(sorted([i, j])) in bonds,
                "is_angle": tuple(sorted([i, j])) in angles,
                "upper": float(bounds[i, j]),
                "lower": float(bounds[j, i]),
            })
    return out


def compute_chiral_atom_constraints(mol, idx_map):
    out = []
    if not all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
        return out
    for center_idx, orientation in Chem.FindMolChiralCenters(mol, includeUnassigned=False):
        center = mol.GetAtomWithIdx(center_idx)
        neighbors = sorted(
            ((n.GetIdx(), int(n.GetProp("_CIPRank"))) for n in center.GetNeighbors()),
            key=lambda n: n[1], reverse=True,
        )
        neighbors = tuple(n[0] for n in neighbors)
        is_r = orientation == "R"
        if len(neighbors) > 4 or center.GetHybridization() != HybridizationType.SP3:
            continue
        atom_idxs = (*neighbors[:3], center_idx)
        if all(i in idx_map for i in atom_idxs):
            out.append({
                "atoms": [int(idx_map[i]) for i in atom_idxs],
                "is_reference": True, "is_r": bool(is_r),
            })
        if len(neighbors) == 4:
            for skip_idx in range(3):
                chiral_set = neighbors[:skip_idx] + neighbors[skip_idx + 1:]
                if skip_idx % 2 == 0:
                    atom_idxs = chiral_set[::-1] + (center_idx,)
                else:
                    atom_idxs = chiral_set + (center_idx,)
                if all(i in idx_map for i in atom_idxs):
                    out.append({
                        "atoms": [int(idx_map[i]) for i in atom_idxs],
                        "is_reference": False, "is_r": bool(is_r),
                    })
    return out


def compute_stereo_bond_constraints(mol, idx_map):
    out = []
    if not all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
        return out
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in {BondStereo.STEREOE, BondStereo.STEREOZ}:
            continue
        start_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        start_neighbors = [
            n[0] for n in sorted(
                ((n.GetIdx(), int(n.GetProp("_CIPRank")))
                 for n in mol.GetAtomWithIdx(start_idx).GetNeighbors() if n.GetIdx() != end_idx),
                key=lambda n: n[1], reverse=True,
            )
        ]
        end_neighbors = [
            n[0] for n in sorted(
                ((n.GetIdx(), int(n.GetProp("_CIPRank")))
                 for n in mol.GetAtomWithIdx(end_idx).GetNeighbors() if n.GetIdx() != start_idx),
                key=lambda n: n[1], reverse=True,
            )
        ]
        is_e = stereo == BondStereo.STEREOE
        atom_idxs = (start_neighbors[0], start_idx, end_idx, end_neighbors[0])
        if all(i in idx_map for i in atom_idxs):
            out.append({
                "atoms": [int(idx_map[i]) for i in atom_idxs],
                "is_check": True, "is_e": bool(is_e),
            })
        if len(start_neighbors) == 2 and len(end_neighbors) == 2:
            atom_idxs = (start_neighbors[1], start_idx, end_idx, end_neighbors[1])
            if all(i in idx_map for i in atom_idxs):
                out.append({
                    "atoms": [int(idx_map[i]) for i in atom_idxs],
                    "is_check": False, "is_e": bool(is_e),
                })
    return out


def compute_flatness_constraints(mol, idx_map):
    planar_double = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    ring5 = Chem.MolFromSmarts("[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1")
    ring6 = Chem.MolFromSmarts("[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1")
    pb, r5, r6 = [], [], []
    for match in mol.GetSubstructMatches(planar_double):
        if all(i in idx_map for i in match):
            pb.append({"atoms": [int(idx_map[i]) for i in match]})
    for match in mol.GetSubstructMatches(ring5):
        if all(i in idx_map for i in match):
            r5.append({"atoms": [int(idx_map[i]) for i in match]})
    for match in mol.GetSubstructMatches(ring6):
        if all(i in idx_map for i in match):
            r6.append({"atoms": [int(idx_map[i]) for i in match]})
    return pb, r5, r6


# ── molecule construction (mirrors boltz SMILES branch) ──────────────────────
def derive_code(canonical_smiles: str) -> str:
    """Stable short ligand code from canonical SMILES (entity id + resname)."""
    return "L" + hashlib.sha1(canonical_smiles.encode()).hexdigest()[:5].upper()


def name_atoms(mol: Mol) -> None:
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(name) > 4:
            raise ValueError(f"atom name longer than 4 chars: {name}")
        atom.SetProp("name", name)


def mol_from_smiles(smiles: str) -> Mol:
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = AllChem.AddHs(mol)
    name_atoms(mol)
    if not compute_3d_conformer(mol):
        raise ValueError(f"Failed to compute 3D conformer for {smiles!r}")
    return mol


def mol_from_molfile(path: str, use_file_conformer: bool = False) -> Mol:
    suffix = path.lower().rsplit(".", 1)[-1]
    if suffix == "pdb":
        raw = Chem.MolFromPDBFile(path, removeHs=False, proximityBonding=False)
    elif suffix in ("sdf", "mol"):
        raw = Chem.MolFromMolFile(path, removeHs=False)
    else:
        raise ValueError(f"Unsupported mol file type: .{suffix} (use .pdb/.sdf/.mol)")
    if raw is None:
        raise ValueError(f"RDKit could not parse mol file: {path}")
    if not use_file_conformer:
        return mol_from_smiles(Chem.MolToSmiles(Chem.RemoveHs(raw)))
    mol = AllChem.AddHs(raw, addCoords=True)
    name_atoms(mol)
    mol.GetConformer().SetProp("name", "Computed")
    return mol


def build_blob(mol: Mol, code: str) -> dict:
    """Heavy-atom blob + 6 constraint groups (mirrors parse_ccd_residue)."""
    conformer = get_conformer(mol)
    atoms, idx_map = [], {}
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() == 1:
            continue
        idx_map[i] = len(atoms)
        pos = conformer.GetAtomPosition(i)
        chir = str(a.GetChiralTag())
        atoms.append({
            "name": a.GetProp("name"),
            "alt_name": "",
            "element": a.GetAtomicNum(),
            "element_sym": a.GetSymbol(),
            "charge": a.GetFormalCharge(),
            "chirality": chir,
            "chirality_id": CHIRALITY_TYPE_IDS.get(chir, CHIRALITY_TYPE_IDS[UNK_CHIRALITY]),
            "leaving": False,
            "ref_pos": [float(pos.x), float(pos.y), float(pos.z)],
        })

    bonds = []
    for b in mol.GetBonds():
        bi, bj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if bi not in idx_map or bj not in idx_map:
            continue
        bname = b.GetBondType().name
        bonds.append({
            "i": idx_map[bi], "j": idx_map[bj],
            "order": b.GetBondTypeAsDouble(),
            "type": bname,
            "type_id": BOND_TYPE_IDS.get(bname, BOND_TYPE_IDS[UNK_BOND]),
            "aromatic": bool(b.GetIsAromatic()),
        })

    pb, r5, r6 = compute_flatness_constraints(mol, idx_map)
    return {
        "ccd": code,
        "num_atoms": len(atoms),
        "atoms": atoms,
        "bonds": bonds,
        "rdkit_bounds": compute_geometry_constraints(mol, idx_map),
        "chiral_atoms": compute_chiral_atom_constraints(mol, idx_map),
        "stereo_bonds": compute_stereo_bond_constraints(mol, idx_map),
        "planar_bonds": pb,
        "planar_rings_5": r5,
        "planar_rings_6": r6,
    }


def build_from_smiles(smiles: str, code: str | None = None) -> dict:
    """SMILES → {code, canonical_smiles, data: blob}. The endpoint's workhorse."""
    mol = mol_from_smiles(smiles)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol))
    code = (code or derive_code(canonical)).upper()
    return {"code": code, "canonical_smiles": canonical, "data": build_blob(mol, code)}


def build_from_molfile(path: str, code: str | None = None, use_file_conformer: bool = False) -> dict:
    mol = mol_from_molfile(path, use_file_conformer)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol))
    code = (code or derive_code(canonical)).upper()
    return {"code": code, "canonical_smiles": canonical, "data": build_blob(mol, code)}
