#!/usr/bin/env python3
"""
Extract RNA + DNA residue topology + nucleic constants from Boltz upstream
into JSON consumable by the TypeScript featurizer.

Mirrors boltz-dev/featurizer_port/scripts/dump_const_tables.py (which produced
the protein topology) but emits only the nucleic pieces.

Outputs into packages/boltz/src/acts/boltz/featurizer/tables/:
    residue_topology_rna.json
    residue_topology_dna.json
    nucleic_constants.json

Requirements:
    - `boltz` Python package importable (use the boltz-dev .venv)
    - ~/.boltz/mols/{A,G,C,U,N,DA,DG,DC,DT,DN}.pkl already cached
      (populated automatically the first time Boltz fetches its CCD data)

Run:
    /path/to/boltz-dev/.venv/bin/python scripts/python/extract_nucleic_topology.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from boltz.data import const

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "src" / "acts" / "boltz" / "featurizer" / "tables"
MOL_DIR = Path.home() / ".boltz" / "mols"

BOND_NAME = {1: "SINGLE", 1.5: "AROMATIC", 2: "DOUBLE", 3: "TRIPLE"}


def extract_residue(res: str) -> dict:
    pkl = MOL_DIR / f"{res}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"{res}.pkl missing under {MOL_DIR}. "
            f"Run Boltz once on a nucleic input to populate the CCD cache, "
            f"or copy from a node that has it."
        )
    mol = pickle.load(open(pkl, "rb"))
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

    name_to_idx = {a["name"]: k for k, a in enumerate(atoms)}
    # Nucleic center/disto convention: C1' for both per Boltz's res_to_center_atom
    # (lookup falls back to C1' if a particular residue is missing from the map).
    center_name = const.res_to_center_atom.get(res, "C1'")
    disto_name = const.res_to_disto_atom.get(res, "C1'")

    return {
        "atoms":            atoms,
        "bonds":            bonds,
        "num_atoms":        len(atoms),
        "center_atom_name": center_name,
        "center_atom_idx":  name_to_idx.get(center_name, -1),
        "disto_atom_name":  disto_name,
        "disto_atom_idx":   name_to_idx.get(disto_name, -1),
        # backbone_atom_idx is indexed against const.nucleic_backbone_atom_names
        # (12 entries: P, OP1, OP2, O5', C5', C4', O4', C3', O3', C2', O2', C1').
        # These line up with atom_backbone_feat channels 5..16.
        "backbone_atom_idx": [name_to_idx.get(n, -1)
                              for n in const.nucleic_backbone_atom_names],
    }


def _write(name: str, data: dict, doc: str) -> None:
    path = OUT_DIR / name
    wrapped = {"_doc": doc, "data": data}
    path.write_text(json.dumps(wrapped, indent=2))
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rna_residues = list(const.rna_token_to_letter.keys())   # ['A','G','C','U','N']
    dna_residues = list(const.dna_token_to_letter.keys())   # ['DA','DG','DC','DT','DN']

    rna_topology = {r: extract_residue(r) for r in rna_residues}
    dna_topology = {r: extract_residue(r) for r in dna_residues}

    _write(
        "residue_topology_rna.json",
        rna_topology,
        doc=(
            "Per-residue topology for the 5 RNA token classes (A, G, C, U, N). "
            "Extracted from ~/.boltz/mols/<CCD>.pkl via Boltz's const module. "
            "Same shape as residue_topology_protein.json. backbone_atom_idx is "
            "indexed against nucleic_backbone_atom_names (12 entries: P, OP1, "
            "OP2, O5', C5', C4', O4', C3', O3', C2', O2', C1'), which align "
            "with atom_backbone_feat channels 5..16. center/disto atom is C1'."
        ),
    )
    _write(
        "residue_topology_dna.json",
        dna_topology,
        doc=(
            "Per-residue topology for the 5 DNA token classes (DA, DG, DC, DT, "
            "DN). Same shape as protein/RNA topologies; nucleic backbone "
            "vocabulary identical to RNA."
        ),
    )
    _write(
        "nucleic_constants.json",
        {
            "rna_letter_to_token":          const.rna_letter_to_token,
            "rna_token_to_letter":          const.rna_token_to_letter,
            "dna_letter_to_token":          const.dna_letter_to_token,
            "dna_token_to_letter":          const.dna_token_to_letter,
            "nucleic_backbone_atom_names":  const.nucleic_backbone_atom_names,
            "nucleic_backbone_atom_index":  const.nucleic_backbone_atom_index,
        },
        doc=(
            "Nucleic featurization constants. Letter↔token maps for RNA "
            "(identity: A→A, G→G, C→C, U→U, N→N) and DNA (prefix-d: "
            "A→DA, …, T→DT, N→DN). nucleic_backbone_atom_names defines the "
            "12-atom vocabulary that lights up atom_backbone_feat channels "
            "5..16 (protein N/CA/C/O occupy channels 1..4)."
        ),
    )


if __name__ == "__main__":
    main()
