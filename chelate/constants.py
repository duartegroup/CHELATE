"""
Description
-----------
Element sets shared by the CHELATE feature schema.

Variables
---------
- `METAL_ATOMS_SET`: Every element classified as a metal, upper-case symbols. Use when
  comparing against PDB/PDBQT text fields.
- `METAL_ELEMENT_SYMBOLS`: The same set in element capitalisation. Use when comparing
  against ``Atom.GetSymbol()``.
- `EXPLICIT_METALS`: The metals frequent enough in the curated dataset to receive their
  own ``has_{Metal}`` one-hot column. Every other metal falls into ``has_GenericMetal``.
  The order matches the bundled models' training schema and must stay stable; see
  ``chelate.schema.get_feature_list``.
"""

METAL_ATOMS_SET: set[str] = {
    "MG",
    "ZN",
    "CA",
    "FE",
    "NA",
    "MN",
    "K",
    "NI",
    "CU",
    "CO",
    "CD",
    "HG",
    "BE",
    "PT",
    "MO",
    "AL",
    "BA",
    "V",
    "RU",
    "SR",
    "CS",
    "W",
    "AU",
    "GD",
    "YB",
    "AG",
    "IR",
    "LI",
    "RB",
    "Y",
    "PB",
    "U",
    "TL",
    "PR",
    "SM",
    "TB",
    "OS",
    "RH",
    "PD",
    "EU",
    "RE",
    "TA",
    "LU",
    "CR",
    "LA",
    "HO",
    "GA",
    "SB",
    "SN",
    "TI",
    "ZR",
    "CE",
    "ER",
    "TH",
    "IN",
    "HF",
    "BI",
    "SC",
    "ND",
    "DY",
    "PA",
    "PU",
    "AM",
    "CM",
    "CF",
    "GE",
    "NB",
    "TC",
    "PM",
    "TM",
    "PO",
    "FR",
    "RA",
    "AC",
    "NP",
    "BK",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
    "RF",
    "DB",
    "SG",
}

EXPLICIT_METALS: list[str] = ["Mn", "Co", "K", "Cu", "Mg", "Zn", "Fe", "Ni", "Ca", "Na"]

# RDKit's Atom.GetSymbol() returns element capitalisation ("Co"), not the upper-case
# spelling used in PDB element fields, so comparisons against parsed molecules need this
# form. generate_features.py performs the same conversion on import.
METAL_ELEMENT_SYMBOLS: set[str] = {symbol.capitalize() for symbol in METAL_ATOMS_SET}
