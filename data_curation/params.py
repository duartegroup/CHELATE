"""
Description
-----------
This module defines variables and parameters used in the data curation process.

Variables
---------
- SPLIT_COLORS: Dictionary mapping dataset splits to their corresponding colors.
- `PLINDER_LOCAL_STORAGE`: Path to the local storage directory for Plinder data.
- `COLS_OF_INTEREST`: List of columns to be included in the final dataset. Available columns are available here:
    https://plinder-org.github.io/plinder/dataset.html#annotation-tables-index
- `METAL_ATOMS_SET`: Set of metal atom symbols used for filtering metalloprotein systems.
"""

from pathlib import Path

from plinder.core import get_config

SPLIT_COLORS: dict = {
    "train": "#FF999A",
    "removed": "#D3D3D3",
    "val": "#99FF99",
    "test": "#74BAFF",
}

PLINDER_LOCAL_STORAGE: Path = Path(get_config().data.plinder_dir)

COLS_OF_INTEREST: list[str] = [
    "system_id",
    "ligand_instance",
    "ligand_asym_id",
    "entry_pdb_id",
    "ligand_ccd_code",
    "system_has_binding_affinity",
    "ligand_interacting_residues",
    "ligand_binding_affinity",
    "ligand_centroid",
    "ligand_is_ion",
    "ligand_is_cofactor",
    "ligand_is_proper",
    "system_type",
    "biounit_num_ligands",
    "ligand_molecular_weight",
    "ligand_num_rot_bonds",
    "ligand_num_heavy_atoms",
]

HETERO_ATOMS_NUM_SET: set = {
    7,  # N
    8,  # O
    16,  # S
}

METAL_ATOMS_SET: set = {
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
