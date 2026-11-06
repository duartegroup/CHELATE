"""
Description
-----------
This module defines variables and parameters used in the docking pipeline.

Paths
------
- `reduce2`: Path to the reduce2 script used to add receptor hydrogens.
- `scrub`: Path to the scrub.py script for cleaning up ligand files.
- `mk_prepare_ligand`: Path to the mk_prepare_ligand.py script for preparing ligands.
- `mk_prepare_receptor`: Path to the mk_prepare_receptor.py script for preparing receptors.
- `PLINDER_SYSTEMS_PATH`: Path to the PLINDER systems directory.
- `GNINA_PATH`: Command that runs GNINA. May contain several words (e.g. a container
  invocation); it is split with ``shlex.split`` before use.
- `VINA_PATH`: Path to the Vina executable.
- `SMINA_PATH`: Path to the SMINA executable.
- `GPDOCK_PATH`: Path to the GPDOCK executable.
- `GPDOCK_MULTI_METAL_PATH`: Path to the GPDOCK executable for multi-metal docking.
- `METALLODOCK_PATH`: Path to the MetalloDock executable.
- `METALLODOCK_MODEL`: Path to the trained MetalloDock model.
- `METALLODOCK_VOC`: Path to the MetalloDock fragment vocabulary dictionary.
- `METALLODOCK_ENV`: Path to the MetalloDock conda environment.
- `CACHED_DATA_PATH`: Path to the `data_curation` cached-data directory.
- `DATASET_CSV`: Path to the curated PLINDER dataset CSV (systems, ligands, splits).
- `CLUSTER_REPRESENTATIVES_CSV`: Path to the cluster-representative CSV used to subsample the train split.
- `METAL_LIGAND_DATA_JSON`: Path to the per-system metal/ligand coordinate JSON.

Variables
--------
- `CHARGE_DICT`: Dictionary mapping all unsupported Meeko metal atom types to their common oxidation states.
- `METAL_ATOMS_SET`: Set of metal elements (re-exported from `chelate.constants`).
- `EXPLICIT_METALS`: Metals with explicit one-hot encodings (re-exported from `chelate.constants`).
- `AUTODOCK_RECEPTOR_TYPES`: List of AutoDock receptor atom types.
- `AUTODOCK_LIGAND_TYPES`: List of AutoDock ligand atom types.
- `GPDOCK_METAL_MAPPING`: Dictionary mapping metal element symbols to GPDOCK ION codes.

Environment Variables
---------------------
- `MMTBX_CCP4_MONOMER_LIB`: Path to the CCP4 monomer library for MMTBX. Defaults to the
  absolute path of `data_curation/geostd` unless already set in the environment.
"""

import os
from pathlib import Path

from chelate.constants import EXPLICIT_METALS, METAL_ATOMS_SET

__all__ = ["EXPLICIT_METALS", "METAL_ATOMS_SET"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEMS_ROOT = str(REPO_ROOT / "docking")

# Paths to various scripts and executables
reduce2: str = os.environ.get(
    "REDUCE2_PATH",
    os.path.expanduser(
        "~/miniconda3/envs/docking_env/lib/python3.11/site-packages/mmtbx/command_line/reduce2.py"
    ),
)
scrub: str = os.environ.get(
    "SCRUB_PATH", "scrub.py"
)  # scrub.py is added to PATH during Scrubber installation
mk_prepare_ligand: str = os.environ.get(
    "MK_PREPARE_LIGAND_PATH", "mk_prepare_ligand.py"
)  # mk_prepare_ligand.py is added to PATH during Meeko installation
mk_prepare_receptor: str = os.environ.get(
    "MK_PREPARE_RECEPTOR_PATH", "mk_prepare_receptor.py"
)  # mk_prepare_receptor.py is added to PATH during Meeko installation

# Reduce2 needs the geostd monomer library. Resolve it relative to this file so the
# scripts work from any working directory, without overriding a user-set value.
os.environ.setdefault(
    "MMTBX_CCP4_MONOMER_LIB",
    str(REPO_ROOT / "data_curation" / "geostd"),
)

PLINDER_SYSTEMS_PATH: str = os.environ.get(
    "PLINDER_SYSTEMS_PATH", "plinder/2024-06/v2/systems/"
)
GNINA_PATH: str = os.environ.get(
    "GNINA_PATH", "singularity run --nv gnina_latest.sif gnina"
)
VINA_PATH: str = "executables/vina_1.2.5_linux_x86_64"
SMINA_PATH: str = "executables/smina.static"
GPDOCK_PATH: str = "executables/gpdock"
GPDOCK_MULTI_METAL_PATH: str = "executables/gpdock_multi_metal"
METALLODOCK_PATH: str = "executables/MetalloDock"
METALLODOCK_MODEL: str = (
    f"{METALLODOCK_PATH}/trained_models/docking_metallodock_time.pkl"
)
METALLODOCK_VOC: str = f"{METALLODOCK_PATH}/dataset/chembl_fragments_voc_dict.pkl"
METALLODOCK_ENV: str = ".conda/envs/MetalloDock_env"

# Curated dataset files produced by the data_curation pipeline. Resolve these from
# this module so docking commands work from any working directory.
CACHED_DATA_PATH: str = str(REPO_ROOT / "data_curation" / "cached_data")
DATASET_CSV: str = (
    f"{CACHED_DATA_PATH}/final_dataset_distTresh=4.0_mwThresh=750.0_"
    "goodApoThresh=70.0_resplit.csv"
)
CLUSTER_REPRESENTATIVES_CSV: str = f"{CACHED_DATA_PATH}/cluster_representatives_5.csv"
METAL_LIGAND_DATA_JSON: str = (
    f"{CACHED_DATA_PATH}/metal_ligand_data_distTresh=4.0_mwThresh=750.0_"
    "goodApoThresh=70.0_resplit.json"
)

# This charge dict contains all unsupported Meeko metal atom types and their common oxidation states.
CHARGE_DICT: dict[str, int] = {
    "LI": 1,  # Lithium
    "BE": 2,  # Beryllium
    "NA": 1,  # Sodium
    "AL": 3,  # Aluminum
    "K": 1,  # Potassium
    "SC": 3,  # Scandium
    "TI": 4,  # Titanium (most common in aqueues solution https://academic.oup.com/metallomics/article/8/1/9/6000637)
    "V": 5,  # Vanadium (most common of vanadate ions, found in vanadium-dependent ions https://rdcu.be/edgyv)
    "CR": 3,  # Chromium (common Cr(III))
    "CO": 2,  # Cobalt (Co(II))
    "NI": 2,  # Nickel (Ni(II))
    "FE2": 2,  # Iron (Fe(II))
    "CU": 2,  # Copper (Cu(II), though Cu(I) exists)
    "GA": 3,  # Gallium
    "GE": 4,  # Germanium
    "AS": 3,  # Arsenic (common As(III))
    "RB": 1,  # Rubidium
    "SR": 2,  # Strontium
    "Y": 3,  # Yttrium
    "ZR": 4,  # Zirconium
    "NB": 5,  # Niobium
    "MO": 6,  # Molybdenum
    "TC": 7,  # Technetium
    "RU": 3,  # Ruthenium (Ru(III) common)
    "RH": 3,  # Rhodium (Rh(III))
    "PD": 2,  # Palladium
    "AG": 1,  # Silver (Ag(I))
    "CD": 2,  # Cadmium
    "IN": 3,  # Indium
    "SN": 4,  # Tin (Sn(IV), though Sn(II) exists)
    "SB": 3,  # Antimony (Sb(III))
    "BA": 2,  # Barium
    "LA": 3,  # Lanthanum
    "CE": 3,  # Cerium (Ce(III))
    "PR": 3,  # Praseodymium
    "ND": 3,  # Neodymium
    "PM": 3,  # Promethium
    "SM": 3,  # Samarium
    "EU": 3,  # Europium
    "GD": 3,  # Gadolinium
    "TB": 3,  # Terbium
    "DY": 3,  # Dysprosium
    "HO": 3,  # Holmium
    "ER": 3,  # Erbium
    "TM": 3,  # Thulium
    "YB": 3,  # Ytterbium
    "LU": 3,  # Lutetium
    "HF": 4,  # Hafnium
    "TA": 5,  # Tantalum
    "W": 6,  # Tungsten
    "RE": 7,  # Rhenium
    "OS": 4,  # Osmium (Os(IV) common)
    "IR": 3,  # Iridium (Ir(III))
    "PT": 2,  # Platinum (Pt(II), though Pt(IV) exists)
    "AU": 3,  # Gold (Au(III))
    "HG": 2,  # Mercury
    "TL": 1,  # Thallium (Tl(I))
    "PB": 2,  # Lead (Pb(II), though Pb(IV) exists)
    "BI": 3,  # Bismuth (Bi(III))
    "PO": 4,  # Polonium
    "FR": 1,  # Francium
    "RA": 2,  # Radium
    "AC": 3,  # Actinium
    "TH": 4,  # Thorium
    "PA": 5,  # Protactinium
    "U": 6,  # Uranium (Uranium(VI) is common as UO2 2+ https://www.sciencedirect.com/science/article/pii/S0162013419306506)
    "NP": 5,  # Neptunium (Np(V))
    "PU": 4,  # Plutonium (Pu(IV) common https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.202300636)
    "AM": 3,  # Americium
    "CM": 3,  # Curium
    "BK": 3,  # Berkelium
    "CF": 3,  # Californium
    "ES": 3,  # Einsteinium
    "FM": 3,  # Fermium
    "MD": 3,  # Mendelevium
    "NO": 2,  # Nobelium (No(II) common)
    "LR": 3,  # Lawrencium
    "RF": 4,  # Rutherfordium
    "DB": 5,  # Dubnium
    "SG": 6,  # Seaborgium
}


# From https://meeko.readthedocs.io/en/develop/tutorial4a.html
AUTODOCK_RECEPTOR_TYPES: list[str] = [
    "HD",
    "C",
    "A",
    "N",
    "NA",
    "OA",
    "F",
    "P",
    "SA",
    "S",
    "Cl",
    "Br",
    "I",
    "Mg",
    "Ca",
    "Mn",
    "Fe",
    "Zn",
]

AUTODOCK_LIGAND_TYPES: list[str] = [
    "HD",
    "C",
    "A",
    "N",
    "NA",
    "OA",
    "F",
    "P",
    "SA",
    "S",
    "Cl",
    "Br",
    "I",
    "Si",
    "B",
]

# Maps metal element symbols (upper-case) to GPDOCK ION codes.
GPDOCK_METAL_MAPPING: dict[str, int] = {
    "ZN": 1,  # Zn2+
    "CA": 2,  # Ca2+  (note: CA atom renamed C0 in PDB before docking)
    "MG": 3,  # Mg2+
    "MN": 4,  # Mn2+
    # FE: resolved at runtime — 5=Fe2+, 6=Fe3+
    "CU": 7,  # Cu2+
    "CO": 8,  # Co2+
    "CD": 9,  # Cd2+  (note: CD atom renamed CAD in PDB before docking)
    "NI": 10,  # Ni2+
}
