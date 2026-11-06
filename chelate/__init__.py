"""
CHELATE — metal-aware re-ranking of docked poses.

Docking programs score poses with general-purpose functions that handle metal
coordination poorly, so the top-ranked pose in a metalloprotein is often placed with the
wrong geometry around the metal. CHELATE re-ranks the poses a docking program already
produced by filtering them: it predicts, for each pose, how far its metal-binding group
is likely to sit from the true one, discards the poses it does not trust, and lets the
docking score rank whatever survives.

Typical use::

    from chelate import predict_poses

    result = predict_poses(
        receptor="Apo_receptor_prepared.pdb",
        poses="VINA_crossdocked.sdf",
        method="VINA",
    )
    result.top1()             # CHELATE's pick
    result.baseline_top1()    # what the docking program ranked first

No crystal reference is required. Models for Vina, Smina and GNINA ship with the
package; :func:`load_model` also accepts a path to your own.
"""

from chelate.features import CUTOFFS, compute_contact_features
from chelate.poses import read_poses
from chelate.predict import DEFAULT_CUTOFFS, PoseSet, load_model, predict_poses
from chelate.receptor import extract_metal_symbols_from_pdb, extract_metals
from chelate.schema import (
    get_feature_list,
    get_metal_cols,
    get_score_col_for_method,
    metal_one_hot,
)

__all__ = [
    "CUTOFFS",
    "DEFAULT_CUTOFFS",
    "PoseSet",
    "compute_contact_features",
    "extract_metal_symbols_from_pdb",
    "extract_metals",
    "get_feature_list",
    "get_metal_cols",
    "get_score_col_for_method",
    "load_model",
    "metal_one_hot",
    "predict_poses",
    "read_poses",
]

__version__ = "0.1.0"
