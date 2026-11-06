"""
Description
-----------
Predict metal-binding-group (MBG) RMSD for docked poses and re-rank them.

This is the CHELATE inference path end to end: read the receptor and the docked poses,
featurise every pose around the metal site, predict how far each pose's metal-binding
group is likely to sit from the true one, and discard the poses whose predicted MBG
RMSD exceeds the cutoff. The docking program's own score then ranks the survivors.
"""

import warnings
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from rdkit import Chem

from chelate.features import compute_contact_features, compute_gasteiger_charges
from chelate.poses import read_poses
from chelate.receptor import extract_metals, load_receptor_pdb
from chelate.schema import get_feature_list, get_score_col_for_method, metal_one_hot

#: Predicted-MBG-RMSD cutoff per docking program, in Angstrom, above which a pose is
#: discarded. Each is the operating point used in the CHELATE paper, chosen on the
#: validation set as the cutoff with the highest top-1 MBG success among retained
#: systems while retaining at least half of them (``model/select_cutoff.py``; must
#: match ``model/params/pred_cutoffs.json``). The models themselves were tuned for a
#: 1.0 A *true* MBG RMSD success threshold.
DEFAULT_CUTOFFS: dict[str, float] = {"VINA": 0.8, "SMINA": 0.85, "GNINA": 0.8}

#: Pass as ``cutoff`` to use :data:`DEFAULT_CUTOFFS` for the pose set's program.
Cutoff = float | Literal["default"] | None

#: Distance, in Angstrom, within which a ligand N/O/S atom of the reference pose
#: belongs to the metal-binding group.
MBG_RADIUS: float = 3.5

_MODEL_FILENAMES = {
    "VINA": "VINA_rmsd=1.0_best_model.cbm",
    "SMINA": "SMINA_rmsd=1.0_best_model.cbm",
    "GNINA": "GNINA_rmsd=1.0_best_model.cbm",
}


def load_model(
    method: str = "VINA", path: str | Path | None = None
) -> CatBoostRegressor:
    """
    Load a trained CHELATE model.

    Parameters
    ----------
    method : str, optional
        Which model to load: ``'VINA'``, ``'SMINA'`` or ``'GNINA'``. Each was
        trained on poses from that docking program, so it should match the poses you
        intend to score.
    path : str or Path, optional
        Load this ``.cbm`` file instead of the bundled one, e.g. a model you trained
        yourself.

    Returns
    -------
    CatBoostRegressor
        The loaded model. Its target is MBG RMSD in Angstrom, so lower is better.

    Raises
    ------
    ValueError
        If ``method`` is not one of the three bundled models.
    FileNotFoundError
        If an explicit ``path`` does not exist.
    """
    model = CatBoostRegressor()

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model.load_model(str(path))
        return model

    method = method.upper()
    if method not in _MODEL_FILENAMES:
        raise ValueError(
            f"No model for method {method!r}. "
            f"Expected one of {tuple(_MODEL_FILENAMES)}, or pass an explicit path."
        )

    with resources.as_file(
        resources.files("chelate.params") / _MODEL_FILENAMES[method]
    ) as model_path:
        model.load_model(str(model_path))
    return model


def _order_columns(
    features: pd.DataFrame, model: CatBoostRegressor, method: str
) -> pd.DataFrame:
    """
    Return ``features`` with its columns in the order the model was trained on.

    Use the model's own ``feature_names_`` to support custom models with a different
    feature order. :func:`chelate.schema.get_feature_list` matches the bundled models'
    training order and provides the fallback for models without feature names.

    Parameters
    ----------
    features : pd.DataFrame
        Feature frame containing at least every column the model expects.
    model : CatBoostRegressor
        Trained model, consulted for ``feature_names_``.
    method : str
        Docking program, used to fall back to the static schema.

    Returns
    -------
    pd.DataFrame
        ``features`` reindexed to the model's column order.

    Raises
    ------
    ValueError
        If the frame is missing columns the model requires.
    """
    expected = list(getattr(model, "feature_names_", []) or [])
    if not expected:
        warnings.warn(
            f"Model {method} has no feature names; assuming the bundled schema order.",
            stacklevel=2,
        )
        expected = get_feature_list(method)

    missing = [c for c in expected if c not in features.columns]
    if missing:
        raise ValueError(
            f"Feature frame is missing {len(missing)} column(s) the model requires, "
            f"e.g. {missing[:5]}."
        )
    return features[expected]


@dataclass
class PoseSet:
    """
    Docked poses with their CHELATE predictions attached.

    Attributes
    ----------
    poses : list of Chem.rdchem.Mol
        The docked poses, in the order the docking program produced them. Index ``i``
        corresponds to ``pose == i + 1`` in :attr:`features`.
    features : pd.DataFrame
        One row per pose: the computed contact features, the metal one-hot columns, the
        docking score, a 1-based ``pose`` column, and ``chelate_prediction`` — the
        predicted MBG RMSD in Angstrom.
    method : str
        Docking program the poses came from.
    metals : list of str
        Metal symbols detected in the receptor.
    metal_coords : list of np.ndarray
        Coordinates of those metals, in the same order. Used by
        :meth:`add_reference_mbg_rmsd`.
    """

    poses: list[Chem.rdchem.Mol]
    features: pd.DataFrame
    method: str
    metals: list[str]
    metal_coords: list[np.ndarray] = field(default_factory=list)

    def _sorted_by_score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Sort poses by the docking score. Poses are ranked in ascending order for Vina
        and smina docking score (lower is better) and descending order for GNINA's
        CNNScore (higher is better).

        Parameters
        ----------
        frame : pd.DataFrame
            Rows of :attr:`features` to order.

        Returns
        -------
        pd.DataFrame
            ``frame`` sorted ascending for Vina and smina ``docking_score`` (lower is
            better) and descending for GNINA ``CNNscore`` (higher is better).
        """
        score_col = get_score_col_for_method(self.method)
        # Stable, so equal scores keep file order, as in the paper's evaluation.
        return frame.sort_values(
            score_col, ascending=score_col != "CNNscore", kind="stable"
        )

    @property
    def default_cutoff(self) -> float:
        """The paper's predicted-RMSD cutoff for this docking program, in Angstrom."""
        return DEFAULT_CUTOFFS[self.method]

    def reranked(self, cutoff: Cutoff = "default") -> pd.DataFrame:
        """
        Poses surviving the predicted-RMSD filter, in docking-score order.

        CHELATE re-ranks by *removing* poses: poses whose predicted MBG RMSD exceeds
        ``cutoff`` are discarded, and whatever remains keeps the docking score ordering.

        Parameters
        ----------
        cutoff : float, ``"default"`` or None, optional
            Predicted-RMSD threshold in Angstrom. ``"default"`` uses the paper's
            cutoff for this docking program (:attr:`default_cutoff`). Pass ``None`` to
            keep every pose, which reduces this to the docking program's own ranking.

        Returns
        -------
        pd.DataFrame
            The surviving rows of :attr:`features`, best-scored first, keeping their
            original index. Empty if no pose passed the cutoff.
        """
        candidates = self.features
        if cutoff == "default":
            cutoff = self.default_cutoff
        if cutoff is not None:
            candidates = candidates[candidates["chelate_prediction"] <= cutoff]
        return self._sorted_by_score(candidates)

    def top1(self, cutoff: Cutoff = "default") -> pd.Series | None:
        """
        Return the pose CHELATE would put first.

        The best-scoring pose among those passing the predicted-RMSD filter.
        Note the ranking is by docking score, not by the prediction; see :meth:`reranked`.

        Parameters
        ----------
        cutoff : float, ``"default"`` or None, optional
            Predicted-RMSD threshold in Angstrom; ``"default"`` uses
            :attr:`default_cutoff`. Pass ``None`` to disable filtering, in which case
            this returns the same pose as :meth:`baseline_top1`.

        Returns
        -------
        pd.Series or None
            The selected row of :attr:`features`, or ``None`` if a cutoff was given and
            no pose passed it, meaning CHELATE trusts none of the poses.
        """
        candidates = self.reranked(cutoff)
        if candidates.empty:
            return None
        return candidates.iloc[0]

    def baseline_top1(self) -> pd.Series:
        """
        Return the pose the docking program itself ranked first.

        The comparison point for `top1`. Ranking direction follows the program:
        lowest docking score for Vina and smina, highest ``CNNscore`` for GNINA.

        Returns
        -------
        pd.Series
            The selected row of :attr:`features`.
        """
        return self._sorted_by_score(self.features).iloc[0]

    def mol_for(self, row: pd.Series) -> Chem.rdchem.Mol:
        """
        Return the RDKit molecule for a row of :attr:`features`.

        Parameters
        ----------
        row : pd.Series
            A row carrying a 1-based ``pose`` value, as returned by :meth:`top1` or
            :meth:`baseline_top1`.

        Returns
        -------
        Chem.rdchem.Mol
            The corresponding pose.
        """
        return self.poses[int(row["pose"]) - 1]

    def write_sdf(self, path: str | Path, row: pd.Series) -> None:
        """
        Write a single selected pose to an SDF file.

        Parameters
        ----------
        path : str or Path
            Destination ``.sdf`` file.
        row : pd.Series
            Row identifying the pose, as returned by :meth:`top1` or
            :meth:`baseline_top1`.
        """
        with Chem.SDWriter(str(path)) as writer:
            writer.write(self.mol_for(row))

    def add_reference_rmsd(
        self, reference_sdf: str | Path, column: str = "reference_rmsd"
    ) -> pd.DataFrame:
        """
        Add the symmetry-corrected whole-ligand RMSD against a crystal reference.

        Only meaningful when the true pose is known, i.e. when benchmarking rather than
        predicting. Requires the optional ``spyrmsd`` dependency. Note that this is
        the RMSD over all heavy atoms; CHELATE predicts the metal-binding-group RMSD,
        which :meth:`add_reference_mbg_rmsd` computes.

        Parameters
        ----------
        reference_sdf : str or Path
            Crystal ligand pose to compare against.
        column : str, optional
            Name of the column to add to :attr:`features`.

        Returns
        -------
        pd.DataFrame
            :attr:`features`, with the new column added in place.
        """
        self.features[column] = [
            rmsd for rmsd, _ in self._symmetry_corrected_matches(reference_sdf)[1]
        ]
        return self.features

    def add_reference_mbg_rmsd(
        self,
        reference_sdf: str | Path,
        column: str = "reference_mbg_rmsd",
        *,
        reference_metal_coords: list | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Add the true metal-binding-group (MBG) RMSD against a crystal reference.

        This is the quantity CHELATE predicts: the RMSD, under the symmetry-corrected
        atom mapping, over the reference ligand's N, O and S atoms within
        :data:`MBG_RADIUS` of any receptor metal. Poses get ``nan`` if the reference
        has no such atom or the mapping fails. Requires the optional ``spyrmsd``
        dependency and assumes the metal coordinates are in the reference pose's frame.
        For cross-docking benchmarks, pass the original holo metal coordinates so
        the reference group matches the training and evaluation labels; an apo
        metal can move enough to select a different set of reference atoms.

        Parameters
        ----------
        reference_sdf : str or Path
            Crystal ligand pose to compare against.
        column : str, optional
            Name of the column to add to :attr:`features`.

        Returns
        -------
        pd.DataFrame
            :attr:`features`, with the new column added in place.
        """
        reference, matches = self._symmetry_corrected_matches(reference_sdf)
        ref_xyz = reference.GetConformer().GetPositions()
        coordinates = (
            self.metal_coords
            if reference_metal_coords is None
            else reference_metal_coords
        )
        metals = np.asarray(coordinates, dtype=float).reshape(-1, 3)
        mbg = {
            atom.GetIdx()
            for atom in reference.GetAtoms()
            if atom.GetSymbol() in {"N", "O", "S"}
            and len(metals)
            and np.linalg.norm(metals - ref_xyz[atom.GetIdx()], axis=1).min()
            < MBG_RADIUS
        }

        values = []
        for pose, (_, mapping) in zip(self.poses, matches, strict=True):
            if not mbg or mapping is None:
                values.append(float("nan"))
                continue
            pose_xyz = pose.GetConformer().GetPositions()
            pairs = [(r, p) for r, p in zip(*mapping, strict=True) if r in mbg]
            sq = [np.sum((ref_xyz[r] - pose_xyz[p]) ** 2) for r, p in pairs]
            values.append(float(np.sqrt(np.mean(sq))))
        self.features[column] = values
        return self.features

    def _symmetry_corrected_matches(
        self, reference_sdf: str | Path
    ) -> tuple[Chem.rdchem.Mol, list[tuple[float, tuple | None]]]:
        """
        Match every pose to the reference with spyrmsd.

        Returns the hydrogen-free reference and, per pose, ``(rmsd, mapping)`` where
        ``mapping`` is spyrmsd's best isomorphism ``(reference_indices, pose_indices)``
        on hydrogen-free molecules, or ``(nan, None)`` if no isomorphism was found.
        """
        try:
            from spyrmsd import rmsd as spyrmsd_rmsd
            from spyrmsd.exceptions import NonIsomorphicGraphs
            from spyrmsd.molecule import Molecule as SpyMolecule
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "Reference RMSD requires spyrmsd. "
                "Install it with: pip install 'chelate[eval]'"
            ) from exc

        reference = next(iter(Chem.SDMolSupplier(str(reference_sdf))), None)
        if reference is None:
            raise ValueError(f"Could not read a molecule from {reference_sdf}")
        # All hydrogens, as for the training labels (see ``poses._read_sdf``).
        reference = Chem.RemoveAllHs(reference)
        ref_mol = SpyMolecule.from_rdkit(reference)

        matches = []
        for pose in self.poses:
            pose_mol = SpyMolecule.from_rdkit(Chem.RemoveAllHs(pose))
            try:
                rmsd, mapping = spyrmsd_rmsd.symmrmsd(
                    ref_mol.coordinates,
                    pose_mol.coordinates,
                    ref_mol.atomicnums,
                    pose_mol.atomicnums,
                    ref_mol.adjacency_matrix,
                    pose_mol.adjacency_matrix,
                    return_best_isomorphism=True,
                )
                matches.append((float(rmsd), mapping))
            except (ValueError, NonIsomorphicGraphs):
                matches.append((float("nan"), None))
        return reference, matches


def predict_poses(
    receptor: str | Path,
    poses: str | Path,
    method: str = "VINA",
    model: CatBoostRegressor | None = None,
) -> PoseSet:
    """
    Featurise docked poses and predict their metal-binding-group RMSD.

    The full CHELATE inference path. Metals are detected from the receptor, so the
    ``has_{Metal}`` columns are derived rather than supplied.

    Parameters
    ----------
    receptor : str or Path
        Prepared receptor PDB, including the metal centre. This is the structure the
        poses were docked into.
    poses : str or Path
        Docking output file, as written by ``method``.
    method : str, optional
        Docking program that produced ``poses``: ``'VINA'``, ``'SMINA'`` or
        ``'GNINA'``.
    model : CatBoostRegressor, optional
        A model from :func:`load_model`. Defaults to the bundled model for ``method``.

    Returns
    -------
    PoseSet
        The poses, their features, and the predicted MBG RMSD per pose.

    Raises
    ------
    ValueError
        If the receptor cannot be parsed, or contains no metal atoms — in which case
        CHELATE has nothing to reason about and the docking score should be used as is.
    """
    method = method.upper()
    if model is None:
        model = load_model(method)

    receptor_mol = load_receptor_pdb(receptor)
    if receptor_mol is None:
        raise ValueError(f"RDKit could not parse the receptor PDB: {receptor}")

    receptor_metals = extract_metals(receptor_mol)
    if not receptor_metals:
        raise ValueError(
            f"No metal atoms found in {receptor}. CHELATE re-ranks by metal-binding "
            "geometry and cannot be applied to a metal-free receptor."
        )

    pose_mols, scores = read_poses(str(poses), method)
    if not pose_mols:
        raise ValueError(f"No readable poses in {poses}")

    score_col = get_score_col_for_method(method)
    one_hot = metal_one_hot([symbol for symbol, _ in receptor_metals])
    receptor_charges = compute_gasteiger_charges(receptor_mol)

    rows = []
    n_poses = len(pose_mols)
    for pose_idx, mol in enumerate(pose_mols, start=1):
        row = compute_contact_features(
            receptor_mol,
            mol,
            receptor_metals,
            pose_idx,
            n_poses,
            receptor_charges=receptor_charges,
        ).to_dict()
        row["pose"] = pose_idx
        row[score_col] = scores[pose_idx - 1]
        row.update(one_hot)
        rows.append(row)

    features = pd.DataFrame(rows)
    features["chelate_prediction"] = model.predict(
        _order_columns(features, model, method)
    )

    return PoseSet(
        poses=pose_mols,
        features=features,
        method=method,
        metals=[symbol for symbol, _ in receptor_metals],
        metal_coords=[coords for _, coords in receptor_metals],
    )
