"""Stable pose ordering without optional docking-tool dependencies."""

from typing import TypeVar

Pose = TypeVar("Pose")


def rank_by_energy(
    mols: list[Pose], energies: list[float]
) -> tuple[list[Pose], list[float]]:
    """
    Order poses by binding energy, lowest first.

    AutoDock writes its ``DOCKED: MODEL`` blocks in LGA run order, not by energy, so
    the pose index would not be a rank. Ties keep their run order. AutoDock-GPU
    reports energies to two decimals, so ties at the best energy do occur (5-14 of
    the 118 common test pairs per run); on the published runs they change the top-1
    MBG outcome of at most one pair per run.
    """
    order = sorted(range(len(energies)), key=lambda i: energies[i])
    return [mols[i] for i in order], [energies[i] for i in order]
