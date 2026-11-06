"""Pass a shared explicit box while preserving each engine's rotation handling."""

from docking.receptor_integrity import distances_to_docking_box


def fixed_box_options(method: str, box: dict) -> list[str]:
    """Keep the reference centre and size; GNINA may expand axes for rotation."""
    if method not in {"VINA", "SMINA", "GNINA"}:
        raise ValueError(f"Unknown docking method: {method}")
    distances_to_docking_box([], box)
    options = []
    for axis in "xyz":
        for name in ("center_" + axis, "size_" + axis):
            options.extend(["--" + name, str(box[name])])
    return options
