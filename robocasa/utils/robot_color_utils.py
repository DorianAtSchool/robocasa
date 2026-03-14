"""
Utilities for visually distinguishing robots by recoloring their materials.

After the MuJoCo model is compiled, robot geoms are namespaced as
``robot0_<name>`` and ``robot1_<name>``.  This module finds the material
IDs used by a robot's geoms and tints the light-coloured ones so the two
robots are easy to tell apart in rendered videos.
"""

from __future__ import annotations

import numpy as np


# Warm amber/orange tint applied to robot1's white/light-grey materials.
# Dark materials (greys < 0.3) are left untouched so the robot retains
# visual contrast between body panels and joints.
_ROBOT1_TINT = np.array([0.95, 0.55, 0.20, 1.0], dtype=np.float32)

# Materials with max RGB channel below this threshold are considered "dark"
# and are left unchanged (joint housings, cable covers, etc.).
_DARK_THRESHOLD = 0.35


def recolor_robot(sim, robot_idx: int = 1, tint: np.ndarray | None = None):
    """Recolour all materials belonging to a robot in the compiled MuJoCo model.

    Finds the robot's materials by iterating over geoms whose names contain
    ``robot{idx}_`` and collecting their material IDs.  This works with
    the robosuite MjModel wrapper which lacks ``mat_id2name``.

    Parameters
    ----------
    sim : MjSim
        The simulator whose ``model.mat_rgba`` will be modified in-place.
    robot_idx : int
        Which robot to recolour (default 1).
    tint : np.ndarray, optional
        RGBA colour to apply.  Defaults to a warm amber.
    """
    if tint is None:
        tint = _ROBOT1_TINT

    prefix = f"robot{robot_idx}_"
    model = sim.model

    # Collect unique material IDs used by this robot's geoms
    robot_mat_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        geom_name = model.geom_id2name(geom_id)
        if geom_name and prefix in geom_name:
            mat_id = int(model.geom_matid[geom_id])
            if mat_id >= 0:
                robot_mat_ids.add(mat_id)

    for mat_id in robot_mat_ids:
        rgba = model.mat_rgba[mat_id]
        # Only tint light-coloured materials; keep dark ones for contrast
        if float(np.max(rgba[:3])) > _DARK_THRESHOLD:
            model.mat_rgba[mat_id] = tint
