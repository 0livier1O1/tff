from __future__ import annotations

import numpy as np
from typing import List


def interpolate_rse(
    rse_left: float,
    rse_mid: float,
    rse_right: float,
    local_step: int,
) -> List[float]:
    """
    3-point linear interpolation of RSE over a candidate grid of size 2*local_step+1.

    The three evaluated points are at relative positions [-local_step, 0, +local_step].
    Two separate linear fits are made — one for the left half and one for the right —
    and their predictions fill in the un-evaluated intermediate positions.

    Returns a list of length 2*local_step+1 (one RSE per candidate rank value).
    """
    # Left half: fit through (-step, rse_left) and (0, rse_mid)
    x_left_known = np.array([-local_step, 0])
    x_left_pred = np.setdiff1d(np.arange(-local_step, 1), x_left_known)
    z_left = np.polyfit(x_left_known, [rse_left, rse_mid], 1)
    interp_left = (z_left[0] * x_left_pred + z_left[1]).tolist()

    # Right half: fit through (0, rse_mid) and (+step, rse_right)
    x_right_known = np.array([0, local_step])
    x_right_pred = np.setdiff1d(np.arange(0, local_step + 1), x_right_known)
    z_right = np.polyfit(x_right_known, [rse_mid, rse_right], 1)
    interp_right = (z_right[0] * x_right_pred + z_right[1]).tolist()

    return [rse_left] + interp_left + [rse_mid] + interp_right + [rse_right]
