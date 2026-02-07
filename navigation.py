import cv2
import numpy as np
import math


def draw_center_arrows(
    glasses_frame: np.ndarray,
    angle_deg: float = 0.0,           # 0 = up, +clockwise, -counter-clockwise
    color_bgr=(0, 0, 255),
    thickness: int = 6,
    tip_length: float = 0.35,
    offset_y: int = 0,
) -> np.ndarray:
    """
    Draws a rotated arrow in the center of each half of a side-by-side glasses frame.
    Rotation pivot is the ARROW CENTER (not the tail).

    Assumes glasses_frame is (H, W, 3) with W ~ 2 * original_width.
    """
    if glasses_frame is None or glasses_frame.size == 0:
        return glasses_frame
    if glasses_frame.ndim != 3 or glasses_frame.shape[2] != 3:
        return glasses_frame

    out = glasses_frame.copy()
    h, w = out.shape[:2]
    half_w = w // 2

    # Arrow length (total length). We draw it centered, so use half for each side.
    arrow_len = int(min(half_w, h) * 0.22)
    half_len = max(1, arrow_len // 2)

    theta = math.radians(angle_deg)

    # Direction vector (0° = up)
    dx = int(half_len * math.sin(theta))
    dy = int(-half_len * math.cos(theta))

    centers = [
        (half_w // 2, h // 2 + offset_y),            # left lens center
        (half_w + half_w // 2, h // 2 + offset_y),   # right lens center
    ]

    for cx, cy in centers:
        # Center-pivot rotation: start and end are symmetric around (cx, cy)
        start = (cx - dx, cy - dy)
        end   = (cx + dx, cy + dy)

        cv2.arrowedLine(
            out,
            start,
            end,
            color_bgr,
            thickness,
            line_type=cv2.LINE_AA,
            tipLength=tip_length,
        )

    return out
