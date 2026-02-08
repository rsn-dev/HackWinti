import cv2
import numpy as np
import math


def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=np.float32)

def _rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [ 0,   0, 1]], dtype=np.float32)

def _project_points(V, K, t):
    Pc = V + t.reshape(1, 3)
    z = np.clip(Pc[:, 2], 1e-3, None)
    x = Pc[:, 0] / z
    y = Pc[:, 1] / z
    uvw = (K @ np.stack([x, y, np.ones_like(x)], axis=0)).T
    return uvw[:, :2]

def _shade_color(color_bgr, intensity):
    c = np.array(color_bgr, dtype=np.float32)
    out = np.clip(c * intensity, 0, 255).astype(np.uint8)
    return (int(out[0]), int(out[1]), int(out[2]))

def _draw_end_on_axis_glyph(img, center, toward_viewer: bool, color_bgr, radius=14):
    """Draw an 'axis still visible' glyph when arrow points almost at camera."""
    cx, cy = center
    outline = (0, 0, 0)

    if toward_viewer:
        # filled dot = toward viewer
        cv2.circle(img, (cx, cy), radius, color_bgr, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, outline, thickness=2, lineType=cv2.LINE_AA)
    else:
        # circle + X = away from viewer
        cv2.circle(img, (cx, cy), radius, color_bgr, thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, outline, thickness=1, lineType=cv2.LINE_AA)
        r = int(radius * 0.7)
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), color_bgr, 3, cv2.LINE_AA)
        cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), color_bgr, 3, cv2.LINE_AA)
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), outline, 1, cv2.LINE_AA)
        cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), outline, 1, cv2.LINE_AA)

def _draw_3d_arrow_at(
    img,
    center_xy,
    length_px,
    angle_deg,
    color_bgr,
    outline_bgr=(0, 0, 0),
    tilt_deg=55.0,         # stronger 3D with higher values
    roll_deg=0.0,
    shaft_w=18,
    head_len_ratio=0.25,
    head_w_ratio=0.70,
    depth_ratio=0.35,
    focal_px=900.0,
    z_offset=900.0,
    smooth_tilt=False,smooth vs hard flip
):
    """
    3D-looking arrow rotating around its CENTER.
    angle_deg: 0=up, +clockwise (screen-space).
    tilt flips depending on whether arrow points up vs down, so:
      - up-ish: recedes (away)
      - down-ish: comes toward you
    """
    cx, cy = center_xy
    L = float(length_px)
    sw = float(shaft_w)
    depth = sw * float(depth_ratio)

    hl = L * float(head_len_ratio)
    hw = max(sw * 1.2, L * float(head_w_ratio))

    # ---- Build model centered at origin along +X (from -L/2 .. +L/2)
    x0 = -L / 2.0
    x_shaft_end = (L / 2.0) - hl
    tip_x = L / 2.0

    def box(x0, x1, y0, y1, z0, z1):
        return np.array([
            [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
            [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
        ], dtype=np.float32)

    shaft = box(x0, x_shaft_end, -sw/2, sw/2, -depth/2, depth/2)

    head = np.array([
        [x_shaft_end, -hw/2, -depth/2],
        [x_shaft_end,  hw/2, -depth/2],
        [tip_x,        0.0,  -depth/2],
        [x_shaft_end, -hw/2,  depth/2],
        [x_shaft_end,  hw/2,  depth/2],
        [tip_x,        0.0,   depth/2],
    ], dtype=np.float32)

    V = np.vstack([shaft, head])

    # ---- Rotations
    a = math.radians(angle_deg)
    tilt = math.radians(tilt_deg)
    roll = math.radians(roll_deg)

    # Map to your convention: 0deg = up.
    # Build points along +X, then rotate -90° so +X becomes up.
    # ---- NEW: tilt depends on up vs down
    if smooth_tilt:
        # continuous: strongest when up/down, near zero when sideways
        tilt_eff = -tilt * math.cos(a)
    else:
        # hard flip: up-ish => away (negative), down-ish => toward (positive)
        tilt_eff = (-tilt) if (math.cos(a) >= 0.0) else (+tilt)

    R = _rot_z(a) @ _rot_x(tilt_eff) @ _rot_z(roll) @ _rot_z(-math.pi / 2.0)
    V = (R @ V.T).T

    # ---- Camera
    K = np.array([[focal_px, 0, cx],
                  [0, focal_px, cy],
                  [0, 0, 1]], dtype=np.float32)
    t = np.array([0.0, 0.0, float(z_offset)], dtype=np.float32)

    # ---- End-on fallback: if projected tip-tail is tiny, draw a glyph
    tip3 = (R @ np.array([+L/2, 0.0, 0.0], dtype=np.float32).reshape(3, 1)).reshape(3)
    tail3 = (R @ np.array([-L/2, 0.0, 0.0], dtype=np.float32).reshape(3, 1)).reshape(3)

    tip2  = _project_points(tip3.reshape(1, 3), K, t)[0]
    tail2 = _project_points(tail3.reshape(1, 3), K, t)[0]
    dist2 = float(np.linalg.norm(tip2 - tail2))

    end_on_thresh = max(10.0, sw * 0.6)
    if dist2 < end_on_thresh:
        toward = (tip3[2] + t[2]) < (tail3[2] + t[2])
        _draw_end_on_axis_glyph(
            img,
            (int(cx), int(cy)),
            toward_viewer=toward,
            color_bgr=color_bgr,
            radius=max(10, int(sw * 0.9)),
        )
        return

    # ---- Project mesh and draw faces
    uv = _project_points(V, K, t).astype(np.int32)

    shaft_faces = [
        [0,1,2,3],
        [4,5,6,7],
        [0,1,5,4],
        [2,3,7,6],
        [1,2,6,5],
        [3,0,4,7],
    ]
    head_faces = [
        [8,9,10],
        [11,12,13],
        [8,9,12,11],
        [9,10,13,12],
        [10,8,11,13],
    ]
    faces = shaft_faces + head_faces

    light = np.array([0.2, -0.4, -1.0], dtype=np.float32)
    light /= (np.linalg.norm(light) + 1e-6)

    def face_avg_z(face):
        idx = np.array(face, dtype=np.int32)
        return float(np.mean((V[idx, 2] + t[2])))

    faces_sorted = sorted(faces, key=face_avg_z, reverse=True)

    for face in faces_sorted:
        idx = np.array(face, dtype=np.int32)
        P = V[idx]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        n /= (np.linalg.norm(n) + 1e-6)
        intensity = float(np.clip(np.dot(n, light) * 0.6 + 0.6, 0.2, 1.0))
        fill = _shade_color(color_bgr, intensity)

        pts2 = uv[idx]
        cv2.fillConvexPoly(img, pts2, fill, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts2], True, outline_bgr, 1, lineType=cv2.LINE_AA)


def draw_center_arrows(
    glasses_frame: np.ndarray,
    angle_deg: float = 0.0,           # 0 = up, +clockwise
    color_bgr=(0, 0, 255),
    thickness: int = 6,               # used to scale shaft width
    tip_length: float = 0.35,         # kept for compatibility (unused)
    offset_y: int = 0,
    tilt_deg: float = 55.0,
    roll_deg: float = 0.0,
    smooth_tilt: bool = False,
) -> np.ndarray:
    """
    3D-looking arrows in the center of both halves.
    Now tilt direction changes for up vs down, so it looks correct when pointing down.
    """
    if glasses_frame is None or glasses_frame.size == 0:
        return glasses_frame
    if glasses_frame.ndim != 3 or glasses_frame.shape[2] != 3:
        return glasses_frame

    out = glasses_frame.copy()
    h, w = out.shape[:2]
    half_w = w // 2

    arrow_len = int(min(half_w, h) * 0.22)
    centers = [
        (half_w // 2, h // 2 + offset_y),
        (half_w + half_w // 2, h // 2 + offset_y),
    ]

    shaft_w = max(8, int(thickness * 3))

    for cx, cy in centers:
        _draw_3d_arrow_at(
            out,
            center_xy=(cx, cy),
            length_px=arrow_len,
            angle_deg=angle_deg,
            color_bgr=color_bgr,
            tilt_deg=tilt_deg,
            roll_deg=roll_deg,
            shaft_w=shaft_w,
            smooth_tilt=smooth_tilt,
        )

    return out
