import numpy as np
import cv2

# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

# Warte auf erstes Frame
ret, first_frame = cap.read()
if not ret:
    print("Fehler: Konnte kein Frame von der Kamera lesen!")
    exit(1)

# Keep looping
while True:
    # Reading the frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)

    # create glasses view
    h, w = frame.shape[:2]

    # 1) Fake stereo (B variant)
    shift = int(w * 0.02)  # 0.01..0.03
    left_eye  = np.roll(frame, -shift, axis=1)
    right_eye = np.roll(frame,  shift, axis=1)

    # remove wrap-around artifacts
    if shift > 0:
        left_eye[:, -shift:] = 0
        right_eye[:, :shift] = 0

    # 2) Combine into one wide image
    glasses_view = np.hstack([left_eye, right_eye])  # (h, 2w, 3)
    H, W = glasses_view.shape[:2]
    half_w = W // 2

    # 3) Lens geometry (BIG lenses, top wider than bottom)
    lens_h = int(H * 0.82)             # big -> less black background
    top_w  = int(half_w * 0.95)        # wide top
    bot_w  = int(top_w * 0.78)         # narrower bottom
    y0 = (H - lens_h) // 2             # centered vertically

    # Centers of each half
    cxL = half_w // 2
    cxR = half_w + half_w // 2

    # 4) Build curved lens mask (outward-bent sides)
    lens_mask = np.zeros((H, W), dtype=np.uint8)

    def add_curved_lens(m, cx, y, top_w, bot_w, h):
        """
        Curved Ray-Ban-ish lens: union of two ellipses
        -> gives outward-bent sides + rounded shape
        """
        tmp = np.zeros_like(m)
        cy = y + h // 2

        # Side curvature ellipse (controls outward bend)
        side_rx = top_w // 2
        side_ry = int(h * 0.52)         # smaller -> more bend; larger -> flatter
        cv2.ellipse(tmp, (cx, cy), (side_rx, side_ry), 0, 0, 360, 255, -1)

        # Bottom narrowing ellipse (makes bottom narrower than top)
        tb_rx = bot_w // 2
        tb_ry = int(h * 0.62)
        cv2.ellipse(tmp, (cx, cy + int(h * 0.05)), (tb_rx, tb_ry), 0, 0, 360, 255, -1)

        # Combine into main mask
        m[:] = cv2.bitwise_or(m, tmp)

    add_curved_lens(lens_mask, cxL, y0, top_w, bot_w, lens_h)
    add_curved_lens(lens_mask, cxR, y0, top_w, bot_w, lens_h)

    # 5) Apply mask: video visible only inside lenses
    bg = np.zeros_like(glasses_view)  # black outside
    glasses_masked = np.where(lens_mask[:, :, None] == 255, glasses_view, bg)

    # 6) Solid nose bridge (NO camera)
    frame_color = (55, 55, 55)  # BGR
    bridge_w = int(half_w * 0.16)
    bridge_h = int(lens_h * 0.18)
    bridge_x = half_w - bridge_w // 2
    bridge_y = y0 + int(lens_h * 0.46)

    cv2.rectangle(
        glasses_masked,
        (bridge_x, bridge_y),
        (bridge_x + bridge_w, bridge_y + bridge_h),
        frame_color,
        -1
    )

    # 7) Outline (no top lines)
    thick = 6
    contours, _ = cv2.findContours(lens_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(glasses_masked, contours, -1, frame_color, thick)

    # Show glasses view
    cv2.imshow("Glasses", glasses_masked)

    # ESC-Taste zum Schließen
    if cv2.waitKey(1) & 0xFF == 27:  # 27 ist ESC
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()