import numpy as np
import cv2
from navigation import draw_center_arrows

# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

# Warte auf erstes Frame
ret, first_frame = cap.read()
if not ret:
    print("Fehler: Konnte kein Frame von der Kamera lesen!")
    exit(1)

# OPTIONAL: Recording aktivieren
recording_enabled = False  # Auf True setzen, wenn Video-Recording gewünscht ist
video_writer = None
recording_stopped = False

if recording_enabled:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = first_frame.shape
    video_writer = cv2.VideoWriter("recording.mp4", fourcc, 30.0, (w, h))
    print("Recording gestartet: recording.mp4")

# navigation arrow angle
arrow_angle = 0.0

# Keep looping
while True:
    # Reading the frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)
    
    # Wenn Recording aktiv und nicht gestoppt: Frame ins Video
    if recording_enabled and (video_writer is not None) and (not recording_stopped):
        video_writer.write(frame)

    # create glasses view
    h, w = frame.shape[:2]

    # Fake stereo effect
    shift = int(w * 0.02)  # 0.01..0.03
    left_eye  = np.roll(frame, -shift, axis=1)
    right_eye = np.roll(frame,  shift, axis=1)

    # remove wrap-around artifacts
    if shift > 0:
        left_eye[:, -shift:] = 0
        right_eye[:, :shift] = 0

    # Combine into one wide image
    glasses_view = np.hstack([left_eye, right_eye])  # (h, 2w, 3)
    H, W = glasses_view.shape[:2]
    half_w = W // 2

    # Lens geometry (BIG lenses, top wider than bottom)
    lens_h = int(H * 0.82)             # big -> less black background
    top_w  = int(half_w * 0.95)        # wide top
    bot_w  = int(top_w * 0.78)         # narrower bottom
    y0 = (H - lens_h) // 2             # centered vertically

    # Centers of each half
    cxL = half_w // 2
    cxR = half_w + half_w // 2

    # Build curved lens mask (outward-bent sides)
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
        side_ry = int(h * 0.52) # smaller -> more bend; larger -> flatter
        cv2.ellipse(tmp, (cx, cy), (side_rx, side_ry), 0, 0, 360, 255, -1)

        # Bottom narrowing ellipse (makes bottom narrower than top)
        tb_rx = bot_w // 2
        tb_ry = int(h * 0.62)
        cv2.ellipse(tmp, (cx, cy + int(h * 0.05)), (tb_rx, tb_ry), 0, 0, 360, 255, -1)

        # Combine into main mask
        m[:] = cv2.bitwise_or(m, tmp)

    add_curved_lens(lens_mask, cxL, y0, top_w, bot_w, lens_h)
    add_curved_lens(lens_mask, cxR, y0, top_w, bot_w, lens_h)

    # Apply mask: video visible only inside lenses
    bg = np.zeros_like(glasses_view)  # black outside
    glasses_masked = np.where(lens_mask[:, :, None] == 255, glasses_view, bg)

    # Solid nose bridge
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

    # Outline
    thick = 6
    contours, _ = cv2.findContours(lens_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(glasses_masked, contours, -1, frame_color, thick)

    # macOS-kompatible Tastenerkennung
    # waitKeyEx gibt erweiterte Key-Codes zurück (inkl. Pfeiltasten)
    key_extended = cv2.waitKeyEx(1)
    key = key_extended & 0xFF  # Nur die unteren 8 Bits für normale Tasten
    
    # ESC-Taste zum Schliessen
    if key == 27 or key_extended == 27:
        break
    
    # Debug: Zeige Key-Codes (kann auskommentiert werden)
    # if key_extended != -1:
    #     print(f"Key pressed: key={key}, key_extended={key_extended}")
    
    # Pfeiltasten für macOS und andere Systeme
    # Verschiedene mögliche Codes für Links-Pfeil:
    # - 81, 2: Standard OpenCV Codes
    # - 2424832: Windows/Linux erweitert
    # - 65361: Linux erweitert  
    # - 63234: macOS Terminal
    # - 'a'/'A': Alternative Tasten
    is_left = (key == 81 or key == 2 or key == ord('a') or key == ord('A') or 
               key_extended == 2424832 or key_extended == 65361 or key_extended == 63234)
    
    # Verschiedene mögliche Codes für Rechts-Pfeil:
    # - 83, 3: Standard OpenCV Codes  
    # - 2555904: Windows/Linux erweitert
    # - 65363: Linux erweitert
    # - 63235: macOS Terminal
    # - 'd'/'D': Alternative Tasten
    is_right = (key == 83 or key == 3 or key == ord('d') or key == ord('D') or 
                key_extended == 2555904 or key_extended == 65363 or key_extended == 63235)
    
    if is_left:
        arrow_angle -= 10
    elif is_right:
        arrow_angle += 10

    # arrow angle normalization
    arrow_angle %= 360

    # after you build glasses_masked
    glasses_masked = draw_center_arrows(
        glasses_masked,
        angle_deg=arrow_angle,
        offset_y=50,            # move arrows x pixels down from the center
        tilt_deg=70.0,          # heigher = stronger 3D
        smooth_tilt=True        # True if you hate the hard flip near 90°
    )

    # Show the window
    cv2.imshow("Tracking", glasses_masked)

# Release the camera and all resources
if video_writer is not None:
    video_writer.release()
    print("Recording beendet.")
cap.release()
cv2.destroyAllWindows()