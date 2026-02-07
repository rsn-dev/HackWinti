import numpy as np
import cv2

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    pyzbar = None
    PYZBAR_AVAILABLE = False

# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

# Warte auf erstes Frame
ret, first_frame = cap.read()
if not ret:
    print("Fehler: Konnte kein Frame von der Kamera lesen!")
    exit(1)

qr_detector = cv2.QRCodeDetector()
last_code = ""
last_signature = None

FOV_TOP_PCT = 0.12
FOV_LEFT_PCT = 0.10
FOV_RIGHT_PCT = 0.10
FOV_BOTTOM_PCT = 0.12

if not PYZBAR_AVAILABLE:
    print("Warnung: pyzbar nicht installiert. Barcode-Scan deaktiviert; QR bleibt aktiv.")


def _clean_text(text):
    return " ".join(text.strip().split())


def decode_codes(frame):
    codes = []

    try:
        retval, decoded_info, _, _ = qr_detector.detectAndDecodeMulti(frame)
        if retval:
            for text in decoded_info:
                text = _clean_text(text)
                if text:
                    codes.append(f"QR: {text}")
    except Exception:
        text, _, _ = qr_detector.detectAndDecode(frame)
        text = _clean_text(text)
        if text:
            codes.append(f"QR: {text}")

    if PYZBAR_AVAILABLE:
        for barcode in pyzbar.decode(frame):
            raw = barcode.data.decode("utf-8", errors="replace")
            text = _clean_text(raw)
            if text:
                codes.append(f"{barcode.type}: {text}")

    seen = set()
    unique = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            unique.append(code)

    return unique


def pick_display_code(codes):
    for code in codes:
        if code.startswith("QR:"):
            return code
    return codes[0] if codes else ""


def update_locked_code(new_code, current_code, current_signature):
    if new_code:
        signature = new_code
        if signature != current_signature:
            return new_code, signature
    return current_code, current_signature


def clamp_text_to_width(lines, max_width, font, font_scale, thickness):
    if max_width <= 0:
        return []

    clamped = []
    for line in lines:
        text = line
        size = cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        if size <= max_width:
            clamped.append(text)
            continue

        ellipsis = "..."
        while text:
            text = text[:-1]
            size = cv2.getTextSize(text + ellipsis, font, font_scale, thickness)[0][0]
            if size <= max_width:
                clamped.append(text + ellipsis)
                break
        else:
            clamped.append(ellipsis)

    return clamped


def draw_text_block(img, x, y, lines, max_width, font_scale=0.55, color=(220, 220, 220), bg=(0, 0, 0)):
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_gap = 6

    lines = clamp_text_to_width(lines, max_width - 8, font, font_scale, thickness)
    if not lines:
        return

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_w = max((w for w, h in sizes), default=0)
    line_h = max((h for w, h in sizes), default=0)

    block_h = len(lines) * line_h + (len(lines) - 1) * line_gap + 8
    block_w = min(max_w + 8, max_width)

    cv2.rectangle(img, (x, y), (x + block_w, y + block_h), bg, -1)

    cursor_y = y + line_h + 4
    for line in lines:
        cv2.putText(img, line, (x + 4, cursor_y), font, font_scale, color, thickness, cv2.LINE_AA)
        cursor_y += line_h + line_gap

# Keep looping
while True:
    # Reading the frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)

    new_codes = decode_codes(frame)
    display_code = pick_display_code(new_codes)
    last_code, last_signature = update_locked_code(display_code, last_code, last_signature)
    display_lines = [last_code] if last_code else ["No code"]

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

    # 8) Display decoded codes in top-left of each lens
    padding = 12
    left_x = int(cxL - top_w // 2 + padding)
    right_x = int(cxR - top_w // 2 + padding)
    text_y = int(y0 + padding)
    lens_w = top_w
    left_lens_x0 = int(cxL - lens_w // 2)
    right_lens_x0 = int(cxR - lens_w // 2)

    fov_left = int(lens_w * FOV_LEFT_PCT)
    fov_right = int(lens_w * FOV_RIGHT_PCT)
    fov_top = int(lens_h * FOV_TOP_PCT)

    left_x = left_lens_x0 + fov_left
    right_x = right_lens_x0 + fov_left
    text_y = int(y0 + fov_top)
    fov_width = int(lens_w - fov_left - fov_right)

    draw_text_block(glasses_masked, left_x, text_y, display_lines, fov_width)
    draw_text_block(glasses_masked, right_x, text_y, display_lines, fov_width)

    # Show glasses view
    cv2.imshow("Glasses", glasses_masked)

    # ESC-Taste zum Schließen
    if cv2.waitKey(1) & 0xFF == 27:  # 27 ist ESC
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()