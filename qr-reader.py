import numpy as np
import cv2
import sys

# pyzbar ist die Hauptbibliothek für alle Barcode-Typen
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    print("FEHLER: pyzbar ist nicht installiert!")
    print("Bitte installieren Sie es mit: pip install pyzbar")
    print("Für macOS zusätzlich: brew install zbar")
    sys.exit(1)

# OpenCV QRCodeDetector als zusätzliche Methode
try:
    qr_detector = cv2.QRCodeDetector()
    OPENCV_QR_AVAILABLE = True
except Exception:
    qr_detector = None
    OPENCV_QR_AVAILABLE = False

# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

# Kamera-Einstellungen für bessere Performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduzierte Auflösung
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Warte auf erstes Frame
ret, first_frame = cap.read()
if not ret:
    print("Fehler: Konnte kein Frame von der Kamera lesen!")
    sys.exit(1)

last_code = ""
last_signature = None

FOV_TOP_PCT = 0.12
FOV_LEFT_PCT = 0.10
FOV_RIGHT_PCT = 0.10
FOV_BOTTOM_PCT = 0.12

# Debug-Modus (auf True setzen für Debug-Ausgaben)
DEBUG_MODE = False

# Performance-Optimierung: Frame-Skipping (scanne nicht jedes Frame)
SCAN_EVERY_N_FRAMES = 2  # Scanne nur jedes 2. Frame
frame_counter = 0

# CLAHE-Objekt einmal erstellen (nicht bei jedem Frame)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Cache für Lens-Mask (wird einmal berechnet und wiederverwendet)
lens_mask_cache = None
lens_mask_size = None

print("Barcode-Scanner initialisiert. Unterstützte Typen:")
print("- QR-Code, CODE128, CODE39, EAN13, EAN8, UPC-A, UPC-E")
print("- I25, DATABAR, PDF417, AZTEC und mehr")
print(f"Performance-Modus: Scanne jedes {SCAN_EVERY_N_FRAMES}. Frame")


def _clean_text(text):
    return " ".join(text.strip().split())


def _format_barcode_type(barcode_type):
    """
    Formatiert Barcode-Typen für bessere Lesbarkeit.
    """
    type_mapping = {
        'QRCODE': 'QR',
        'CODE128': 'CODE128',
        'CODE39': 'CODE39',
        'EAN13': 'EAN-13',
        'EAN8': 'EAN-8',
        'UPC-A': 'UPC-A',
        'UPC-E': 'UPC-E',
        'I25': 'Interleaved 2/5',
        'DATABAR': 'DataBar',
        'DATABAR_EXP': 'DataBar Expanded',
        'PDF417': 'PDF417',
        'AZTEC': 'Aztec',
    }
    return type_mapping.get(barcode_type, barcode_type)


def decode_codes(frame):
    """
    Erkennt alle Barcode-Typen im Frame.
    Optimiert für Performance - verwendet nur die effektivsten Methoden.
    """
    codes = []
    
    # Reduziere Auflösung für Erkennung (schneller, aber immer noch genau genug)
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640.0 / w
        small_frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_frame = frame
    
    # Bildvorverarbeitung - nur die wichtigsten Methoden
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Nur 2 Methoden verwenden: Graustufen und CLAHE (die effektivsten)
    processed_frames = [
        ('gray', gray),
        ('clahe', clahe.apply(gray))
    ]
    
    # Pyzbar - Hauptmethode für alle Barcode-Typen
    # Stoppe sofort wenn Codes gefunden wurden (Early Exit)
    for name, processed_frame in processed_frames:
        try:
            decoded = pyzbar.decode(processed_frame)
            if decoded:  # Wenn Codes gefunden, verarbeite sie
                for barcode in decoded:
                    try:
                        raw = barcode.data.decode("utf-8", errors="replace")
                        text = _clean_text(raw)
                        if text:
                            barcode_type = barcode.type
                            formatted_type = _format_barcode_type(barcode_type)
                            code_str = f"{formatted_type}: {text}"
                            codes.append(code_str)
                            if DEBUG_MODE:
                                print(f"  [{name}] {code_str}")
                    except Exception:
                        pass
                # Early Exit: Wenn Codes gefunden, nicht weiter suchen
                if codes:
                    break
        except Exception:
            pass
    
    # OpenCV QRCodeDetector nur wenn noch keine Codes gefunden wurden
    if not codes and OPENCV_QR_AVAILABLE:
        for name, processed_frame in processed_frames:
            try:
                retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(processed_frame)
                if retval and decoded_info is not None:
                    for text in decoded_info:
                        text = _clean_text(text)
                        if text:
                            codes.append(f"QR: {text}")
                            if DEBUG_MODE:
                                print(f"  [OpenCV-Multi, {name}] QR: {text}")
                            break  # Early Exit
                    if codes:
                        break
            except Exception:
                try:
                    text, points, _ = qr_detector.detectAndDecode(processed_frame)
                    if text:
                        text = _clean_text(text)
                        if text:
                            codes.append(f"QR: {text}")
                            if DEBUG_MODE:
                                print(f"  [OpenCV-Single, {name}] QR: {text}")
                            break
                except Exception:
                    pass
    
    # Entferne Duplikate
    seen = set()
    unique = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            unique.append(code)
    
    return unique


def pick_display_code(codes):
    """
    Wählt einen Code zur Anzeige aus.
    Behandelt alle Barcode-Typen gleichwertig.
    """
    if not codes:
        return ""
    
    # Wenn mehrere Codes gefunden wurden, bevorzuge QR-Codes,
    # ansonsten nimm den ersten gefundenen Code
    for code in codes:
        if code.startswith("QR:") or code.startswith("QRCODE:"):
            return code
    
    # Ansonsten den ersten Code nehmen
    return codes[0]


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


def draw_text_block(img, x, y, lines, max_width, font_scale=0.5, color=(255, 255, 255), bg=(20, 20, 30), center_x=False):
    """
    Zeichnet einen modernen Textblock mit Glassmorphism-Effekt (iOS/Jarvis-Stil).
    center_x: Wenn True, wird der Block horizontal zentriert um x
    """
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_gap = 8
    padding = 16

    # Clamp lines to fit width - verwende größte Schriftgröße für Clamping
    max_scale = font_scale * 1.2  # Erste Zeile ist größer
    clamped_lines = []
    for i, line in enumerate(lines):
        current_scale = font_scale * 1.2 if i == 0 else font_scale * 0.9
        clamped = clamp_text_to_width([line], max_width - padding * 2, font, current_scale, thickness)
        if clamped:
            clamped_lines.extend(clamped)
    
    if not clamped_lines:
        return

    # Berechne Größen mit korrekten Skalierungen
    sizes = []
    for i, line in enumerate(clamped_lines):
        current_scale = font_scale * 1.2 if i == 0 else font_scale * 0.9
        current_thickness = 2 if i == 0 else 1
        size = cv2.getTextSize(line, font, current_scale, current_thickness)[0]
        sizes.append(size)
    
    max_w = max((w for w, h in sizes), default=0)
    line_h = max((h for w, h in sizes), default=0)

    block_h = len(clamped_lines) * line_h + (len(clamped_lines) - 1) * line_gap + padding * 2
    block_w = min(max_w + padding * 2, max_width)
    
    # Zentriere horizontal wenn gewünscht (x ist dann der Mittelpunkt)
    if center_x:
        x = x - block_w // 2

    # Stelle sicher, dass der Block innerhalb des Bildes bleibt
    h_img, w_img = img.shape[:2]
    
    # Begrenze Position und Größe
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + block_w > w_img:
        block_w = max(20, w_img - x - 2)
    if y + block_h > h_img:
        block_h = max(20, h_img - y - 2)
    
    if block_w < 20 or block_h < 20 or x < 0 or y < 0:
        return

    # GLASSMORPHISM-EFFEKT (iOS/Jarvis-Stil)
    # 1. Erstelle Blur-Effekt auf dem Hintergrund-Bereich
    roi = img[y:y+block_h, x:x+block_w].copy()
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
    
    # 2. Erstelle halbtransparenten Overlay mit Gradient-Effekt
    overlay = img[y:y+block_h, x:x+block_w].copy()
    
    # Galaktischer Hintergrund mit Gradient (dunkelblau zu lila)
    for i in range(block_h):
        alpha = 0.15 + (i / block_h) * 0.1  # Leichter Gradient
        # Dunkelblau zu Lila Gradient
        r = int(30 + (i / block_h) * 20)
        g = int(20 + (i / block_h) * 30)
        b = int(40 + (i / block_h) * 40)
        cv2.rectangle(overlay, (0, i), (block_w, i+1), (b, g, r), -1)
    
    # 3. Kombiniere Blur + Overlay (Glassmorphism)
    alpha_blend = 0.7  # Transparenz
    glass_bg = cv2.addWeighted(blurred_roi, 0.4, overlay, 0.6, 0)
    
    # 4. Zeichne subtilen weißen Rahmen für Glanz-Effekt (oben heller)
    border_thickness = 1
    # Obere Kante - heller
    cv2.line(glass_bg, (0, 0), (block_w, 0), (255, 255, 255), border_thickness)
    # Seitliche und untere Kanten - dunkler
    cv2.line(glass_bg, (0, block_h-1), (block_w, block_h-1), (100, 100, 120), border_thickness)
    cv2.line(glass_bg, (0, 0), (0, block_h), (150, 150, 170), border_thickness)
    cv2.line(glass_bg, (block_w-1, 0), (block_w-1, block_h), (150, 150, 170), border_thickness)
    
    # 5. Füge subtilen Schatten hinzu
    shadow_mask = np.zeros((block_h + 4, block_w + 4, 3), dtype=np.uint8)
    cv2.rectangle(shadow_mask, (2, 2), (block_w + 2, block_h + 2), (0, 0, 0), -1)
    shadow_blur = cv2.GaussianBlur(shadow_mask, (9, 9), 0)
    
    # Kopiere Glassmorphism-Hintergrund zurück ins Bild
    img[y:y+block_h, x:x+block_w] = glass_bg

    # Zeichne Text mit unterschiedlichen Stilen
    cursor_y = y + line_h + padding
    for i, line in enumerate(clamped_lines):
        # Erste Zeile (Parcel/Code) größer und fetter
        if i == 0:
            current_scale = font_scale * 1.2
            current_color = (255, 255, 100)  # Gelblich für Code
            current_thickness = 2
        # Zweite Zeile (Dest)
        elif i == 1:
            current_scale = font_scale * 0.9
            current_color = (150, 255, 150)  # Grünlich für Dest
            current_thickness = 1
        # Dritte Zeile (Placement)
        elif i == 2:
            current_scale = font_scale * 0.9
            current_color = (150, 200, 255)  # Bläulich für Placement
            current_thickness = 1
        # Vierte Zeile (Require: Signature)
        else:
            current_scale = font_scale * 1.1  # Größer als andere Zeilen
            current_color = (0, 100, 255)  # Helles Rot für Signature (BGR Format)
            current_thickness = 2  # Dicker für bessere Sichtbarkeit
        
        # Text mit Schatten für bessere Lesbarkeit
        shadow_x = x + padding + 1
        shadow_y = cursor_y + 1
        cv2.putText(img, line, (shadow_x, shadow_y), font, current_scale, (0, 0, 0), current_thickness + 1, cv2.LINE_AA)
        cv2.putText(img, line, (x + padding, cursor_y), font, current_scale, current_color, current_thickness, cv2.LINE_AA)
        
        cursor_y += line_h + line_gap

# Keep looping
while True:
    # Reading the frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)

    # Frame-Skipping: Scanne nicht jedes Frame für bessere Performance
    frame_counter += 1
    if frame_counter % SCAN_EVERY_N_FRAMES == 0:
        new_codes = decode_codes(frame)
    else:
        # Verwende die letzten gefundenen Codes
        new_codes = []
    
    if DEBUG_MODE:
        if new_codes:
            print(f"Erkannte Codes ({len(new_codes)}): {new_codes}")
        else:
            print("Keine Codes erkannt")
    
    display_code = pick_display_code(new_codes)
    last_code, last_signature = update_locked_code(display_code, last_code, last_signature)
    
    # Erstelle moderne Anzeige mit Code, Dest und Placement
    if last_code:
        # Extrahiere nur den Code-Inhalt (ohne Typ-Präfix)
        code_content = last_code.split(": ", 1)[1] if ": " in last_code else last_code
        display_lines = [
            f"Parcel: {code_content}",
            "Dest: Teststrasse 23 8400 Winterthur",
            "Placement: Entrance Code 1234",
            "Require: Signature"
        ]
    else:
        display_lines = [
            "Parcel: "
        ]

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
    # Cache die Mask - berechne nur neu wenn Größe sich ändert
    current_size = (H, W)
    if lens_mask_cache is None or lens_mask_size != current_size:
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
        
        lens_mask_cache = lens_mask
        lens_mask_size = current_size
    else:
        lens_mask = lens_mask_cache

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

    # 8) Display decoded codes - OBEN zentriert in jeder Linse
    lens_w = top_w
    left_lens_x0 = int(cxL - lens_w // 2)
    right_lens_x0 = int(cxR - lens_w // 2)
    
    # Verwende die gesamte Linsenbreite (mit etwas Padding)
    text_padding = int(lens_w * 0.1)  # 10% Padding auf jeder Seite
    text_width = lens_w - text_padding * 2
    
    # Positioniere oben in der Linse
    text_y = int(y0 + lens_h * 0.05)  # 5% von oben
    
    # Zentriere horizontal in jeder Linse (cxL und cxR sind die Zentren)
    left_x = cxL
    right_x = cxR
    
    # Erstelle temporäres Bild für Text-Overlay
    # Zeichne Text auf separatem Overlay, dann maskiere mit Lens-Mask
    text_overlay = glasses_masked.copy()
    
    # Zeichne Text-Block zentriert auf Overlay
    draw_text_block(text_overlay, left_x, text_y, display_lines, text_width, center_x=True)
    draw_text_block(text_overlay, right_x, text_y, display_lines, text_width, center_x=True)
    
    # Wende Lens-Mask an: Text nur dort sichtbar, wo Lens-Mask aktiv ist
    # Außerhalb der Linsen: Original-Video, innerhalb: Video + Text
    glasses_masked = np.where(
        lens_mask[:, :, None] == 255, 
        text_overlay,  # Innerhalb Linsen: Video mit Text
        glasses_masked  # Außerhalb Linsen: Nur Video (schwarz)
    )

    # Show glasses view
    cv2.imshow("Glasses", glasses_masked)

    # ESC-Taste zum Schließen
    if cv2.waitKey(1) & 0xFF == 27:  # 27 ist ESC
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()
