import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import datetime
import time

# MediaPipe Initialisierung für Hand-Erkennung
def download_model_if_needed():
    """Lädt das Hand Landmarker Modell herunter, falls es nicht vorhanden ist."""
    model_dir = os.path.join(os.path.expanduser("~"), ".mediapipe_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hand_landmarker.task")
    
    if not os.path.exists(model_path):
        print("Lade Hand Landmarker Modell herunter...")
        try:
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Modell erfolgreich heruntergeladen!")
        except Exception as e:
            print(f"Fehler beim Herunterladen des Modells: {e}")
            print("Bitte stellen Sie sicher, dass Sie eine Internetverbindung haben.")
            exit(1)
    
    return model_path

# Modell-Pfad
model_path = download_model_if_needed()

# Hand Landmarker Optionen - 2 Hände für Foto-Geste
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,  # Zwei Hände für Foto-Geste
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Hand Landmarker erstellen
hand_landmarker = vision.HandLandmarker.create_from_options(options)

def is_photo_gesture(hand_landmarks_list):
    """
    Prüft, ob die "Fotografieren"-Geste erkannt wird:
    - Zwei Hände müssen erkannt werden
    - Jede Hand: Daumen und Zeigefinger bilden einen Rahmen (gekrümmt, L-Form)
    - Mittel-, Ring- und kleiner Finger sind eingeklappt
    """
    if len(hand_landmarks_list) < 2:
        return False
    
    # Landmark-Indizes
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_MCP = 9
    RING_TIP = 16
    RING_MCP = 13
    PINKY_TIP = 20
    PINKY_MCP = 17
    
    def is_frame_hand(hand_landmarks):
        """Prüft ob eine Hand die Rahmen-Position zeigt"""
        if len(hand_landmarks) < 21:
            return False
        
        # Daumen: TIP sollte höher sein als IP (nach oben gekrümmt)
        thumb_curved_up = hand_landmarks[THUMB_TIP].y < hand_landmarks[THUMB_IP].y
        
        # Zeigefinger: gekrümmt (PIP sollte zwischen TIP und MCP sein)
        # TIP sollte nicht vollständig ausgestreckt sein (nicht viel höher als PIP)
        index_curved = (hand_landmarks[INDEX_TIP].y > hand_landmarks[INDEX_PIP].y - 0.02) and \
                       (hand_landmarks[INDEX_PIP].y < hand_landmarks[INDEX_MCP].y)
        
        # Daumen und Zeigefinger sollten relativ nah beieinander sein (Rahmen bilden)
        thumb_tip = hand_landmarks[THUMB_TIP]
        index_tip = hand_landmarks[INDEX_TIP]
        thumb_index_distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        frame_formed = thumb_index_distance < 0.15  # Schwellenwert für Rahmen-Form
        
        # Andere Finger müssen eingeklappt sein
        middle_closed = hand_landmarks[MIDDLE_TIP].y > hand_landmarks[MIDDLE_MCP].y
        ring_closed = hand_landmarks[RING_TIP].y > hand_landmarks[RING_MCP].y
        pinky_closed = hand_landmarks[PINKY_TIP].y > hand_landmarks[PINKY_MCP].y
        
        return thumb_curved_up and index_curved and frame_formed and \
               middle_closed and ring_closed and pinky_closed
    
    # Beide Hände müssen die Rahmen-Position zeigen
    return is_frame_hand(hand_landmarks_list[0]) and is_frame_hand(hand_landmarks_list[1])

def save_camera_screenshot(frame, prefix="screenshot"):
    """
    Speichert einen Screenshot des Kamerafeeds als PNG.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    ok = cv2.imwrite(filename, frame)
    if ok:
        print(f"Screenshot gespeichert: {os.path.abspath(filename)}")
        return filename
    print("Fehler: Screenshot konnte nicht gespeichert werden.")
    return None


# Variable um zu verhindern, dass Screenshot bei jedem Frame ausgelöst wird
was_showing_photo_gesture_previous_frame = False

# Variable für Screenshot-Anzeige in Gläsern
screenshot_to_show = None
screenshot_timestamp = None
SCREENSHOT_DISPLAY_DURATION = 5.0  # 5 Sekunden

# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

# Warte auf erstes Frame
ret, first_frame = cap.read()
if not ret:
    print("Fehler: Konnte kein Frame von der Kamera lesen!")
    exit(1)

frame_timestamp_ms = 0

# Keep looping
while True:
    # Reading the frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process the frame with MediaPipe
    detection_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 33  # ~30 FPS
    
    is_showing_photo_gesture_current_frame = False
    
    # Prüfe auf Foto-Geste (zwei Hände im Rahmen)
    if detection_result.hand_landmarks and len(detection_result.hand_landmarks) >= 2:
        h, w, _ = frame.shape
        
        # Zeichne Hand-Landmarks zur Visualisierung
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Zeichne Verbindungen zwischen Landmarks
            connections = [
                [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
                [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
                [0, 9], [9, 10], [10, 11], [11, 12],  # Middle finger
                [0, 13], [13, 14], [14, 15], [15, 16],  # Ring finger
                [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
                [5, 9], [9, 13], [13, 17]  # Palm
            ]
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    end = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                    cv2.line(frame, start, end, (0, 0, 255), 2)
        
        # Prüfe auf Foto-Geste
        if is_photo_gesture(detection_result.hand_landmarks):
            is_showing_photo_gesture_current_frame = True
            cv2.putText(frame, "FOTO-GESTE ERKANNT!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Screenshot nur einmal auslösen, wenn die Geste gerade erkannt wurde
            if not was_showing_photo_gesture_previous_frame:
                save_camera_screenshot(frame, prefix="screenshot")
                # Screenshot für Anzeige in Gläsern speichern
                screenshot_to_show = frame.copy()
                screenshot_timestamp = time.time()
    
    # Aktualisiere den Status für den nächsten Frame
    was_showing_photo_gesture_previous_frame = is_showing_photo_gesture_current_frame

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
    
    # Zeige Screenshot in Gläsern, falls vorhanden und noch innerhalb der 3 Sekunden
    if screenshot_to_show is not None and screenshot_timestamp is not None:
        elapsed_time = time.time() - screenshot_timestamp
        if elapsed_time < SCREENSHOT_DISPLAY_DURATION:
            # Erstelle Screenshot-Glasses-View (gleiche Transformation wie normales Video)
            screenshot_h, screenshot_w = screenshot_to_show.shape[:2]
            screenshot_shift = int(screenshot_w * 0.02)
            screenshot_left_eye = np.roll(screenshot_to_show, -screenshot_shift, axis=1)
            screenshot_right_eye = np.roll(screenshot_to_show, screenshot_shift, axis=1)
            
            if screenshot_shift > 0:
                screenshot_left_eye[:, -screenshot_shift:] = 0
                screenshot_right_eye[:, :screenshot_shift] = 0
            
            screenshot_glasses_view = np.hstack([screenshot_left_eye, screenshot_right_eye])
            
            # Verwende Screenshot statt Live-Video für Gläser
            glasses_view = screenshot_glasses_view
        else:
            # Zeit abgelaufen, Screenshot zurücksetzen
            screenshot_to_show = None
            screenshot_timestamp = None
    
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
    
    # 8) Zeige roten Text innerhalb der Linsen auf beiden Seiten, wenn Screenshot angezeigt wird
    if screenshot_to_show is not None and screenshot_timestamp is not None:
        elapsed_time = time.time() - screenshot_timestamp
        if elapsed_time < SCREENSHOT_DISPLAY_DURATION:
            # Roter Text innerhalb der Linsen
            text = "Parcel placement evidence submitted"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # Viel größer
            color = (0, 0, 255)  # Rot in BGR
            thickness = 3
            
            # Text-Größe berechnen
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position innerhalb der linken Linse (oben)
            text_y = y0 + int(lens_h * 0.15)  # Oben innerhalb der Linse
            text_x_left = cxL - text_width // 2  # Zentriert in linker Linse
            
            # Position innerhalb der rechten Linse (oben)
            text_x_right = cxR - text_width // 2  # Zentriert in rechter Linse
            
            # Text auf beiden Seiten mit schwarzem Outline für bessere Lesbarkeit
            # Linke Linse
            cv2.putText(glasses_masked, text, (text_x_left, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(glasses_masked, text, (text_x_left, text_y), font, font_scale, color, thickness)
            
            # Rechte Linse
            cv2.putText(glasses_masked, text, (text_x_right, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(glasses_masked, text, (text_x_right, text_y), font, font_scale, color, thickness)

    # Show the tracking window
    cv2.imshow("Tracking", glasses_masked)

    # ESC-Taste zum Schließen aller Fenster
    if cv2.waitKey(1) & 0xFF == 27:  # 27 ist ESC
        break

# Release the camera and all resources
hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()