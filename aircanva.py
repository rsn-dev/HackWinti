import numpy as np
import cv2
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
from navigation import draw_center_arrows

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

# Hand Landmarker Optionen
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Hand Landmarker erstellen
hand_landmarker = vision.HandLandmarker.create_from_options(options)

def is_only_index_finger_extended(hand_landmarks):
    """
    Prüft, ob nur der Zeigefinger ausgestreckt ist.
    Returns True wenn:
    - Zeigefinger ausgestreckt ist (Tip höher als MCP)
    - Mittel-, Ring- und kleiner Finger eingeklappt sind (Tip niedriger als MCP)
    """
    if len(hand_landmarks) < 21:
        return False
    
    # Landmark-Indizes
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_TIP = 20
    
    # Prüfe Zeigefinger: Tip muss höher sein als MCP (kleinere y-Koordinate)
    index_extended = hand_landmarks[INDEX_TIP].y < hand_landmarks[INDEX_MCP].y
    
    # Prüfe andere Finger: Tips müssen niedriger sein als MCPs (größere y-Koordinate)
    middle_closed = hand_landmarks[MIDDLE_TIP].y > hand_landmarks[MIDDLE_MCP].y
    ring_closed = hand_landmarks[RING_TIP].y > hand_landmarks[RING_MCP].y
    pinky_closed = hand_landmarks[PINKY_TIP].y > hand_landmarks[PINKY_MCP].y
    
    return index_extended and middle_closed and ring_closed and pinky_closed

def are_all_fingers_extended(hand_landmarks):
    """
    Prüft, ob alle Finger vollständig ausgestreckt sind (Handfläche zeigen).
    Returns True wenn alle Finger vollständig ausgestreckt sind.
    """
    if len(hand_landmarks) < 21:
        return False
    
    # Landmark-Indizes
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_TIP = 20
    
    # Prüfe alle Finger: Für vollständig ausgestreckte Finger müssen:
    # 1. TIP deutlich höher sein als MCP (kleinere y-Koordinate)
    # 2. PIP höher sein als MCP
    # 3. TIP höher sein als PIP
    # 4. Mindestabstand zwischen TIP und MCP für Sicherheit
    
    def is_fully_extended(tip, pip, mcp):
        """Prüft ob ein Finger vollständig ausgestreckt ist"""
        tip_above_pip = tip.y < pip.y
        pip_above_mcp = pip.y < mcp.y
        tip_above_mcp = tip.y < mcp.y
        # Mindestabstand von 0.05 zwischen TIP und MCP (normalisiert)
        sufficient_distance = (mcp.y - tip.y) > 0.05
        return tip_above_pip and pip_above_mcp and tip_above_mcp and sufficient_distance
    
    index_fully_extended = is_fully_extended(
        hand_landmarks[INDEX_TIP], 
        hand_landmarks[INDEX_PIP], 
        hand_landmarks[INDEX_MCP]
    )
    middle_fully_extended = is_fully_extended(
        hand_landmarks[MIDDLE_TIP], 
        hand_landmarks[MIDDLE_PIP], 
        hand_landmarks[MIDDLE_MCP]
    )
    ring_fully_extended = is_fully_extended(
        hand_landmarks[RING_TIP], 
        hand_landmarks[RING_PIP], 
        hand_landmarks[RING_MCP]
    )
    pinky_fully_extended = is_fully_extended(
        hand_landmarks[PINKY_TIP], 
        hand_landmarks[PINKY_PIP], 
        hand_landmarks[PINKY_MCP]
    )
    
    return index_fully_extended and middle_fully_extended and ring_fully_extended and pinky_fully_extended

def is_ok_sign(hand_landmarks):
    """
    Prüft, ob ein OK-Zeichen gezeigt wird:
    - Daumen und Zeigefinger bilden einen Ring (sind nah beieinander)
    - Mittel-, Ring- und kleiner Finger sind ausgestreckt
    """
    if len(hand_landmarks) < 21:
        return False
    
    # Landmark-Indizes
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_MCP = 5
    INDEX_PIP = 6
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_TIP = 20
    
    # Berechne euklidische Distanz zwischen Daumen- und Zeigefingerspitze
    thumb_tip = hand_landmarks[THUMB_TIP]
    index_tip = hand_landmarks[INDEX_TIP]
    
    # 3D-Distanz berechnen
    distance = np.sqrt(
        (thumb_tip.x - index_tip.x)**2 + 
        (thumb_tip.y - index_tip.y)**2 + 
        (thumb_tip.z - index_tip.z)**2
    )
    
    # Daumen und Zeigefinger müssen nah beieinander sein (Ring bilden)
    # Schwellenwert: 0.03 (normalisiert)
    thumb_index_connected = distance < 0.03
    
    # Prüfe ob andere Finger ausgestreckt sind
    def is_fully_extended(tip, pip, mcp):
        tip_above_pip = tip.y < pip.y
        pip_above_mcp = pip.y < mcp.y
        tip_above_mcp = tip.y < mcp.y
        sufficient_distance = (mcp.y - tip.y) > 0.05
        return tip_above_pip and pip_above_mcp and tip_above_mcp and sufficient_distance
    
    middle_fully_extended = is_fully_extended(
        hand_landmarks[MIDDLE_TIP], 
        hand_landmarks[MIDDLE_PIP], 
        hand_landmarks[MIDDLE_MCP]
    )
    ring_fully_extended = is_fully_extended(
        hand_landmarks[RING_TIP], 
        hand_landmarks[RING_PIP], 
        hand_landmarks[RING_MCP]
    )
    pinky_fully_extended = is_fully_extended(
        hand_landmarks[PINKY_TIP], 
        hand_landmarks[PINKY_PIP], 
        hand_landmarks[PINKY_MCP]
    )
    
    return thumb_index_connected and middle_fully_extended and ring_fully_extended and pinky_fully_extended

def is_thumbs_up(hand_landmarks):
    """
    Prüft Daumen hoch (👍):
    - Daumen TIP höher als IP und MCP (kleinere y => höher im Bild)
    - Andere Finger eingeklappt (TIP unter MCP)
    """
    if len(hand_landmarks) < 21:
        return False

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP  = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_TIP = 20

    thumb_tip = hand_landmarks[THUMB_TIP]
    thumb_ip  = hand_landmarks[THUMB_IP]
    thumb_mcp = hand_landmarks[THUMB_MCP]

    # Daumen deutlich nach oben
    thumb_up = (thumb_tip.y < thumb_ip.y) and (thumb_tip.y < thumb_mcp.y) and ((thumb_mcp.y - thumb_tip.y) > 0.05)

    # Andere Finger zu (TIP tiefer als MCP)
    index_closed  = hand_landmarks[INDEX_TIP].y  > hand_landmarks[INDEX_MCP].y
    middle_closed = hand_landmarks[MIDDLE_TIP].y > hand_landmarks[MIDDLE_MCP].y
    ring_closed   = hand_landmarks[RING_TIP].y   > hand_landmarks[RING_MCP].y
    pinky_closed  = hand_landmarks[PINKY_TIP].y  > hand_landmarks[PINKY_MCP].y

    return thumb_up and index_closed and middle_closed and ring_closed and pinky_closed

def show_canvas_popup(paintWindow):
    """
    Zeigt das Canvas (paintWindow) in einem Popup mit weißem Hintergrund.
    Speichert NICHT.
    """
    canvas = paintWindow.copy()
    
    # Sicherstellen, dass es uint8 ist
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    
    cv2.namedWindow("Canvas Popup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Canvas Popup", canvas.shape[1], canvas.shape[0])
    cv2.imshow("Canvas Popup", canvas)
    
    print("Canvas im Popup angezeigt. Schließen Sie das Fenster manuell.")

def save_camera_screenshot(frame, prefix="camera"):
    """
    Speichert einen Screenshot des Kamerafeeds als PNG.
    """
    import datetime
    import os
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    ok = cv2.imwrite(filename, frame)
    if ok:
        print(f"Camera-Screenshot gespeichert: {os.path.abspath(filename)}")
        return filename
    print("Fehler: Camera-Screenshot konnte nicht gespeichert werden.")
    return None

def show_screenshot_popup(frame):
    """
    Zeigt einen Screenshot des aktuellen Kamerafeeds in einem separaten Popup-Fenster.
    Das Fenster bleibt offen, bis der Benutzer es manuell schließt.
    """
    # Kopiere den aktuellen Frame
    screenshot = frame.copy()
    
    # Erstelle ein neues Fenster für den Screenshot
    cv2.namedWindow("Screenshot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Screenshot", screenshot.shape[1], screenshot.shape[0])
    
    # Zeige den Screenshot im Popup-Fenster
    cv2.imshow("Screenshot", screenshot)
    
    print("Screenshot im Popup-Fenster angezeigt. Schließen Sie das Fenster manuell.")
    
    return screenshot

def export_drawing_from_canvas(paintWindow):
    """
    Exportiert die Zeichnung direkt aus dem Canvas (paintWindow).
    Das ist der robusteste Weg, da paintWindow bereits alle Linien enthält.
    """
    import datetime
    import os
    
    try:
        if paintWindow is None or paintWindow.size == 0:
            print("Fehler: paintWindow ist leer!")
            return None
        
        # Stelle sicher, dass das Canvas uint8 ist
        canvas_u8 = paintWindow
        if canvas_u8.dtype != np.uint8:
            canvas_u8 = np.clip(canvas_u8, 0, 255).astype(np.uint8)
        
        # Generiere Dateinamen mit Zeitstempel
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{timestamp}.png"
        
        # Speichere das Bild direkt
        success = cv2.imwrite(filename, canvas_u8)
        if success:
            full_path = os.path.abspath(filename)
            print(f"Zeichnung erfolgreich exportiert als: {full_path}")
            return filename
        else:
            print(f"Fehler: Konnte Bild nicht speichern: {filename}")
            return None
            
    except Exception as e:
        print(f"Fehler beim Exportieren: {e}")
        import traceback
        traceback.print_exc()
        return None


# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Variable um zu verfolgen, ob im vorherigen Frame gezeichnet wurde
was_drawing_previous_frame = False
# Variable um zu verhindern, dass Clear All bei jedem Frame ausgelöst wird
was_showing_palm_previous_frame = False
# Variable um zu verhindern, dass OK-Zeichen bei jedem Frame ausgelöst wird
was_showing_ok_previous_frame = False
# Variable um zu verhindern, dass Thumbs Up bei jedem Frame ausgelöst wird
was_showing_thumbs_previous_frame = False

# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

# Warte auf erstes Frame, um Canvas-Größe zu bestimmen
ret, first_frame = cap.read()
if not ret:
    print("Fehler: Konnte kein Frame von der Kamera lesen!")
    exit(1)

# Canvas setup - dynamisch an Frame-Größe angepasst
paintWindow = np.ones((first_frame.shape[0], first_frame.shape[1], 3), dtype=np.uint8) * 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# OPTIONAL: Recording aktivieren
recording_enabled = False  # Auf True setzen, wenn Video-Recording gewünscht ist
video_writer = None
recording_stopped = False

if recording_enabled:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = first_frame.shape
    video_writer = cv2.VideoWriter("recording.mp4", fourcc, 30.0, (w, h))
    print("Recording gestartet: recording.mp4")
frame_timestamp_ms = 0

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
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process the frame with MediaPipe
    detection_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 33  # ~30 FPS
    
    center = None
    is_drawing_current_frame = False
    is_showing_palm_current_frame = False
    is_showing_ok_current_frame = False
    is_showing_thumbs_current_frame = False
    
    # If hand is detected
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, _ = frame.shape
            
            # Draw hand landmarks (optional, for visualization)
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections between landmarks
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
            
            # Get the index finger tip (landmark 8)
            # MediaPipe hand landmarks: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
            if len(hand_landmarks) >= 21:
                # 1) THUMBS UP (👍) - Popup Canvas + Recording stoppen + Camera Screenshot (priorisiert)
                if is_thumbs_up(hand_landmarks):
                    is_showing_thumbs_current_frame = True
                    print("THUMBS UP - Show canvas + stop rec + screenshot")
                    # cv2.putText(frame, "THUMBS UP - Show canvas + stop rec + screenshot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Nur einmal auslösen
                    if not was_showing_thumbs_previous_frame:
                        # 1) Canvas Popup (NICHT speichern)
                        show_canvas_popup(paintWindow)
                        
                        # 2) Recording beenden
                        if recording_enabled and (video_writer is not None) and (not recording_stopped):
                            video_writer.release()
                            recording_stopped = True
                            video_writer = None
                            print("Recording beendet.")
                        
                        # 3) Camera-Screenshot schießen (Tracking frame)
                        save_camera_screenshot(frame, prefix="camera")
                
                # 2) Prüfe, ob alle Finger ausgestreckt sind (Clear All Geste)
                elif are_all_fingers_extended(hand_landmarks):
                    is_showing_palm_current_frame = True
                    # Zeige Hinweis
                    print("CLEAR ALL - Show palm")
                    # cv2.putText(frame, "CLEAR ALL - Show palm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Clear All nur einmal auslösen, wenn die Geste gerade erkannt wurde
                    if not was_showing_palm_previous_frame:
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0

                        paintWindow[:,:,:] = np.uint8(255)
                        print("Canvas cleared!")
                
                # Prüfe, ob OK-Zeichen gezeigt wird (Daumen und Zeigefinger verbunden, andere Finger ausgestreckt)
                elif is_ok_sign(hand_landmarks):
                    is_showing_ok_current_frame = True
                    # Zeige Hinweis
                    print("OK - Taking screenshot...")
                    # cv2.putText(frame, "OK - Taking screenshot...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    # Zeige Screenshot im Popup nur einmal, wenn die Geste gerade erkannt wurde
                    if not was_showing_ok_previous_frame:
                        show_screenshot_popup(frame)
                
                # Prüfe, ob nur der Zeigefinger ausgestreckt ist
                elif is_only_index_finger_extended(hand_landmarks):
                    index_finger_tip = hand_landmarks[8]
                    
                    # Convert normalized coordinates to pixel coordinates
                    center = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                    
                    # Draw circle at finger tip (grün wenn aktiv)
                    cv2.circle(frame, center, 10, (0, 255, 0), -1)
                    
                    # Nur zeichnen wenn nur Zeigefinger ausgestreckt ist
                    is_drawing_current_frame = True
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)
                else:
                    # Zeigefinger nicht allein ausgestreckt - zeige gelben Kreis zur Info
                    index_finger_tip = hand_landmarks[8]
                    center = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                    cv2.circle(frame, center, 10, (0, 255, 255), -1)
                    print("Only point index finger")
                    # cv2.putText(frame, "Only point index finger", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Wenn im vorherigen Frame gezeichnet wurde, aber jetzt nicht mehr, neue deques erstellen
    # um die Linien nicht zu verbinden
    if was_drawing_previous_frame and not is_drawing_current_frame:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
    
    # Wenn keine Hand erkannt wird, auch neue deques erstellen
    if not detection_result.hand_landmarks:
        if was_drawing_previous_frame:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
    
    # Wenn Recording aktiv und nicht gestoppt: Frame ins Video
    if recording_enabled and (video_writer is not None) and (not recording_stopped):
        video_writer.write(frame)
    
    # Aktualisiere den Status für den nächsten Frame
    was_drawing_previous_frame = is_drawing_current_frame
    was_showing_palm_previous_frame = is_showing_palm_current_frame
    was_showing_ok_previous_frame = is_showing_ok_current_frame
    was_showing_thumbs_previous_frame = is_showing_thumbs_current_frame

    # Draw lines of all the colors on the canvas and frame 
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

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

    key = cv2.waitKeyEx(1)
    if key == 27:  # ESC-Taste zum Schliessen aller Fenster
        break
    elif key == 2424832:      # LEFT arrow
        arrow_angle -= 10
    elif key == 2555904:      # RIGHT arrow
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

    # Show all the windows
    cv2.imshow("Tracking", glasses_masked)
    cv2.imshow("Paint", paintWindow)

# Release the camera and all resources
if video_writer is not None:
    video_writer.release()
    print("Recording beendet.")
hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()