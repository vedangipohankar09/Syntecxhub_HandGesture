import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gesture_core import HandTracker, classify_gesture, gesture_to_action, ACTION_FUNCTIONS

HOLD_SECONDS = 1.2
FONT_SIZE = 14
FONT_PATHS = ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"]

text_font = None
for path in FONT_PATHS:
    try:
        text_font = ImageFont.truetype(path, FONT_SIZE)
        break
    except Exception:
        continue
if text_font is None:
    text_font = ImageFont.load_default()

# Cyberpunk palette: Icy Blue / Purple / Orange / Red
ICY_BLUE_RGB = (79, 195, 247)     # #4FC3F7
PURPLE_RGB = (155, 89, 182)       # #9B59B6
ORANGE_RGB = (255, 140, 66)       # #FF8C42
RED_RGB = (230, 57, 80)           # #E63950

ICY_BLUE_BGR = ICY_BLUE_RGB[::-1]
ORANGE_BGR = ORANGE_RGB[::-1]

PANEL_X, PANEL_Y = 10, 10
PANEL_W, PANEL_H = 230, 75
OSD_FADE_SECONDS = 2.0

def draw_hud(frame, gesture, action, progress):
    # Angular cut-corner panel outline (no fill, just accent border over a dim wash)
    x, y, w, h = PANEL_X, PANEL_Y, PANEL_W, PANEL_H
    panel_pts = np.array([
        [x + 10, y], [x + w, y], [x + w, y + h - 10],
        [x + w - 10, y + h], [x, y + h], [x, y + 10]
    ])
    overlay = frame.copy()
    cv2.fillPoly(overlay, [panel_pts], (30, 15, 10))
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.polylines(frame, [panel_pts], True, ICY_BLUE_BGR, 1, cv2.LINE_AA)

    bar_x = PANEL_X + 14
    bar_y = PANEL_Y + 58
    bar_w = PANEL_W - 28
    bar_fill = int(bar_w * progress)
    cv2.line(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y), color=(60, 60, 60), thickness=2)
    cv2.line(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y), color=ORANGE_BGR, thickness=2)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((PANEL_X + 14, PANEL_Y + 10), f"GESTURE: {gesture.upper()}", font=text_font, fill=PURPLE_RGB)
    draw.text((PANEL_X + 14, PANEL_Y + 34), f"ACTION: {action.upper()}", font=text_font, fill=ORANGE_RGB)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def draw_osd(frame, label, level_percent):
    """Native-style pop-up showing current level, drawn bottom-center, caller controls when/how long it's visible."""
    h, w = frame.shape[:2]
    osd_w, osd_h = 220, 50
    ox = (w - osd_w) // 2
    oy = h - osd_h - 30

    overlay = frame.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + osd_w, oy + osd_h), color=(25, 20, 15), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    cv2.rectangle(frame, (ox, oy), (ox + osd_w, oy + osd_h), color=ICY_BLUE_BGR, thickness=1, lineType=cv2.LINE_AA)

    bar_x, bar_y, bar_w = ox + 14, oy + osd_h - 14, osd_w - 28
    bar_fill = int(bar_w * (level_percent / 100))
    cv2.line(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y), color=(60, 60, 60), thickness=3)
    cv2.line(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y), color=ORANGE_BGR, thickness=3)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((ox + 14, oy + 8), f"{label}: {level_percent}%", font=text_font, fill=ICY_BLUE_RGB)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

tracker = HandTracker()
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

pending_gesture = None
pending_start_time = 0
confirmed_gesture = None
confirmed_action = "No Action"

osd_label = None
osd_level = None
osd_shown_at = 0

while True:
    success, frame = cap.read()
    if not success:
        print("Could not access webcam")
        break

    frame = cv2.flip(frame, 1)
    result = tracker.process_frame(frame)

    gesture = "None"
    progress = 0

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = tracker.get_landmark_points(hand_landmarks, frame.shape)
            tracker.draw_landmarks(frame, points)

            gesture = classify_gesture(points)
            now = time.time()

            if gesture != pending_gesture:
                pending_gesture = gesture
                pending_start_time = now

            held_duration = now - pending_start_time
            progress = min(held_duration / HOLD_SECONDS, 1.0)

            if held_duration >= HOLD_SECONDS and gesture != confirmed_gesture:
                confirmed_gesture = gesture
                confirmed_action = gesture_to_action(gesture)
                if confirmed_action in ACTION_FUNCTIONS:
                    try:
                        result_value = ACTION_FUNCTIONS[confirmed_action]()
                        if confirmed_action in ("Volume Up", "Volume Down") and result_value is not None:
                            osd_label, osd_level, osd_shown_at = "VOLUME", result_value, now
                        elif confirmed_action in ("Brightness Up", "Brightness Down") and result_value is not None:
                            osd_label, osd_level, osd_shown_at = "BRIGHTNESS", result_value, now
                        elif confirmed_action in ("Play", "Pause"):
                            osd_label, osd_level, osd_shown_at = confirmed_action.upper(), 100, now
                    except Exception as error:
                        print("Action failed:", error)
    else:
        pending_gesture = None

    frame = draw_hud(frame, gesture, confirmed_action, progress)

    if osd_label is not None and (time.time() - osd_shown_at) < OSD_FADE_SECONDS:
        frame = draw_osd(frame, osd_label, osd_level)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
