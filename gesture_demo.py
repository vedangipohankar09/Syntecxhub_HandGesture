import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gesture_core import HandTracker, classify_gesture, gesture_to_action, ACTION_FUNCTIONS

HOLD_SECONDS = 1.2
FONT_SIZE = 12
FONT_PATHS = ["C:/Windows/Fonts/times.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]

text_font = None
for path in FONT_PATHS:
    try:
        text_font = ImageFont.truetype(path, FONT_SIZE)
        break
    except Exception:
        continue
if text_font is None:
    text_font = ImageFont.load_default()

ACCENT_RGB = (255, 140, 0)
WHITE_RGB = (240, 240, 240)
PANEL_X, PANEL_Y = 10, 10
PANEL_W, PANEL_H = 220, 75

def draw_hud(frame, gesture, action, progress):
    overlay = frame.copy()
    cv2.rectangle(overlay, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + PANEL_H), color=(20, 20, 20), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    bar_x = PANEL_X + 12
    bar_y = PANEL_Y + 58
    bar_w = PANEL_W - 24
    bar_fill = int(bar_w * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 3), color=(60, 60, 60), thickness=-1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 3), color=ACCENT_RGB[::-1], thickness=-1)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((PANEL_X + 12, PANEL_Y + 10), f"Gesture: {gesture}", font=text_font, fill=WHITE_RGB)
    draw.text((PANEL_X + 12, PANEL_Y + 30), f"Action: {action}", font=text_font, fill=ACCENT_RGB)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

tracker = HandTracker()
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

pending_gesture = None
pending_start_time = 0
confirmed_gesture = None
confirmed_action = "No Action"

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
                        ACTION_FUNCTIONS[confirmed_action]()
                    except Exception as error:
                        print("Action failed:", error)
    else:
        pending_gesture = None

    frame = draw_hud(frame, gesture, confirmed_action, progress)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()