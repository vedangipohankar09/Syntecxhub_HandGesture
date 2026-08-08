import os
import math
import urllib.request
import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities
import screen_brightness_control as sbc
import keyboard

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
ACCENT_COLOR = (255, 140, 0)
LINE_THICKNESS = 2
DOT_RADIUS = 4

class HandTracker:
    def __init__(self, max_hands=2):
        if not os.path.exists(MODEL_PATH):
            print("Downloading hand landmark model (first run only)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        return result

    def get_landmark_points(self, hand_landmarks, frame_shape):
        height, width, _ = frame_shape
        points = []
        for landmark in hand_landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            points.append((x, y))
        return points

    def draw_landmarks(self, frame, points):
        for connection in HAND_CONNECTIONS:
            start_point = tuple(int(v) for v in points[connection.start])
            end_point = tuple(int(v) for v in points[connection.end])
            cv2.line(frame, start_point, end_point, color=ACCENT_COLOR, thickness=int(LINE_THICKNESS), lineType=cv2.LINE_AA)

        for point in points:
            point = tuple(int(v) for v in point)
            cv2.circle(frame, point, int(DOT_RADIUS), color=ACCENT_COLOR, thickness=-1, lineType=cv2.LINE_AA)


FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def fingers_up(points):
    wrist = points[0]
    fingers = []

    if points[4][0] > points[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip_distance = distance(points[tip_id], wrist)
        pip_distance = distance(points[pip_id], wrist)
        if tip_distance > pip_distance * 1.15:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def classify_gesture(points):
    fingers = fingers_up(points)

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    if fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    if fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "Peace Sign"
    if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
        return "One Finger"
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
        return "Three Fingers"
    return "Unknown Gesture"

def gesture_to_action(gesture):
    actions = {
        "Fist": "Pause",
        "Open Palm": "Play",
        "Thumbs Up": "Volume Up",
        "Peace Sign": "Volume Down",
        "One Finger": "Brightness Down",
        "Three Fingers": "Brightness Up",
    }
    return actions.get(gesture, "No Action")


speaker = AudioUtilities.GetSpeakers()
volume_interface = speaker.EndpointVolume

def volume_up():
    current = volume_interface.GetMasterVolumeLevelScalar()
    new_level = min(current + 0.05, 1.0)
    volume_interface.SetMasterVolumeLevelScalar(new_level, None)

def volume_down():
    current = volume_interface.GetMasterVolumeLevelScalar()
    new_level = max(current - 0.05, 0.0)
    volume_interface.SetMasterVolumeLevelScalar(new_level, None)

def brightness_up():
    current = sbc.get_brightness()[0]
    sbc.set_brightness(min(current + 10, 100))

def brightness_down():
    current = sbc.get_brightness()[0]
    sbc.set_brightness(max(current - 10, 0))

def toggle_play_pause():
    keyboard.send("play/pause media")

ACTION_FUNCTIONS = {
    "Volume Up": volume_up,
    "Volume Down": volume_down,
    "Brightness Up": brightness_up,
    "Brightness Down": brightness_down,
    "Play": toggle_play_pause,
    "Pause": toggle_play_pause,
}