# Hand Gesture Recognition 🖐️

Control your PC with hand gestures — no mouse, no keyboard. Just show your hand to the webcam.

Built with MediaPipe + OpenCV, with a cyberpunk-themed HUD.

## What it does

Show a gesture, hold it for a second, and it triggers an action:

| Gesture | Action |
|---|---|
|  Fist | Pause |
|  Open Palm | Play |
|  Thumbs Up | Volume Up |
|  Peace Sign | Volume Down |
|  One Finger | Brightness Down |
|  Three Fingers | Brightness Up |

## Setup

```bash
pip install -r requirements.txt
python gesture_demo.py
```

Press `q` to quit.

## Notes

- Windows only (uses `pycaw` for volume, `screen-brightness-control` for brightness)
- If it crashes silently with no error, you probably have both `opencv-python` and `opencv-contrib-python` installed — remove the plain one:
  ```bash
  pip uninstall opencv-python -y
  ```
- `hand_landmarker.task` is already in this repo, so no download needed on first run.

---
