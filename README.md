# Hand Gesture Recognition 🖐️

Control your PC with hand gestures — no mouse, no keyboard. Just show your hand to the webcam.

Built with MediaPipe + OpenCV, with a cyberpunk tactical-HUD theme.

## What it does

Show a gesture, hold it for a second, and it triggers an action:

| Gesture | Action |
|---|---|
| ✊ Fist | Pause |
| ✋ Open Palm | Play |
| 👍 Thumbs Up | Volume Up |
| ✌️ Peace Sign | Volume Down |
| ☝️ One Finger | Brightness Down |
| 🤟 Three Fingers | Brightness Up |

When Volume/Brightness/Play/Pause fires, a native-style popup shows the
current level (like Windows' built-in OSD) and fades out after 2 seconds.

## Theme

Angular cut-corner HUD panel, icy blue border, monospace text. Hand
skeleton draws as a gradient: icy blue → purple → orange → red, with
smooth ~1.5px connecting lines and hollow diamond joint markers.

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
Made for the Syntecxhub internship project.
