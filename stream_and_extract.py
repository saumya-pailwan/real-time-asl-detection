# stream_and_extract.py
import cv2
import tempfile
import subprocess
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def stream_video_to_frames(url, start_time, end_time):
    duration = end_time - start_time
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    try:
        # Force best mp4 format
        command = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "-o", temp_video,
            url
        ]
        subprocess.run(command, check=True)

        # Check file size
        if os.path.getsize(temp_video) < 1024 * 100:
            raise ValueError("Downloaded file is too small or empty.")

        # Trim video segment using ffmpeg
        trimmed_video = tempfile.NamedTemporaryFile(suffix="_trimmed.mp4", delete=False).name
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
            "-i", temp_video, "-vf", "fps=25", "-loglevel", "quiet", trimmed_video
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(temp_video)

        # Read frames with OpenCV
        cap = cv2.VideoCapture(trimmed_video)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        os.remove(trimmed_video)

        return frames

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[FFMPEG ERROR] {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        raise RuntimeError(f"[Video Fetch Error] {e}")

def extract_keypoints_from_frames(frames):
    keypoints_sequence = []
    for frame in frames:
        results = mp_hands.process(frame)
        frame_kp = []
        for hand in results.multi_hand_landmarks or []:
            for lm in hand.landmark:
                frame_kp.extend([lm.x, lm.y, lm.z])
        frame_kp += [0.0] * (126 - len(frame_kp))
        keypoints_sequence.append(frame_kp)
    return np.array(keypoints_sequence)

def process_video_clip(sample):
    url = sample["url"]
    start = sample["start_time"]
    end = sample["end_time"]
    try:
        frames = stream_video_to_frames(url, start, end)
        keypoints = extract_keypoints_from_frames(frames)
        return keypoints
    except Exception as e:
        print(f"[Error] {sample['file']} - {e}")
        return None
