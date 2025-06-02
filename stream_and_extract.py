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

def pad_or_truncate_kp(kp_array: np.ndarray, T_max: int = 100):
    """
    kp_array: shape (n_frames, 126)
    Returns:
      - fixed_x: np.ndarray of shape (T_max, 126)
      - mask:    np.ndarray of shape (T_max,), dtype=bool,
                 where mask[t]=True if t < original_length, else False
    """
    n_frames, D = kp_array.shape
    if n_frames >= T_max:
        # Truncate to the first T_max frames
        fixed_x = kp_array[:T_max, :]
        mask    = np.ones((T_max,), dtype=bool)
    else:
        # Pad with zeros for (T_max - n_frames) rows
        pad_amt = T_max - n_frames
        padding = np.zeros((pad_amt, D), dtype=kp_array.dtype)
        fixed_x = np.vstack([kp_array, padding])
        mask    = np.concatenate([
            np.ones((n_frames,), dtype=bool),
            np.zeros((pad_amt,), dtype=bool)
        ])
    return fixed_x, mask

def build_dataset(samples: list, T_max: int = 100):
    """
    samples: list of dicts, each containing at least:
             {
               "url":        "<video URL>",
               "start_time": <float seconds>,
               "end_time":   <float seconds>,
               "label":      <int or string label>
               # (optionally "file" or other metadata)
             }
    Returns:
      X_all: np.ndarray of shape (N, T_max, 126)
      mask:  np.ndarray of shape (N, T_max), dtype=bool
      y_all: np.ndarray of shape (N,)
    """
    X_list    = []
    mask_list = []
    y_list    = []

    for sample in samples:
        kp = process_video_clip(sample)   # → None or an array of shape (n_frames, 126)
        if kp is None:
            # Skip any sample that failed processing
            continue

        fixed_x, m = pad_or_truncate_kp(kp, T_max=T_max)
        X_list.append(fixed_x)
        mask_list.append(m)
        y_list.append(sample["label"])

    # Stack into big arrays
    X_all = np.stack(X_list, axis=0)     # shape = (N, T_max, 126)
    mask  = np.stack(mask_list, axis=0)  # shape = (N, T_max)
    y_all = np.array(y_list)             # shape = (N,)

    return X_all, mask, y_all
