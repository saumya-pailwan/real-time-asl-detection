import json
import numpy as np
import os
import cv2
import mediapipe as mp
import numpy as np
import json
from io import BytesIO
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import argparse
from collections import deque
import time

MAX_FRAMES     = 100
HAND_LANDMARKS = 21
FEATURE_DIM    = HAND_LANDMARKS * 2 * 3  # 21 points × 2 hands × (x,y,z)

def load_and_tabulate_with_mask(json_path):
    """
    Reads a split JSON, skips entries with processed_frames == 0, and returns:
      - X:    np.float32 array of shape (n_samples, MAX_FRAMES, FEATURE_DIM)
      - masks: np.uint8 array of shape (n_samples, MAX_FRAMES), with 1=real frame, 0=padded
      - labels: list of string labels, length = n_samples
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X_list    = []
    mask_list = []
    labels    = []

    for entry in data:
        keypoints_data   = entry.get("keypoints_data", {})
        video_info       = keypoints_data.get("video_info", {})
        processed_frames = video_info.get("processed_frames", 0)

        # Skip any sample with zero processed frames
        if processed_frames == 0:
            continue

        # We'll fill up to MAX_FRAMES rows of 126 features
        sample_matrix = np.zeros((MAX_FRAMES, FEATURE_DIM), dtype=np.float32)
        # Mask = 1 for real frames, 0 for padding
        sample_mask   = np.zeros((MAX_FRAMES,), dtype=np.uint8)

        frames = keypoints_data.get("keypoints", [])
        # How many actual frames do we have?  (cap it at MAX_FRAMES)
        num_real = min(len(frames), MAX_FRAMES)

        for i in range(num_real):
            feature_vec = np.zeros((FEATURE_DIM,), dtype=np.float32)

            frame_info = frames[i]

            # LEFT HAND (21 points × 3)
            left = frame_info.get("left_hand")
            if left:
                for idx, (x, y, z) in enumerate(left):
                    base = idx * 3
                    feature_vec[base : base+3] = [x, y, z]
            # If left is None, those 63 slots stay at zero

            # RIGHT HAND (21 points × 3)
            right = frame_info.get("right_hand")
            if right:
                for idx, (x, y, z) in enumerate(right):
                    base = (HAND_LANDMARKS * 3) + idx * 3
                    feature_vec[base : base+3] = [x, y, z]
            # If right is None, those 63 slots stay at zero

            sample_matrix[i] = feature_vec
            sample_mask[i]   = 1  # mark this row as “real”

        # Remaining rows [num_real:MAX_FRAMES] stay zero and mask stays 0

        X_list.append(sample_matrix)
        mask_list.append(sample_mask)
        labels.append(entry["original_data"]["clean_text"])

    X_array    = np.stack(X_list, axis=0)    # shape = (n_samples, MAX_FRAMES, FEATURE_DIM)
    mask_array = np.stack(mask_list, axis=0) # shape = (n_samples, MAX_FRAMES)
    return X_array, mask_array, labels

def prepare_input_for_model(hand_matrix: np.ndarray, max_frames: int = MAX_FRAMES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare real-time hand_matrix for ML model (padding & masking).

    Args:
        hand_matrix: shape (num_frames, FEATURE_DIM)
        max_frames: max length of input (e.g., 100)

    Returns:
        Tuple (X, mask):
            X: shape (1, MAX_FRAMES, FEATURE_DIM)
            mask: shape (1, MAX_FRAMES)
    """
    num_frames = hand_matrix.shape[0]
    feature_dim = hand_matrix.shape[1]
    
    # Initialize
    X = np.zeros((1, max_frames, feature_dim), dtype=np.float32)
    mask = np.zeros((1, max_frames), dtype=np.uint8)

    frames_to_use = min(num_frames, max_frames)
    X[0, :frames_to_use, :] = hand_matrix[:frames_to_use]
    mask[0, :frames_to_use] = 1

    return X, mask


class ASLKeypointExtractor:
    """
    Extract keypoints from ASL videos on the fly without downloading them permanently.
    Supports YouTube URLs and direct video URLs.
    """
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # Reduced for better performance
            enable_segmentation=False,
            refine_face_landmarks=False  # Disabled for better performance
        )
    
    def _extract_frame_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract keypoints from a single frame"""
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.holistic.process(rgb_frame)
        
        keypoints = {}
        
        # Extract hand keypoints
        if results.left_hand_landmarks:
            keypoints["left_hand"] = self._landmarks_to_array(results.left_hand_landmarks)
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            keypoints["right_hand"] = self._landmarks_to_array(results.right_hand_landmarks)
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        return keypoints if keypoints else None
    
    def _landmarks_to_array(self, landmarks) -> List[List[float]]:
        """Convert MediaPipe landmarks to array format"""
        return [[lm.x, lm.y, lm.z] 
                for lm in landmarks.landmark]
    
    def extract_keypoints_from_webcam(self, duration_sec: int = 4, fps: int = 30) -> Tuple[np.ndarray, int]:
        """
        Extract hand keypoints from webcam in real time.

        Args:
            duration_sec: Number of seconds to capture.
            fps: Target FPS (frames per second).
        
        Returns:
            Tuple of:
              - np.ndarray of shape (num_frames, FEATURE_DIM)
              - actual number of frames processed
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to access webcam")

        print("📸 Starting webcam capture. Press 'q' to quit early.")
        keypoints_list = []
        frame_count = 0
        max_frames = duration_sec * fps

        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                #30 fps
                #maybe flip here, maybe not?
                #frame = cv2.flip(frame, 1)
                if not ret:
                    print("⚠️ Failed to grab frame")
                    break
                keypoints = self._extract_frame_keypoints(frame)
                if keypoints:
                    feature_vec = np.zeros((FEATURE_DIM,), dtype=np.float32)

                    if "left_hand" in keypoints:
                        for idx, (x, y, z) in enumerate(keypoints["left_hand"]):
                            base = idx * 3
                            feature_vec[base:base+3] = [x, y, z]

                    if "right_hand" in keypoints:
                        for idx, (x, y, z) in enumerate(keypoints["right_hand"]):
                            base = HAND_LANDMARKS * 3 + idx * 3
                            feature_vec[base:base+3] = [x, y, z]

                    keypoints_list.append(feature_vec)
                    frame_count += 1

                # Show the frame for user feedback
                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        hand_matrix = np.stack(keypoints_list, axis=0) if keypoints_list else np.zeros((0, FEATURE_DIM))
        return hand_matrix, frame_count

    def run_realtime_extraction(self):
        HOP = 30

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to access webcam")

        print("📸 Capturing. Press 'q' to quit.")
        buffer = deque(maxlen=MAX_FRAMES)
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to grab frame")
                    break

                keypoints = self._extract_frame_keypoints(frame)
                feature_vec = np.zeros((FEATURE_DIM,), dtype=np.float32)

                if keypoints:
                    if "left_hand" in keypoints:
                        for idx, (x, y, z) in enumerate(keypoints["left_hand"]):
                            base = idx * 3
                            feature_vec[base:base + 3] = [x, y, z]

                    if "right_hand" in keypoints:
                        for idx, (x, y, z) in enumerate(keypoints["right_hand"]):
                            base = HAND_LANDMARKS * 3 + idx * 3
                            feature_vec[base:base + 3] = [x, y, z]

                buffer.append(feature_vec)

                # Display the frame with landmarks
                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Process input every HOP frames 
                frame_count += 1

                if frame_count % HOP == 0 and len(buffer) == MAX_FRAMES:
                    hand_matrix = np.stack(buffer, axis=0)
                    X, mask = prepare_input_for_model(hand_matrix)
                    yield X, mask

        finally:
            cap.release()
            cv2.destroyAllWindows()
