import warnings
import pickle
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

warnings.filterwarnings("ignore")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_FRAMES     = 100                   # how many frames the model expects as input
HOP            = 30                    # run inference every 30 frames (≈1 second at 30 FPS)
HAND_LANDMARKS = 21
FEATURE_DIM    = HAND_LANDMARKS * 2 * 3 # 21 points × 2 hands × (x,y,z) = 126
# ────────────────────────────────────────────────────────────────────────────────


def prepare_input_for_model(
    hand_matrix: np.ndarray,
    max_frames: int = MAX_FRAMES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare a single (num_frames, FEATURE_DIM) hand_matrix to the form:
      X:    (1, MAX_FRAMES, FEATURE_DIM)
      mask: (1, MAX_FRAMES)
    where mask[i] = 1 for “real” frames, 0 for padding.

    If hand_matrix has fewer than max_frames rows, we pad with zeros.
    """
    num_frames = hand_matrix.shape[0]
    feature_dim = hand_matrix.shape[1]

    X = np.zeros((1, max_frames, feature_dim), dtype=np.float32)
    mask = np.zeros((1, max_frames), dtype=np.uint8)

    frames_to_use = min(num_frames, max_frames)
    X[0, :frames_to_use, :] = hand_matrix[:frames_to_use, :]
    mask[0, :frames_to_use] = 1
    return X, mask


class ASLKeypointExtractor:
    """
    Extracts left‐ and right‐hand keypoints from a live webcam frame,
    draws them on the frame, and returns a 126‐dimensional vector
    (or None if no hands were detected).
    """

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Use MediaPipe Holistic in non‐static mode (video)
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        )

    def _landmarks_to_array(self, landmarks) -> List[List[float]]:
        return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]

    def _extract_frame_keypoints(self, frame: np.ndarray) -> Optional[Dict[str, List[List[float]]]]:
        """
        Runs MediaPipe on the BGR frame, draws the landmarks for both hands
        onto the frame, and returns a dict with keys "left_hand" and/or "right_hand",
        mapping to a list of 21 [x,y,z] lists each. If neither hand is found, returns None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        keypoints: Dict[str, List[List[float]]] = {}

        # Left hand
        if results.left_hand_landmarks:
            keypoints["left_hand"] = self._landmarks_to_array(results.left_hand_landmarks)
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        # Right hand
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


def run_live_inference(
    model,
    le,
    extractor: ASLKeypointExtractor,
    camera_index: int = 0
):
    """
    Opens the front‐facing camera, maintains a rolling buffer of the last MAX_FRAMES keypoint vectors,
    and every HOP frames runs one forward pass through `model`. It then overlays the predicted sign,
    confidence, and latency (ms) on the same window where hand landmarks are drawn.

    Args:
      - model : a compiled Keras sign‐recognition model
      - le    : a fitted sklearn LabelEncoder mapping integer→sign‐word
      - extractor: an ASLKeypointExtractor for generating 126‐dim features/frame
      - camera_index: which cv2.VideoCapture device to open (0=default front camera)
    """
    # On macOS, you can force AVFoundation to get the front camera:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera (index {}).".format(camera_index))

    print("📸 Starting live ASL inference. Press 'q' to quit.")
    buffer = deque(maxlen=MAX_FRAMES)
    frame_count = 0

    latest_label: Optional[str] = None
    latest_confidence: Optional[float] = None
    latest_latency_ms: Optional[float] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to grab a frame.")
                break

            frame_count += 1

            # 1) Extract hand keypoints and build a 126‐vector for this frame
            kpts = extractor._extract_frame_keypoints(frame)
            feature_vec = np.zeros((FEATURE_DIM,), dtype=np.float32)

            if kpts:
                # Fill left hand slots if available
                if "left_hand" in kpts:
                    for idx, (x, y, z) in enumerate(kpts["left_hand"]):
                        base = idx * 3
                        feature_vec[base : base + 3] = [x, y, z]
                # Fill right hand slots if available
                if "right_hand" in kpts:
                    for idx, (x, y, z) in enumerate(kpts["right_hand"]):
                        base = (HAND_LANDMARKS * 3) + idx * 3
                        feature_vec[base : base + 3] = [x, y, z]

            # Append this frame’s 126‐vector (or zero if no hands) to the rolling buffer
            buffer.append(feature_vec)

            # 2) Every HOP frames, if we have built up MAX_FRAMES in the buffer, run inference
            if (frame_count % HOP == 0) and (len(buffer) == MAX_FRAMES):
                hand_matrix = np.stack(buffer, axis=0)  # shape=(MAX_FRAMES, FEATURE_DIM)
                X_input, mask_input = prepare_input_for_model(hand_matrix)

                t0 = time.time()
                y_proba = model.predict(X_input, verbose=0)  # shape=(1,NUM_CLASSES)
                t1 = time.time()

                pred_idx = int(np.argmax(y_proba, axis=1)[0])
                conf = float(y_proba[0, pred_idx])
                lab = le.inverse_transform([pred_idx])[0]

                latest_label = lab
                latest_confidence = conf
                latest_latency_ms = (t1 - t0) * 1000.0

            # 3) Overlay “latest_label (XX.X%)” and “Latency: YYY ms” on top‐left of frame
            if latest_label is not None:
                # Draw a translucent rectangle behind text (so it’s readable)
                cv2.rectangle(frame, (5, 5), (300, 60), (0, 0, 0), thickness=-1)
                text_label = f"{latest_label} ({latest_confidence*100:.1f}%)"
                text_latency = f"Latency: {latest_latency_ms:.0f} ms"

                cv2.putText(
                    frame,
                    text_label,
                    org=(10,  30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    text_latency,
                    org=(10, 55),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

            # 4) Show the annotated frame
            cv2.imshow("ASL Live Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # ── 1) Load your label encoder (pickled previously) ──────────────────────────
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # ── 2) Load your trained Keras model ─────────────────────────────────────────
    model = load_model("sign_model_test_47.keras")

    # ── 3) Instantiate the MediaPipe‐based keypoint extractor ──────────────────
    extractor = ASLKeypointExtractor()

    # ── 4) Run the live loop (carries on until you press 'q') ───────────────────
    run_live_inference(model=model, le=le, extractor=extractor, camera_index=0)