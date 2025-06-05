import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import Counter
import pyttsx3
import threading


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load trained model
with open('Alphabet/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


def get_hand_roi(landmarks, frame_shape):
    """Get bounding box coordinates from hand landmarks"""
    h, w = frame_shape[:2]
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    return (int(min(x_coords)), int(min(y_coords))), (int(max(x_coords)), int(max(y_coords)))

cap = cv2.VideoCapture(0)
tts_engine = pyttsx3.init()


buffered_predictions = []
window_start = None
sentence = ""
max_sentence_len = 20
window_duration = 3.0
# Helper to run tts async to prevent blocking
def speak_async(text: str):
    def _worker(txt):
        tts_engine.say(txt)
        tts_engine.runAndWait()
    threading.Thread(target=_worker, args=(text,), daemon=True).start()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Mirror the frame for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()

    
    if results.multi_hand_landmarks:
        #If we see a hand but haven't started the buffer, start collecting
        if window_start is None:
            window_start = current_time
            buffered_predictions = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract features for prediction
            positions = []
            for landmark in hand_landmarks.landmark:
                positions.extend([landmark.x, landmark.y])
            
            # Predict
            if len(positions) == 42:  # 21 landmarks * 2 (x,y)
                prediction = model.predict([positions])[0]
                proba = model.predict_proba([positions])[0].max()

                if proba > 0.15:
                    buffered_predictions.append(prediction)
                
                # Only show predictions with high confidence
                if proba > 0.0:  # Adjust threshold as needed
                    # Get hand region of interest
                    (x1, y1), (x2, y2) = get_hand_roi(hand_landmarks, frame.shape)
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{prediction} ({proba:.2f})", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2, cv2.LINE_AA)
                    
        # Calculate how much time has passed since the current window has started
        elapsed = current_time - window_start
        # If the window has been open for window_duration seconds, close it
        if elapsed >= window_duration:
            if buffered_predictions:
                # Find the most common letter in the current buffer
                most_common, count = Counter(buffered_predictions).most_common(1)[0]
                # Append it to the UI Output
                if most_common == "Space":
                    sentence += "' '"
                    speak_async(sentence)
                else:
                    sentence += most_common
                window_start = None
                if len(sentence) >= max_sentence_len:
                    sentence = sentence[-max_sentence_len:]
    else:
        # We don't see a hand, so reset the window_start and the buffer
        window_start = None
        buffered_predictions = []

    frame_h, frame_w = frame.shape[:2]
    text_position = (10, frame_h - 20)
    cv2.putText(
        frame,
        f"Spelled: {sentence}",
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Display frame
    cv2.imshow('ASL Detection', frame)
    
    # Exit on 'q' press
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    # Clear displayed sentence on 'c' press
    elif key == ord('c'):
        sentence = ""

cap.release()
cv2.destroyAllWindows()
