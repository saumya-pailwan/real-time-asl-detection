import cv2
import mediapipe as mp
import numpy as np
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load trained model
with open('random_forest_model_lexasl_3D.pkl', 'rb') as f:
    model = pickle.load(f)


def get_hand_roi(landmarks, frame_shape):
    """Get bounding box coordinates from hand landmarks"""
    h, w = frame_shape[:2]
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    return (int(min(x_coords)), int(min(y_coords))), (int(max(x_coords)), int(max(y_coords)))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Mirror the frame for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
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
                positions.extend([landmark.x, landmark.y, landmark.z])
            
            # Predict
            if len(positions) == 63:  # 21 landmarks * 2 (x,y)
                prediction = model.predict([positions])[0]
                proba = model.predict_proba([positions])[0].max()
                
                # Only show predictions with high confidence
                if proba > 0.0:  # Adjust threshold as needed
                    # Get hand region of interest
                    (x1, y1), (x2, y2) = get_hand_roi(hand_landmarks, frame.shape)
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{prediction} ({proba:.2f})", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow('ASL Detection', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
