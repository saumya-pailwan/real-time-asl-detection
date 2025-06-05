import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import pickle
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure page
st.set_page_config(
    page_title="ASL Detection System",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        margin: 10px 0;
    }
    .prediction-box {
        border: 2px solid #2E86AB;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f0f8ff;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load your trained Random Forest model
@st.cache_resource
def load_model():
    """Load your trained Random Forest model"""
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize MediaPipe
@st.cache_resource
def initialize_mediapipe():
    """Initialize MediaPipe hands detection"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing, mp_hands

# Define your ASL classes (based on your classification report)
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def extract_hand_landmarks(image, hands):
    """Extract hand landmarks from image using MediaPipe - matches original training preprocessing"""
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Handle different image formats (RGB/RGBA)
        if len(image_np.shape) != 3:
            st.error(f"Image shape is {image_np.shape}, expected 3D image")
            return None, None
        
        # Convert RGBA to RGB if needed (remove alpha channel)
        if image_np.shape[2] == 4:
            # RGBA image - convert to RGB by removing alpha channel
            image_np = image_np[:, :, :3]
        elif image_np.shape[2] != 3:
            st.error(f"Image has {image_np.shape[2]} channels, expected 3 or 4")
            return None, None
        
        # Match the original preprocessing exactly:
        # PIL images are RGB, but we need to ensure consistent processing
        if isinstance(image, Image.Image):
            # PIL image is RGB, but let's convert to BGR then back to RGB to match training
            # This ensures the same color space processing as the original training data
            bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        else:
            # OpenCV image (camera) - convert BGR to RGB like in training
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe (same as training)
        results = hands.process(rgb_image)
        
        # Skip if no hand or multiple hands detected (same logic as training)
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 1:
            return None, results
        
        # Extract landmarks exactly like in training
        hand_landmarks = results.multi_hand_landmarks[0]
        positions = []
        for landmark in hand_landmarks.landmark:
            positions.extend([landmark.x, landmark.y, landmark.z])
        
        # Should have 63 features (21 landmarks × 3 coordinates)
        if len(positions) != 63:
            st.error(f"Expected 63 features, got {len(positions)}")
            return None, results
        
        return np.array(positions).reshape(1, -1), results
            
    except Exception as e:
        st.error(f"Error extracting landmarks: {str(e)}")
        return None, None

def predict_asl(model, landmarks):
    """Make prediction using the Random Forest model"""
    try:
        if landmarks is not None:
            prediction = model.predict(landmarks)
            probabilities = model.predict_proba(landmarks)
            
            predicted_class = prediction[0]
            confidence = np.max(probabilities[0])
            
            return predicted_class, confidence, probabilities[0]
        return None, None, None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def display_prediction(predicted_class, confidence):
    """Display prediction with styling based on confidence"""
    if confidence > 0.8:
        confidence_class = "confidence-high"
    elif confidence > 0.5:
        confidence_class = "confidence-medium"
    else:
        confidence_class = "confidence-low"
    
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Detected ASL Sign: <span style="color: #2E86AB; font-size: 2rem;">{predicted_class}</span></h3>
        <p>Confidence: <span class="{confidence_class}">{confidence:.2%}</span></p>
    </div>
    """, unsafe_allow_html=True)

def draw_landmarks_on_image(image, results, mp_drawing, mp_hands):
    """Draw hand landmarks on the image"""
    if results.multi_hand_landmarks:
        # Ensure we have a copy of the image
        if isinstance(image, Image.Image):
            annotated_image = np.array(image.copy())
        else:
            annotated_image = image.copy()
        
        # Handle RGBA to RGB conversion if needed
        if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 4:
            annotated_image = annotated_image[:, :, :3]
        
        # MediaPipe drawing expects RGB format
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # Red landmarks
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green connections
            )
        return annotated_image
    return np.array(image) if isinstance(image, Image.Image) else image

def main():
    # Header
    st.markdown('<h1 class="main-header"> ASL Detection System </h1>', unsafe_allow_html=True)
    
    # Load model and initialize MediaPipe
    model = load_model()
    if model is None:
        st.error("Please ensure your 'random_forest_model.pkl' file is in the correct path and try again.")
        return
    
    hands, mp_drawing, mp_hands = initialize_mediapipe()
    
    # Sidebar for mode selection
    st.sidebar.title("Detection Mode")
    mode = st.sidebar.radio(
        "Choose detection mode:",
        ["Image Upload", "Live Camera"],
        index=0
    )
    
    if mode == "Image Upload":
        st.header("📸 Image Upload Mode")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing ASL sign language"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.info(f"📄 **File:** {uploaded_file.name} | 📏 **Size:** {uploaded_file.size} bytes")
            
            with col2:
                st.subheader("Hand Detection & Prediction")
                landmarks, results = extract_hand_landmarks(image, hands)
                
                if landmarks is not None:
                    predicted_class, confidence, all_probabilities = predict_asl(model, landmarks)
                    
                    if predicted_class is not None:
                        display_prediction(predicted_class, confidence)
                        
                        st.subheader("Top 3 Predictions")
                        top_3_indices = np.argsort(all_probabilities)[-3:][::-1]
                        
                        for i, idx in enumerate(top_3_indices):
                            class_name = ASL_CLASSES[idx]
                            conf = all_probabilities[idx]
                            st.write(f"{i+1}. **{class_name}** - {conf:.2%}")
                    
                    st.subheader("Hand Landmarks Detection")
                    image_np = np.array(image)
                    if image_np.shape[2] == 4:
                        image_np = image_np[:, :, :3]
                    annotated_image = draw_landmarks_on_image(image_np, results, mp_drawing, mp_hands)
                    st.image(annotated_image, caption="Hand Landmarks", use_container_width=True)
                
                else:
                    st.warning("No hand detected in the image. Please upload an image with a clear hand sign.")
                    st.info("**Tips for better detection:**\n- Ensure good lighting\n- Hand should be clearly visible\n- Try different hand positions\n- Avoid cluttered backgrounds")

    elif mode == "Live Camera":
        st.header("📹 Live Camera Mode")
        
        # Camera controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_camera = st.button("Start Camera", key="start_cam")
        with col2:
            stop_camera = st.button("Stop Camera", key="stop_cam")
        with col3:
            predict_button = st.button("Predict Current Frame", key="predict")
        
        # Initialize session state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'cap' not in st.session_state:
            st.session_state.cap = None
        
        # Camera frame placeholder
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        prediction_placeholder = st.empty()
        
        if start_camera:
            try:
                st.session_state.cap = cv2.VideoCapture(0)
                if st.session_state.cap.isOpened():
                    st.session_state.camera_active = True
                    status_placeholder.success("Camera started successfully!")
                else:
                    status_placeholder.error("Could not access camera. Please check if camera is available.")
            except Exception as e:
                status_placeholder.error(f"Error starting camera: {str(e)}")
        
        if stop_camera:
            st.session_state.camera_active = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            status_placeholder.info("📹 Camera stopped.")
            frame_placeholder.empty()
            prediction_placeholder.empty()
        
        # Live camera feed
        if st.session_state.camera_active and st.session_state.cap is not None:
            ret, frame = st.session_state.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Predict on current frame
                if predict_button:
                    with prediction_placeholder.container():
                        st.subheader("Real-time Prediction")
                        
                        # Extract landmarks and predict
                        landmarks, results = extract_hand_landmarks(frame_rgb, hands)
                        
                        if landmarks is not None:
                            predicted_class, confidence, all_probabilities = predict_asl(model, landmarks)
                            
                            if predicted_class is not None:
                                # Display prediction
                                display_prediction(predicted_class, confidence)
                                
                                # Show top 3 predictions
                                st.subheader("Top 3 Predictions")
                                top_3_indices = np.argsort(all_probabilities)[-3:][::-1]
                                
                                for i, idx in enumerate(top_3_indices):
                                    class_name = ASL_CLASSES[idx]
                                    conf = all_probabilities[idx]
                                    st.write(f"{i+1}. **{class_name}** - {conf:.2%}")
                        else:
                            st.warning(" No hand detected in current frame.")
                            st.info(" Position your hand clearly in front of the camera")
                
                # Auto-refresh to keep camera feed live
                time.sleep(0.1)
                st.rerun()
        elif st.session_state.camera_active:
            status_placeholder.warning("📹 Camera is active but no frame received. Please check camera connection.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "ASL Detection System | Built with Streamlit & MediaPipe | Random Forest Model"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()