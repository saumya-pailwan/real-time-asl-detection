import cv2
import mediapipe as mp
import numpy as np
import pickle
import warnings
from realtime_helper import ASLKeypointExtractor, prepare_input_for_model

warnings.filterwarnings('ignore')

extractor = ASLKeypointExtractor()
for X, mask in extractor.run_realtime_extraction():
    print(f"X.shape: {X.shape}, mask.shape: {mask.shape}")

