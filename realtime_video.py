import warnings
import pickle
import numpy as np
from realtime_helper import ASLKeypointExtractor, prepare_input_for_model
from keras.models import load_model

warnings.filterwarnings('ignore')

# 1) Initialize your keypoint extractor as before:
extractor = ASLKeypointExtractor()

# 2) Load the pickled LabelEncoder (that was fitted on train+val+test labels):
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# 3) Load your trained Keras model (the .keras file you saved as “sign_model_test_47.keras”):
model = load_model("sign_model_test_47.keras")

# 4) In your real‐time loop, predict and then map back via le.inverse_transform():
for X, mask in extractor.run_realtime_extraction():
    #   X should be shaped (1, T_MAX, D_FEATURE) (or a batch of frames)
    #   “mask” is unused here, but you’re free to pass it if your model expects it.
    y_proba = model.predict(X)           # shape = (1, NUM_CLASSES)
    y_pred  = np.argmax(y_proba, axis=1) # shape = (1,)   (e.g. array([17]))
    label   = le.inverse_transform(y_pred)

    print(f"Predicted label: {label[0]}")