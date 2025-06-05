import warnings
from realtime_helper import ASLKeypointExtractor, prepare_input_for_model
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

warnings.filterwarnings('ignore')

extractor = ASLKeypointExtractor()
le = LabelEncoder()

top100_signs = ['about', 'again', 'ask', 'bad', 'boy', 'but', 'buy', 'can', 'come',
 'different', 'drink', 'easy', 'eat', 'family', 'feel', 'few', 'find',
 'fine', 'finish', 'for', 'forget', 'friend', 'get', 'girl', 'give', 'go',
 'good', 'happy', 'hard', 'have', 'he', 'hello', 'help', 'home', 'how',
 'know', 'later', 'like', 'little', 'live', 'look', 'make', 'many', 'me',
 'meet', 'more', 'my', 'name', 'need', 'new', 'no', 'not', 'now', 'ok',
 'old', 'other', 'play', 'please', 'remember', 'right', 'sad', 'same',
 'say', 'school', 'see', 'she', 'sign', 'slow', 'some', 'sorry', 'stay',
 'take', 'talk', 'tell', 'thank you', 'their', 'they', 'thing', 'think',
 'time', 'tired', 'try', 'understand', 'use', 'wait', 'want', 'what',
 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'work', 'write',
 'wrong', 'yes', 'you', 'your']

le.fit(top100_signs)

modelpath = 'sign_model_test_47.keras'
model = load_model(modelpath)


for X, mask in extractor.run_realtime_extraction():
    #print(f"X.shape: {X.shape}, mask.shape: {mask.shape}")
    y_proba = model.predict(X)  
    y_pred = np.argmax(y_proba, axis=1)
    label = le.inverse_transform(y_pred)
    print(f"Predicted label: {label[0]}")
