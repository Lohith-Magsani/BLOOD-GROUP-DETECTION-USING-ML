import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Load the model
try:
    model = load_model('model/blood_group_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

labels = ['A', 'B', 'AB', 'O']

def predict_blood_group(img_path):
    try:
        # Load the image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Preprocess the image: resize and normalize
        img = cv2.resize(img, (100, 100))
        img = img.reshape(1, 100, 100, 1) / 255.0  # Normalize

        # Predict the blood group
        pred = model.predict(img)
        return labels[np.argmax(pred)]
    except Exception as e:
        return f"Error in prediction: {e}"

# Example usage
print(predict_blood_group('test_fingerprint.png'))
