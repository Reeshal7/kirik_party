from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import logging

# Suppress TensorFlow logging and other INFO messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Configuration ---
# 1. UPDATE THIS PATH to your .keras model file
MODEL_PATH = "emotion_model.keras"
# 2. UPDATE THIS PATH to the Haar Cascade XML file
HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# 3. Define the emotion labels in the order your model was trained
EMOTION_DICT = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}
# 4. Define the input image size for your model
MODEL_INPUT_SIZE = (48, 48)


# --- Flask Setup ---
app = Flask(__name__, template_folder='.')

# --- Model & Detector Loading ---
try:
    emotion_model = load_model(MODEL_PATH)
    face_detector = cv2.CascadeClassifier(HAARCASCADE_PATH)
    print("✅ Model and face detector loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or cascade classifier: {e}")
    print("Please ensure the MODEL_PATH and HAARCASCADE_PATH are correct.")
    emotion_model = None
    face_detector = None


# --- Web Routes ---
@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive a video frame from the client, process it,
    and send back the emotion prediction.
    """
    if not emotion_model or not face_detector:
        return jsonify({'label': 'Server Error', 'confidence': 0}), 500

    try:
        # Get the base64 image data from the POST request
        data = request.get_json()
        header, encoded = data['image'].split(',', 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except (ValueError, TypeError, KeyError) as e:
        print(f"⚠️ Frame decoding failed or bad request: {e}")
        return jsonify({'error': 'Invalid image data'}), 400

    # Default values
    label = "No Face"
    confidence = 0.0

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Process the first detected face
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        
        # Extract the region of interest (ROI)
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)

        if roi_gray.size > 0:
            # Prepare the ROI for the model
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make a prediction
            prediction = emotion_model.predict(roi, verbose=0)[0]
            
            # Get the emotion label and confidence
            max_index = np.argmax(prediction)
            label = EMOTION_DICT.get(max_index, "Unknown")
            confidence = float(np.max(prediction)) * 100
    
    # Return the prediction as a JSON response
    return jsonify({'label': label, 'confidence': confidence})

# --- Main Execution ---
if __name__ == '__main__':
    print("🚀 Starting Flask server in debug mode...")
    app.run(port=5000, debug=True)

