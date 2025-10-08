from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Import the CORS library
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
import os

# --- Initialization ---
# Use Flask's convention for templates folder
app = Flask(__name__, template_folder='templates')
CORS(app) # Enable CORS for your entire Flask app

# --- Load Models and Classifiers ---
try:
    # Load your trained Keras model
    emotion_model = load_model("emotion_model_full.keras")
    emotion_dict = {
        0: "Angry", 1: "Disgusted", 2: "Fearful",
        3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
    }

    # Load face detector from OpenCV
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("✅ Model and face detector loaded successfully.")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    emotion_model = None
    face_detector = None


# --- Routes ---
@app.route('/')
def serve_index():
    """
    This route serves the main HTML file using render_template,
    which looks for 'index.html' inside the 'templates' folder.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    This route handles the emotion prediction on an image frame.
    """
    if not emotion_model or not face_detector:
        return jsonify({"error": "Model or face detector not loaded"}), 500

    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # --- Emotion detection logic ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_gray = gray_frame[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            
            # Prepare image for the model
            img_pixels = np.expand_dims(cropped_img, axis=-1)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels = img_pixels / 255.0

            # Get prediction
            prediction = emotion_model.predict(img_pixels, verbose=0)[0]
            max_index = int(np.argmax(prediction))
            
            label = emotion_dict.get(max_index, "Unknown")
            confidence = float(np.max(prediction)) * 100

            return jsonify({"label": label, "confidence": confidence})

        else:
            return jsonify({"label": "No Face", "confidence": 0.0})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# --- Main Execution ---
if __name__ == '__main__':
    print("🚀 Starting Flask server in debug mode...")
    app.run(port=5000, debug=True)

