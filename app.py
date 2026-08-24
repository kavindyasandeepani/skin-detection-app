from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Trained Model Load 
model = load_model('skin_model.h5')


classes = ['acne', 'dry', 'normal', 'oily']

def get_recommendations(condition):
    recommendations = {
        "acne": ["Salicylic Acid Face Wash", "Benzoyl Peroxide Spot Treatment", "Oil-Free Moisturizer"],
        "dry": ["Hyaluronic Acid Serum", "Rich Hydrating Moisturizer", "Gentle Cream Cleanser"],
        "oily": ["Oil-Free Cleanser", "Mattifying Moisturizer", "Clay Mask"],
        "normal": ["Daily SPF Moisturizer", "Gentle Face Wash"]
    }
    return recommendations.get(condition, [])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # OpenCV - Face Detection
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        os.remove(filepath)
        return jsonify({"error": "No face detected. Please upload a clear selfie."})

    x, y, w, h = faces[0]
    face_region = image[y:y+h, x:x+w]
    face_resized = cv2.resize(face_region, (128, 128))

    # TensorFlow Model - Prediction
    face_array = face_resized / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    prediction = model.predict(face_array)
    result_index = np.argmax(prediction)
    condition = classes[result_index]
    confidence = float(np.max(prediction)) * 100

    products = get_recommendations(condition)
    os.remove(filepath)

    return jsonify({
        "condition": condition,
        "confidence": round(confidence, 2),
        "recommendations": products
    })

if __name__ == '__main__':
    app.run(debug=True)