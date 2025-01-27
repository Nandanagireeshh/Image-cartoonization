from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

@app.route('/cartoonize', methods=['POST'])
def cartoonize():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Read the image
    img = cv2.imread(image_path)

    # Ensure image was read successfully
    if img is None:
        return jsonify({"error": "Failed to process the image"}), 500

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur
    gray_blurred = cv2.medianBlur(gray, 7)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )

    # Apply bilateral filter to smooth colors
    color = cv2.bilateralFilter(img, 9, 300, 300)

    # Combine edges and smoothed color
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Convert result to PIL Image and save to in-memory file
    cartoon_image = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    img_io = BytesIO()
    cartoon_image.save(img_io, 'JPEG')
    img_io.seek(0)

    # Remove the uploaded file to keep the folder clean
    os.remove(image_path)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
