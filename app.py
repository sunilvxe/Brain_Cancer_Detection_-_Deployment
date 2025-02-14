from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image  # Import Pillow for image processing
import numpy as np
import io
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_cancer_model.h5")
try:
    model = load_model(MODEL_PATH)
    CLASS_LABELS = list(range(model.output_shape[-1]))  # Dynamically determine number of classes
except Exception as e:
    raise RuntimeError(f"Failed to load the model from {MODEL_PATH}. Error: {str(e)}")

# If the model's class indices are predefined, update CLASS_LABELS
CLASS_LABELS = ['Healthy', 'Benign', 'Malignant', 'Unknown']  # Replace or update if needed

# Define the route for the home page
@app.route("/")
def index():
    return render_template("index.html")

# Define the route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Check if file is uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please provide an image file."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected. Please choose an image file."}), 400

    # Validate file type to support PNG, JPG, JPEG, and WEBP
    if not (file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        return jsonify({"error": "Invalid file type. Please upload a .png, .jpg, .jpeg, or .webp image."}), 400

    try:
        # Convert the uploaded file to BytesIO and open with PIL
        img = Image.open(io.BytesIO(file.read()))

        # Convert to RGB to ensure no alpha channel (especially for WEBP)
        img = img.convert('RGB')

        # Resize to the model's input size
        img = img.resize((224, 224))

        # Convert image to array and normalize
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
