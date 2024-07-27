import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load YOLOv8 model
model = YOLO("trained_30.pt")  # load a custom model

@app.route("/", methods=["GET"])
def sayHello():
    return jsonify("Hello and welcome to plant disease classification!")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Perform the prediction
        results = model(image)

        # Debugging: Print results
        print("Results type:", type(results))
        print("Results content:", results)

        # Access the probabilities and class names
        probs = results[0].probs
        names = results[0].names

        # Get the top prediction
        top_index = probs.top1  # Get index of the top prediction
        top_confidence = probs.top1conf  # Get confidence score of the top prediction

        # Form the top prediction result
        top_prediction = {
            'label': names[int(top_index)],
            'confidence': float(top_confidence)
        }
        return jsonify({'prediction': top_prediction}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)