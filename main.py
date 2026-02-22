from flask import Flask, request, jsonify
import os
from src.customcnn import preprocess_image, model
from src.convnext import convnext_predict, convnext_model, device

app = Flask(__name__)

# --------------------- FLASK ROUTES ---------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Forgery Detection API is running"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img_bytes = file.read()

    # preprocess
    img = preprocess_image(img_bytes)

    # run CNN
    prediction = model.predict(img)

    # run ConvNext
    convnext_label, confidence = convnext_predict(img_bytes)

    # Assuming model outputs probability for "fake"
    prob = float(prediction[0][0])
    label = "Forged" if prob > 0.5 else "Real"

    return jsonify({
        "forgery_analysis": {
            "prediction": label,
            "confidence": round(prob if prob > 0.5 else 1 - prob, 4)
        },
        "ai_image_analysis": {
            "prediction": convnext_label,
            "confidence": confidence
        }
    })


# --------------------- MAIN FUNCTION ---------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)