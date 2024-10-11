
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io


app = Flask(__name__)


model = tf.keras.models.load_model('model_epoch_09_val_acc_0.96.keras')

angle_mapping = {
    0: 0,
    1: 130,
    2: 180,
    3: 230,
    4: 270,
    5: 320,
    6: 40,
    7: 90
}


def preprocess_image(image, target_size=(224, 224)):

    image = image.resize(target_size)
    image = np.array(image)
    
    
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    try:
      
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_angle = angle_mapping[predicted_class_idx]
        confidence_score = np.max(predictions)

        return jsonify({
            "predicted_angle": int(predicted_angle),
            "confidence_score": float(confidence_score)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)