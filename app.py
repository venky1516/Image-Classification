from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model("cnn_model.h5")
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def preprocess_image(img_path):
    image = Image.open(img_path).resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img = request.files["image"]
        path = "static/" + img.filename
        img.save(path)
        image = preprocess_image(path)
        prediction = model.predict(image)
        result = class_names[np.argmax(prediction)]
        return render_template("index.html", prediction=result, image_path=path)
    return render_template("index.html", prediction=None)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
