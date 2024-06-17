import os
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from pix2pix_model import build_pix2pix

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULT_FOLDER'] = './results'
model = build_pix2pix()  

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img) / 127.5 - 1.0 
    return np.expand_dims(img, axis=0)

def postprocess_image(img_array):
    img_array = (img_array.squeeze() + 1.0) * 127.5
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    input_image = preprocess_image(filepath)

    translated_image = model.predict(input_image)

    translated_image = postprocess_image(translated_image[0])

    result_filename = 'result_' + filename
    result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    translated_image.save(result_filepath)

    return send_file(result_filepath, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
