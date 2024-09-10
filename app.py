from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import zipfile
import os

# Ruta del archivo .zip y carpeta de extracción
zip_path = 'C:/Users/Luis Martinez/Desktop/ApiMicorriza/micorriza_detector.zip'
extract_path = 'C:/Users/Luis Martinez/Desktop/ApiMicorriza/'

# Descomprimir el archivo .zip si no está descomprimido
if not os.path.exists(os.path.join(extract_path, 'micorriza_detector.h5')):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Cargar el modelo descomprimido
model_path = os.path.join(extract_path, 'C:/Users/Luis Martinez/Desktop/ApiMicorriza/micorriza_detector.h5')
model = load_model(model_path)

app = Flask(__name__)


def preprocess_image(img):
    img = img.resize((256, 256))  # Ajusta el tamaño según tu modelo
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255  # Normaliza la imagen
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        result = int(np.round(prediction[0][0]))  # Redondear y convertir a entero
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


