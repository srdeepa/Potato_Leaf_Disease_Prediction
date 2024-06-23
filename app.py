from flask import Flask, request, redirect, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Update this path to the absolute path of your model file
model_path = r'C:\Users\deepu\Downloads\Potato-Disease-Classification-master\models\potatoes.h5'  
# Load the pre-trained model
try:
    model = tf.keras.models.load_model(model_path) 
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Define disease names corresponding to class indices
disease_names = ['POTATO EARLY BLIGHT LEAF', 'POTATO LATE BLIGHT LEAF', 'POTATO HEALTHY LEAF']

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image_path, target_size=(256, 256)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize the image array
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        flash(f"Failed to preprocess image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            input_image = preprocess_image(file_path)
            if input_image is None:
                return redirect(request.url)

            predictions = model.predict(input_image)
            class_index = np.argmax(predictions[0])
            predicted_disease = disease_names[class_index]
            confidence = round(100 * np.max(predictions[0]), 2)
            
            if predicted_disease == 'POTATO HEALTHY LEAF':
                result_message = "The leaf is not infected."
            elif predicted_disease == 'POTATO EARLY BLIGHT LEAF':
                result_message = "The leaf is infected with early blight"
            elif predicted_disease == 'POTATO LATE BLIGHT LEAF':
                result_message = "The leaf is infected with late blight."

            return f"""
            <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-image: url('https://t4.ftcdn.net/jpg/07/06/68/01/360_F_706680135_NUlFJTWllvo8VEEQW1HXZUSioNaOtGFS.jpg');
                        background-size: cover;
                        text-align: center;
                        padding: 50px;
                        color: #fff;
                    }}
                    h1 {{
                        color: Black;
                    }}
                    .result {{
                        background-color: rgba(0, 0, 0, 0.7);
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        display: inline-block;
                        margin-top: 20px;
                    }}
                </style>
            </head>
            <body>
                <h1>Prediction Result</h1>
                <div class="result">
                    <p><strong>Predicted Disease:</strong> {predicted_disease}</p>
                    <p><strong>{result_message}</strong></p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <a href="/" style="color: #ffeb3b;">Back to Home</a>
                </div>
            </body>
            </html>
            """
    
    # If GET request or no file uploaded, return the form HTML
    return f"""
    <html>
    <head>
        <title>Potato Leaf Disease Classifier</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-image: url('https://media.istockphoto.com/id/1148295776/photo/potato-field-where-flowers-bloomed.jpg?s=612x612&w=0&k=20&c=WD9_Xxgukvu77sWQAs4az2Xguf0nFluJSYmryt5YUaQ=');
                background-size: cover;
                text-align: center;
                padding: 50px;
                color: #fff;
            }}
            h1 {{
                color: Black;
            }}
            .upload-form {{
                background-color: rgba(0, 0, 0, 0.7);
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                display: inline-block;
                margin-top: 50px;
                width: 50%;
                max-width: 300px;
            }}
            input[type="file"] {{
                margin: 10px 0;
                color: #fff;
            }}
            button {{
                background-color: #6b8e23;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }}
            button:hover {{
                background-color: #556b2f;
            }}
        </style>
    </head>
    <body>
        <h1>Potato Leaf Disease Classifier</h1>
        <div class="upload-form">
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <br>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
