from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

from contextlib import contextmanager
import pathlib

@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup

import torch

EXPORT_PATH = pathlib.Path("C:/Users/shubh/Desktop/Ham10k/model/best.pt")

with set_posix_windows():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=EXPORT_PATH, force_reload=True)

model.eval()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('process_image', filename=filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def process_image(filename):
    # Load and process the image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    # Predict using the YOLOv5 model
    with torch.no_grad():
        prediction = model(img_tensor)

    # Process the prediction to get the classification
    # Assuming the model returns a list of detections
    # You'll need to adjust this part based on the actual output of your model
    # For simplicity, let's assume the model returns a list of class names
    class_names = prediction.xyxyn[0][:, -1].tolist()

    # Render the result
    return render_template('results.html', class_names=class_names)

if __name__ == '__main__':
    app.run(debug=True)
