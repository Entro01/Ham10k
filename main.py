from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='../model/best.pt', force_reload=True)
#model.eval()

@app.route('/', methods=['GET', 'POST'])
def landing():
    return render_template('mockup.html')

@app.route('/page', methods=['GET', 'POST'])
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
    # This part depends on how you want to display the classification
    # For simplicity, let's assume the model returns a list of class names
    class_names = prediction.xyxyn[0][:, -1].tolist()

    # Render the result
    return render_template('result.html', class_names=class_names)

if __name__ == '__main__':
    app.run(debug=True)
