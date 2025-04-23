# app.py (archivo principal de Flask)
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limitar a 16MB

# Definir las clases disponibles
class_names = ["fogsmog", "hail", "lightning", "rain", "rainbow", "sandstorm", "snow"]

# Definir las transformaciones de imagen igual que en el entrenamiento
transformations = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Definir la arquitectura del modelo CNN
class CNNModel(nn.Module):
    def __init__(self, num_classes=len(class_names)):
        super(CNNModel, self).__init__()
        # Primera capa convolucional
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Tercera capa convolucional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calcula el tamaño después de las capas convolucionales y pooling
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Cargar el modelo
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = CNNModel(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

# Ruta al modelo guardado (ajusta esto según donde guardes tu modelo)
model_path = 'cnn_weather_model.pth'
model, device = load_model(model_path)

# Función para predecir
def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transformations(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top3 = torch.topk(probs, 3)
        
    results = []
    for i in range(3):
        class_idx = top3.indices[0][i].item()
        prob = top3.values[0][i].item()
        results.append({
            'class': class_names[class_idx],
            'probability': f"{prob*100:.2f}%"
        })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Leer la imagen y convertirla a base64 para mostrarla en frontend
        img_bytes = file.read()
        encoded_img = base64.b64encode(img_bytes).decode('utf-8')
        
        # Hacer la predicción
        results = predict_image(img_bytes)
        
        return jsonify({
            'predictions': results,
            'image': encoded_img
        })

if __name__ == '__main__':
    app.run(debug=True)