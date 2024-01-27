# app.py
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import torch
from model import SimpleCNN
app = Flask(__name__)

# 모델을 미리 로드합니다. 이 부분은 실제 모델을 사용하는 것으로 대체해야 합니다.
model = SimpleCNN()
# model.load_state_dict(torch.load('simple_cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # 배치 차원 추가
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="이미지를 업로드하세요.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="파일을 선택하세요.")

    try:
        img = preprocess_image(file)
        with torch.no_grad():
            prediction = model(img)
            result = torch.argmax(prediction).item()
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"예측 중 오류 발생: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
