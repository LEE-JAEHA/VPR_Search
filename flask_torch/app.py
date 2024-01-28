from torch_model.model import SimpleCNN

# app.py

from flask import Flask, render_template, request, g
from PIL import Image
import sqlite3
import torchvision.transforms as transforms
import random
import pdb

app = Flask(__name__)

DATABASE = 'sample_data.db'

# 데이터베이스 연결 함수
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

# 데이터베이스 teardown 함수
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# 딥러닝 모델 로드 및 전처리 함수
def load_and_process_image(file_path):
    image = Image.open(file_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# 예측 함수
def predict_image(input_tensor):
    # 여기에 모델을 로드하고 예측하는 코드를 추가하세요
    # model = load_model()  # 모델을 로드하는 코드
    # result = model(input_tensor)  # 모델을 통한 예측
    # return result

    # 임시로 랜덤한 예측 결과를 반환하도록 설정
    return random.randint(0, 10)

@app.route('/predict',methods=['GET'])
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
        # 파일명을 기반으로 데이터베이스에서 레이블 및 id 가져오기
        db = get_db()
        cursor = db.cursor()

        # # 파일명을 소문자로 변환하여 비교
        # filename_lower = file.filename.lower()
        # cursor.execute("SELECT id, label FROM images WHERE LOWER(filename)=?", (filename_lower,))
        # result = cursor.fetchone()

        # if result is not None:
        # image_id, label = result

        # 딥러닝 모델을 통한 예측
        # pdb.set_trace()
        # input_tensor = load_and_process_image(file)
        print(file)
        prediction_result = predict_image("et")

        # 숫자 예측 결과와 비교하여 작거나 같은 id 값들을 가져오기
        
        cursor.execute("SELECT * FROM images WHERE id <= ?", (prediction_result,))
        grid_data = cursor.fetchall()
        # pdb.set_trace()
        if grid_data is not None:
            return render_template('index.html', result=prediction_result, grid_data=grid_data)
        else:
            return render_template('index.html', result="이미지에 대한 레이블을 찾을 수 없습니다.")
    except Exception as e:
        return render_template('index.html', result=f"예측 중 오류 발생: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
