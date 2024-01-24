from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 이미지 폴더 경로 설정
IMAGE_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# 이미지 파일 확장자 필터링
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cosine_similarity_score(img1, img2):
    # 이미지를 그레이 스케일로 변환
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 이미지를 벡터로 펼치기
    img1_vector = img1_gray.flatten()
    img2_vector = img2_gray.flatten()

    # 코사인 유사성 계산
    similarity_score = cosine_similarity([img1_vector], [img2_vector])

    return similarity_score[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 업로드된 파일 처리
        if 'file' not in request.files:
            return redirect(request.url)
        # import pdb;pdb.set_trace()
        file = request.files['file']

        if file and allowed_file(file.filename):
            # 이미지 파일 읽기
            uploaded_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

            # 이미지 리스트에서 유사성을 기반으로 필터링
            image_list = []
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if allowed_file(filename):
                    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    existing_image = cv2.imread(path)

                    # 유사성 비교
                    similarity_score = cosine_similarity_score(uploaded_image, existing_image)

                    # 예시: 유사성이 0.8 이상인 이미지만 포함
                    if similarity_score >= 0.8:
                        image_list.append({
                            'name': filename,
                            'path': path
                        })

            return render_template('index.html', image_list=image_list)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
