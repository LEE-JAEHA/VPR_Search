from flask import Flask, render_template, request
import os
from skimage.metrics import structural_similarity as ssim
from skimage import io, color
from PIL import Image
import numpy as np
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

class ImageInfo:
    def __init__(self, name, path, similarity=None):
        self.name = name
        self.path = path
        self.similarity = similarity

# 수정 후 코드
def compare_images(img1_path, img2_path):
    # 이미지를 Pillow를 사용해 열고 RGB로 변환
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # 이미지 크기를 맞춤
    img2 = img2.resize(img1.size)

    # 이미지를 NumPy 배열로 변환
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    # 이미지를 그레이스케일로 변환
    img1_gray = color.rgb2gray(img1_np)
    img2_gray = color.rgb2gray(img2_np)

    # Structural Similarity Index (SSI) 계산
    similarity_index, _ = ssim(img1_gray, img2_gray, full=True)

    return similarity_index

def get_similar_images(input_image_path):
    input_image_name = os.path.basename(input_image_path)
    similar_images = []

    for image_name in os.listdir(app.config['UPLOAD_FOLDER']):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

            similarity = compare_images(input_image_path, image_path)
            similar_images.append(ImageInfo(image_name, image_path, similarity))

    # Sort similar images by similarity (higher similarity first)
    similar_images.sort(key=lambda x: x.similarity, reverse=True)

    return similar_images


# def cosine_similarity_score(img1, img2):
#     # 이미지를 그레이 스케일로 변환
#     img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # 이미지를 벡터로 펼치기
#     img1_vector = img1_gray.flatten()
#     img2_vector = img2_gray.flatten()

#     # 코사인 유사성 계산
#     similarity_score = cosine_similarity([img1_vector], [img2_vector])

#     return similarity_score[0][0]


@app.route('/', methods=['GET', 'POST'])
def index():
    similar_images = None
    input_image_path = None

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            # Save uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Get similar images
            input_image_path = os.path.join('static/images', file.filename)
            similar_images = get_similar_images(input_image_path)
    # import pdb;pdb.set_trace()
    return render_template('index.html', similar_images=similar_images, input_image_path=input_image_path)

if __name__ == '__main__':
    app.run(debug=True,port=8080)
