import cv2
import numpy as np
import os
import zipfile
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

label_idx = ["0~12시간", "12~24시간", "24~36시간", "36~48시간", "48~60시간", "60~72시간", "그 이상"]
image_height = 174
image_width = 320 # 80
num_classes = 7

def extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def read_images_from_directory(directory):
    images = []
    labels = []
    for root, _, files in os.walk(directory):
        idx = 0
        first = datetime.strptime("20220818174850", "%Y%m%d%H%M%S")
        for filename in sorted(files):
            if filename.endswith(".jpg"):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # 그레이스케일로 변환
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    images.append(gray_image)

                    buffer = filename.split("_")[1].split(".")[0]
                    if buffer.isdigit():
                        buffer = "20" + str(buffer)
                        if idx == 0:
                            first = datetime.strptime(buffer, "%Y%m%d%H%M%S")
                            labels.append(0)
                        else:
                            now = datetime.strptime(buffer, "%Y%m%d%H%M%S")
                            diff = now - first
                            diff_hours = diff.total_seconds() / 3600
                            if diff_hours < 12:
                                labels.append(0)
                            elif diff_hours < 24:
                                labels.append(1)
                            elif diff_hours < 36:
                                labels.append(2)
                            elif diff_hours < 48:
                                labels.append(3)
                            elif diff_hours < 60:
                                labels.append(4)
                            elif diff_hours < 72:
                                labels.append(5)
                            else:
                                labels.append(6)
            idx += 1

    return np.array(images), np.array(labels)

# # train.zip 파일 압축 해제
print("Extracting train.zip ...")
extract_zip("drive/MyDrive/Colab Notebooks/train.zip", "train")

# # train.zip 파일 압축 해제
print("Extracting test.zip ...")
extract_zip("drive/MyDrive/Colab Notebooks/test.zip", "test")

# 이미지를 읽어와서 전처리 및 라벨링
print("Loading train data ...")
images, labels = read_images_from_directory("train")

# 이미지 데이터 정규화
images = images / 255.0

# CNN 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (image_height, image_width, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(num_classes, activation = 'softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 모델 학습
print("Training ...")
history = model.fit(images, labels, epochs=10)

# 정확도 그래프 그리기
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# 테스트 데이터 로딩 및 전처리
print("Loading test data ...")
test_images, test_labels = read_images_from_directory("test")
test_images = test_images / 255.0

# 모델 평가
print("Evaluating ...")
_, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)