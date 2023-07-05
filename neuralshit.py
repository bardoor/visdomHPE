import os

import cv2
from ultralytics import YOLO
import numpy as np
import keras
from keras import layers
from keras.utils import to_categorical


def find_videos(dataset_path):
    # Найдём все поддиректории, содержащие видео с упражнениями
    videos_dirs = []
    for f in os.scandir(dataset_path):
        if f.is_dir():
            videos_dirs.append(f.path)

    # Находим все видео и помечаем, какое упражнение на каждом из них
    videos = []
    for videos_dir in videos_dirs:
        for f in os.scandir(videos_dir):
            if f.is_file():
                # Обрабатываем только видео .mp4 и .mov
                _, ext = os.path.splitext(f.path)
                if ext.lower() not in [".mp4", ".mov"]:
                    continue

                videos.append({
                    "exercise": os.path.basename(videos_dir),
                    "path": os.path.abspath(f.path)
                })

    return videos


def generate_train_data(yolo_model, videos_path):
    # Извлекаем все видео
    videos = find_videos(videos_path)

    # Извлекаем метки для каждого из видео
    labels = [v["exercise"] for v in videos]
    encoded_labels = to_categorical(list(range(len(labels))))
    labels_mapping = dict(zip(labels, encoded_labels))

    for video in videos:
        cap = cv2.VideoCapture(video["path"])

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = yolo_model(frame, verbose=False)

                coordinates = []
                for xy in results[0].keypoints.xyn[0].numpy(force=True):
                    coordinates.extend(xy)
                coordinates = np.array(coordinates).reshape((1, -1))
                encoded_label = np.array(labels_mapping[video["exercise"]]).reshape((1, -1))

                yield (coordinates, encoded_label)
            else:
                break
        
        cap.release()
    
    cv2.destroyAllWindows()


def get_classifier_model():
    inputs = keras.Input(shape=(34, 1))
    x = layers.GRU(128, activation="tanh", return_sequences=True)(inputs)
    x = layers.GRU(64, activation="tanh")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="pose_estimation_classifier")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


model = get_classifier_model()

yolo_model = YOLO('yolov8n-pose.pt')
for (x_train, y_train) in generate_train_data(yolo_model, "dataset/dataset"):
    model.fit(x_train, y_train)
