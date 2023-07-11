import csv
import os
from random import shuffle

import pandas as pd
from tensorflow import convert_to_tensor
from tensorflow.keras.layers import StringLookup

from keypoints import VideoKeypointsLoader, Keypoints


def _find_videos(path):
    videos = []

    for directory in os.scandir(path):
        if not directory.is_dir():
            continue

        for video_file in os.scandir(directory.path):
            if not video_file.is_file():
                continue

            _, ext = os.path.splitext(video_file.path)

            if ext.lower() not in [".mp4", ".mov"]:
                continue

            videos.append({
                "activity": os.path.basename(directory),
                "path": os.path.abspath(video_file.path)
            })

    return videos


def to_csv(path_to_videos, output_file_name, verbose=True):
    """Создаёт csv-файл, содержащий датасет из обработанных видео

    Аргументы:
        path_to_videos: Путь к директории, содержащей папки с видео

        output_file_name: Имя сгенерированного csv-файла

        verbose: Флаг, указывающий, выводить ли подробности об обработке видео (по умолчанию - True)
    """

    if not output_file_name.endswith(".csv"):
        output_file_name += ".csv"

    videos = _find_videos(path_to_videos)
    shuffle(videos)

    if verbose:
        print(f"Found {len(videos)} videos")

    keypoints_loader = VideoKeypointsLoader()

    mode = "w"

    if os.path.exists(output_file_name):
        ans = input(
            f"[WARNING]: There's already a file called \"{output_file_name}\"\n" +
            "The data found will be added to the file. Continue? (у\n): "
        )
        if ans.lower().startswith("y"):
            mode = "a"
        else:
            print("Creating csv file aborted")
            return

    column_names = ["activity"] + [keypoint.name for keypoint in Keypoints]

    with open(output_file_name, mode) as dataset_file:
        dataset_file_writer = csv.DictWriter(dataset_file, column_names)

        if mode == "w":
            dataset_file_writer.writeheader()

        for video in videos:
            activity, video_path = video["activity"], video["path"]

            if verbose:
                print(f"Processing \"{video_path}\"...")

            video_keypoints = keypoints_loader.load(video_path)
            for frame_keypoints in video_keypoints:
                row = dict(
                    zip(column_names, [activity] + list(frame_keypoints))
                )
                dataset_file_writer.writerow(row)


def create_dataset(path_to_csv, time_steps=20, step=5):
    """Считывает датасет из .csv файла

    Аргументы:
        path_do_dataset: Путь к .csv файлу

        time_steps: Сколько кадров будет содержаться в одном
        примере (по умолчанию - 20)

        step: Насколько кадров следующий пример опережает
        текущий (по умолчанию - 5)

    Возвращает:
        Два тензора: примеры и соотвествующие метки упражнений
    """

    Xs, ys = [], []

    df = pd.read_csv(path_to_csv)

    X = df.drop(columns=["activity"]).to_numpy()
    y = df.activity.to_numpy()

    for i in range(0, len(X) - time_steps, step):
        series = X[i:(i + time_steps)]
        labels = y[i:(i + time_steps)]

        if len(pd.unique(labels)) != 1:
            continue

        Xs.append(series)
        ys.append(labels[0])

    labels = ys

    one_hot_encoder = StringLookup(output_mode="one_hot")
    one_hot_encoder.adapt(ys)

    Xs = convert_to_tensor(Xs)
    ys = one_hot_encoder(ys)

    return Xs, ys, labels, one_hot_encoder


def generate(path_to_video, times_steps=20, step=5):
    """Генерирует последовательности кадров (кейпоинтов) из видео
    с заданным размером и шагом

    Аргументы:
        path_to_video: Путь к видео

        time_steps: Сколько массивов кейпоинтов будет содержаться
        в одной последовательности (по умолчанию - 20)

        step: Насколько кадров (кейпоинтов) следующая последовательность
        опережает текущую (по умолчанию - 5)
    """

    keypoints_loader = VideoKeypointsLoader()

    keypoints = keypoints_loader.load(path_to_video)
    for i in range(0, len(keypoints) - times_steps, step):
        seq = convert_to_tensor(keypoints[i:(i + times_steps)])
        yield seq
