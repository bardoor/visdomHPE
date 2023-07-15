import enum

import cv2
from ultralytics import YOLO


class VideoKeypointsLoader:

    def __init__(self, yolo_model_name="yolov8n-pose.pt"):
        self.yolo_model = YOLO(yolo_model_name)
        self.yolo_model.to("cuda")

    def load(self, path):
        video_keypoints = []

        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = self.yolo_model(frame, classes=0, verbose=False)

                if keypoints.size > 0:
                    keypoints = keypoints.reshape(34)
                    video_keypoints.append(keypoints)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        return video_keypoints
