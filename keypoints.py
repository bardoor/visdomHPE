import enum

import cv2
from ultralytics import YOLO


class Keypoints(enum.Enum):
    nose_x = 0
    nose_y = 1
    left_eye_x = 2
    left_eye_y = 3
    right_eye_x = 4
    right_eye_y = 5
    left_ear_x = 6
    left_ear_y = 7
    right_ear_x = 8
    right_ear_y = 9
    left_shoulder_x = 10
    left_shoulder_y = 11
    right_shoulder_x = 12
    right_shoulder_y = 13
    left_elbow_x = 14
    left_elbow_y = 15
    right_elbow_x = 16
    right_elbow_y = 17
    left_wrist_x = 18
    left_wrist_y = 19
    right_wrist_x = 20
    right_wrist_y = 21
    left_hip_x = 22
    left_hip_y = 23
    right_hip_x = 24
    right_hip_y = 25
    left_knee_x = 26
    left_knee_y = 27
    right_knee_x = 28
    right_knee_y = 29
    left_ankle_x = 30
    left_ankle_y = 31
    right_ankle_x = 32
    right_ankle_y = 33


class VideoKeypointsLoader:

    def __init__(self, yolo_model_name="yolov8n-pose.pt"):
        self.yolo_model = YOLO(yolo_model_name)

    def load(self, path):
        video_keypoints = []

        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = self.yolo_model(frame, classes=0, verbose=False)

                if results[0].keypoints:
                    keypoints = \
                        results[0].keypoints.xyn[0].numpy(
                            force=True)

                    if keypoints.size > 0:
                        keypoints = keypoints.reshape(34)
                        video_keypoints.append(keypoints)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        return video_keypoints
