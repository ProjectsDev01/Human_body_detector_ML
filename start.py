from ultralytics import YOLO
import cv2

# model = YOLO("./runs/detect/detector_model/weights/last.pt")
model = YOLO("yolov8n-seg.pt")

results = model.predict(source="0", show=True)

print(results)