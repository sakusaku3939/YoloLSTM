from ultralytics import YOLO


def crop_dataset(path):
    model = YOLO("yolov8n.pt")
    model(path, project=path, name="yolo", save_crop=True, conf=0.1)
