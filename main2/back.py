import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

models = {}
def load_models():
    REPO_ID = "fiktusfffffff/models01072026"
    path_fast = hf_hub_download(repo_id=REPO_ID, filename="yolo26s.pt")
    path_accurate = hf_hub_download(repo_id=REPO_ID, filename="accurate.pt")
    models['fast'] = YOLO(path_fast)
    models['accurate'] = YOLO(path_accurate)

def train_like_size(img, max_side=1600, stride=32):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    new_h = int(round(h * scale / stride)) * stride
    new_w = int(round(w * scale / stride)) * stride
    return (new_h, new_w)

def predict(image, model_name, conf, max_bbox):
    if not models:
        load_models()
    model = models.get(model_name)
    if model is None:
        raise Exception('Model not found')
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    max_native = max(h, w)
    maxsz = 1080
    if model_name == 'accurate':#КОСТЫЛЬ, т.к. я дообучал модель на высоком разрешении (imgsz = 1600) для повышения map,
        if max_native <= 1080:#  при апсемпле изображения 1360x765 модель ведёт себя плохо (в остальном - отлично)
            maxsz = max_native#  очень даже хорошо помолго.
        elif max_native <= 1400:
            maxsz = 1280
        else:
            maxsz = 1600
    result = model(image, conf=conf, imgsz = train_like_size(image, maxsz))

    det = []
    for r in result:
        boxes = r.boxes.xyxy.tolist() if r.boxes else []
        classes = r.boxes.cls.tolist() if r.boxes else []
        confs = r.boxes.conf.tolist() if r.boxes else []
        for box, cls, conf in zip(boxes, classes, confs):
            det.append({
                'box': box,
                'class': cls,
                'conf': round(conf,2),
            })
    return det[:max_bbox]