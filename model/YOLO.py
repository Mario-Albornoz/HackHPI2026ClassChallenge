from ultralytics.models.yolo import YOLO


class YOLOModel:
    def __init__(self) -> None:
        self.custom_model = YOLO("yolo26n.yaml")  # build a new model from YAML
        self.pretrained_model = YOLO(
            "./pretrained_models/yolo26n.pt"
        )  # load a pretrained model (recommended for training)
