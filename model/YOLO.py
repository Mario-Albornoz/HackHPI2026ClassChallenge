from ultralytics.models.yolo import YOLO


class YOLOModel:
    def __init__(self) -> None:
        self.custom_model = YOLO("yolo26n.yaml")  # build a new model from YAML
        self.pretrained_model = YOLO(
            "./pretrained_models/yolo26n.pt"
        )  # load a pretrained model (recommended for training)


def test():
    # Load a model
    model = YOLO("yolo26n.pt")  # load an official model

    # Predict with the model
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = (
            result.boxes.xyxy
        )  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [
            result.names[cls.item()] for cls in result.boxes.cls.int()
        ]  # class name of each box
        confs = result.boxes.conf
