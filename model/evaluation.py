import json
from pathlib import Path

from ultralytics.models.yolo import YOLO


def evaluate_yolo_model(
    model_path: str,
    data_yaml: str,
    split: str = "train",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
    batch: int = 16,
    device: str | int | None = None,
    save_json: bool = True,
    project: str | None = None,
    name: str = "evaluation",
):
    """
    Evaluate a trained YOLO detection model on a dataset split.

    Args:
        model_path: path to trained weights, e.g. ".../best.pt"
        data_yaml: path to dataset.yaml
        split: which split to evaluate on: "val" or "test"
        imgsz: evaluation image size
        conf: confidence threshold used during validation/inference matching
        iou: IoU threshold for NMS during validation
        batch: batch size
        device: device string or index, e.g. "cpu", "0", 0
        save_json: whether to save COCO-style predictions JSON when supported
        project: optional output directory root for Ultralytics runs
        name: run name

    Returns:
        A plain Python dict with the most useful summary metrics.
    """
    model_path = str(Path(model_path).expanduser().resolve())
    data_yaml = str(Path(data_yaml).expanduser().resolve())

    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        batch=batch,
        device=device,
        save_json=save_json,
        project=project,
        name=name,
        plots=True,
        verbose=True,
    )

    summary = {
        "model_path": model_path,
        "data_yaml": data_yaml,
        "split": split,
        "precision_B": float(metrics.box.p),
        "recall_B": float(metrics.box.r),
        "mAP50_B": float(metrics.box.map50),
        "mAP50_95_B": float(metrics.box.map),
    }

    # Include extra metrics if exposed by the installed Ultralytics version
    if hasattr(metrics, "results_dict"):
        try:
            summary["results_dict"] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.results_dict.items()
            }
        except Exception:
            summary["results_dict"] = metrics.results_dict

    print("\nEvaluation summary:")
    for key, value in summary.items():
        if key == "results_dict":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    return summary


evaluate_yolo_model(
    model_path="/Users/marioandresalbornoz/Desktop/hackathon2.0/HackHPI2026ClassChallenge/pretrained_models/yolo26m.pt",
    data_yaml="/Users/marioandresalbornoz/Desktop/hackathon2.0/HackHPI2026ClassChallenge/clean_data.yaml",
    split="val",
)
