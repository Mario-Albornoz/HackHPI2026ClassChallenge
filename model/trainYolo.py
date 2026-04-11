import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import (
    get_hash,
    load_dataset_cache_file,
    save_dataset_cache_file,
)
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.tqdm import TQDM

BASE_DIR = Path(__file__).resolve().parent
DATASET_YAML = BASE_DIR / "dataset.yaml"
MODEL_WEIGHTS = BASE_DIR / "yolo26n.pt"


class COCODataset(YOLODataset):
    """Dataset that reads COCO JSON annotations directly without conversion to .txt files."""

    def __init__(self, *args, json_file="", **kwargs):
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        categories = {
            cat["id"]: i
            for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))
        }

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = (
                np.array(bboxes, dtype=np.float32)
                if bboxes
                else np.zeros((0, 5), dtype=np.float32)
            )
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (
            FileNotFoundError,
            AssertionError,
            AttributeError,
            KeyError,
            ModuleNotFoundError,
        ):
            cache = self.cache_labels(cache_path)
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]


class COCOTrainer(DetectionTrainer):
    """Trainer that uses COCODataset for direct COCO JSON training."""

    def find_jsons(path):
        return list(Path(path).rglob("*.json"))

    def build_dataset(self, img_path, mode="train", batch=None):

        json_file = (
            self.data["train_json"]
            if mode == "train"
            else self.data.get("val_json", self.data["train_json"])
        )
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=(
                int(self.model.stride.max())
                if hasattr(self, "model") and self.model
                else 32
            ),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


model = YOLO("yolo26n.pt")
model.train(data=DATASET_YAML, epochs=100, imgsz=640, trainer=COCOTrainer)
