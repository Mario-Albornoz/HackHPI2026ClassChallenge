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


def find_json_files(path: str | Path) -> list[Path]:
    root_dir = Path(path)
    json_files = list(root_dir.rglob("*.json"))
    return json_files


def build_image_index(img_root: Path):
    index = {}
    for p in img_root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            index.setdefault(p.name, []).append(p)
    return index


def debug_print_image_match(
    img_root: Path, file_name: str, matches: list[Path], limit_state: dict
):
    if limit_state["count"] < limit_state["limit"]:
        print("\nDEBUG IMAGE LOOKUP")
        print("split root:", img_root)
        print("json file_name:", file_name)
        print("matches found:", len(matches))
        for m in matches[:5]:
            print("  ", m)
        limit_state["count"] += 1


class MultiCOCODataset(YOLODataset):
    """
    YOLO dataset that reads multiple COCO JSON files directly.

    Expected YAML fields:
        train: path to training image root
        val: path to validation image root
        train_json_root: directory containing training COCO JSON files
        val_json_root: directory containing validation COCO JSON files

    Assumption:
        For each image entry in each COCO file, img_info["file_name"] is relative to
        the split image root (train or val).
    """

    def __init__(self, *args, json_files=None, **kwargs):
        self.json_files = [Path(p).expanduser().resolve() for p in (json_files or [])]
        self._image_info_by_file = {}
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        # We do not scan images from disk here.
        # Images are discovered from the COCO JSON files in cache_labels().
        return []

    def _build_categories_mapping(self):
        """
        Build a consistent category mapping across all JSON files.

        Maps category name -> contiguous YOLO class index.
        This is safer than trusting raw category_id values to be globally consistent.
        """
        category_name_to_idx = {}
        next_idx = 0

        for json_file in self.json_files:
            with json_file.open("r", encoding="utf-8") as f:
                coco = json.load(f)

            for cat in coco.get("categories", []):
                name = str(cat.get("name", "")).strip()
                if not name:
                    continue
                if name not in category_name_to_idx:
                    category_name_to_idx[name] = next_idx
                    next_idx += 1

        if not category_name_to_idx:
            raise ValueError("No categories found across provided COCO JSON files")

        return category_name_to_idx

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Build Ultralytics label cache from all COCO files in this split.
        """
        x = {"labels": []}
        img_root = Path(self.img_path)
        image_index = build_image_index(img_root)
        debug_state = {"count": 0, "limit": 10}
        category_name_to_idx = self._build_categories_mapping()

        n_json = 0
        n_images_total = 0
        n_images_used = 0
        n_boxes_total = 0
        n_boxes_used = 0

        for json_file in TQDM(self.json_files, desc=f"{self.prefix}reading COCO jsons"):
            n_json += 1

            with json_file.open("r", encoding="utf-8") as f:
                coco = json.load(f)

            # Map local category_id -> YOLO class index via category name
            local_cat_id_to_yolo_idx = {}
            for cat in coco.get("categories", []):
                cat_name = str(cat.get("name", "")).strip()
                if cat_name in category_name_to_idx:
                    local_cat_id_to_yolo_idx[cat["id"]] = category_name_to_idx[cat_name]

            img_to_anns = defaultdict(list)
            for ann in coco.get("annotations", []):
                img_to_anns[ann["image_id"]].append(ann)

            for img_info in coco.get("images", []):
                n_images_total += 1

                if "id" not in img_info or "file_name" not in img_info:
                    continue
                if "width" not in img_info or "height" not in img_info:
                    continue

                h, w = img_info["height"], img_info["width"]
                if h <= 0 or w <= 0:
                    continue

                matches = image_index.get(Path(img_info["file_name"]).name, [])
                debug_print_image_match(
                    img_root, img_info["file_name"], matches, debug_state
                )

                if len(matches) == 1:
                    im_file = matches[0]
                elif len(matches) == 0:
                    continue
                else:
                    print(
                        f"Ambiguous filename '{img_info['file_name']}' -> {len(matches)} matches, skipping"
                    )
                    continue
                n_images_used += 1
                self.im_files.append(str(im_file))

                bboxes = []
                for ann in img_to_anns.get(img_info["id"], []):
                    n_boxes_total += 1

                    if ann.get("iscrowd", False):
                        continue
                    if "bbox" not in ann or "category_id" not in ann:
                        continue
                    if ann["category_id"] not in local_cat_id_to_yolo_idx:
                        continue

                    # COCO bbox format: [x_min, y_min, width, height]
                    box = np.array(ann["bbox"], dtype=np.float32)
                    if box.shape[0] != 4:
                        continue

                    x_min, y_min, bw, bh = box.tolist()
                    if bw <= 0 or bh <= 0:
                        continue

                    # Convert to normalized xywh
                    x_center = (x_min + bw / 2.0) / w
                    y_center = (y_min + bh / 2.0) / h
                    norm_w = bw / w
                    norm_h = bh / h

                    # Clamp slightly out-of-bounds boxes
                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    norm_w = min(max(norm_w, 0.0), 1.0)
                    norm_h = min(max(norm_h, 0.0), 1.0)

                    if norm_w <= 0 or norm_h <= 0:
                        continue

                    cls = local_cat_id_to_yolo_idx[ann["category_id"]]
                    bboxes.append([cls, x_center, y_center, norm_w, norm_h])
                    n_boxes_used += 1

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

        x["hash"] = get_hash([str(p) for p in self.json_files] + [str(self.img_path)])

        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)

        print(f"{self.prefix}Loaded {n_json} JSON files")
        print(f"{self.prefix}Images referenced: {n_images_total}")
        print(f"{self.prefix}Images found on disk: {n_images_used}")
        print(f"{self.prefix}Annotations seen: {n_boxes_total}")
        print(f"{self.prefix}Annotations used: {n_boxes_used}")
        print(f"{self.prefix}Classes: {category_name_to_idx}")

        return x

    def get_labels(self):
        """
        Load labels from cache if valid, otherwise rebuild cache.
        """
        cache_name = f"{Path(self.img_path).name}_multi_coco.cache"
        cache_path = Path(self.img_path) / cache_name

        expected_hash = get_hash(
            [str(p) for p in self.json_files] + [str(self.img_path)]
        )

        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == expected_hash
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


class MultiCOCOTrainer(DetectionTrainer):
    """
    Custom trainer that discovers many COCO JSON files per split by walking a directory tree.
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "train":
            json_root = self.data.get("train_json_root")
            if not json_root:
                raise ValueError("dataset.yaml is missing 'train_json_root'")
        else:
            json_root = self.data.get("val_json_root")
            if not json_root:
                raise ValueError("dataset.yaml is missing 'val_json_root'")

        json_files = find_json_files(json_root)
        if not json_files:
            raise ValueError(f"No COCO JSON files found under: {json_root}")

        return MultiCOCODataset(
            img_path=img_path,
            json_files=json_files,
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


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_YAML = BASE_DIR / "dataset.yaml"

    model = YOLO("yolo26n.pt")
    model.train(
        data=str(DATASET_YAML),
        epochs=10,
        imgsz=640,
        trainer=MultiCOCOTrainer,
        device="cpu",  # change if you want
    )
