import json
from collections import defaultdict
from pathlib import Path


def find_annotations(path: str):

    root_dir = Path(path)
    json_files = list(root_dir.rglob("*.json"))
    return json_files


class Transformer:
    def __init__(self) -> None:
        pass

    def coco_to_yolo(
        self,
        json_paths,
        input_root,
        output_root,
        class_mapping=None,
        use_category_ids_directly=True,
    ):
        input_root = Path(input_root)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for json_path in json_paths:
            json_path = Path(json_path)

            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if "images" not in data or "annotations" not in data:
                raise ValueError(f"{json_path} is not a valid COCO annotations file")

            relative_path = json_path.relative_to(input_root)

            # Example: train/a.json → train/
            out_dir = output_root / relative_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # Map image_id -> metadata
            image_map = {
                img["id"]: {
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                }
                for img in data["images"]
            }

            # Group annotations by image_id
            anns_by_image = defaultdict(list)
            for ann in data["annotations"]:
                anns_by_image[ann["image_id"]].append(ann)

            for image_id, img_info in image_map.items():
                label_name = Path(img_info["file_name"]).with_suffix("").name + ".txt"
                label_path = out_dir / label_name

                lines = []
                for ann in anns_by_image.get(image_id, []):
                    if "bbox" not in ann:
                        continue

                    x_min, y_min, box_w, box_h = ann["bbox"]
                    img_w = img_info["width"]
                    img_h = img_info["height"]

                    if img_w <= 0 or img_h <= 0 or box_w <= 0 or box_h <= 0:
                        continue

                    x_center = (x_min + box_w / 2.0) / img_w
                    y_center = (y_min + box_h / 2.0) / img_h
                    norm_w = box_w / img_w
                    norm_h = box_h / img_h

                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    norm_w = min(max(norm_w, 0.0), 1.0)
                    norm_h = min(max(norm_h, 0.0), 1.0)

                    coco_cat_id = ann["category_id"]

                    if class_mapping is not None:
                        if coco_cat_id not in class_mapping:
                            continue
                        class_id = class_mapping[coco_cat_id]
                    elif use_category_ids_directly:
                        class_id = coco_cat_id
                    else:
                        raise ValueError(
                            "Either provide class_mapping or set use_category_ids_directly=True"
                        )

                    lines.append(
                        f"{class_id} "
                        f"{x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                    )

                with label_path.open("w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

            print(f"Converted {json_path} -> {out_dir}")


if __name__ == "__main__":

    # If your category IDs are already 0,1,2,... you can keep them directly.
    # Example custom mapping if needed:
    # class_mapping = {0: 0, 3: 1}
    class_mapping = {0: 1}

    transformer = Transformer()

    transformer.coco_to_yolo(
        json_paths=find_annotations(
            "/Users/marioandresalbornoz/Desktop/annotationCoco"
        ),
        output_root="/Users/marioandresalbornoz/Desktop/clean_dataset_annotations/annotation",
        input_root="/Users/marioandresalbornoz/Desktop/annotationCoco",
        class_mapping=class_mapping,
        use_category_ids_directly=True,
    )
