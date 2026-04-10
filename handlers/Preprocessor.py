import json
import os

from PIL import Image, UnidentifiedImageError


class Preprocessor:
    """
    Cleans COCO-format annotation files by removing corrupted or invalid entries.

    Checks performed per annotation file:
      - Missing images: file_name not found anywhere under images_dir
      - Corrupted images: file exists but cannot be opened by PIL
      - Invalid bboxes: zero/negative width or height
      - Orphaned annotations: annotation references an image_id not in the images list
    """

    def __init__(self) -> None:
        pass

    def _find_image(self, images_dir: str, file_name: str) -> str | None:
        """Search for file_name recursively under images_dir. Returns path or None."""
        for dirpath, _, filenames in os.walk(images_dir):
            if file_name in filenames:
                return os.path.join(dirpath, file_name)
        return None

    def _is_valid_bbox(self, bbox: list[float]) -> bool:
        """Return False if bbox has negative size """
        if len(bbox) != 4:
            return False
        x, y, w, h = bbox
        if w < 0 or h < 0:
            return False
        if x < 0 or y < 0:
            return False
        return True

    def clean(
        self,
        coco_path: str,
        images_dir: str,
        output_path: str | None = None,
    ) -> dict:
        """
        Remove corrupted or invalid entries from a single COCO annotation file.

        Args:
            coco_path:   Path to the COCO JSON file.
            images_dir:  Root directory to search for images (searched recursively).
            output_path: Where to save the cleaned JSON. If None, overwrites coco_path.

        Returns:
            The cleaned COCO dict.
        """
        with open(coco_path) as f:
            coco = json.load(f)

        original_image_count = len(coco["images"])
        original_ann_count = len(coco["annotations"])

        valid_images = []
        valid_image_ids = set()
        image_sizes: dict[int, tuple[int, int]] = {}

        for img in coco["images"]:
            path = self._find_image(images_dir, img["file_name"])

            if path is None:
                print(f"  REMOVED image {img['id']}: file not found — {img['file_name']}")
                continue

            try:
                with Image.open(path) as im:
                    im.verify()
                # Re-open after verify (verify closes the file)
                with Image.open(path) as im:
                    image_sizes[img["id"]] = (im.width, im.height)
            except (UnidentifiedImageError, Exception) as e:
                print(f"  REMOVED image {img['id']}: corrupted — {img['file_name']} ({e})")
                continue

            valid_images.append(img)
            valid_image_ids.add(img["id"])

        valid_annotations = []

        for ann in coco["annotations"]:
            if ann["image_id"] not in valid_image_ids:
                print(f"  REMOVED annotation {ann['id']}: references invalid image {ann['image_id']}")
                continue

            if not self._is_valid_bbox(ann.get("bbox", [])):
                print(f"  REMOVED annotation {ann['id']}: invalid bbox {ann.get('bbox')} for image {ann['image_id']}")
                continue

            valid_annotations.append(ann)

        coco["images"] = valid_images
        coco["annotations"] = valid_annotations

        removed_images = original_image_count - len(valid_images)
        removed_anns = original_ann_count - len(valid_annotations)
        print(
            f"  Removed {removed_images} image(s), {removed_anns} annotation(s) "
            f"— {len(valid_images)} image(s) and {len(valid_annotations)} annotation(s) remaining."
        )

        output_path = output_path or coco_path
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        return coco

    def clean_all(
        self,
        annotations_root: str,
        images_root: str,
        output_root: str | None = None,
    ) -> None:
        """
        Clean all COCO JSON files found recursively under annotations_root.

        Directory structure expected:
            annotations_root/place_date/something.json
            images_root/place_date/subday/image1.jpg

        Args:
            annotations_root: Root directory containing COCO JSON files.
            images_root:      Root directory containing the corresponding images.
            output_root:      Where to save cleaned JSONs, preserving subfolder
                              structure. If None, files are overwritten in place.
        """
        json_paths = []
        for dirpath, _, filenames in os.walk(annotations_root):
            for filename in filenames:
                if filename.endswith(".json"):
                    json_paths.append(os.path.join(dirpath, filename))

        abs_root = os.path.abspath(annotations_root)
        if not os.path.isdir(abs_root):
            raise NotADirectoryError(f"annotations_root does not exist: {abs_root}")
        print(f"Found {len(json_paths)} annotation file(s) under {abs_root}\n")

        for json_path in json_paths:
            relative = os.path.relpath(os.path.dirname(json_path), annotations_root)
            images_dir = os.path.join(images_root, relative)

            if output_root is not None:
                out_dir = os.path.join(output_root, relative)
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, os.path.basename(json_path))
            else:
                output_path = json_path

            print(f"--- {json_path}")
            self.clean(json_path, images_dir, output_path)
            print()
