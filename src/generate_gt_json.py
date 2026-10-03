import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image, UnidentifiedImageError


def yolo_to_xyxy(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Converts YOLO normalized [x_center, y_center, width, height]
    to pixel coordinates [x1, y1, x2, y2] clamped to image dimensions.
    """
    x_center, y_center, width, height = bbox

    # Denormalize coordinates to pixel scale
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    # Calculate corner points and clamp within valid pixel range [0, dimension]
    x1 = max(0.0, round(x_center - width / 2, 2))
    y1 = max(0.0, round(y_center - height / 2, 2))
    x2 = min(float(image_width), round(x_center + width / 2, 2))
    y2 = min(float(image_height), round(y_center + height / 2, 2))

    return [x1, y1, x2, y2]


def generate_gt_json(
    images_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    output_dir: Union[str, Path],
    class_names: Optional[Union[List[str], Dict[int, str]]] = None,
) -> None:
    """
    Parses YOLO annotations and image metadata into individual ground-truth JSON files.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize class_names to an int -> str lookup map
    label_map: Dict[int, str] = {}
    if isinstance(class_names, list):
        label_map = {idx: name for idx, name in enumerate(class_names)}
    elif isinstance(class_names, dict):
        label_map = class_names

    # Collect images case-insensitively (.jpg, .JPG, .png, .PNG, etc.)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    )

    processed_count = 0

    for image_path in image_paths:
        # Read image dimensions safely without loading full image into RAM
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except (UnidentifiedImageError, OSError) as e:
            print(f"[Warning] Skipping unreadable image {image_path.name}: {e}")
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        annotations = []

        if label_path.is_file():
            with open(label_path, "r", encoding="utf-8") as file:
                for line_idx, line in enumerate(file, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue

                    # Validate line structure
                    if len(parts) < 5:
                        print(
                            f"[Warning] Incomplete annotation in {label_path.name} "
                            f"line {line_idx}. Expected 5 values, got {len(parts)}."
                        )
                        continue

                    # Parse class ID and bbox values
                    try:
                        class_id = int(parts[0])
                        raw_bbox = [float(val) for val in parts[1:5]]
                    except ValueError:
                        print(
                            f"[Warning] Non-numeric entry in {label_path.name} line {line_idx}."
                        )
                        continue

                    # Validate normalized coordinates range
                    if not all(0.0 <= val <= 1.0 for val in raw_bbox):
                        print(
                            f"[Warning] Bounding box values outside [0, 1] in "
                            f"{label_path.name} line {line_idx}."
                        )

                    xyxy_box = yolo_to_xyxy(raw_bbox, image_width, image_height)
                    label_name = label_map.get(class_id, f"class_{class_id}")

                    annotations.append(
                        {
                            "class_id": class_id,
                            "label": label_name,
                            "bbox": xyxy_box,
                        }
                    )

        # Assemble ground truth metadata dictionary
        gt_data = {
            "image_id": image_path.stem,
            "image_path": str(image_path.resolve()),
            "image_width": image_width,
            "image_height": image_height,
            "annotations": annotations,
        }

        # Write output JSON
        output_path = output_dir / f"{image_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(gt_data, file, indent=4)

        processed_count += 1

    print(f"Generated {processed_count} GT JSON files in: {output_dir}")


if __name__ == "__main__":
    # Example usage
    CLASSES = ["car", "pedestrian", "traffic_light", "truck"]

    generate_gt_json(
        images_dir="data/images",
        labels_dir="data/labels",
        output_dir="data/gt_jsons",
        class_names=CLASSES,
    )