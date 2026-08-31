import json
from pathlib import Path

from PIL import Image


CLASS_NAMES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole",
]


def convert_yolo_to_coco(
    images_dir,
    labels_dir,
    output_json,
    class_names=CLASS_NAMES,
):
    """
    Convert YOLO object detection annotations to COCO format.

    YOLO format:
        class_id x_center y_center width height

    COCO format:
        category_id x_min y_min width height
    """

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_json = Path(output_json)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Create category information
    for class_id, class_name in enumerate(class_names):
        coco["categories"].append(
            {
                "id": class_id + 1,
                "name": class_name,
                "supercategory": "road_damage",
            }
        )

    image_files = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.png"))
    )

    annotation_id = 1

    # Process each image
    for image_id, image_path in enumerate(image_files, start=1):

        # Get image dimensions
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        # Add image information
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image_width,
                "height": image_height,
            }
        )

        # Find corresponding YOLO label
        label_path = labels_dir / f"{image_path.stem}.txt"

        # Images without annotations are valid
        if not label_path.exists():
            continue

        with open(label_path, "r") as file:
            lines = file.readlines()

        # Process annotations
        for line in lines:

            values = line.strip().split()

            if not values:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            bbox_width = float(values[3])
            bbox_height = float(values[4])

            # Convert normalized YOLO coordinates to pixels
            x_center *= image_width
            y_center *= image_height

            bbox_width *= image_width
            bbox_height *= image_height

            # Convert center coordinates to top-left coordinates
            x_min = x_center - bbox_width / 2
            y_min = y_center - bbox_height / 2

            # Keep bounding boxes inside image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)

            bbox_width = min(
                bbox_width,
                image_width - x_min,
            )

            bbox_height = min(
                bbox_height,
                image_height - y_min,
            )

            # Skip invalid bounding boxes
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            # Add COCO annotation
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [
                        round(x_min, 2),
                        round(y_min, 2),
                        round(bbox_width, 2),
                        round(bbox_height, 2),
                    ],
                    "area": round(
                        bbox_width * bbox_height,
                        2,
                    ),
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

    # Create output directory
    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save COCO JSON
    with open(output_json, "w") as file:
        json.dump(
            coco,
            file,
            indent=4,
        )

    print(f"\nSaved: {output_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")

    return coco


def convert_dataset(
    dataset_root,
    output_root,
):
    """
    Convert train, validation and test splits.
    """

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    splits = ["train", "val", "test"]

    for split in splits:

        print(f"\n{'=' * 50}")
        print(f"Converting {split} split")
        print(f"{'=' * 50}")

        convert_yolo_to_coco(
            images_dir=dataset_root / split / "images",
            labels_dir=dataset_root / split / "labels",
            output_json=output_root / "annotations" / f"{split}.json",
        )


