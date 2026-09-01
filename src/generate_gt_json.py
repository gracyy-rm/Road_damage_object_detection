import json
from pathlib import Path
from PIL import Image


def yolo_to_xyxy(bbox, image_width, image_height):
    x_center, y_center, width, height = bbox

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [x1, y1, x2, y2]


def generate_gt_json(images_dir, labels_dir, output_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.png"))
    )

    for image_path in image_paths:

        label_path = labels_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        annotations = []

        if label_path.exists():

            with open(label_path, "r") as file:

                for line in file:

                    values = line.strip().split()

                    if not values:
                        continue

                    class_id = int(values[0])

                    bbox = list(
                        map(float, values[1:5])
                    )

                    bbox = yolo_to_xyxy(
                        bbox,
                        image_width,
                        image_height,
                    )

                    annotations.append(
                        {
                            "class_id": class_id,
                            "bbox": bbox,
                        }
                    )

        gt_data = {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "image_width": image_width,
            "image_height": image_height,
            "annotations": annotations,
        }

        output_path = (
            output_dir
            / f"{image_path.stem}.json"
        )

        with open(output_path, "w") as file:
            json.dump(
                gt_data,
                file,
                indent=4,
            )

    print(
        f"Generated {len(image_paths)} GT JSON files "
        f"in: {output_dir}"
    )