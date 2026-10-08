import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ultralytics import YOLO


class YOLOInference:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def predict_image(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
        confidence: float = 0.05,
        iou_threshold: float = 0.60,
        visualize: bool = False,
        save_visualization: bool = False,
        visualization_path: str | Path | None = None,
    ) -> dict:
        image_path = Path(image_path)
        results = self.model.predict(
            source=str(image_path),
            conf=confidence,
            iou=iou_threshold,
            agnostic_nms=False,
            verbose=False,
        )
        result = results[0]
        image_height, image_width = result.orig_shape

        predictions = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().int().tolist()

            for bbox, score, class_id in zip(boxes, confidences, class_ids):
                predictions.append({
                    "class_id": class_id,
                    "label": self.class_names.get(class_id, str(class_id)),
                    "confidence": round(float(score), 4),
                    "bbox": [round(float(coord), 2) for coord in bbox],  # [x1, y1, x2, y2]
                })
        prediction_data = {
            "image_id": image_path.stem,
            "image_path": str(image_path.resolve()),
            "image_width": image_width,
            "image_height": image_height,
            "predictions": predictions,
        }
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(prediction_data, f, indent=4)

        if visualize:
            annotated_image = result.plot()[..., ::-1]
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_image)
            plt.axis("off")
            plt.show()
            plt.close() 

        if save_visualization:
            if visualization_path is None:
                visualization_path = image_path.parent / f"{image_path.stem}_prediction.jpg"
            visualization_path = Path(visualization_path)
            visualization_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(visualization_path))

        return prediction_data

    def predict_dataset(
        self,
        images_dir: str | Path,
        output_dir: str | Path,
        confidence: float = 0.05,
        iou_threshold: float = 0.60,
    ) -> None:
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted([
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ])

        if not image_paths:
            print(f"No valid image files found in {images_dir}")
            return

        for image_path in tqdm(image_paths, desc="Processing images"):
            output_path = output_dir / f"{image_path.stem}.json"
            self.predict_image(
                image_path=image_path,
                output_path=output_path,
                confidence=confidence,
                iou_threshold=iou_threshold,
            )

        print(f"Generated {len(image_paths)} prediction JSON files in: {output_dir}")