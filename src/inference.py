import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

class YOLOInference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict_image(self, image_path, output_path=None, confidence=0.001, visualize=False, save_visualization=False):
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        results = self.model.predict(source=str(image_path), conf=confidence, verbose=False)
        result = results[0]
        predictions = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()

            for bbox, confidence_score, class_id in zip(boxes, confidences, class_ids):
                predictions.append({
                    "class_id": int(class_id),
                    "confidence": float(confidence_score),
                    "bbox": [float(value) for value in bbox],
                })

        prediction_data = {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "image_width": image_width,
            "image_height": image_height,
            "predictions": predictions,
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as file:
                json.dump(prediction_data, file, indent=4)

        if visualize:
            annotated_image = result.plot()
            if save_visualization:
                visualization_path = image_path.parent / f"{image_path.stem}_prediction.jpg"
                Image.fromarray(annotated_image[..., ::-1]).save(visualization_path)
            return prediction_data, annotated_image

        return prediction_data

    def predict_dataset(self, images_dir, output_dir, confidence=0.001):
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            list(images_dir.glob("*.jpg")) + 
            list(images_dir.glob("*.jpeg")) + 
            list(images_dir.glob("*.png"))
        )

        for image_path in image_paths:
            output_path = output_dir / f"{image_path.stem}.json"
            self.predict_image(image_path=image_path, output_path=output_path, confidence=confidence)

        print(f"Generated {len(image_paths)} prediction JSON files in: {output_dir}")