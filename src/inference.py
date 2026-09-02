import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ultralytics import YOLO

class YOLOInference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict_image(self, image_path, output_path=None, confidence=0.001, 
                      visualize=False, save_visualization=False, visualization_path=None):
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
                    "bbox": [float(value) for value in bbox]
                })

        prediction_data = {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "image_width": image_width,
            "image_height": image_height,
            "predictions": predictions
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as file:
                json.dump(prediction_data, file, indent=4)

        if visualize or save_visualization:
            annotated_image = result.plot()[..., ::-1]

            if visualize:
                plt.figure(figsize=(12, 8))
                plt.imshow(annotated_image)
                plt.axis("off")
                plt.show()

            if save_visualization:
                if visualization_path is None:
                    visualization_path = image_path.parent / f"{image_path.stem}_prediction.jpg"
                
                visualization_path = Path(visualization_path)
                visualization_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(annotated_image).save(visualization_path)

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

        for image_path in tqdm(image_paths, desc="Generating predictions"):
            output_path = output_dir / f"{image_path.stem}.json"
            self.predict_image(image_path=image_path, output_path=output_path, confidence=confidence)

        print(f"Generated {len(image_paths)} prediction JSON files in: {output_dir}")