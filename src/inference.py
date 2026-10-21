import json
from pathlib import Path
from typing import Optional, Union
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    from ultralytics import RTDETR
except ImportError:
    RTDETR = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class ModelInference:
    """
    Inference runner supporting YOLOv8 and RT-DETR models.
    Generates standardized JSON prediction files compatible with DatasetEvaluator.
    """

    def __init__(self, model_path: Union[str, Path], device: Optional[Union[int, str]] = None):
        self.model_path = Path(model_path)
        # Ensure device falls back to CUDA if not manually passed
        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        model_name = self.model_path.name.lower()
        if "rtdetr" in model_name and RTDETR is not None:
            self.model = RTDETR(str(self.model_path))
        else:
            self.model = YOLO(str(self.model_path))

        self.class_names = self.model.names

    def predict_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        confidence: float = 0.1,
        iou_threshold: float = 0.5,
        agnostic_nms: bool = False,
        max_det: int = 300,
        visualize: bool = False,
        save_visualization: bool = False,
        visualization_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """
        Runs inference on a single image and optionally writes the JSON prediction file.
        """
        image_path = Path(image_path)
        
        predict_kwargs = {
            "source": str(image_path),
            "conf": confidence,
            "max_det": max_det,
            "device": self.device,
            "verbose": False,
        }

        if isinstance(self.model, YOLO):
            predict_kwargs.update({
                "iou": iou_threshold,
                "agnostic_nms": agnostic_nms,
            })

        results = self.model.predict(**predict_kwargs)
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
                    "bbox": [round(float(coord), 2) for coord in bbox],
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
                visualization_path = image_path.parent / f"{image_path.stem}_pred.jpg"
            visualization_path = Path(visualization_path)
            visualization_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(visualization_path))

        return prediction_data

    def predict_dataset(
        self,
        images_dir: Union[str, Path],
        output_dir: Union[str, Path],
        batch_size: int = 32,
        confidence: float = 0.1,
        iou_threshold: float = 0.5,
        agnostic_nms: bool = False,
        max_det: int = 300,
    ) -> None:
        """
        Streams batched GPU predictions across the folder and writes an individual
        <image_id>.json file for each image in real-time.
        """
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

        predict_kwargs = {
            "source": str(images_dir),  # Pass the directory directly (native Ultralytics batching)
            "conf": confidence,
            "max_det": max_det,
            "device": self.device,
            "batch": batch_size,        # Runs batches on GPU
            "stream": True,             # Streams results one-by-one so RAM never fills up
            "verbose": False,
        }

        if isinstance(self.model, YOLO):
            predict_kwargs.update({
                "iou": iou_threshold,
                "agnostic_nms": agnostic_nms,
            })

        desc = f"Inference [{self.model_path.stem}]"
        
        # Generator yields one result at a time; each individual JSON is written immediately
        for result in tqdm(self.model.predict(**predict_kwargs), total=len(image_paths), desc=desc, unit="img"):
            current_image_path = Path(result.path)
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
                        "bbox": [round(float(coord), 2) for coord in bbox],
                    })

            prediction_data = {
                "image_id": current_image_path.stem,
                "image_path": str(current_image_path.resolve()),
                "image_width": image_width,
                "image_height": image_height,
                "predictions": predictions,
            }

            # Save the individual JSON file
            output_file = output_dir / f"{current_image_path.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(prediction_data, f, indent=4)

        print(f"\nGenerated {len(image_paths)} prediction files in: {output_dir}")