import json
from pathlib import Path
from typing import Optional, Union
import torch
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
        # 1. Explicitly select CUDA if available
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

    def _format_prediction(self, result, image_path: Path) -> dict:
        """Helper to parse a single Ultralytics Results object into the target dict format."""
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

        return {
            "image_id": image_path.stem,
            "image_path": str(image_path.resolve()),
            "image_width": image_width,
            "image_height": image_height,
            "predictions": predictions,
        }

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
        Runs batched GPU streaming inference across a folder and writes individual JSON files.
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
            "source": [str(p) for p in image_paths],  # Feed image paths as a list
            "conf": confidence,
            "max_det": max_det,
            "device": self.device,
            "batch": batch_size,                      # Batch images together
            "stream": True,                           # Generator mode avoids loading all into RAM
            "verbose": False,
        }

        if isinstance(self.model, YOLO):
            predict_kwargs.update({
                "iou": iou_threshold,
                "agnostic_nms": agnostic_nms,
            })

        desc = f"Inference [{self.model_path.stem}]"
        results_generator = self.model.predict(**predict_kwargs)

        # Iterate through streamed batch results
        for idx, result in enumerate(tqdm(results_generator, total=len(image_paths), desc=desc, unit="img")):
            img_path = image_paths[idx]
            pred_data = self._format_prediction(result, img_path)

            output_path = output_dir / f"{img_path.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pred_data, f, indent=2)

        print(f"Generated {len(image_paths)} prediction files in: {output_dir}")