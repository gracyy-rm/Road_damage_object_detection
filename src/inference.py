import json
from pathlib import Path
from typing import Optional, Union

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
        self.device = device

        # Load RTDETR or YOLO based on model filename
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
        confidence: float = 0.05,
        iou_threshold: float = 0.60,
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
            "iou": iou_threshold,
            "agnostic_nms": agnostic_nms,
            "max_det": max_det,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

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
                    "bbox": [round(float(coord), 2) for coord in bbox],  # [x1, y1, x2, y2]
                })

        prediction_data = {
            "image_id": image_path.stem,
            "image_path": str(image_path.resolve()),
            "image_width": image_width,
            "image_height": image_height,
            "predictions": predictions,
        }

        # Save JSON file
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(prediction_data, f, indent=4)

        # Optional on-screen display
        if visualize:
            annotated_image = result.plot()[..., ::-1]
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_image)
            plt.axis("off")
            plt.show()
            plt.close()

        # Optional visualization image save
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
        confidence: float = 0.05,
        iou_threshold: float = 0.60,
        agnostic_nms: bool = False,
        max_det: int = 300,
    ) -> None:
        """
        Runs batch inference across an image folder and saves an <image_id>.json file for each.
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

        desc = f"Inference [{self.model_path.stem}]"
        for image_path in tqdm(image_paths, desc=desc, unit="img"):
            output_path = output_dir / f"{image_path.stem}.json"
            self.predict_image(
                image_path=image_path,
                output_path=output_path,
                confidence=confidence,
                iou_threshold=iou_threshold,
                agnostic_nms=agnostic_nms,
                max_det=max_det,
            )

        print(f"Generated {len(image_paths)} prediction files in: {output_dir}")


if __name__ == "__main__":
    IMAGES_DIR = "dataset/images"
    BASE_PRED_DIR = "dataset/predictions"

    models = {
        "yolov8s": "weights/yolov8s.pt",
        "yolov8l": "weights/yolov8l.pt",
        "rtdetr_l": "weights/rtdetr-l.pt",
    }

    CONFIDENCE = 0.05
    IOU_THRESHOLD = 0.60

    for model_name, weights_path in models.items():
        print(f"\n---> Running predictions for: {model_name}")
        out_dir = Path(BASE_PRED_DIR) / model_name

        infer_engine = ModelInference(weights_path)
        infer_engine.predict_dataset(
            images_dir=IMAGES_DIR,
            output_dir=out_dir,
            confidence=CONFIDENCE,
            iou_threshold=IOU_THRESHOLD,
        )