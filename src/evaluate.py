import json
from pathlib import Path
from tabulate import tabulate

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import numpy as np
import pandas as pd

class ImageEvaluator:

    def __init__(self, gt_path, pred_path, iou_threshold=0.5):
        self.gt_path = Path(gt_path)
        self.pred_path = Path(pred_path)
        self.iou_threshold = iou_threshold
        self.gt_data = self._load_json(self.gt_path)
        self.pred_data = self._load_json(self.pred_path)

    @staticmethod
    def _load_json(path):
        with open(path, "r") as file:
            return json.load(file)

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def match_predictions(self, confidence_threshold):
        predictions = [
            (index, prediction)
            for index, prediction in enumerate(self.pred_data["predictions"])
            if prediction["confidence"] >= confidence_threshold
        ]

        predictions.sort(
            key=lambda item: item[1]["confidence"],
            reverse=True
        )

        ground_truths = self.gt_data["annotations"]
        matched_gt_indices = set()
        matched_ious = []
        matches = []
        tp = fp = 0

        for pred_index, prediction in predictions:
            best_iou = 0.0
            best_gt_index = None

            for gt_index, ground_truth in enumerate(ground_truths):

                if gt_index in matched_gt_indices:
                    continue

                if prediction["class_id"]!=ground_truth["class_id"]:
                    continue
                
                iou = self.calculate_iou(
                    prediction["bbox"],
                    ground_truth["bbox"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index is not None and best_iou >= self.iou_threshold:
                matched_gt_indices.add(best_gt_index)
                matched_ious.append(best_iou)
                tp += 1
                status = "TP"
            else:
                fp += 1
                status = "FP"

            matches.append({
                "prediction_index": pred_index,
                "gt_index": best_gt_index if status == "TP" else None,
                "confidence": prediction["confidence"],
                "iou": best_iou,
                "status": status
            })

        fn = len(ground_truths) - len(matched_gt_indices)

        return {
            "gt_count": len(ground_truths),
            "prediction_count": len(predictions),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "matched_ious": matched_ious,
            "matches": matches
        }

    def match_class_predictions(self, class_id, confidence_threshold):
        ground_truths = [
            (index, ground_truth)
            for index, ground_truth in enumerate(self.gt_data["annotations"])
            if ground_truth["class_id"] == class_id
        ]

        predictions = [
            (index, prediction)
            for index, prediction in enumerate(self.pred_data["predictions"])
            if prediction["class_id"] == class_id
            and prediction["confidence"] >= confidence_threshold
        ]

        predictions.sort(
            key=lambda item: item[1]["confidence"],
            reverse=True
        )

        matched_gt_indices = set()
        matched_ious = []
        tp = fp = 0

        for _, prediction in predictions:
            best_iou = 0.0
            best_gt_index = None

            for gt_index, ground_truth in ground_truths:

                if gt_index in matched_gt_indices:
                    continue

              
                iou = self.calculate_iou(
                    prediction["bbox"],
                    ground_truth["bbox"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index is not None and best_iou >= self.iou_threshold:
                matched_gt_indices.add(best_gt_index)
                matched_ious.append(best_iou)
                tp += 1
            else:
                fp += 1

        return {
            "class_id": class_id,
            "gt_count": len(ground_truths),
            "prediction_count": len(predictions),
            "tp": tp,
            "fp": fp,
            "fn": len(ground_truths) - len(matched_gt_indices),
            "matched_ious": matched_ious
        }

    @staticmethod
    def calculate_metrics(match_result):
        tp = match_result["tp"]
        fp = match_result["fp"]
        fn = match_result["fn"]

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        matched_ious = match_result["matched_ious"]
        mean_iou = (
            sum(matched_ious) / len(matched_ious)
            if matched_ious else 0.0
        )

        return {
            "gt_count": match_result["gt_count"],
            "prediction_count": match_result["prediction_count"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mean_iou": mean_iou
        }

    def get_per_class_stat(self, confidence_threshold):
        filtered_predictions = [
            prediction
            for prediction in self.pred_data["predictions"]
            if prediction["confidence"] >= confidence_threshold
        ]

        class_ids = {
            ground_truth["class_id"]
            for ground_truth in self.gt_data["annotations"]
        }

        class_ids.update(
            prediction["class_id"]
            for prediction in filtered_predictions
        )

        results = []

        for class_id in sorted(class_ids):
            result = self.match_class_predictions(
                class_id,
                confidence_threshold
            )

            metrics = self.calculate_metrics(result)
            metrics["class_id"] = class_id
            results.append(metrics)

        return results

    def get_stat(self, iou=0.5, confidence_thresholds=None, return_stat=True):

        if confidence_thresholds is None:
            confidence_thresholds = [0.25, 0.5, 0.75]

        self.iou_threshold = iou
        overall_results = []
        per_class_results = []

        for confidence_threshold in confidence_thresholds:

            overall_match = self.match_predictions(confidence_threshold)
            overall_metrics = self.calculate_metrics(overall_match)

            overall_metrics["confidence_threshold"] = confidence_threshold
            overall_results.append(overall_metrics)

            class_results = self.get_per_class_stat(confidence_threshold)

            for result in class_results:
                result["confidence_threshold"] = confidence_threshold
                per_class_results.append(result)

        results = {
            "overall": overall_results,
            "per_class": per_class_results
        }

        if not return_stat:
            return results

        overall_table = [
            [
                result["confidence_threshold"],
                result["gt_count"],
                result["prediction_count"],
                result["tp"],
                result["fp"],
                result["fn"],
                round(result["precision"], 4),
                round(result["recall"], 4),
                round(result["f1_score"], 4),
                round(result["mean_iou"], 4)
            ]
            for result in overall_results
        ]

        print("\nSINGLE-IMAGE OVERALL DETECTION METRICS\n")

        print(tabulate(
            overall_table,
            headers=[
                "Confidence",
                "GT",
                "Predictions",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "Mean IoU"
            ],
            tablefmt="grid"
        ))

        per_class_table = [
            [
                result["confidence_threshold"],
                result["class_id"],
                result["gt_count"],
                result["prediction_count"],
                result["tp"],
                result["fp"],
                result["fn"],
                round(result["precision"], 4),
                round(result["recall"], 4),
                round(result["f1_score"], 4),
                round(result["mean_iou"], 4)
            ]
            for result in per_class_results
        ]

        print("\nSINGLE-IMAGE PER-CLASS DETECTION METRICS\n")

        print(tabulate(
            per_class_table,
            headers=[
                "Confidence",
                "Class",
                "GT",
                "Predictions",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "Mean IoU"
            ],
            tablefmt="grid"
        ))

        return results
    
    def plot(self,confidence_threshold=0.25,class_names=None):
        image_path=self.gt_data["image_path"]

        image=Image.open(image_path).convert("RGB")

        predictions=[
            prediction
            for prediction in self.pred_data["predictions"]
            if prediction["confidence"]>=confidence_threshold
        ]

        fig,axes=plt.subplots(1,2,figsize=(16,8))

        # Ground Truth
        axes[0].imshow(image)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        for annotation in self.gt_data["annotations"]:
            x1,y1,x2,y2=annotation["bbox"]
            class_id=annotation["class_id"]

            rectangle=patches.Rectangle(
                (x1,y1),
                x2-x1,
                y2-y1,
                fill=False,
                linewidth=2
            )

            axes[0].add_patch(rectangle)

            class_name=(
                class_names[class_id]
                if class_names is not None
                else str(class_id)
            )

            axes[0].text(
                x1,
                max(0,y1-5),
                class_name,
                fontsize=10,
                bbox=dict(alpha=0.7)
            )

        # Predictions
        axes[1].imshow(image)
        axes[1].set_title(
            f"Predictions (Confidence ≥ {confidence_threshold})"
        )
        axes[1].axis("off")

        for prediction in predictions:
            x1,y1,x2,y2=prediction["bbox"]
            class_id=prediction["class_id"]
            confidence=prediction["confidence"]

            rectangle=patches.Rectangle(
                (x1,y1),
                x2-x1,
                y2-y1,
                fill=False,
                linewidth=2
            )

            axes[1].add_patch(rectangle)

            class_name=(
                class_names[class_id]
                if class_names is not None
                else str(class_id)
            )

            label=f"{class_name} {confidence:.2f}"

            axes[1].text(
                x1,
                max(0,y1-5),
                label,
                fontsize=10,
                bbox=dict(alpha=0.7)
            )

        plt.tight_layout()
        plt.show()

class DatasetEvaluator:
    def __init__(self,gt_dir,pred_dir,iou_threshold=0.5):
        self.gt_dir=Path(gt_dir)
        self.pred_dir=Path(pred_dir)
        self.iou_threshold=iou_threshold
        self.data=self._load_dataset()
        self.df=None

    def _load_dataset(self):
        data={}

        for gt_path in self.gt_dir.glob("*.json"):
            image_id=gt_path.stem
            pred_path=self.pred_dir/f"{image_id}.json"

            if not pred_path.exists():
                continue

            with open(gt_path,"r") as file:
                gt_data=json.load(file)

            with open(pred_path,"r") as file:
                pred_data=json.load(file)

            data[image_id]={
                "gt":gt_data,
                "pred":pred_data
            }

        return data

    def _get_class_ids(self):
        class_ids=set()

        for image_data in self.data.values():
            for annotation in image_data["gt"]["annotations"]:
                class_ids.add(annotation["class_id"])

            for prediction in image_data["pred"]["predictions"]:
                class_ids.add(prediction["class_id"])

        return sorted(class_ids)

    @staticmethod
    def _calculate_metrics(gt_count,prediction_count,tp,fp,fn,matched_ious):
        precision=tp/(tp+fp) if tp+fp>0 else 0.0
        recall=tp/(tp+fn) if tp+fn>0 else 0.0

        f1_score=(
            2*precision*recall/(precision+recall)
            if precision+recall>0 else 0.0
        )

        mean_iou=(
            sum(matched_ious)/len(matched_ious)
            if matched_ious else 0.0
        )

        return {
            "gt_count":gt_count,
            "prediction_count":prediction_count,
            "tp":tp,
            "fp":fp,
            "fn":fn,
            "precision":precision,
            "recall":recall,
            "f1_score":f1_score,
            "mean_iou":mean_iou
        }

    def evaluate_dataset(self,confidence_threshold):
        gt_count=0
        prediction_count=0
        tp=0
        fp=0
        fn=0
        matched_ious=[]

        for image_data in self.data.values():
            gt_data=image_data["gt"]
            pred_data=image_data["pred"]

            predictions=[
                prediction
                for prediction in pred_data["predictions"]
                if prediction["confidence"]>=confidence_threshold
            ]

            predictions=sorted(
                predictions,
                key=lambda prediction:prediction["confidence"],
                reverse=True
            )

            ground_truths=gt_data["annotations"]
            matched_gt_indices=set()

            gt_count+=len(ground_truths)
            prediction_count+=len(predictions)

            for prediction in predictions:
                best_iou=0.0
                best_gt_index=None

                for gt_index,ground_truth in enumerate(ground_truths):
                    if gt_index in matched_gt_indices:
                        continue

                    if prediction["class_id"]!=ground_truth["class_id"]:
                        continue

                    iou=ImageEvaluator.calculate_iou(
                        prediction["bbox"],
                        ground_truth["bbox"]
                    )

                    if iou>best_iou:
                        best_iou=iou
                        best_gt_index=gt_index

                if best_gt_index is not None and best_iou>=self.iou_threshold:
                    tp+=1
                    matched_gt_indices.add(best_gt_index)
                    matched_ious.append(best_iou)
                else:
                    fp+=1

            fn+=len(ground_truths)-len(matched_gt_indices)

        return self._calculate_metrics(
            gt_count,
            prediction_count,
            tp,
            fp,
            fn,
            matched_ious
        )

    def evaluate_per_class(self,confidence_threshold):
        results=[]

        for class_id in self._get_class_ids():
            gt_count=0
            prediction_count=0
            tp=0
            fp=0
            fn=0
            matched_ious=[]

            for image_data in self.data.values():
                ground_truths=[
                    annotation
                    for annotation in image_data["gt"]["annotations"]
                    if annotation["class_id"]==class_id
                ]

                predictions=[
                    prediction
                    for prediction in image_data["pred"]["predictions"]
                    if prediction["class_id"]==class_id
                    and prediction["confidence"]>=confidence_threshold
                ]

                predictions=sorted(
                    predictions,
                    key=lambda prediction:prediction["confidence"],
                    reverse=True
                )

                matched_gt_indices=set()

                gt_count+=len(ground_truths)
                prediction_count+=len(predictions)

                for prediction in predictions:
                    best_iou=0.0
                    best_gt_index=None

                    for gt_index,ground_truth in enumerate(ground_truths):
                        if gt_index in matched_gt_indices:
                            continue

                        iou=ImageEvaluator.calculate_iou(
                            prediction["bbox"],
                            ground_truth["bbox"]
                        )

                        if iou>best_iou:
                            best_iou=iou
                            best_gt_index=gt_index

                    if best_gt_index is not None and best_iou>=self.iou_threshold:
                        tp+=1
                        matched_gt_indices.add(best_gt_index)
                        matched_ious.append(best_iou)
                    else:
                        fp+=1

                fn+=len(ground_truths)-len(matched_gt_indices)

            metrics=self._calculate_metrics(
                gt_count,
                prediction_count,
                tp,
                fp,
                fn,
                matched_ious
            )

            metrics["class_id"]=class_id
            results.append(metrics)

        return results

    def calculate_ap(self,class_id,iou_threshold=0.5):
        predictions=[]
        gt_by_image={}
        gt_count=0

        for image_id,image_data in self.data.items():
            ground_truths=[
                annotation
                for annotation in image_data["gt"]["annotations"]
                if annotation["class_id"]==class_id
            ]

            gt_by_image[image_id]=ground_truths
            gt_count+=len(ground_truths)

            for prediction in image_data["pred"]["predictions"]:
                if prediction["class_id"]==class_id:
                    predictions.append({
                        "image_id":image_id,
                        "confidence":prediction["confidence"],
                        "bbox":prediction["bbox"]
                    })

        if gt_count==0 or not predictions:
            return 0.0

        predictions=sorted(
            predictions,
            key=lambda prediction:prediction["confidence"],
            reverse=True
        )

        matched_gt={
            image_id:set()
            for image_id in gt_by_image
        }

        tp=[]
        fp=[]

        for prediction in predictions:
            image_id=prediction["image_id"]
            ground_truths=gt_by_image[image_id]

            best_iou=0.0
            best_gt_index=None

            for gt_index,ground_truth in enumerate(ground_truths):
                if gt_index in matched_gt[image_id]:
                    continue

                iou=ImageEvaluator.calculate_iou(
                    prediction["bbox"],
                    ground_truth["bbox"]
                )

                if iou>best_iou:
                    best_iou=iou
                    best_gt_index=gt_index

            if best_gt_index is not None and best_iou>=iou_threshold:
                tp.append(1)
                fp.append(0)
                matched_gt[image_id].add(best_gt_index)
            else:
                tp.append(0)
                fp.append(1)

        tp=np.cumsum(tp)
        fp=np.cumsum(fp)

        recall=tp/gt_count
        precision=tp/(tp+fp)

        recall=np.concatenate(([0.0],recall,[1.0]))
        precision=np.concatenate(([0.0],precision,[0.0]))

        for index in range(len(precision)-1,0,-1):
            precision[index-1]=max(
                precision[index-1],
                precision[index]
            )

        indices=np.where(
            recall[1:]!=recall[:-1]
        )[0]

        ap=np.sum(
            (recall[indices+1]-recall[indices])
            *precision[indices+1]
        )

        return float(ap)

    def get_ap(self,iou_threshold=0.5):
        results=[]

        for class_id in self._get_class_ids():
            results.append({
                "class_id":class_id,
                "iou_threshold":iou_threshold,
                "ap":self.calculate_ap(
                    class_id,
                    iou_threshold
                )
            })

        return results

    def calculate_map(self,iou_threshold=0.5):
        ap_results=self.get_ap(iou_threshold)

        if not ap_results:
            return 0.0

        return float(
            np.mean(
                [result["ap"] for result in ap_results]
            )
        )

    def calculate_map_50_95(self):
        iou_thresholds=np.arange(0.5,1.0,0.05)

        map_scores=[
            self.calculate_map(round(float(iou),2))
            for iou in iou_thresholds
        ]

        return float(np.mean(map_scores))

    def get_stat(
        self,
        iou=0.5,
        confidence_thresholds=None,
        return_stat=True
    ):
        if confidence_thresholds is None:
            confidence_thresholds=[0.25,0.5,0.75]
        self.iou_threshold=iou
        overall_results=[]
        per_class_results=[]

        for confidence_threshold in confidence_thresholds:
            overall=self.evaluate_dataset(confidence_threshold)
            overall["confidence_threshold"]=confidence_threshold
            overall_results.append(overall)

            class_results=self.evaluate_per_class(confidence_threshold)

            for result in class_results:
                result["confidence_threshold"]=confidence_threshold
                per_class_results.append(result)

        ap_results=self.get_ap(0.5)
        map_50=self.calculate_map(0.5)
        map_50_95=self.calculate_map_50_95()

        results={
            "overall":overall_results,
            "per_class":per_class_results,
            "ap":ap_results,
            "map_50":map_50,
            "map_50_95":map_50_95
        }

        if not return_stat:
            return results

        overall_table=[
            [
                result["confidence_threshold"],
                result["gt_count"],
                result["prediction_count"],
                result["tp"],
                result["fp"],
                result["fn"],
                round(result["precision"],4),
                round(result["recall"],4),
                round(result["f1_score"],4),
                round(result["mean_iou"],4)
            ]
            for result in overall_results
        ]

        self.df = pd.DataFrame(
            overall_table,
            columns=[
                "Confidence",
                "GT",
                "Predictions",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "Mean IoU"
            ]
        )
        print("\nDATASET-LEVEL OVERALL METRICS\n")
        print(tabulate(
            overall_table,
            headers=[
                "Confidence","GT","Predictions","TP","FP","FN",
                "Precision","Recall","F1","Mean IoU"
            ],
            tablefmt="grid"
        ))

        per_class_table=[
            [
                result["confidence_threshold"],
                result["class_id"],
                result["gt_count"],
                result["prediction_count"],
                result["tp"],
                result["fp"],
                result["fn"],
                round(result["precision"],4),
                round(result["recall"],4),
                round(result["f1_score"],4),
                round(result["mean_iou"],4)
            ]
            for result in per_class_results
        ]

        print("\nDATASET-LEVEL PER-CLASS METRICS\n")
        print(tabulate(
            per_class_table,
            headers=[
                "Confidence","Class","GT","Predictions","TP","FP","FN",
                "Precision","Recall","F1","Mean IoU"
            ],
            tablefmt="grid"
        ))

        ap_table=[
            [
                result["class_id"],
                result["iou_threshold"],
                round(result["ap"],4)
            ]
            for result in ap_results
        ]

        print("\nDATASET-LEVEL AP\n")
        print(tabulate(
            ap_table,
            headers=["Class","IoU","AP"],
            tablefmt="grid"
        ))

        print("\nDATASET-LEVEL mAP\n")
        print(tabulate(
            [
                ["mAP@0.5",round(map_50,4)],
                ["mAP@0.5:0.95",round(map_50_95,4)]
            ],
            headers=["Metric","Score"],
            tablefmt="grid"
        ))

        return results

    def plot(
        self,
        num_samples=5,
        confidence_threshold=0.25,
        class_names=None,
        random_sample=True
    ):
        image_ids=list(self.data.keys())
        num_samples=min(num_samples,len(image_ids))

        if random_sample:
            image_ids=random.sample(image_ids,num_samples)
        else:
            image_ids=image_ids[:num_samples]

        for image_id in image_ids:
            evaluator=ImageEvaluator(
                gt_path=self.gt_dir/f"{image_id}.json",
                pred_path=self.pred_dir/f"{image_id}.json",
                iou_threshold=self.iou_threshold
            )

            evaluator.plot(
                confidence_threshold=confidence_threshold,
                class_names=class_names
            )