# Road Damage Object Detection 

An end-to-end computer vision pipeline for detecting and evaluating road surface distress (cracks, potholes, and surface corruptions) using state-of-the-art object detection models: **YOLOv8-S**, **YOLOv8-L**, and **RT-DETR-L**.

---

## Overall Dataset Evaluation Summary

The models were evaluated using a custom evaluation pipeline across the complete dataset at optimized confidence thresholds:

| Model | Confidence Threshold | Ground Truth (GT) | Predictions | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-S**[cite: 2] | 0.2[cite: 2] | 9,675[cite: 2] | 9,646[cite: 2] | 6,768[cite: 2] | 2,878[cite: 2] | 2,907[cite: 2] | 0.7016[cite: 2] | 0.6995[cite: 2] | 0.7006[cite: 2] | 0.9491[cite: 2] |
| **YOLOv8-L**[cite: 2] | 0.2[cite: 2] | 9,675[cite: 2] | 9,803[cite: 2] | 6,850[cite: 2] | 2,953[cite: 2] | 2,825[cite: 2] | 0.6988[cite: 2] | 0.7080[cite: 2] | **0.7034**[cite: 2] | **0.9501**[cite: 2] |
| **RT-DETR-L**[cite: 2] | 0.4[cite: 2] | 9,675[cite: 2] | 12,486[cite: 2] | 7,618[cite: 2] | 4,868[cite: 2] | 2,057[cite: 2] | 0.6101[cite: 2] | **0.7874**[cite: 2] | 0.6875[cite: 2] | 0.9425[cite: 2] |

---

## 🔍 Key Findings & Model Selection
* **YOLOv8-L** achieves the most balanced overall performance, securing the highest **F1-Score (0.7034)**[cite: 2] and **Mean IoU (0.9501)**[cite: 2] with stable precision-recall trade-offs.
* **RT-DETR-L** displays exceptional recall (0.7874)[cite: 2] and captures the lowest false negatives, but suffers from a higher false positive rate at its tuned threshold.

**For deep-dive class breakdowns, qualitative analysis, annotation limitations, failure cases, and visual comparisons, see the [Evaluation & Analysis Document](./EVALUATION_AND_ANALYSIS.md).**

---

## ⚙️ Quick Start Installation

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt