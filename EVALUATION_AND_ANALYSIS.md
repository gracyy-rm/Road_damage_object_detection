---

### Part 2: `EVALUATION_AND_ANALYSIS.md` Template

This secondary document dives deep into class-wise stats, qualitative analysis, annotation quirks, failure cases, and visual plots, with heavy emphasis on your selected model.

```markdown
# Comprehensive Model Evaluation & Qualitative Analysis

This document outlines the class-wise quantitative results, qualitative error analysis (including annotation inconsistencies), failure cases, and visual comparisons for **YOLOv8-S**, **YOLOv8-L**, and **RT-DETR-L**.

---

## 📈 Class-Wise Performance Breakdown

The table below details class-specific performance metrics across all three models at their respective tuned confidence thresholds:

| Model & Class | YOLOv8-S (conf: 0.2)[cite: 1] | YOLOv8-L (conf: 0.2)[cite: 1] | RT-DETR-L (conf: 0.4)[cite: 1] |
| :--- | :---: | :---: | :---: |
| **Longitudinal Crack** | P: 0.6948 \| R: 0.6716 \| F1: 0.6830[cite: 1] | P: 0.7084 \| R: 0.6721 \| F1: **0.6898**[cite: 1] | P: 0.6049 \| R: 0.7837 \| F1: 0.6828[cite: 1] |
| **Transverse Crack** | P: 0.6526 \| R: 0.6830 \| F1: **0.6674**[cite: 1] | P: 0.6420 \| R: 0.6949 \| F1: **0.6674**[cite: 1] | P: 0.5897 \| R: 0.7475 \| F1: 0.6593[cite: 1] |
| **Alligator Crack** | P: 0.7676 \| R: 0.7521 \| F1: **0.7531**[cite: 1] | P: 0.7479 \| R: 0.7586 \| F1: 0.7532[cite: 1] | P: 0.6448 \| R: 0.8230 \| F1: 0.7231[cite: 1] |
| **Other Corruption** | P: 0.7007 \| R: 0.8141 \| F1: **0.7531**[cite: 1] | P: 0.6858 \| R: 0.8223 \| F1: 0.7479[cite: 1] | P: 0.6287 \| R: 0.8633 \| F1: 0.7276[cite: 1] |
| **Pothole** | P: 0.7210 \| R: 0.5678 \| F1: 0.6353[cite: 1] | P: 0.7177 \| R: 0.6067 \| F1: **0.6575**[cite: 1] | P: 0.5761 \| R: 0.6887 \| F1: 0.6274[cite: 1] |
| **AVERAGE** | P: 0.7073 \| R: 0.6977 \| F1: 0.6984[cite: 1] | P: 0.7004 \| R: 0.7109 \| F1: **0.7032**[cite: 1] | P: 0.6088 \| R: 0.7812 \| F1: 0.6840[cite: 1] |

---

## 🔎 Qualitative Analysis & Annotation Observations

During visual inspection of predictions against ground truth labels, several key qualitative insights emerged:
1. **Annotation Quality Noise:** A subset of ground-truth annotations displayed loose bounding boxes or missed fine hairline cracks entirely. Consequently, the model occasionally predicted correct distress patterns that were penalized as False Positives due to imperfect human labeling.
2. **Crack Intersections:** Complex intersections where longitudinal and transverse cracks meet often resulted in overlapping class confusion.

---

## 🖼️ Ground Truth vs. Prediction Visualizations

The following plots contrast Ground Truth (left/top) with Model Predictions (right/bottom) under specific edge cases:

> *Placeholder: Insert your visual comparison image here*
> `![GT vs Predictions](./assets/gt_vs_pred_yolov8l.png)`

* **Observation:** Notice how the model accurately segments minor edge cracking even when the original ground truth mask omitted peripheral boundaries.

---

## 🛠️ Failure Cases & Scope for Improvement

* **False Positives on Shadows/Stains:** Oil stains and sharp pavement color transitions are occasionally misclassified as *Other Corruption* or *Longitudinal Cracks*.
* **Small Pothole Detection:** Highly shadowed potholes show lower recall rates due to lack of distinct depth features.
* **Scope for Improvement:** 
  * Implementing test-time augmentation (TTA) and hard negative mining.
  * Refining dataset annotations to remove ambiguous or loose bounding boxes.
  * Exploring hybrid loss functions to better handle class imbalance in fine-grained cracks.