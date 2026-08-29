# MissingPieces: Computer Vision Pipeline for Fashion Recognition

Final Project for the Introduction to Computer Vision course (EPICODE Institute of Technology).

---

## 1. Project Overview

The objective of this project is to automate garment cataloging from unconstrained user photographs taken in home environments (such as wooden floors, patterned rugs, and non-uniform lighting).

The pipeline takes a raw image, segments the garment from the background, extracts dominant color attributes, and classifies the item into one of the top 10 categories, generating a structured JSON payload for downstream REST API consumption.

---

## 2. Dataset

* Source: agrigorev/clothing-dataset-full (Kaggle)
* Validated Samples: 5,175 verified physical images
* Evaluated Classes (Top 10): T-Shirt, Longsleeve, Pants, Shoes, Shirt, Dress, Outwear, Shorts, Hat, Skirt
* Data Cleaning: Excluded noisy samples with "Not sure" labels or missing annotations

---

## 3. Architecture & Methodology

The pipeline consists of four modular stages:

```text
+-------------------+
|  Raw Image Input  |
+---------+---------+
          |
          v
+------------------------------------------+
| Preprocessing & GrabCut Segmentation     |
+----+-------------------+-----------------+
     |                   |                 |
     v                   v                 v
+-----------------+ +-----------------+ +-----------------+
| Color Attribute | | Classical CV    | | Deep Learning   |
| K-Means (k=3)   | | HOG + LBP + SVM | | ResNet18 Head   |
+--------+--------+ +--------+--------+ +--------+--------+
         |                   |                   |
         +-------------------+-------------------+
                             |
                             v
                  +---------------------+
                  | Unified JSON Output |
                  +---------------------+
```

### A. Preprocessing & Foreground Segmentation
1. Uniform resizing to 224 x 224 pixels in RGB color space.
2. Background isolation using the GrabCut algorithm with a 90% bounding box initialization (1 iteration) to create a binary mask over garment pixels.
3. Morphological cleaning (opening, closing, largest connected component) to remove residual mask noise.

### B. Color Feature Extraction
* Unsupervised K-Means (k=3): Fitted exclusively on active foreground pixels. The largest cluster centroid is exported as standard RGB and Hexadecimal (HEX) values.

### C. Classification Benchmark
1. Classical Approach (Handcrafted Descriptors):
   * HOG (Histogram of Oriented Gradients): 8 orientations, 16 x 16 pixels/cell, 2 x 2 cells/block to capture boundary shapes.
   * LBP (Local Binary Patterns): Uniform patterns (r=2, 16 points) capturing micro-texture patterns of fabric weaves.
   * Classifier: Support Vector Machine (SVM) with RBF kernel (C=10.0) trained on standard-scaled [HOG + LBP] vectors.
2. Deep Learning Approach (Representation Learning):
   * Backbone: ResNet18 pre-trained on ImageNet.
   * Strategy: Feature extraction with frozen base convolutional layers (requires_grad = False).
   * Custom Classification Head: Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 10).
   * Training Configuration: 5 epochs, CrossEntropyLoss, Adam optimizer (learning rate = 0.001), batch size 32.

---

## 4. Experimental Results

Both models were evaluated on the same 20% validation split:
```text
| Model               | Accuracy   | Macro F1 | Avg Latency       |
| **SVM (HOG + LBP)** | 60.00%     | 0.46     | ~221 ms (CPU)     |
| **ResNet18**        | **78.85%** | **0.76** | **~107 ms (GPU)** |
```
### Failure Analysis
* Skew toward T-Shirts: Because T-Shirt is the most frequent class in the dataset, the network tends to predict T-Shirt when it is uncertain.
* Shirt vs. Longsleeve: Longsleeves are confused with Shirts (18 cases) because both have long sleeves and a similar shape when laid flat.
* Skirts: Skirt remains the hardest category. The classical SVM failed completely (F1 = 0.00), while ResNet18 reaches 51.6% recall.

---

## 5. REST Ingestion JSON Output

Calling predict_fashion_item(image_path) outputs the standardized schema:

```json
{
    "predicted_category": "Shoes",
    "confidence": 0.4375,
    "dominant_color_hex": "#d0c7bf",
    "dominant_color_rgb": [
        208,
        199,
        191
    ],
    "avg_inference_latency_ms": 107.53
}
