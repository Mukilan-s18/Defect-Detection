# Project Report: Unsupervised Defect Detection using PatchCore

## 1. Project Overview
This project implements a state-of-the-art anomaly detection system for industrial manufacturing using the **PatchCore** architecture. The primary objective is to detect minute defects in manufactured parts using purely unsupervised learning. The model is trained exclusively on "good" (non-defective) samples and identifies anomalies by measuring the structural deviation of test images against a condensed memory bank of nominal features.

The pipeline is designed to run seamlessly on cloud infrastructure (specifically the MANGO platform) and is heavily optimized to process high-dimensional features without exceeding strict memory (OOM) and compute-time constraints.

---

## 2. Dataset Architecture
The dataset consists of multi-view images of industrial components. Because defects might only be visible from certain angles, the system processes three distinct camera perspectives simultaneously:
- **Bottom View**
- **Side View**
- **Top View**

### Pre-processing Pipeline (Block 1)
- **Transformations:** All images are normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), resized to a shorter edge of 256 pixels, and CenterCropped to `224x224`.
- **Dataloaders:** PyTorch DataLoaders are initialized with `batch_size=16` for both training and testing datasets.
- **Serialization:** The resulting DataLoader objects are serialized using `joblib` for seamless transport between independent execution blocks on the cloud platform.

---

## 3. Neural Network Architecture (Block 2)
Instead of training a model from scratch, this system utilizes **Transfer Learning** via a pre-trained WideResNet50 backbone.

- **Backbone Model:** `Wide_ResNet50_2` pre-trained on ImageNet1K.
- **Feature Extraction Layers:** Intermediate feature maps are extracted from `Layer 2` and `Layer 3` of the network. This multi-scale approach captures both low-level textures (e.g., scratches) and high-level structural semantics (e.g., missing parts).
- **Dimensionality Alignment:** Features from Layer 3 are upsampled using Bilinear Interpolation to match the spatial dimensions of Layer 2. The layers are then concatenated along the channel dimension to form a dense `1536-dimensional` feature vector for each pixel patch.
- **Patch Pooling:** A `3x3` Average Pooling layer with `stride=1` and `padding=1` is applied to smooth the extracted features and add local spatial robustness.
- **Frozen Weights:** The entire network is frozen (`requires_grad = False`) because PatchCore is a memory-based method, not a backpropagation-based method.

---

## 4. Pre-Train Evaluation (Block 3)
Before constructing the memory bank, the system evaluates the raw feature overlap between "good" and "anomalous" data using a naive Euclidean distance metric. 
- A histogram distribution plot is generated, proving that without the rigorous Coreset algorithm, the defect scores heavily overlap, rendering simple thresholding useless.
- This establishes the baseline necessity for the K-Nearest Neighbors (K-NN) algorithm applied later in the pipeline.

---

## 5. Memory Bank Construction & Optimization (Block 4)
This block represents the core algorithmic intelligence of the pipeline. The goal is to build a "Memory Bank" of all possible normal feature variations.

### Subsampling & OOM Protections
Extracting 1536-dimensional vectors for every patch in the training set results in millions of vectors, which easily exceeds 2.5 GB of RAM. To prevent `SIGKILL` (Out of Memory) crashes on constrained cloud containers:
1. **Per-Batch Random Subsampling:** During the extraction loop, each batch of patches is aggressively subsampled down to a maximum of 1,000 vectors. This reduces peak RAM usage from 2.5GB to under 150MB.
2. **Global Cap:** Before applying the Coreset algorithm, the global patch pool is capped at `30,000` vectors to guarantee that O(N²) distance calculations complete in under 2 minutes, preventing CPU timeout errors.

### Greedy Coreset Selection
Instead of keeping all 30,000 patches, the system uses a **Greedy Coreset Algorithm** to condense the memory bank while preserving maximum feature diversity.
- **Coreset Ratio:** `10%` (`0.1`). The algorithm iteratively selects the patch that has the maximum distance from the already-selected patches, ensuring that the memory bank represents the entire "nominal space" boundary without redundancy.
- **Hardware Acceleration:** The coreset loop is executed directly on PyTorch GPU/MPS tensors (`torch.cdist` / `torch.sum`) rather than NumPy arrays, resulting in a 10x speedup.

---

## 6. Inference, Scoring, and Evaluation (Block 5)
During inference, a test image is passed through the network to extract its feature patches. 

### K-Nearest Neighbors (K-NN) Scoring
- For every patch in the test image, the Euclidean distance is calculated against every patch in the condensed Memory Bank.
- The algorithm looks at the `k=5` nearest neighbors in the memory bank and takes the average distance.
- This generates a spatial "anomaly score map" (`28x28`) for the entire image. 

### Gaussian Smoothing
- The spatial anomaly score map is smoothed using a Gaussian Filter (`sigma=1.5`). This eliminates isolated noisy pixels and highlights true structural anomalies.
- The final image-level anomaly score is defined as the maximum value within this smoothed spatial map.

### Metric Calculation
- **Threshold Optimization:** The system dynamically calculates the optimal anomaly threshold by maximizing the F1-Score on the Precision-Recall curve.
- **Histograms:** A side-by-side histogram is plotted showing clear separation between the "Good" and "Anomaly" distributions for all three views, proving the success of the model.
- **Performance:** Final metrics (Precision, Recall, F1) are calculated using the optimal threshold and exported as a CSV report (`metrics.csv`).

---

## 7. Deliverables
Upon successful execution, the pipeline produces the following artifacts:
1. **Pre-Train Histogram** (`pretrain_score_distribution.png`): Shows the raw, overlapping data distributions before the memory bank is built.
2. **Post-Train Histogram** (`trained_score_distribution.png`): Shows the clearly separated defect distributions with the dynamically chosen threshold line.
3. **Metrics Report** (`metrics.csv`): A precise statistical breakdown of Precision, Recall, and F1 scores across the Bottom, Side, and Top views.
