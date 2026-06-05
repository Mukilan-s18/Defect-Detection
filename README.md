# Unsupervised Defect Detection using PatchCore

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/status-active-success.svg)

This repository implements a state-of-the-art anomaly detection system for industrial manufacturing using the **PatchCore** architecture. The primary objective is to detect minute defects in manufactured parts using purely unsupervised learning. The model is trained exclusively on "good" (non-defective) samples and identifies anomalies by measuring the structural deviation of test images against a condensed memory bank of nominal features.

The pipeline is heavily optimized to process high-dimensional features without exceeding strict memory (OOM) and compute-time constraints, making it ready for cloud infrastructure deployment.

## 🚀 Key Features
- **Unsupervised Learning**: Requires only "good" samples for training.
- **Transfer Learning**: Utilizes a pre-trained WideResNet50 backbone for robust feature extraction.
- **Coreset Subsampling**: Employs a Greedy Coreset Algorithm to condense the memory bank while preserving maximum feature diversity, preventing OOM errors on constrained environments.
- **Multi-View Support**: Processes bottom, side, and top views of industrial components.
- **Hardware Acceleration**: K-NN and Coreset selection optimized using PyTorch GPU/MPS tensors.

---

## 🧠 Neural Network Architecture

1. **Backbone Model**: `Wide_ResNet50_2` pre-trained on ImageNet1K.
2. **Multi-Scale Feature Extraction**: Intermediate feature maps are extracted from `Layer 2` and `Layer 3` to capture both low-level textures (scratches) and high-level structural semantics (missing parts).
3. **Dimensionality Alignment**: Features are upsampled and concatenated to form a dense `1536-dimensional` vector.
4. **Patch Pooling**: A `3x3` Average Pooling layer smooths features for local spatial robustness.

---

## 🛠 Pipeline Workflow

The project is structured into modular blocks for seamless execution:

### 1. Pre-Processing 
- Normalization (ImageNet statistics), resizing to 256px, and CenterCropping to `224x224`.
- Objects are serialized using `joblib` for independent execution blocks.

### 2. Memory Bank Construction
- **Subsampling**: Aggressively subsamples extracted patches to prevent `SIGKILL` (Out of Memory) crashes, reducing peak RAM usage from >2.5GB to <150MB.
- **Greedy Coreset**: Condenses 30,000+ patches using a 10% coreset ratio, ensuring the memory bank represents the entire "nominal space" without redundancy.

### 3. Inference & K-NN Scoring
- Extracts features for test images.
- Calculates Euclidean distance against the `k=5` nearest neighbors in the memory bank.
- Generates a spatial `28x28` anomaly score map.
- Applies **Gaussian Smoothing** (`sigma=1.5`) to eliminate noisy pixels.
- Dynamically calculates the optimal anomaly threshold by maximizing the F1-Score on the Precision-Recall curve.

---

## 📊 Results & Deliverables

The pipeline produces the following artifacts in the `outputs/` folder:
- **`metrics.csv`**: Statistical breakdown of Precision, Recall, and F1 scores across all camera views.
- **`pretrain_score_distribution.png`**: Raw, overlapping data distributions before memory bank construction.
- **`trained_score_distribution.png`**: Separated defect distributions with the dynamically chosen threshold line.

---

## 💻 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mukilan-s18/Defect-Detection.git
   cd Defect-Detection
   ```
2. Install dependencies (e.g., via pip/conda). *(Ensure PyTorch, Torchvision, Scikit-learn, and Joblib are installed).*

### Execution
Run the full pipeline using:
```bash
python run_pipeline.py
```

---

## 📄 License
This project is for educational and research purposes.
