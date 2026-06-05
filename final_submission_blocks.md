-----------------------------
Block 1: Data Pre Processing
Block 2: Model Definition (PatchCore Anomaly Detector)
Block 3: Pre Train Prediction Visualization
Block 4: Training (Memory Bank Construction)
Block 5: Evaluation & Deliverables Generation
----------------------------------

# Loading Libraries - Common to all blocks
import os
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score,
    precision_recall_curve, average_precision_score,
)
from scipy.ndimage import gaussian_filter
import joblib
from torch.utils.data import Dataset

# Dataset Classes (Common to all blocks)
class SingleClassImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        if os.path.exists(root):
            for f in sorted(os.listdir(root)):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    self.samples.append(os.path.join(root, f))
                    
    def __len__(self): 
        return len(getattr(self, 'samples', getattr(self, 'paths', [])))
        
    def __getitem__(self, idx):
        items = getattr(self, 'samples', getattr(self, 'paths', []))
        path = items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, 0, path

class LabelledImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        for label_name, label_idx in [('good', 0), ('anomaly', 1)]:
            d = os.path.join(root, label_name)
            if os.path.exists(d):
                for f in sorted(os.listdir(d)):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        self.samples.append((os.path.join(d, f), label_idx))
                        
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label, path

### **************************************************************************************************************************************

# Block 1 : Data Pre Processing
## Input
- *Data Method*: Data Source; *Data Type*: String; *Variable*: **zip_path**; *Default Value*: ; *Description*: Path to uploaded dataset zip;
- *Data Method*: Static Data; *Data Type*: String; *Variable*: **output_root**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset; *Description*: Shared output path;
- *Data Method*: Static Data; *Data Type*: Number; *Variable*: **image_size**; *Default Value*: 256; *Description*: Resize shorter edge;
- *Data Method*: Static Data; *Data Type*: Number; *Variable*: **batch_size**; *Default Value*: 16; *Description*: DataLoader batch size;
- *Data Method*: Static Data; *Data Type*: Number; *Variable*: **random_state**; *Default Value*: 42; *Description*: Random seed;

## Output
*Type*: String; *Variable*: **train_loaders_path**;
*Type*: String; *Variable*: **test_loaders_path**;
*Type*: String; *Variable*: **test_meta_path**;

## Code
def main(zip_path, output_root, image_size=256, batch_size=16, random_state=42):
    VIEWS = ['bottom', 'side', 'top']
    zip_path = str(zip_path).strip()
    output_root = str(output_root).strip()
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # The MANGO platform automatically extracts the uploaded zip file directly into the output_root.
    # We do NOT need to extract the zip file manually.
    base = os.path.join(output_root, 'Defect Detection', 'Dataset')

    if not os.path.exists(base):
        # Fallback debug message
        found_files = os.listdir(output_root) if os.path.exists(output_root) else []
        raise FileNotFoundError(
            f"Cannot find the expected image folders at {base}\n"
            f"Contents of {output_root} are: {found_files}\n"
            "Please check the folder structure."
        )

    print(f"Dataset successfully located at: {base}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    train_loaders = {}
    test_loaders = {}
    test_meta = {}

    for view in VIEWS:
        train_dir = os.path.join(base, 'train', view, 'good')
        test_dir = os.path.join(base, 'test', view)
        
        train_dataset = SingleClassImageFolder(root=train_dir, transform=train_transform)
        test_dataset = LabelledImageFolder(root=test_dir, transform=test_transform)
        
        train_loaders[view] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loaders[view] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        test_meta[view] = {'dataset_size': len(test_dataset)}

    artefacts_dir = os.path.join(output_root, 'saved_artefacts')
    os.makedirs(artefacts_dir, exist_ok=True)
    
    train_loaders_path = os.path.join(artefacts_dir, 'train_loaders.joblib')
    test_loaders_path = os.path.join(artefacts_dir, 'test_loaders.joblib')
    test_meta_path = os.path.join(artefacts_dir, 'test_meta.joblib')
    
    joblib.dump(train_loaders, train_loaders_path)
    joblib.dump(test_loaders, test_loaders_path)
    joblib.dump(test_meta, test_meta_path)
    
    # Print summary to MANGO Logs
    print("\n" + "="*50)
    print("           DATA PRE-PROCESSING SUMMARY")
    print("="*50)
    for view in VIEWS:
        train_len = len(train_loaders[view].dataset)
        test_len = len(test_loaders[view].dataset)
        print(f"View: {view.ljust(10)} | Train Set: {train_len} | Test Set: {test_len}")
    print("="*50 + "\n")
    print(f"Data pre-processing complete. Deliverables saved.")
    
    return train_loaders_path, test_loaders_path, test_meta_path

### **************************************************************************************************************************************

# Block 2 : Model Definition (PatchCore Anomaly Detector)
## Input
- *Data Method*: Static Data; *Data Type*: String; *Variable*: **output_root**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset;
- *Data Method*: Static Data; *Data Type*: String; *Variable*: **backbone_name**; *Default Value*: wide_resnet50_2;
- *Data Method*: Static Data; *Data Type*: Number; *Variable*: **coreset_ratio**; *Default Value*: 0.1;

## Output
*Type*: String; *Variable*: **model_config_path**;

## Code
### Model Definition Blueprint
class PatchCoreExtractor(nn.Module):
    def __init__(self, backbone_name='wide_resnet50_2'):
        super().__init__()
        weights_enum = tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = tv_models.wide_resnet50_2(weights=weights_enum)
        self.layer2 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2
        )
        self.layer3 = nn.Sequential(backbone.layer3)
        self.patch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        features = torch.cat([f2, f3_up], dim=1)
        return self.patch_pool(features)

def main(output_root, backbone_name='wide_resnet50_2', coreset_ratio=0.1):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = PatchCoreExtractor(backbone_name=backbone_name).to(device)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Feature extractor output shape: {out.shape}")

    artefacts_dir = os.path.join(output_root, 'saved_artefacts')
    os.makedirs(artefacts_dir, exist_ok=True)

    model_config = {
        'backbone_name': backbone_name,
        'feature_dim': 1536,
        'patch_h': 28,
        'patch_w': 28,
        'coreset_ratio': coreset_ratio,
        'image_size': 224,
    }

    model_config_path = os.path.join(artefacts_dir, 'model_config.joblib')
    joblib.dump(model_config, model_config_path)
    
    # Print summary to MANGO Logs
    print("\n" + "="*50)
    print("           MODEL CONFIGURATION SUMMARY")
    print("="*50)
    for k, v in model_config.items():
        print(f"{k.ljust(15)} : {v}")
    print("="*50 + "\n")
    print(f"Model config saved to {model_config_path}")
    return model_config_path

### **************************************************************************************************************************************

# Block 3 : Pre Train Prediction Visualization
## Input
- *Data Method*: Previous Block; *Data Type*: String; *Variable*: **model_config_path**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset/saved_artefacts/model_config.joblib;
- *Data Method*: Previous Block; *Data Type*: String; *Variable*: **test_loaders_path**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset/saved_artefacts/test_loaders.joblib;
- *Data Method*: Static Data; *Data Type*: String; *Variable*: **output_root**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset;

## Output
*Type*: String; *Variable*: **pretrain_vis_path**;

## Code
### Model Definition Blueprint
class PatchCoreExtractor(nn.Module):
    def __init__(self, backbone_name='wide_resnet50_2'):
        super().__init__()
        weights_enum = tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = tv_models.wide_resnet50_2(weights=weights_enum)
        self.layer2 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2
        )
        self.layer3 = nn.Sequential(backbone.layer3)
        self.patch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        features = torch.cat([f2, f3_up], dim=1)
        return self.patch_pool(features)

def compute_naive_scores(model, loader, device):
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            feats = model(imgs.to(device))
            score = feats.pow(2).mean(dim=[1, 2, 3]).cpu().numpy()
            scores.extend(score.tolist())
            labels.extend(lbls.numpy().tolist())
    return np.array(scores), np.array(labels)

def main(model_config_path, test_loaders_path, output_root):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    mc = joblib.load(model_config_path)
    test_loaders = joblib.load(test_loaders_path)

    model = PatchCoreExtractor(backbone_name=mc['backbone_name']).to(device)
    VIEWS = ['bottom', 'side', 'top']
    
    outputs_dir = os.path.join(output_root, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(VIEWS), figsize=(16, 4.5))
    fig.suptitle('Block 3 — Pre-Train Naive Score Distribution (Expected Overlap)', fontsize=12)

    for ax, view in zip(axes, VIEWS):
        scores, labels = compute_naive_scores(model, test_loaders[view], device)
        ax.hist(scores[labels == 0], bins=25, alpha=0.70, color='green', label='Good', density=True)
        ax.hist(scores[labels == 1], bins=25, alpha=0.70, color='red', label='Anomaly', density=True)
        ax.set_title(f'{view.capitalize()} View')
        ax.legend()

    pretrain_vis_path = os.path.join(outputs_dir, 'pretrain_score_distribution.png')
    plt.savefig(pretrain_vis_path, dpi=150, bbox_inches='tight')
    plt.show()  # Renders the graph directly in MANGO UI
    plt.close()
    
    # Print summary to MANGO Logs
    print("\n" + "="*50)
    print("          PRE-TRAIN EXECUTION SUMMARY")
    print("="*50)
    print("Naive distance scores calculated for all views.")
    print("Pre-training visualization plot generated.")
    print("="*50 + "\n")
    print(f"Pre-train visualisation saved to {pretrain_vis_path}")
    return pretrain_vis_path

### **************************************************************************************************************************************

# Block 4 : Training
## Input
- *Data Method*: Previous Block; *Data Type*: String; *Variable*: **train_loaders_path**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset/saved_artefacts/train_loaders.joblib;
- *Data Method*: Previous Block; *Data Type*: String; *Variable*: **model_config_path**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset/saved_artefacts/model_config.joblib;
- *Data Method*: Static Data; *Data Type*: String; *Variable*: **output_root**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset;

## Output
*Type*: String; *Variable*: **memory_bank_path**;

## Code
### Model Definition Blueprint
class PatchCoreExtractor(nn.Module):
    def __init__(self, backbone_name='wide_resnet50_2'):
        super().__init__()
        weights_enum = tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = tv_models.wide_resnet50_2(weights=weights_enum)
        self.layer2 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2
        )
        self.layer3 = nn.Sequential(backbone.layer3)
        self.patch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        features = torch.cat([f2, f3_up], dim=1)
        return self.patch_pool(features)

def extract_patch_features(model, loader, device):
    all_patches = []
    model.eval()
    with torch.no_grad():
        for imgs, _, _ in loader:
            feats = model(imgs.to(device))
            B, C, H, W = feats.shape
            patches = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            
            # CRITICAL OOM FIX: Subsample patches per-batch before appending.
            # This prevents the list from growing to 2GB+ and causing a SIGKILL.
            if patches.shape[0] > 1000:
                idx = np.random.choice(patches.shape[0], 1000, replace=False)
                patches = patches[idx]
                
            all_patches.append(patches)
            
    features = np.concatenate(all_patches, axis=0)
    
    # CRITICAL SPEED OPTIMIZATION FOR CLOUD CPUs
    # If there are >30,000 patches, randomly drop the excess before greedy coreset.
    # This prevents the platform from taking 30+ minutes on O(N^2) distance calculations.
    max_patches = 30000
    if features.shape[0] > max_patches:
        idx = np.random.choice(features.shape[0], max_patches, replace=False)
        features = features[idx]
        
    return features

def greedy_coreset_subsample(features, coreset_ratio=0.1, random_state=42):
    # Improvement 1: Hardware Accelerated Subsampling
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(random_state)
    
    N = features.shape[0]
    M = max(1, int(coreset_ratio * N))
    
    features_t = torch.tensor(features, device=device)
    selected = [torch.randint(0, N, (1,)).item()]
    min_dists = torch.full((N,), float('inf'), device=device)
    
    for step in range(1, M):
        last = features_t[selected[-1]:selected[-1]+1]
        dists = torch.sum((features_t - last)**2, dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(int(torch.argmax(min_dists).item()))
        
    return features_t[selected].cpu().numpy()

def main(train_loaders_path, model_config_path, output_root):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    mc = joblib.load(model_config_path)
    train_loaders = joblib.load(train_loaders_path)
    model = PatchCoreExtractor(backbone_name=mc['backbone_name']).to(device)
    
    VIEWS = ['bottom', 'side', 'top']
    memory_banks = {}

    for view in VIEWS:
        print(f"Constructing memory bank for view: {view} ...")
        all_features = extract_patch_features(model, train_loaders[view], device)
        coreset = greedy_coreset_subsample(all_features, coreset_ratio=mc['coreset_ratio'])
        memory_banks[view] = coreset

    artefacts_dir = os.path.join(output_root, 'saved_artefacts')
    os.makedirs(artefacts_dir, exist_ok=True)
    memory_bank_path = os.path.join(artefacts_dir, 'memory_bank.joblib')
    joblib.dump(memory_banks, memory_bank_path)
    
    # Print summary to MANGO Logs
    print("\n" + "="*50)
    print("       MEMORY BANK CONSTRUCTION SUMMARY")
    print("="*50)
    for view in VIEWS:
        print(f"View: {view.ljust(10)} | Final Memory Bank Size: {len(memory_banks[view])} patches")
    print("="*50 + "\n")
    print(f"Memory bank saved to {memory_bank_path}")
    return memory_bank_path

### **************************************************************************************************************************************

# Block 5 : Post Train Prediction Visualization
## Input
- *Data Method*: Previous Block; *Data Type*: String; *Variable*: **memory_bank_path**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset/saved_artefacts/memory_bank.joblib;
- *Data Method*: Previous Block; *Data Type*: String; *Variable*: **test_loaders_path**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset/saved_artefacts/test_loaders.joblib;
- *Data Method*: Previous Block; *Data Type*: String; *Variable*: **model_config_path**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset/saved_artefacts/model_config.joblib;
- *Data Method*: Static Data; *Data Type*: String; *Variable*: **output_root**; *Default Value*: /shared/datasource/mukilan.s.2024.aiml@rajalakshmi.edu.in/Project2_DefectDataset;

## Output
*Type*: String; *Variable*: **trained_vis_save_path**;

## Code
### Model Definition Blueprint
class PatchCoreExtractor(nn.Module):
    def __init__(self, backbone_name='wide_resnet50_2'):
        super().__init__()
        weights_enum = tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = tv_models.wide_resnet50_2(weights=weights_enum)
        self.layer2 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2
        )
        self.layer3 = nn.Sequential(backbone.layer3)
        self.patch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        features = torch.cat([f2, f3_up], dim=1)
        return self.patch_pool(features)

def nn_distance_scores(query_features, memory_bank, k=5, chunk_size=256):
    # Improvement 1 & 2: Hardware Accelerated K-NN
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    q_t = torch.tensor(query_features, device=device)
    mb_t = torch.tensor(memory_bank, device=device)
    
    dists = []
    for i in range(0, len(q_t), chunk_size):
        q_chunk = q_t[i:i + chunk_size]
        d = torch.cdist(q_chunk, mb_t)
        # K-Nearest Neighbors Average
        topk_dists, _ = torch.topk(d, k=k, dim=1, largest=False)
        avg_dists = topk_dists.mean(dim=1)
        dists.append(avg_dists.cpu().numpy())
    return np.concatenate(dists)

def run_inference(model, test_loaders, memory_banks, device):
    VIEWS = ['bottom', 'side', 'top']
    results = {v: {'scores': [], 'labels': [], 'paths': [], 'score_maps': []} for v in VIEWS}
    model.eval()
    
    for view in VIEWS:
        mb = memory_banks[view]
        with torch.no_grad():
            for imgs, lbls, paths in test_loaders[view]:
                for img_t, lbl, path in zip(imgs, lbls, paths):
                    feats = model(img_t.unsqueeze(0).to(device))
                    B, C, H, W = feats.shape
                    patches = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
                    
                    patch_dists = nn_distance_scores(patches, mb, k=5)
                    score_map = patch_dists.reshape(H, W)
                    
                    # Improvement 3: Gaussian Smoothing
                    score_map_smoothed = gaussian_filter(score_map, sigma=1.5)
                    image_score = np.max(score_map_smoothed)
                    
                    results[view]['scores'].append(image_score)
                    results[view]['labels'].append(lbl.item())
                    results[view]['paths'].append(path)
                    results[view]['score_maps'].append(score_map_smoothed)
                    
        results[view]['scores'] = np.array(results[view]['scores'])
        results[view]['labels'] = np.array(results[view]['labels'])
    
    return results

def main(memory_bank_path, test_loaders_path, model_config_path, output_root):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    memory_banks = joblib.load(memory_bank_path)
    test_loaders = joblib.load(test_loaders_path)
    mc = joblib.load(model_config_path)
    
    model = PatchCoreExtractor(backbone_name=mc['backbone_name']).to(device)
    VIEWS = ['bottom', 'side', 'top']
    
    outputs_dir = os.path.join(output_root, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("Running inference and extracting metrics (this may take a moment)...")
    results = run_inference(model, test_loaders, memory_banks, device)
    
    fig, axes = plt.subplots(1, len(VIEWS), figsize=(16, 4.5))
    fig.suptitle('Block 5 — Trained PatchCore F1-Score Distribution', fontsize=12)
    
    thresholds = {}
    for ax, view in zip(axes, VIEWS):
        sc = results[view]['scores']
        lbl = results[view]['labels']
        
        # Calculate optimal threshold
        precision, recall, ths = precision_recall_curve(lbl, sc)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1)
        thresholds[view] = ths[best_idx] if best_idx < len(ths) else ths[-1]
        
        sc, lbl = results[view]['scores'], results[view]['labels']
        ax.hist(sc[lbl == 0], bins=28, alpha=0.70, color='green', label='Good', density=True)
        ax.hist(sc[lbl == 1], bins=28, alpha=0.70, color='red', label='Anomaly', density=True)
        ax.axvline(thresholds[view], color='black', linestyle='--', label=f'Threshold={thresholds[view]:.3f}')
        ax.set_title(f'{view.capitalize()} View')
        ax.legend()
        
    trained_vis_save_path = os.path.join(outputs_dir, 'trained_score_distribution.png')
    plt.savefig(trained_vis_save_path, dpi=150, bbox_inches='tight')
    plt.show()  # Renders the graph directly in MANGO UI
    plt.close()
    
    # Save Metrics CSV
    rows = []
    for view in VIEWS:
        lbl = results[view]['labels']
        preds = (results[view]['scores'] >= thresholds[view]).astype(int)
        precision = precision_score(lbl, preds, zero_division=0)
        recall = recall_score(lbl, preds, zero_division=0)
        f1 = f1_score(lbl, preds, zero_division=0)
        rows.append({'View': view, 'Threshold': thresholds[view], 'Precision': precision, 'Recall': recall, 'F1': f1})
    
    metrics_path = os.path.join(outputs_dir, 'metrics.csv')
    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False)
    
    # Print the table to standard output so it shows up in the MANGO UI logs!
    print("\n" + "="*60)
    print("                 FINAL EVALUATION METRICS")
    print("="*60)
    print(df.to_string())
    print("="*60 + "\n")
    
    print(f"Evaluation complete. Deliverables saved to {outputs_dir}")
    return trained_vis_save_path

### **************************************************************************************************************************************
