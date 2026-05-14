# GalaxEye Binary Change Detection on EO-SAR Image Pairs

## Overview

This repository contains my solution to the GalaxEye AI Research Intern technical assignment on binary change detection using paired pre-event and post-event EO-SAR satellite imagery.

The objective is to classify each pixel as:

- **0** → No Change
- **1** → Change

A Siamese-style U-Net with a ResNet34 encoder was implemented using PyTorch and segmentation_models_pytorch. The model takes concatenated pre-event and post-event images as input and predicts a binary change mask.

---

## Repository Structure

```text
galaxeye-binary-change-detection/
├── configs/
│   └── config.yaml
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── utils.py
│   ├── train.py
│   └── eval.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- segmentation-models-pytorch
- albumentations
- rasterio
- NumPy
- scikit-learn
- matplotlib
- pandas
- PyYAML
- tqdm

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows (Command Prompt)

```cmd
venv\Scripts\activate.bat
```

#### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Place the provided dataset in the following structure:

```text
data/
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

---

## Label Remapping

The original annotations contain four classes. They are remapped to binary labels as follows:

| Original Value | Original Class | Binary Value | Binary Class |
|---------------:|----------------|-------------:|--------------|
| 0 | Background | 0 | No Change |
| 1 | Intact | 0 | No Change |
| 2 | Damaged | 1 | Change |
| 3 | Destroyed | 1 | Change |

---

## Model Architecture

### Siamese U-Net with ResNet34 Encoder

The implemented model uses:

- U-Net decoder from `segmentation_models_pytorch`
- ResNet34 encoder pretrained on ImageNet
- 4 input channels:
  - 2 channels from pre-event image
  - 2 channels from post-event image
- 1 output channel representing change probability

### Loss Function

A combination of:

- Binary Cross Entropy Loss
- Dice Loss

### Optimizer

- AdamW

---

## Training Configuration

| Parameter | Value |
|---------|------:|
| Image Size | 256 × 256 |
| Batch Size | 4 |
| Epochs | 10 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Threshold | 0.5 |
| Random Seed | 42 |
| Encoder | ResNet34 |
| Encoder Weights | ImageNet |

---

## Configuration File

All hyperparameters are stored in:

~~~text
configs/config.yaml
~~~

---

## Training

Run the following command from the project root:

```bash
cd src
python train.py
```

The best model checkpoint is saved automatically to:

```text
outputs/checkpoints/best_model.pth
```

---

## Evaluation

Run:

```bash
cd src
python eval.py
```

This evaluates the saved checkpoint on the test split and reports:

- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Model Weights

Download the trained model weights from:

**Replace this with your public Google Drive or Hugging Face link**

---

## Results

### Validation Set (Best Checkpoint)

| Metric | Score |
|------:|------:|
| IoU | 0.2327 |
| Precision | 0.3918 |
| Recall | 0.3643 |
| F1 Score | 0.3776 |

### Test Set

| Metric | Score |
|------:|------:|
| IoU | 0.0053 |
| Precision | 0.0110 |
| Recall | 0.0102 |
| F1 Score | 0.0106 |

---

## Discussion

The model achieved a validation F1 score of 0.3776, demonstrating that it successfully learned meaningful change patterns from the training data.

However, the model generalized poorly to the held-out test set, achieving an F1 score of 0.0106. This indicates significant domain shift between the training/validation and test distributions, which is a common challenge in remote sensing change detection.

Potential causes include:

- Different disaster types and geographic regions
- Variations in sensor characteristics
- Severe class imbalance
- Limited training time
- Use of a relatively simple baseline architecture

---

## Future Work

Potential improvements include:

- Class-weighted BCE or Focal Loss
- Threshold tuning on validation data
- Test-time augmentation
- Event-specific normalization
- More advanced architectures such as:
  - SNUNet
  - ChangeFormer
  - BIT
  - FCCDN

---

## Time and Resource Summary

- Hardware: CPU-based training on a Windows machine
- Average training time per epoch: ~15 minutes
- Total training time: ~2.5 hours
- Framework: PyTorch

---

## References

1. U-Net: https://arxiv.org/abs/1505.04597
2. ResNet: https://arxiv.org/abs/1512.03385
3. segmentation_models_pytorch: https://github.com/qubvel/segmentation_models.pytorch
4. GalaxEye Technical Assignment Document