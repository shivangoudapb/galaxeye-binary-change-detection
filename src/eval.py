import os
import yaml
import torch
from torch.utils.data import DataLoader

from dataset import ChangeDetectionDataset
from model import SiameseUNet
from train import evaluate


def main():
    with open("../configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    test_dataset = ChangeDetectionDataset(
        os.path.join(cfg["data_root"], "test"),
        image_size=cfg["image_size"],
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    model = SiameseUNet(
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
    ).to(device)

    checkpoint_path = os.path.join(
        "..",
        cfg["save_dir"],
        "checkpoints",
        "best_model.pth",
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    metrics = evaluate(
        model,
        test_loader,
        device,
        threshold=cfg["threshold"],
    )

    print("\nTest Results")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()