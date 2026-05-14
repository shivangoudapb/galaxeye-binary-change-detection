import os
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import ChangeDetectionDataset
from model import SiameseUNet
from losses import BCEDiceLoss
from metrics import compute_metrics
from utils import set_seed, save_checkpoint


def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return compute_metrics(all_preds, all_targets)


def main():
    with open("../configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_dataset = ChangeDetectionDataset(
        os.path.join(cfg["data_root"], "train"),
        image_size=cfg["image_size"],
        augment=True,
    )

    val_dataset = ChangeDetectionDataset(
        os.path.join(cfg["data_root"], "val"),
        image_size=cfg["image_size"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    model = SiameseUNet(
        encoder_name=cfg["encoder_name"],
        encoder_weights=cfg["encoder_weights"],
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_f1 = 0.0

    checkpoint_path = os.path.join(
        "..",
        cfg["save_dir"],
        "checkpoints",
        "best_model.pth",
    )

    if os.path.exists(checkpoint_path):
        print("Loading existing checkpoint...")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        metrics = evaluate(
            model,
            val_loader,
            device,
            threshold=cfg["threshold"],
        )

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val IoU: {metrics['iou']:.4f}")
        print(f"Val Precision: {metrics['precision']:.4f}")
        print(f"Val Recall: {metrics['recall']:.4f}")
        print(f"Val F1: {metrics['f1']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]

            save_checkpoint(
                model,
                os.path.join(
                    "..",
                    cfg["save_dir"],
                    "checkpoints",
                    "best_model.pth",
                ),
            )

            print("Saved best model.")

    print(f"\nBest Validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()