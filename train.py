"""
MLP classifier for ASL hand signs.

Usage:
    # Train
    python train.py

    # Inference on a single image
    python train.py --infer path/to/image.jpg --checkpoint checkpoints/best.pt
"""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BATCH_SIZE, CKPT_DIR, DEVICE, EPOCHS, HIDDEN, IMG_SIZE, LR, DATA_ROOT
from model import MLP


def get_transforms(train: bool) -> transforms.Compose:
    t = [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    if train:
        t = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ] + t
    return transforms.Compose(t)


def load_data() -> tuple[DataLoader, DataLoader, list[str]]:
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=get_transforms(True))
    test_ds = datasets.ImageFolder(DATA_ROOT / "test", transform=get_transforms(False))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, train_ds.classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def plot_results(history: dict, save_path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()


def train() -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    train_loader, val_loader, classes = load_data()
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Device: {DEVICE}")

    model = MLP(IMG_SIZE * IMG_SIZE, HIDDEN, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        flag = ""
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "classes": classes,
                    "hidden": HIDDEN,
                },
                CKPT_DIR / "best.pt",
            )
            flag = "  ← best"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train loss {t_loss:.4f} acc {t_acc:.4f} | "
            f"val loss {v_loss:.4f} acc {v_acc:.4f}{flag}"
        )

    torch.save(
        {
            "epoch": EPOCHS,
            "model": model.state_dict(),
            "classes": classes,
            "hidden": HIDDEN,
        },
        CKPT_DIR / "last.pt",
    )
    plot_results(history, CKPT_DIR / "training_curves.png")
    print(f"\nBest val accuracy: {best_acc:.4f}")


@torch.no_grad()
def infer(image_path: str, checkpoint: str) -> None:
    from PIL import Image

    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    hidden = ckpt.get("hidden", HIDDEN)

    model = MLP(IMG_SIZE * IMG_SIZE, hidden, len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(image_path)
    tensor = get_transforms(train=False)(img).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze()
    top5 = probs.topk(5)

    print(f"\nImage: {image_path}")
    print(f"Prediction: {classes[top5.indices[0]]}  ({top5.values[0]*100:.1f}%)")
    print("Top-5:")
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  {classes[idx]:>4s}  {prob*100:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "best.pt"))
    args = parser.parse_args()

    if args.infer:
        infer(args.infer, args.checkpoint)
    else:
        train()
