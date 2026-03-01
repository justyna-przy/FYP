import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd 
import torch
import torch.nn.functional as F

from src.config import CONFIG
from src.dataset.dataloader import (AugmentConfig, LoaderConfig,
                                    make_dataloaders)
from src.models.bird_resnet import BirdResNet


def resolve_num_workers(requested_num_workers: int) -> int:
    if requested_num_workers <= 0:
        return 0

    shm_manager = Path(torch.__file__).resolve().parent / "bin" / "torch_shm_manager"
    if not shm_manager.exists():
        return requested_num_workers

    if os.access(shm_manager, os.X_OK):
        return requested_num_workers

    print(
        "Warning: torch_shm_manager is not executable at "
        f"{shm_manager}. Falling back to --num-workers 0."
    )
    return 0


def resolve_data_dir(config_data_dir: str) -> Path:
    configured = Path(config_data_dir)
    repo_root = Path(__file__).resolve().parents[1]

    if configured.is_absolute():
        candidates = [configured]
    else:
        candidates = [
            Path.cwd() / configured,
            repo_root / configured,
            repo_root / "src" / "dataset" / configured,
        ]

    # Prefer a directory that already has training splits.
    for candidate in candidates:
        split_marker = candidate / "spectrograms_v1" / "splits" / "classes.json"
        if split_marker.exists():
            return candidate.resolve()

    # Fallback to first existing directory.
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Last resort: preserve original configured location under repo root.
    return (repo_root / configured).resolve()


def resolve_device(device_arg: str, no_cuda: bool) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    # auto mode: prefer CUDA, then Apple MPS, then CPU
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(train_csv: Path, classes_json: Path, device: torch.device) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    class_to_idx: Dict[str, int] = json.loads(classes_json.read_text(encoding="utf-8"))

    # Count occurrences of each class
    counts: Dict[str, int] = df["class_name"].value_counts().to_dict()
    num_classes = len(class_to_idx)
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls, idx in class_to_idx.items():
        count = counts.get(cls, 0)
        # Avoid division by zero; if a class is absent (should not happen) use a tiny count
        weights[idx] = 1.0 / max(count, 1e-6)
    # Normalize so that mean weight is 1.0
    weights *= float(num_classes) / weights.sum()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model: BirdResNet,
                    dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    class_weights: torch.Tensor,
                    device: torch.device) -> Tuple[float, float, float]:
    model.train()
    losses = []
    all_preds = []
    all_targets = []
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    # Compute metrics on CPU
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    return float(np.mean(losses)), acc, macro_f1


@torch.no_grad()
def evaluate(model: BirdResNet,
             dataloader: torch.utils.data.DataLoader,
             class_weights: torch.Tensor,
             device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets, weight=class_weights)
        losses.append(loss.item())
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    # Compute metrics on CPU
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    return float(np.mean(losses)), acc, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BirdResNet on bird spectrogram dataset")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="number of DataLoader workers")
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA training")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="compute device (default: auto -> cuda, mps, cpu)",
    )
    parser.add_argument("--out-dir", type=str, default="runs/bird_resnet_v1",
                        help="output directory for checkpoints and logs")
    args = parser.parse_args()

    device = resolve_device(args.device, args.no_cuda)
    print(f"Using device: {device}")

    # Dataset and split paths
    data_dir = resolve_data_dir(CONFIG.paths.data_dir)
    spec_root = data_dir / "spectrograms_v1"
    splits_dir = spec_root / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    classes_json = splits_dir / "classes.json"
    print(f"Using data dir: {data_dir}")
    num_workers = resolve_num_workers(args.num_workers)
    if num_workers != args.num_workers:
        print(f"Using num_workers={num_workers} (requested {args.num_workers})")
    # Build data loaders (augmentation disabled for now)
    loader_cfg = LoaderConfig(batch_size=args.batch_size,
                              num_workers=num_workers,
                              pin_memory=(device.type == "cuda"))
    augment_cfg = AugmentConfig(enable=False)  # no augmentation
    loaders = make_dataloaders(train_csv=train_csv, val_csv=val_csv, test_csv=None,
                               classes_json=classes_json, loader_cfg=loader_cfg,
                               aug_cfg=augment_cfg, data_dir=data_dir, seed=22)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    # Compute class weights
    class_weights = compute_class_weights(train_csv, classes_json, device)
    # Model
    num_classes = len(json.loads(classes_json.read_text()))
    model = BirdResNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer,
                                                 class_weights, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, class_weights, device)
        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "args": vars(args)
            }
            torch.save(ckpt, out_dir / "best.pt")
    print(f"Best validation F1: {best_val_f1:.3f}")


if __name__ == "__main__":
    main()
