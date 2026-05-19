#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the optional left / right / rest EEG model for arm rotation mode."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from brain_arm_control import build_feature_vector


LOGGER = logging.getLogger("train_eeg_model")

LABEL_FILES = (
    ("left", 0, "actionleft.txt"),
    ("right", 1, "actionright.txt"),
    ("rest", 2, "rest.txt"),
)

FEATURE_NAMES = [
    "attention",
    "meditation",
    "delta",
    "theta",
    "low_alpha",
    "high_alpha",
    "low_beta",
    "high_beta",
    "low_gamma",
    "mid_gamma",
    "blink_strength",
    "beta_theta_ratio",
    "alpha_theta_ratio",
    "engagement",
]


@dataclass
class TrainingConfig:
    data_dir: Path = Path("data")
    output: Path = Path("model/FinalModel.pth")
    window_size: int = 30
    stride: int = 1
    poor_signal_threshold: int = 20
    epochs: int = 150
    patience: int = 25
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.008
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    log_level: str = "INFO"

    def validate(self) -> None:
        self.window_size = max(2, int(self.window_size))
        self.stride = max(1, int(self.stride))
        self.poor_signal_threshold = max(0, int(self.poor_signal_threshold))
        self.epochs = max(1, int(self.epochs))
        self.patience = max(1, int(self.patience))
        self.batch_size = max(2, int(self.batch_size))
        self.learning_rate = max(1e-6, float(self.learning_rate))
        self.weight_decay = max(0.0, float(self.weight_decay))
        if not 0.1 <= self.train_ratio < 1.0:
            raise ValueError("--train-ratio must be in [0.1, 1.0)")
        if not 0.0 <= self.val_ratio < 0.5:
            raise ValueError("--val-ratio must be in [0.0, 0.5)")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("--train-ratio + --val-ratio must be < 1.0")


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_feature_line(line: str, poor_signal_threshold: int) -> Optional[List[float]]:
    parts = line.strip().split(",")
    if len(parts) != 2:
        return None
    raw = parts[1].split("|")
    if len(raw) != 12:
        return None
    try:
        values = list(map(float, raw))
    except ValueError:
        return None

    (
        attention,
        meditation,
        delta,
        theta,
        low_alpha,
        high_alpha,
        low_beta,
        high_beta,
        low_gamma,
        mid_gamma,
        poor_signal,
        blink_strength,
    ) = values

    if poor_signal > poor_signal_threshold:
        return None
    if attention == 0 or meditation == 0:
        return None

    return build_feature_vector(
        int(attention),
        int(meditation),
        int(delta),
        int(theta),
        int(low_alpha),
        int(high_alpha),
        int(low_beta),
        int(high_beta),
        int(low_gamma),
        int(mid_gamma),
        int(blink_strength),
    )


def load_windowed_samples(
    np: Any,
    file_path: Path,
    label: int,
    *,
    window_size: int,
    stride: int,
    poor_signal_threshold: int,
) -> List[Tuple[Any, int]]:
    if not file_path.exists():
        LOGGER.warning("Missing data file: %s", file_path)
        return []

    features = [
        feature
        for feature in (
            parse_feature_line(line, poor_signal_threshold)
            for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        )
        if feature is not None
    ]
    if len(features) < window_size:
        LOGGER.warning(
            "%s has only %d valid rows, fewer than window_size=%d",
            file_path,
            len(features),
            window_size,
        )
        return []

    samples: List[Tuple[Any, int]] = []
    for start in range(0, len(features) - window_size + 1, stride):
        window = np.array(features[start : start + window_size], dtype=np.float32)
        combined = np.concatenate([window.mean(axis=0), window.std(axis=0)], axis=0)
        samples.append((combined, label))
    return samples


def load_all_samples(np: Any, config: TrainingConfig) -> List[Tuple[Any, int]]:
    all_samples: List[Tuple[Any, int]] = []
    for label_name, label_id, filename in LABEL_FILES:
        path = config.data_dir / filename
        samples = load_windowed_samples(
            np,
            path,
            label_id,
            window_size=config.window_size,
            stride=config.stride,
            poor_signal_threshold=config.poor_signal_threshold,
        )
        LOGGER.info("%-5s label=%d samples=%d source=%s", label_name, label_id, len(samples), path)
        all_samples.extend(samples)
    return all_samples


def stratified_split(
    samples: Sequence[Tuple[Any, int]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]], List[Tuple[Any, int]]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[Tuple[Any, int]]] = {}
    for features, label in samples:
        by_label.setdefault(label, []).append((features, label))

    train_samples: List[Tuple[Any, int]] = []
    val_samples: List[Tuple[Any, int]] = []
    test_samples: List[Tuple[Any, int]] = []

    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        count = len(items)
        if count < 3:
            train_samples.extend(items)
            continue
        train_count = max(1, int(count * train_ratio))
        val_count = int(count * val_ratio)
        if val_ratio > 0 and val_count == 0 and count - train_count > 1:
            val_count = 1
        if train_count + val_count >= count:
            val_count = max(0, count - train_count - 1)
        train_samples.extend(items[:train_count])
        val_samples.extend(items[train_count : train_count + val_count])
        test_samples.extend(items[train_count + val_count :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def stack_features(np: Any, samples: Sequence[Tuple[Any, int]]) -> Any:
    return np.stack([sample[0] for sample in samples], axis=0)


def class_weights(np: Any, torch: Any, samples: Sequence[Tuple[Any, int]]) -> Any:
    counts = np.bincount([label for _features, label in samples], minlength=3)
    weights = counts.sum() / (counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def make_dataset_class_with_numpy(torch: Any, np: Any):
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, samples: Sequence[Tuple[Any, int]], mean: Any, std: Any, augment: bool = False):
            self.samples = list(samples)
            self.mean = mean
            self.std = std
            self.augment = augment

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> Tuple[Any, int]:
            features, label = self.samples[index]
            normalized = (features - self.mean) / (self.std + 1e-8)
            if self.augment:
                normalized = normalized + np.random.normal(0, 0.008, normalized.shape)
            return torch.tensor(normalized, dtype=torch.float32), int(label)

    return FeatureDataset


def build_final_model(torch: Any, input_dim: int) -> Any:
    nn = torch.nn

    class FinalUnifiedModel(nn.Module):
        def __init__(self, model_input_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(model_input_dim, 384)
            self.bn1 = nn.BatchNorm1d(384)
            self.dropout1 = nn.Dropout(0.12)
            self.fc2 = nn.Linear(384, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.dropout2 = nn.Dropout(0.15)
            self.fc3 = nn.Linear(256, 256)
            self.bn3 = nn.BatchNorm1d(256)
            self.dropout3 = nn.Dropout(0.15)
            self.fc4 = nn.Linear(256, 128)
            self.bn4 = nn.BatchNorm1d(128)
            self.dropout4 = nn.Dropout(0.18)
            self.fc5 = nn.Linear(128, 3)

        def forward(self, x: Any) -> Any:
            x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
            x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
            identity = x
            x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
            x = x + identity
            x = self.dropout4(torch.relu(self.bn4(self.fc4(x))))
            return self.fc5(x)

    return FinalUnifiedModel(input_dim)


def evaluate(torch: Any, model: Any, data_loader: Any, criterion: Any) -> Tuple[float, float]:
    if data_loader is None:
        return 0.0, 0.0
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            total_count += len(labels)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, correct / total_count


def copy_state_dict(model: Any) -> Dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run_training(config: TrainingConfig) -> None:
    try:
        import numpy as np
        import torch
    except Exception as exc:
        raise RuntimeError("训练模型需要安装 numpy 和 torch：pip install numpy torch") from exc

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    samples = load_all_samples(np, config)
    if not samples:
        raise RuntimeError("没有可用训练样本，请先运行 collect_training_data.py 采集数据。")

    labels_present = sorted({label for _features, label in samples})
    if labels_present != [0, 1, 2]:
        raise RuntimeError("训练数据必须同时包含 left/right/rest 三类样本。")

    train_samples, val_samples, test_samples = stratified_split(
        samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    if len(train_samples) < 2:
        raise RuntimeError("训练样本太少，至少需要 2 个训练窗口。")

    train_features = stack_features(np, train_samples)
    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = train_features.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    input_dim = int(train_features.shape[1])

    Dataset = make_dataset_class_with_numpy(torch, np)
    train_dataset = Dataset(train_samples, feature_mean, feature_std, augment=True)
    val_dataset = Dataset(val_samples, feature_mean, feature_std, augment=False) if val_samples else None
    test_dataset = Dataset(test_samples, feature_mean, feature_std, augment=False) if test_samples else None

    drop_last = len(train_dataset) > config.batch_size and len(train_dataset) % config.batch_size == 1
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = (
        torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )
    test_loader = (
        torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        if test_dataset is not None
        else None
    )

    LOGGER.info(
        "samples: train=%d val=%d test=%d input_dim=%d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
        input_dim,
    )

    model = build_final_model(torch, input_dim)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(np, torch, train_samples))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, config.patience // 3),
    )

    best_metric = float("inf")
    best_epoch = 0
    best_state = copy_state_dict(model)
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        correct = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_count += len(labels)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / max(total_count, 1)
        train_acc = correct / max(total_count, 1)
        val_loss, val_acc = evaluate(torch, model, val_loader, criterion)
        metric = val_loss if val_loader is not None else train_loss
        scheduler.step(metric)

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = copy_state_dict(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            LOGGER.info(
                "epoch=%03d train_loss=%.4f train_acc=%.2f%% val_loss=%.4f val_acc=%.2f%% patience=%d/%d",
                epoch,
                train_loss,
                train_acc * 100,
                val_loss,
                val_acc * 100,
                patience_counter,
                config.patience,
            )

        if patience_counter >= config.patience:
            LOGGER.info("Early stopping at epoch %d; best epoch=%d", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(torch, model, test_loader, criterion)
    train_loss, train_acc = evaluate(torch, model, train_loader, criterion)
    val_loss, val_acc = evaluate(torch, model, val_loader, criterion)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "input_dim": input_dim,
        "labels": {"0": "left", "1": "right", "2": "rest"},
        "feature_names": FEATURE_NAMES,
        "window_size": config.window_size,
        "stride": config.stride,
        "best_epoch": best_epoch,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy_strict": test_acc,
        "test_accuracy_relaxed": test_acc,
        "model_architecture": "FinalUnifiedModel",
        "training_method": "single supervised model for arm rotation mode",
    }
    torch.save(checkpoint, config.output)

    LOGGER.info("模型已保存: %s", config.output.resolve())
    LOGGER.info("accuracy: train=%.2f%% val=%.2f%% test=%.2f%%", train_acc * 100, val_acc * 100, test_acc * 100)
    LOGGER.info("该模型只用于 mode 1 左右旋转: 0=left, 1=right, 2=rest")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FinalModel.pth for the optional mode-1 left/right/rest classifier."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="model/FinalModel.pth")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--poor-signal-threshold", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.008)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        data_dir=Path(args.data_dir),
        output=Path(args.output),
        window_size=args.window_size,
        stride=args.stride,
        poor_signal_threshold=args.poor_signal_threshold,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        log_level=args.log_level,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        config = config_from_args(args)
        config.validate()
        run_training(config)
        return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
