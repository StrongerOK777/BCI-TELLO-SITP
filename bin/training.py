"""Reusable training utilities for the BCI classifier pipeline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg import build_feature_vector
from .models import FinalUnifiedModel, combine_feature_window


Sample = Tuple[np.ndarray, int]


@dataclass
class RecordedTrial:
    """One contiguous recording block (one cued round of train_user.py)."""

    label: int
    rows: np.ndarray  # (n_rows, 14) feature rows from build_feature_vector
    start_time: float


def parse_feature_line(line: str, poor_signal_threshold: int = 20) -> Optional[List[float]]:
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
    file_path: str | Path,
    label: int,
    *,
    window_size: int = 20,
    stride: int = 1,
) -> List[Sample]:
    path = Path(file_path)
    features_list = [parse_feature_line(line) for line in path.read_text().splitlines()]
    features = [features for features in features_list if features is not None]
    samples: List[Sample] = []
    if len(features) < window_size:
        return samples
    for start in range(0, len(features) - window_size + 1, stride):
        window = np.array(features[start : start + window_size], dtype=np.float32)
        combined = np.concatenate([window.mean(axis=0), window.std(axis=0)], axis=0)
        samples.append((combined, label))
    return samples


def stratified_split(
    samples: Sequence[Sample],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[Sample]] = {}
    for features, label in samples:
        by_label.setdefault(label, []).append((features, label))
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    test_samples: List[Sample] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        count = len(items)
        train_count = int(count * train_ratio)
        val_count = int(count * val_ratio)
        train_samples.extend(items[:train_count])
        val_samples.extend(items[train_count : train_count + val_count])
        test_samples.extend(items[train_count + val_count :])
    return train_samples, val_samples, test_samples


def load_trials(
    file_path: str | Path,
    label: int,
    *,
    gap_sec: float = 1.0,
    max_rows: int = 30,
) -> List[RecordedTrial]:
    """Split a recording file into trials so windows never span two rounds.

    A new trial starts when the timestamp jumps by more than ``gap_sec`` (the
    countdown between rounds), goes backwards (a later session appended to the
    same file), or the current trial already holds ``max_rows`` rows.
    """

    trials: List[RecordedTrial] = []
    rows: List[List[float]] = []
    start_time = 0.0
    last_time: Optional[float] = None

    def flush() -> None:
        if rows:
            trials.append(RecordedTrial(label, np.array(rows, dtype=np.float32), start_time))

    for line in Path(file_path).read_text().splitlines():
        features = parse_feature_line(line)
        if features is None:
            continue
        try:
            timestamp = float(line.split(",", 1)[0])
        except ValueError:
            continue
        new_trial = (
            last_time is None
            or timestamp - last_time > gap_sec
            or timestamp < last_time
            or len(rows) >= max_rows
        )
        if new_trial:
            flush()
            rows = []
            start_time = timestamp
        rows.append(features)
        last_time = timestamp
    flush()
    return trials


def trial_grouped_split(
    trials: Sequence[RecordedTrial],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[RecordedTrial], List[RecordedTrial], List[RecordedTrial]]:
    """Stratified split at trial level: every trial lands in exactly one set.

    Each class keeps at least one test trial (and one validation trial when it
    has three or more trials), so accuracy is always measured on unseen rounds.
    """

    rng = random.Random(seed)
    by_label: Dict[int, List[RecordedTrial]] = {}
    for trial in trials:
        by_label.setdefault(trial.label, []).append(trial)
    train: List[RecordedTrial] = []
    val: List[RecordedTrial] = []
    test: List[RecordedTrial] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        count = len(items)
        n_test = max(1, round(count * (1 - train_ratio - val_ratio))) if count >= 2 else 0
        n_val = max(1, round(count * val_ratio)) if count >= 3 else 0
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    return train, val, test


def trial_windows(
    trials: Sequence[RecordedTrial],
    *,
    window_size: int = 30,
    stride: int = 5,
) -> List[Sample]:
    """Mean+std windows that slide only inside one trial.

    A trial shorter than ``window_size`` contributes one window over all of its
    rows, provided it has at least half a window; shorter trials are skipped.
    """

    samples: List[Sample] = []
    for trial in trials:
        count = len(trial.rows)
        if count < max(1, window_size // 2):
            continue
        size = min(window_size, count)
        for start in range(0, count - size + 1, max(1, stride)):
            samples.append((combine_feature_window(trial.rows[start : start + size]), trial.label))
    return samples


def significance_threshold(n: int, n_classes: int, alpha: float = 0.05) -> float:
    """Smallest accuracy k/n with P(X >= k) <= alpha under chance (X ~ Bin(n, 1/n_classes)).

    Accuracy at or above this value on ``n`` held-out trials is significantly
    better than guessing at level ``alpha``.
    """

    if n <= 0 or n_classes < 2:
        return 1.0
    p = 1.0 / n_classes
    log_p, log_q, log_n_fact = math.log(p), math.log(1 - p), math.lgamma(n + 1)
    tail = 0.0
    # accumulate P(X >= k) from the top down; stop before it exceeds alpha
    for k in range(n, -1, -1):
        log_term = log_n_fact - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * log_p + (n - k) * log_q
        next_tail = tail + math.exp(log_term)
        if next_tail > alpha:
            return min(1.0, (k + 1) / n)
        tail = next_tail
    return 0.0


def feature_stats(samples: Sequence[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std of the raw training vectors (what ModelPredictor uses)."""

    matrix = np.stack([features for features, _ in samples], axis=0).astype(np.float32)
    return matrix.mean(axis=0), matrix.std(axis=0)


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[Sample], mean: np.ndarray, std: np.ndarray, augment: bool = False):
        self.samples = list(samples)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        features, label = self.samples[index]
        normalized = (features - self.mean) / (self.std + 1e-8)
        if self.augment:
            normalized = normalized + np.random.normal(0, 0.008, normalized.shape)
        return torch.tensor(normalized, dtype=torch.float32), label


class WideModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


class DeepModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 192)
        self.bn2 = nn.BatchNorm1d(192)
        self.dropout2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(192, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.15)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        return self.fc5(x)


class BalancedModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(0.12)
        self.fc2 = nn.Linear(384, 192)
        self.bn2 = nn.BatchNorm1d(192)
        self.dropout2 = nn.Dropout(0.18)
        self.fc3 = nn.Linear(192, 96)
        self.bn3 = nn.BatchNorm1d(96)
        self.dropout3 = nn.Dropout(0.18)
        self.fc4 = nn.Linear(96, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)


class ResidualModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.15)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        out = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        identity = out
        out = self.dropout2(F.relu(self.bn2(self.fc2(out))))
        out = out + identity
        out = self.dropout3(F.relu(self.bn3(self.fc3(out))))
        return self.fc4(out)


class LightModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 320)
        self.bn1 = nn.BatchNorm1d(320)
        self.dropout1 = nn.Dropout(0.08)
        self.fc2 = nn.Linear(320, 160)
        self.bn2 = nn.BatchNorm1d(160)
        self.dropout2 = nn.Dropout(0.12)
        self.fc3 = nn.Linear(160, 80)
        self.bn3 = nn.BatchNorm1d(80)
        self.dropout3 = nn.Dropout(0.12)
        self.fc4 = nn.Linear(80, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)


MODEL_ARCHITECTURES = [
    ("Wide Model", WideModel),
    ("Deep Model", DeepModel),
    ("Balanced Model", BalancedModel),
    ("Residual Model", ResidualModel),
    ("Light Model", LightModel),
]


def distillation_loss(
    student_outputs: torch.Tensor,
    teacher_soft_labels: torch.Tensor,
    hard_labels: torch.Tensor,
    *,
    temperature: float = 3.0,
    alpha: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_soft = F.log_softmax(student_outputs / temperature, dim=1)
    soft_loss = F.kl_div(student_soft, teacher_soft_labels, reduction="batchmean")
    soft_loss = soft_loss * (temperature**2)
    hard_loss = F.cross_entropy(student_outputs, hard_labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss, soft_loss, hard_loss


def main() -> int:
    print(
        "bin.training exposes reusable training utilities. "
        "Run the interactive data-collection training workflow from TrainUser/train_user.py "
        "until it is split into a non-interactive pipeline."
    )
    return 0
