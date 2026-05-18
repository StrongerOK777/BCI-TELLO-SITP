"""Model architectures and prediction helpers for BCI classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LABELS = ["left", "right", "rest"]


class FinalUnifiedModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 384)
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
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        identity = x
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = x + identity
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        return self.fc5(x)


def resolve_model_path(checkpoint_path: Optional[str | Path] = None) -> Path:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    path = Path(checkpoint_path).expanduser()
    return path


def combine_feature_window(feature_window: Sequence[Sequence[float]]) -> np.ndarray:
    if len(feature_window) == 0:
        return np.array([], dtype=np.float32)
    window_data = np.array(feature_window, dtype=np.float32)
    mean_feat = window_data.mean(axis=0)
    std_feat = window_data.std(axis=0)
    return np.concatenate([mean_feat, std_feat], axis=0)


def load_final_model(
    checkpoint_path: Optional[str | Path] = None,
    *,
    map_location: str = "cpu",
) -> Tuple[FinalUnifiedModel, Any, Any, dict]:
    path = resolve_model_path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint does not contain a state dict: {path}")
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    input_dim = checkpoint.get("input_dim") if isinstance(checkpoint, dict) else None
    if input_dim is None and "fc1.weight" in state_dict:
        input_dim = int(state_dict["fc1.weight"].shape[1])
    if input_dim is None:
        input_dim = 28
    model = FinalUnifiedModel(int(input_dim))
    model.load_state_dict(state_dict)
    model.eval()
    feature_mean = checkpoint.get("feature_mean") if isinstance(checkpoint, dict) else None
    feature_std = checkpoint.get("feature_std") if isinstance(checkpoint, dict) else None
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, feature_mean, feature_std, metadata


def predict_features(
    model: FinalUnifiedModel,
    features: Sequence[float],
    feature_mean: Any = None,
    feature_std: Any = None,
) -> int:
    input_data = np.array(features, dtype=np.float32)
    if input_data.size == 0:
        return 2
    if feature_mean is not None and feature_std is not None:
        input_data = (input_data - feature_mean) / (feature_std + 1e-6)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        return int(torch.argmax(output, dim=1).item())


def predict_window(
    model: FinalUnifiedModel,
    feature_window: Sequence[Sequence[float]],
    feature_mean: Any = None,
    feature_std: Any = None,
) -> int:
    combined = combine_feature_window(feature_window)
    return predict_features(model, combined, feature_mean, feature_std)


class ModelPredictor:
    def __init__(self, checkpoint_path: Optional[str | Path] = None, *, autoload: bool = True) -> None:
        self.checkpoint_path = checkpoint_path
        self.model: Optional[FinalUnifiedModel] = None
        self.feature_mean: Any = None
        self.feature_std: Any = None
        self.metadata: dict = {}
        if autoload:
            self.load()

    def load(self) -> None:
        model, feature_mean, feature_std, metadata = load_final_model(self.checkpoint_path)
        self.model = model
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.metadata = metadata

    def predict_window(self, feature_window: Sequence[Sequence[float]]) -> int:
        if self.model is None:
            self.load()
        assert self.model is not None
        return predict_window(self.model, feature_window, self.feature_mean, self.feature_std)

    def predict_label(self, feature_window: Sequence[Sequence[float]]) -> str:
        return LABELS[self.predict_window(feature_window)]


def print_model_metadata(metadata: dict) -> None:
    strict = metadata.get("test_accuracy_strict")
    relaxed = metadata.get("test_accuracy_relaxed")
    print("Model loaded successfully!")
    if strict is not None:
        print(f"Test Accuracy (strict): {strict * 100:.2f}%")
    if relaxed is not None:
        print(f"Test Accuracy (relaxed): {relaxed * 100:.2f}%")
