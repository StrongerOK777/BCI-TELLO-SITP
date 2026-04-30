#!/usr/bin/env python3
"""Compatibility exports for the shared final model loader."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.models import FinalUnifiedModel, load_final_model, predict_features


def load_model(checkpoint_path="FinalModel.pth"):
    model, feature_mean, feature_std, _metadata = load_final_model(checkpoint_path)
    return model, feature_mean, feature_std


def predict(features, checkpoint_path="FinalModel.pth"):
    model, feature_mean, feature_std = load_model(checkpoint_path)
    return predict_features(model, features, feature_mean, feature_std)


if __name__ == "__main__":
    _model, _feature_mean, _feature_std, _metadata = load_final_model()
    print("Model loaded successfully!")
