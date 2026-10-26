"""Download and load public EEG datasets in the shape our NeuroSky pipeline expects.

Two datasets are supported:

* ``mendeley`` - "NeuroSky EEG Signal Collection Dataset" (Mendeley Data,
  doi:10.17632/mbbmpb5h66.1, CC BY 4.0). Recorded with a real MindWave Mobile,
  so every row already carries the same eSense fields our headset reports.
* ``physionet`` - "EEG Motor Movement/Imagery Dataset" (PhysioNet EEGMMIDB,
  doi:10.13026/C28G6P). 64-channel, 160 Hz. We keep one channel at a time and
  re-compute NeuroSky-style band powers so it can be fed into the same features.

Everything here is stdlib + numpy so no new dependency is needed.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.eeg import build_feature_vector  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
USER_AGENT = {"User-Agent": "BCI-TELLO-benchmark/1.0"}

# NeuroSky ThinkGear band edges (Hz), in the order build_feature_vector expects.
NEUROSKY_BANDS = [
    ("delta", 0.5, 2.75),
    ("theta", 3.5, 6.75),
    ("lowAlpha", 7.5, 9.25),
    ("highAlpha", 10.0, 11.75),
    ("lowBeta", 13.0, 16.75),
    ("highBeta", 18.0, 29.75),
    ("lowGamma", 31.0, 39.75),
    ("midGamma", 41.0, 49.75),
]

# PhysioNet imagery runs: 4/8/12 = imagine opening/closing left or right fist.
PHYSIONET_IMAGERY_RUNS = (4, 8, 12)
PHYSIONET_EVENT_LABELS = {"T1": 0, "T2": 1, "T0": 2}  # left, right, rest (bin.models.LABELS)


@dataclass
class Trial:
    """One labelled trial: a sequence of 14-dim NeuroSky feature rows (~1 row/s)."""

    subject: str
    trial_id: str
    label: int
    rows: np.ndarray  # (n_rows, 14) from bin.eeg.build_feature_vector
    raw: np.ndarray | None = None  # optional raw single-channel signal
    fs: float | None = None
    meta: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mendeley NeuroSky dataset
# ---------------------------------------------------------------------------

MENDELEY_API = "https://data.mendeley.com/public-api/datasets/mbbmpb5h66"


def _get_json(url: str):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT)))


def download_mendeley(data_dir: Path) -> Path:
    out = data_dir / "mendeley_neurosky"
    folders = _get_json(f"{MENDELEY_API}/folders/1")
    by_id = {folder["id"]: folder for folder in folders}

    def folder_path(folder_id):
        parts = []
        while folder_id in by_id:
            parts.append(by_id[folder_id]["name"])
            folder_id = by_id[folder_id].get("parent_id")
        return Path(*reversed(parts))

    count = 0
    for folder in folders:
        for item in _get_json(f"{MENDELEY_API}/files?folder_id={folder['id']}&version=1"):
            target = out / folder_path(folder["id"]) / item["filename"]
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(item["content_details"]["download_url"], headers=USER_AGENT)
            target.write_bytes(urllib.request.urlopen(request).read())
            count += 1
    print(f"Mendeley: downloaded {count} new files into {out}")
    return out


def load_mendeley(data_dir: Path = DEFAULT_DATA_DIR) -> List[Trial]:
    """Return every CSV as a Trial; label is filled in later per task (see meta)."""

    root = data_dir / "mendeley_neurosky"
    trials: List[Trial] = []
    for path in sorted(root.rglob("*.csv")):
        parts = path.relative_to(root).parts  # Subject / Focus|Fokus|Not ... / Gaze / file
        subject, state, gaze = parts[0], parts[1], parts[2]
        focus = "not" not in state.lower()
        gaze = {"fornt": "Front"}.get(gaze.lower(), gaze)
        rows = []
        with path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                try:
                    if int(record["poor_signal"]) > 20:
                        continue
                    values = [int(record[key]) for key in (
                        "attention", "meditation", "delta", "theta", "low_alpha", "high_alpha",
                        "low_beta", "high_beta", "low_gamma", "mid_gamma")]
                except (KeyError, ValueError):
                    continue
                rows.append(build_feature_vector(*values, blink_strength=0))
        if not rows:
            continue
        trials.append(
            Trial(
                subject=subject,
                trial_id=f"{subject}/{state}/{gaze}/{path.stem}",
                label=-1,
                rows=np.array(rows, dtype=np.float64),
                meta={"focus": "focus" if focus else "not_focus", "gaze": gaze},
            )
        )
    return trials


# ---------------------------------------------------------------------------
# PhysioNet EEGMMIDB
# ---------------------------------------------------------------------------

# PhysioNet's AWS open-data mirror is ~100x faster than physionet.org from China.
PHYSIONET_URL = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/S{s:03d}/S{s:03d}R{r:02d}.edf"


def download_physionet(data_dir: Path, subjects: Sequence[int] = range(1, 110), workers: int = 8) -> Path:
    out = data_dir / "physionet_eegmmidb"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, r) for s in subjects for r in PHYSIONET_IMAGERY_RUNS]

    def fetch(job):
        s, r = job
        target = out / f"S{s:03d}R{r:02d}.edf"
        if target.exists() and target.stat().st_size > 0:
            return 0
        request = urllib.request.Request(PHYSIONET_URL.format(s=s, r=r), headers=USER_AGENT)
        payload = urllib.request.urlopen(request).read()
        partial = target.with_suffix(".part")
        partial.write_bytes(payload)
        partial.rename(target)  # never leave a truncated .edf behind
        return 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        count = sum(pool.map(fetch, jobs))
    print(f"PhysioNet: downloaded {count} new files into {out}")
    return out


def read_edf(path: Path):
    """Minimal EDF/EDF+ reader: returns ({label: signal_uV}, fs_by_label, annotations)."""

    data = path.read_bytes()
    header_bytes = int(data[184:192])
    n_records = int(data[236:244])
    record_duration = float(data[244:252])
    ns = int(data[252:256])

    def field_list(offset, width):
        return [data[offset + i * width : offset + (i + 1) * width].decode("ascii").strip() for i in range(ns)]

    offset = 256
    labels = field_list(offset, 16); offset += ns * 16
    offset += ns * 80  # transducer
    offset += ns * 8  # physical dimension
    phys_min = [float(v) for v in field_list(offset, 8)]; offset += ns * 8
    phys_max = [float(v) for v in field_list(offset, 8)]; offset += ns * 8
    dig_min = [float(v) for v in field_list(offset, 8)]; offset += ns * 8
    dig_max = [float(v) for v in field_list(offset, 8)]; offset += ns * 8
    offset += ns * 80  # prefiltering
    samples = [int(v) for v in field_list(offset, 8)]

    record_len = sum(samples)
    raw = np.frombuffer(data[header_bytes : header_bytes + n_records * record_len * 2], dtype="<i2")
    raw = raw.reshape(n_records, record_len)

    signals: Dict[str, np.ndarray] = {}
    fs: Dict[str, float] = {}
    annotations: List[tuple] = []
    start = 0
    for i, label in enumerate(labels):
        chunk = raw[:, start : start + samples[i]]
        start += samples[i]
        if label == "EDF Annotations":
            text = chunk.astype("<i2").tobytes().decode("latin-1")
            for match in re.finditer(r"([+-]\d+(?:\.\d+)?)\x15(\d+(?:\.\d+)?)?\x14([^\x14\x00]*)\x14", text):
                if match.group(3):
                    annotations.append((float(match.group(1)), float(match.group(2) or 0.0), match.group(3)))
            continue
        gain = (phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i])
        signals[label.rstrip(".").upper()] = (chunk.reshape(-1) - dig_min[i]) * gain + phys_min[i]
        fs[label.rstrip(".").upper()] = samples[i] / record_duration
    return signals, fs, annotations


def neurosky_band_rows(signal: np.ndarray, fs: float, *, epoch_sec: float = 1.0, hop_sec: float = 0.5) -> np.ndarray:
    """Emulate ThinkGear: one row of 8 band powers per 1 s epoch (Hann-windowed FFT)."""

    n = int(round(epoch_sec * fs))
    hop = int(round(hop_sec * fs))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    window = np.hanning(n)
    masks = [(freqs >= lo) & (freqs <= hi) for _, lo, hi in NEUROSKY_BANDS]
    rows = []
    for begin in range(0, len(signal) - n + 1, hop):
        segment = signal[begin : begin + n]
        segment = (segment - segment.mean()) * window
        power = np.abs(np.fft.rfft(segment)) ** 2
        rows.append([power[mask].sum() for mask in masks])
    return np.array(rows, dtype=np.float64)


def _band_rows_to_features(bands: np.ndarray) -> np.ndarray:
    # attention / meditation / blink are proprietary NeuroSky outputs and cannot be
    # reproduced from raw EEG, so they are zero here (constant -> carries no info).
    return np.array([build_feature_vector(0, 0, *row, blink_strength=0) for row in bands], dtype=np.float64)


def load_physionet(
    data_dir: Path = DEFAULT_DATA_DIR,
    channels: Sequence[str] = ("FP1", "C3", "C4", "CZ"),
    subjects: Sequence[int] = range(1, 110),
) -> Dict[str, List[Trial]]:
    """Return {channel: [Trial, ...]} with left/right/rest trials from imagery runs."""

    root = data_dir / "physionet_eegmmidb"
    trials: Dict[str, List[Trial]] = {channel: [] for channel in channels}
    for s in subjects:
        for r in PHYSIONET_IMAGERY_RUNS:
            path = root / f"S{s:03d}R{r:02d}.edf"
            if not path.exists():
                continue
            signals, fs_map, annotations = read_edf(path)
            for k, (onset, duration, code) in enumerate(annotations):
                if code not in PHYSIONET_EVENT_LABELS:
                    continue
                for channel in channels:
                    fs = fs_map[channel]
                    begin = int(round(onset * fs))
                    end = int(round((onset + min(duration, 4.0)) * fs))
                    segment = signals[channel][begin:end]
                    if len(segment) < int(3.5 * fs):
                        continue
                    bands = neurosky_band_rows(segment, fs)
                    trials[channel].append(
                        Trial(
                            subject=f"S{s:03d}",
                            trial_id=f"S{s:03d}R{r:02d}E{k:02d}",
                            label=PHYSIONET_EVENT_LABELS[code],
                            rows=_band_rows_to_features(bands),
                            raw=segment,
                            fs=fs,
                        )
                    )
    return trials


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download public EEG datasets for the benchmark")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", choices=["mendeley", "physionet", "all"], default="all")
    parser.add_argument("--subjects", type=int, default=109, help="PhysioNet: first N subjects")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("mendeley", "all"):
        download_mendeley(args.data_dir)
    if args.dataset in ("physionet", "all"):
        download_physionet(args.data_dir, range(1, args.subjects + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
