"""Benchmark our single-channel NeuroSky pipeline on public EEG datasets.

The pipeline under test is exactly what TrainUser/train_user.py does: 14-dim
``build_feature_vector`` rows -> window mean+std (``combine_feature_window``)
-> z-score -> ``FinalUnifiedModel``. We additionally report simple sklearn
baselines and compare the *random window split* used in training today with
*trial-grouped* and *cross-subject* splits that a real online session faces.

Usage (from repo root):
    python experiments/public_datasets/datasets.py            # download
    python experiments/public_datasets/run_benchmark.py       # evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[1]
for path in (ROOT_DIR, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bin.models import FinalUnifiedModel, combine_feature_window  # noqa: E402
from bin.training import significance_threshold, stratified_split  # noqa: E402
from datasets import DEFAULT_DATA_DIR, Trial, load_mendeley, load_physionet  # noqa: E402

SEED = 42


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def windows_from_trial(trial: Trial, window: int, stride: int = 1, log_bands: bool = False) -> List[np.ndarray]:
    rows = trial.rows
    if log_bands:
        rows = rows.copy()
        rows[:, 2:10] = np.log1p(np.maximum(rows[:, 2:10], 0))
    if len(rows) < window:
        window = len(rows)
    return [combine_feature_window(rows[i : i + window]) for i in range(0, len(rows) - window + 1, stride)]


def trial_vector(trial: Trial, log_bands: bool = False) -> np.ndarray:
    """One vector per trial: mean+std over all rows (matches the 3 s online window)."""

    return windows_from_trial(trial, window=len(trial.rows), log_bands=log_bands)[0]


def fine_psd_vector(trial: Trial, fmin: float = 4.0, fmax: float = 40.0, step: float = 2.0) -> np.ndarray:
    """Log PSD in 2 Hz bins from the raw signal - what reading raw 512 Hz would allow."""

    from scipy.signal import welch

    freqs, power = welch(trial.raw, fs=trial.fs, nperseg=int(trial.fs), noverlap=int(trial.fs) // 2)
    edges = np.arange(fmin, fmax + step, step)
    return np.array([np.log(power[(freqs >= lo) & (freqs < hi)].mean() + 1e-12) for lo, hi in zip(edges[:-1], edges[1:])])


def build_matrix(trials: Sequence[Trial], featurize: Callable[[Trial], np.ndarray]):
    X = np.array([featurize(t) for t in trials], dtype=np.float32)
    y = np.array([t.label for t in trials])
    groups = np.array([t.subject for t in trials])
    return X, y, groups


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_final_unified(X_train, y_train, X_val=None, y_val=None, *, epochs=200, patience=25, seed=SEED):
    """Train bin.models.FinalUnifiedModel the way train_user.py does (z-score + CE)."""

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if X_val is None:
        idx = rng.permutation(len(X_train))
        n_val = max(1, int(0.1 * len(idx)))
        X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
        X_train, y_train = X_train[idx[n_val:]], y_train[idx[n_val:]]
    mean, std = X_train.mean(0), X_train.std(0)
    norm = lambda a: torch.tensor((a - mean) / (std + 1e-8), dtype=torch.float32)  # noqa: E731
    xt, yt = norm(X_train), torch.tensor(y_train, dtype=torch.long)
    xv, yv = norm(X_val), torch.tensor(y_val, dtype=torch.long)
    model = FinalUnifiedModel(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_state, wait = float("inf"), None, 0
    batch = 64
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            if len(ids) < 2:
                continue
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xt[ids] + 0.008 * torch.randn_like(xt[ids])), yt[ids])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(xv), yv).item()
        if val_loss < best - 1e-4:
            best, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return model(norm(X)).argmax(1).numpy()

    return predict


def sklearn_model(name: str):
    if name == "LDA":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    raise ValueError(name)


def fit_predict(model_name, X_train, y_train, X_test):
    if model_name == "FinalUnifiedModel":
        return train_final_unified(X_train, y_train)(X_test)
    model = sklearn_model(model_name)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Evaluation protocols
# ---------------------------------------------------------------------------

def chance_threshold(n: int, k: int, alpha: float = 0.05) -> float:
    """Smallest accuracy significantly above 1/k chance for n test trials (exact binomial tail)."""

    return significance_threshold(n, k, alpha)


def score(y_true, y_pred) -> Dict[str, float]:
    return {
        "acc": float(np.mean(y_true == y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def cv_eval(X, y, splits, model_name) -> Tuple[Dict[str, float], np.ndarray]:
    y_true, y_pred = [], []
    for train_idx, test_idx in splits:
        y_pred.append(fit_predict(model_name, X[train_idx], y[train_idx], X[test_idx]))
        y_true.append(y[test_idx])
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    labels = np.unique(y)
    return score(y_true, y_pred), confusion_matrix(y_true, y_pred, labels=labels)


def random_window_split_eval(trials, window, model_name, log_bands=False):
    """Reproduce train_user.py: all overlapping windows pooled, random 80/10/10 split."""

    samples = [(w, t.label) for t in trials for w in windows_from_trial(t, window, 1, log_bands)]
    train, _, test = stratified_split(samples, train_ratio=0.8, val_ratio=0.1, seed=SEED)
    X_train = np.array([s[0] for s in train], dtype=np.float32)
    y_train = np.array([s[1] for s in train])
    X_test = np.array([s[0] for s in test], dtype=np.float32)
    y_test = np.array([s[1] for s in test])
    return score(y_test, fit_predict(model_name, X_train, y_train, X_test))


def grouped_window_eval(trials, window, model_name, group_of, n_splits=5, log_bands=False):
    """Same windows, but every window of a trial (or subject) stays on one side."""

    X, y, g = [], [], []
    for t in trials:
        for w in windows_from_trial(t, window, 1, log_bands):
            X.append(w); y.append(t.label); g.append(group_of(t))
    X, y, g = np.array(X, dtype=np.float32), np.array(y), np.array(g)
    n_groups = len(np.unique(g))
    splitter = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=SEED)
    return cv_eval(X, y, list(splitter.split(X, y, g)), model_name)[0]


def within_subject_eval(trials, featurize, model_name, n_splits=5):
    """Per-subject k-fold on trials; returns mean/std over subjects + per-subject accs."""

    by_subject: Dict[str, List[Trial]] = {}
    for t in trials:
        by_subject.setdefault(t.subject, []).append(t)
    accs = []
    for subject, items in sorted(by_subject.items()):
        X, y, _ = build_matrix(items, featurize)
        if len(np.unique(y)) < 2 or np.bincount(y).min() < n_splits:
            continue
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        accs.append(cv_eval(X, y, list(splitter.split(X, y)), model_name)[0]["acc"])
    accs = np.array(accs)
    return {"mean": float(accs.mean()), "std": float(accs.std()), "n_subjects": int(len(accs)), "per_subject": accs.tolist()}


def cross_subject_eval(trials, featurize, model_name, n_splits=5):
    X, y, groups = build_matrix(trials, featurize)
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    result, cm = cv_eval(X, y, splits, model_name)
    result["confusion"] = cm.tolist()
    return result


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_mendeley(data_dir: Path) -> Dict:
    trials = load_mendeley(data_dir)
    print(f"[Mendeley] {len(trials)} trials, rows/trial ~{np.mean([len(t.rows) for t in trials]):.1f}")
    out: Dict = {"n_trials": len(trials)}

    tasks = {}
    focus = [Trial(t.subject, t.trial_id, int(t.meta["focus"] == "not_focus"), t.rows, meta=t.meta) for t in trials]
    tasks["focus_vs_notfocus"] = (focus, 2)
    gaze_map = {"Left": 0, "Right": 1, "Front": 2}
    gaze = [Trial(t.subject, t.trial_id, gaze_map[t.meta["gaze"]], t.rows, meta=t.meta)
            for t in trials if t.meta["gaze"] in gaze_map]
    tasks["gaze_left_right_front"] = (gaze, 3)

    for task, (items, k) in tasks.items():
        print(f"  task {task}: {len(items)} trials, {k} classes")
        res: Dict = {"n_trials": len(items), "classes": k, "chance": 1 / k,
                     "sig_threshold_trials": chance_threshold(len(items), k)}
        for model_name in ("FinalUnifiedModel", "LogReg", "RF"):
            res[model_name] = {
                # train_user.py protocol: window=5 rows, stride 1, random split
                "random_window_split": random_window_split_eval(items, 5, model_name),
                # honest: windows of one trial never on both sides
                "trial_grouped_cv": grouped_window_eval(items, 5, model_name, lambda t: t.trial_id),
                # hardest: train on one person, test on the other
                "cross_subject": grouped_window_eval(items, 5, model_name, lambda t: t.subject, n_splits=2),
            }
            print(f"    {model_name:18s} " + " | ".join(f"{p}: {v['acc']:.3f}" for p, v in res[model_name].items()))
        out[task] = res
    return out


def run_physionet(data_dir: Path, subjects: Sequence[int]) -> Dict:
    t0 = time.time()
    by_channel = load_physionet(data_dir, subjects=subjects)
    print(f"[PhysioNet] loaded in {time.time() - t0:.0f}s: " + ", ".join(f"{c}={len(v)}" for c, v in by_channel.items()))
    out: Dict = {}

    def lr_only(trials):
        return [t for t in trials if t.label in (0, 1)]

    def lr_rest_balanced(trials):
        # T0 rest is ~2x as frequent as each task; keep every other rest trial.
        rest = [t for t in trials if t.label == 2][::2]
        return lr_only(trials) + rest

    feature_sets = {
        "neurosky_style": lambda t: trial_vector(t),
        "neurosky_log": lambda t: trial_vector(t, log_bands=True),
        "fine_psd": fine_psd_vector,
    }

    # 1) Within-subject, left vs right, every channel x feature set (fast linear model)
    within: Dict = {}
    for channel, trials in by_channel.items():
        items = lr_only(trials)
        for feat_name, featurize in feature_sets.items():
            res = within_subject_eval(items, featurize, "LDA")
            within[f"{channel}/{feat_name}"] = res
            print(f"  within-subject L/R {channel:4s} {feat_name:15s} LDA acc {res['mean']:.3f} ± {res['std']:.3f}")
    # Two-channel motor-cortex reference (C3+C4), the classic MI setup
    paired = [
        Trial(a.subject, a.trial_id, a.label, a.rows, meta={})
        for a in lr_only(by_channel["C3"])
    ]
    c4 = {t.trial_id: t for t in lr_only(by_channel["C4"])}
    c3 = {t.trial_id: t for t in lr_only(by_channel["C3"])}
    paired = [p for p in paired if p.trial_id in c4]
    both = lambda t: np.concatenate([fine_psd_vector(c3[t.trial_id]), fine_psd_vector(c4[t.trial_id])])  # noqa: E731
    within["C3+C4/fine_psd"] = within_subject_eval(paired, both, "LDA")
    print(f"  within-subject L/R C3+C4 fine_psd LDA acc {within['C3+C4/fine_psd']['mean']:.3f}")
    n_per_subject = int(np.median([sum(1 for t in lr_only(by_channel['FP1']) if t.subject == s)
                                   for s in {t.subject for t in by_channel['FP1']}]))
    out["within_subject_lr"] = within
    out["within_subject_lr_sig_threshold"] = chance_threshold(n_per_subject, 2)
    out["within_subject_lr_trials_per_subject"] = n_per_subject

    # 2) Our exact pipeline (Fp1 = MindWave position), cross-subject, 2- and 3-class
    cross: Dict = {}
    for task, select, k in (("left_right", lr_only, 2), ("left_right_rest", lr_rest_balanced, 3)):
        for channel in ("FP1", "C3"):
            items = select(by_channel[channel])
            for model_name in ("FinalUnifiedModel", "LogReg"):
                res = cross_subject_eval(items, feature_sets["neurosky_log"], model_name)
                cross[f"{task}/{channel}/{model_name}"] = res
                print(f"  cross-subject {task:15s} {channel:4s} {model_name:18s} acc {res['acc']:.3f} (n={res['n']})")
        cross[f"{task}/chance"] = 1 / k
    out["cross_subject"] = cross

    # 3) Is "imagining anything" vs rest detectable from one channel? (2-class)
    detect: Dict = {}
    for channel in ("FP1", "C3"):
        items = [Trial(t.subject, t.trial_id, int(t.label != 2), t.rows, t.raw, t.fs)
                 for t in lr_rest_balanced(by_channel[channel])]
        for feat_name in ("neurosky_log", "fine_psd"):
            res = within_subject_eval(items, feature_sets[feat_name], "LDA")
            detect[f"{channel}/{feat_name}"] = res
            print(f"  within-subject MI-vs-rest {channel:4s} {feat_name:15s} LDA acc {res['mean']:.3f} ± {res['std']:.3f}")
    out["within_subject_mi_vs_rest"] = detect

    # 4) Sanity check of the data/labels: classic 64-channel CSP + LDA, left vs right
    out["within_subject_lr_csp64"] = csp_sanity_check(data_dir, subjects)
    print(f"  within-subject L/R 64ch CSP+LDA acc {out['within_subject_lr_csp64']['mean']:.3f}")

    # 5) Which headsets could do left/right MI? Same CSP pipeline on channel subsets
    out["within_subject_lr_csp_subsets"] = run_csp_subsets(data_dir, subjects)

    # 6) Leakage check on Fp1: overlapping windows, random split vs subject-grouped split
    items = lr_rest_balanced(by_channel["FP1"])
    leak = {
        "random_window_split": random_window_split_eval(items, 3, "FinalUnifiedModel", log_bands=True),
        "subject_grouped": grouped_window_eval(items, 3, "FinalUnifiedModel", lambda t: t.subject, log_bands=True),
    }
    print("  leakage FP1 3-class: " + " | ".join(f"{p}: {v['acc']:.3f}" for p, v in leak.items()))
    out["leakage_fp1_3class"] = leak
    return out


# Channel subsets for CSP: what a given headset could see of the motor cortex.
EMOTIV_EPOC_X = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")
CSP_SUBSETS = {
    "all64": None,
    "emotiv14": EMOTIV_EPOC_X,
    "emotiv_fc5_fc6": ("FC5", "FC6"),
    "c3_c4": ("C3", "C4"),
    "c3_cz_c4": ("C3", "CZ", "C4"),
}


def csp_channel_subsets(data_dir: Path, subjects: Sequence[int], subsets: Dict, n_filters: int = 6) -> Dict:
    """CSP (8-30 Hz, 0.5-4 s) + LDA, left vs right, per subject 5-fold, for each channel subset.

    ``subsets`` maps a name to a channel tuple (``None`` = all channels). Each
    subject is loaded and filtered once and every subset is scored on the same
    epochs and folds, so the subsets are directly comparable.
    """

    from scipy.linalg import eigh
    from scipy.signal import butter, sosfiltfilt

    from datasets import PHYSIONET_IMAGERY_RUNS, read_edf

    accs: Dict[str, List[float]] = {name: [] for name in subsets}
    for s in subjects:
        epochs, labels, names = [], [], None
        for r in PHYSIONET_IMAGERY_RUNS:
            path = data_dir / "physionet_eegmmidb" / f"S{s:03d}R{r:02d}.edf"
            if not path.exists():
                continue
            signals, fs_map, annotations = read_edf(path)
            names = sorted(signals)
            fs = fs_map[names[0]]
            sos = butter(4, [8, 30], btype="band", fs=fs, output="sos")
            data = sosfiltfilt(sos, np.array([signals[n] for n in names]), axis=1)
            for onset, _, code in annotations:
                if code in ("T1", "T2"):
                    begin, end = int((onset + 0.5) * fs), int((onset + 4.0) * fs)
                    if end <= data.shape[1]:
                        epochs.append(data[:, begin:end]); labels.append(int(code == "T2"))
        if len(epochs) < 20:
            continue
        X_all, y = np.array(epochs), np.array(labels)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_all, y))
        for subset_name, channels in subsets.items():
            X = X_all if channels is None else X_all[:, [names.index(c) for c in channels]]
            k = min(n_filters, X.shape[1]) // 2 or 1
            correct = 0
            for train_idx, test_idx in folds:
                covs = [np.mean([e @ e.T / np.trace(e @ e.T) for e in X[train_idx][y[train_idx] == c]], axis=0) for c in (0, 1)]
                _, vecs = eigh(covs[0], covs[0] + covs[1])
                W = np.concatenate([vecs[:, :k], vecs[:, -k:]], axis=1).T
                feats = lambda E: np.log(np.var(np.einsum("fc,ncs->nfs", W, E), axis=2))  # noqa: E731
                clf = LinearDiscriminantAnalysis().fit(feats(X[train_idx]), y[train_idx])
                correct += int((clf.predict(feats(X[test_idx])) == y[test_idx]).sum())
            accs[subset_name].append(correct / len(y))
    out = {}
    for subset_name, values in accs.items():
        values = np.array(values)
        out[subset_name] = {
            "channels": list(subsets[subset_name]) if subsets[subset_name] is not None else "all",
            "mean": float(values.mean()), "std": float(values.std()),
            "n_subjects": int(len(values)), "per_subject": values.tolist(),
        }
    return out


def csp_sanity_check(data_dir: Path, subjects: Sequence[int], n_filters: int = 6) -> Dict:
    """64-channel CSP + LDA. If this is well above chance the labels/epoching are
    correct, so near-chance single-channel results are real."""

    result = csp_channel_subsets(data_dir, subjects, {"all64": None}, n_filters)["all64"]
    result.pop("channels")
    return result


def run_csp_subsets(data_dir: Path, subjects: Sequence[int]) -> Dict:
    result = csp_channel_subsets(data_dir, subjects, CSP_SUBSETS)
    for name, res in result.items():
        print(f"  within-subject L/R CSP {name:15s} acc {res['mean']:.3f} ± {res['std']:.3f}")
    return result


def run_own_data(window: int = 5) -> Dict:
    """Re-score the repo's own src/data recordings under both split protocols."""

    from bin.training import parse_feature_line

    trials = []
    for label, name in enumerate(("actionleft", "actionright", "rest")):
        lines = (ROOT_DIR / "src" / "data" / f"{name}.txt").read_text().splitlines()
        rows = [r for r in (parse_feature_line(line) for line in lines) if r is not None]
        # cut each recording into 3 s blocks (30 rows) so a grouped split is possible
        for b in range(0, len(rows) - 29, 30):
            trials.append(Trial("self", f"{name}_{b // 30}", label, np.array(rows[b : b + 30], dtype=np.float64)))
    out = {"n_blocks": len(trials)}
    for model_name in ("FinalUnifiedModel", "LogReg"):
        out[model_name] = {
            "random_window_split": random_window_split_eval(trials, window, model_name),
            "block_grouped_cv": grouped_window_eval(trials, window, model_name, lambda t: t.trial_id),
        }
        print(f"[own src/data] {model_name:18s} " + " | ".join(f"{p}: {v['acc']:.3f}" for p, v in out[model_name].items()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the NeuroSky pipeline on public datasets")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--subjects", type=int, default=109, help="PhysioNet: first N subjects")
    parser.add_argument("--only", choices=["mendeley", "physionet", "own", "csp"], default=None,
                        help="csp: only re-run the channel-subset CSP check and merge it into --output")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.set_num_threads(4)

    excluded = {88, 92, 100, 104}  # known bad sampling/annotation in EEGMMIDB
    subjects = [s for s in range(1, args.subjects + 1) if s not in excluded]
    if args.only == "csp":
        results = json.loads(args.output.read_text()) if args.output.exists() else {}
        results.setdefault("physionet", {})["within_subject_lr_csp_subsets"] = run_csp_subsets(args.data_dir, subjects)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"结果已合并：{args.output}")
        return 0

    results: Dict = {}
    if args.only in (None, "own"):
        results["own_data"] = run_own_data()
    if args.only in (None, "mendeley"):
        results["mendeley"] = run_mendeley(args.data_dir)
    if args.only in (None, "physionet"):
        results["physionet"] = run_physionet(args.data_dir, subjects)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
