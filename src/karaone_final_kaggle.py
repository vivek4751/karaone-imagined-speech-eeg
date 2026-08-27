#!/usr/bin/env python3
"""KaraOne final Kaggle pipeline: phonemic vowel vs consonant.

Task
----
Class 0: vowels ``iy`` and ``uw``.
Class 1: consonant/syllabic prompts ``m``, ``n``, ``piy``, ``tiy``, ``diy``.
Word prompts ``gnaw``, ``knew``, ``pat``, and ``pot`` are always excluded.

This file is intentionally self-contained. It uses the authoritative raw SET,
epoch_inds.mat, and kinect_data/labels.txt files; it never guesses a fixed
5-second window. The actual thinking_inds intervals are used for every trial.

Default execution discovers the available KaraOne candidate subjects, removes
trials whose audited post-ASR noise fraction exceeds 10%, and retains the eight
lowest-noise subjects with usable vowel and consonant trials. Subjects with more
than 10% rejected trials remain flagged in the audit unless strict subject-level
exclusion is explicitly enabled. Selection uses noise quality only, never
classification performance.
Set KARAONE_SUBJECTS explicitly to run a declared candidate list. The default
output is not an accuracy guarantee:
balanced accuracy, vowel recall, consonant recall, ROC-AUC, macro-F1, and every
fold confusion matrix are saved so consonant-majority behavior is visible.

For practical Kaggle runtime, ``KARAONE_FAST_MODE=1`` is the default. It still
runs every requested branch, but uses smaller convolutional ensembles and a
shorter inner adaptive search. Set ``KARAONE_FAST_MODE=0`` for the larger,
slower configuration.

Kaggle usage
------------
1. Add the full KaraOne raw dataset as a Kaggle input.
2. Set the root correctly, for example:

   import os
   os.environ['KARAONE_ROOT'] = '/kaggle/input/datasets/vivekranjan4751/karaone'

3. Run:

   !python /kaggle/working/karaone_final_kaggle.py

The script can install missing packages when KARAONE_AUTO_INSTALL=1. The Aeon
version installed in the tested environment exposes HydraClassifier,
MiniRocketClassifier, MultiRocketClassifier, and MultiRocketHydraClassifier.
There is no portable MiniRocketHydraClassifier export in that version, so the
requested MiniRocket-Hydra branch is implemented as a clearly labeled
MiniRocket+Hydra probability fusion. Every requested classifier branch has a
vowel-recall adaptive mode. In the default fast mode, the inverse-frequency
class weight for class 0 (vowels) is multiplied by a selected vowel-weight
candidate using inner training folds, the selected model is refit on all
outer-training subjects, and a training-only decision threshold is applied.
Balanced-accuracy and consonant-recall floors prevent an all-vowel degenerate
solution. Threshold-only adaptation is an explicit sensitivity mode and is
blocked by default. The optional ``catch22+summary+mr hydra`` branch uses the
same vowel-recall-focused objective when combining score streams.

Important preprocessing note
----------------------------
The default ``asr_notch_car`` mode applies a 50 Hz notch, a 1–50 Hz
band-pass, label-independent continuous EOG regression when EOG channels are
available, MEEGkit ASR when available (with clearing/rest intervals used for
calibration), CAR, a small fixed-montage Laplacian, robust per-trial channel
normalization, and resampling to 256 Hz / 1280 points. Trial QC is applied by default: trials above the configured
10% post-ASR noise fraction are removed before the noise-only best-eight
selection. If the Kaggle input contains already processed official SET files, run the explicit
control with KARAONE_PREPROCESSING_MODE=official and report both variants; do
not silently call a processed SET file raw. The loader also accepts official CNT
recordings when those are the files supplied for the selected subjects.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample, sosfiltfilt
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("KARAONE_ROOT", "/kaggle/input/datasets/vivekranjan4751/karaone"))
OUT = Path(os.environ.get("KARAONE_OUTPUT", "/kaggle/working/karaone_final_results"))
DEFAULT_SUBJECTS = ["MM05", "MM08", "MM16", "MM20", "MM21"]
KNOWN_SUBJECTS = ["MM05", "MM08", "MM09", "MM10", "MM11", "MM12", "MM14", "MM15", "MM16", "MM18", "MM19", "MM20", "MM21", "P02"]
SUBJECT_OVERRIDE = [x.strip() for x in os.environ.get("KARAONE_SUBJECTS", "").split(",") if x.strip()]
SUBJECTS = list(SUBJECT_OVERRIDE) if SUBJECT_OVERRIDE else list(KNOWN_SUBJECTS)
KEEP_PROMPTS = {"iy", "uw", "m", "n", "piy", "tiy", "diy"}
WORD_PROMPTS = {"gnaw", "knew", "pat", "pot"}
VOWELS = {"iy", "uw"}
CONSONANTS = {"m", "n", "piy", "tiy", "diy"}
SEED = int(os.environ.get("KARAONE_SEED", "42"))
FAST_MODE = os.environ.get("KARAONE_FAST_MODE", "1").lower() in {"1", "true", "yes"}
TARGET_CHANNELS = 62
TARGET_LEN = int(os.environ.get("KARAONE_TARGET_LEN", "1280"))
EFFECTIVE_FS = float(os.environ.get("KARAONE_EFFECTIVE_FS", "256"))
LINE_NOISE_HZ = float(os.environ.get("KARAONE_LINE_NOISE_HZ", "50"))
LINE_NOISE_Q = float(os.environ.get("KARAONE_LINE_NOISE_Q", "30"))
PREPROCESSING_MODE = os.environ.get("KARAONE_PREPROCESSING_MODE", "asr_notch_car").lower().strip()
ASR_ENABLED = os.environ.get("KARAONE_ASR_ENABLED", "1").lower() in {"1", "true", "yes"}
ASR_CUTOFF = float(os.environ.get("KARAONE_ASR_CUTOFF", "3"))
ASR_MAX_BAD_CHANS = float(os.environ.get("KARAONE_ASR_MAX_BAD_CHANS", "0.30"))
ASR_CALIBRATION_TRIALS = int(os.environ.get("KARAONE_ASR_CALIBRATION_TRIALS", "20"))
EOG_REGRESSION = os.environ.get("KARAONE_EOG_REGRESSION", "1").lower() in {"1", "true", "yes"}
ROBUST_NORMALIZE = os.environ.get("KARAONE_ROBUST_NORMALIZE", "1").lower() in {"1", "true", "yes"}
ROBUST_CLIP = float(os.environ.get("KARAONE_ROBUST_CLIP", "8"))
# The official KaraOne description applies a small spatial Laplacian after
# channel-wise mean removal. It is label-independent and uses only the fixed
# 10-20 montage geometry, never the held-out labels.
SPATIAL_FILTER = os.environ.get("KARAONE_SPATIAL_FILTER", "laplacian").lower().strip()
LAPLACIAN_NEIGHBORS = int(os.environ.get("KARAONE_LAPLACIAN_NEIGHBORS", "4"))
LAPLACIAN_MIX = float(os.environ.get("KARAONE_LAPLACIAN_MIX", "0.50"))
if SPATIAL_FILTER not in {"none", "laplacian"}:
    raise ValueError("KARAONE_SPATIAL_FILTER must be none or laplacian")
if LAPLACIAN_NEIGHBORS < 1 or not (0.0 <= LAPLACIAN_MIX <= 1.0):
    raise ValueError("KARAONE_LAPLACIAN_NEIGHBORS must be >=1 and KARAONE_LAPLACIAN_MIX must be in [0,1]")
APPLY_QC = os.environ.get("KARAONE_APPLY_QC_REJECTION", "1").lower() in {"1", "true", "yes"}
QC_THRESHOLD = float(os.environ.get("KARAONE_NOISE_THRESHOLD", "0.10"))
SUBJECT_REJECTION_FRACTION = float(os.environ.get("KARAONE_SUBJECT_REJECTION_FRACTION", "0.10"))
MAX_SUBJECTS = int(os.environ.get("KARAONE_MAX_SUBJECTS", "8"))
REQUIRE_BEST_EIGHT = os.environ.get("KARAONE_REQUIRE_BEST_EIGHT", "1").lower() in {"1", "true", "yes"}
# Default behavior satisfies the requested exact best-eight run: individual
# trials above 10% noise are removed, then the eight lowest-noise subjects
# with usable post-QC data are selected. Subjects above the 10% rejection
# fraction remain visibly flagged in the audit. Set this to 1 only when a
# strict subject-level exclusion is desired, accepting that fewer than eight
# subjects may remain.
ENFORCE_SUBJECT_REJECTION = os.environ.get("KARAONE_ENFORCE_SUBJECT_REJECTION", "0").lower() in {"1", "true", "yes"}
QC_Z = float(os.environ.get("KARAONE_QC_Z", "5"))
QC_WINDOW_SECONDS = float(os.environ.get("KARAONE_QC_WINDOW_SECONDS", "0.50"))
QC_STEP_SECONDS = float(os.environ.get("KARAONE_QC_STEP_SECONDS", "0.25"))
REQUIRE_EXPECTED = os.environ.get("KARAONE_REQUIRE_EXPECTED_COHORT", "0").lower() in {"1", "true", "yes"}
EXPECTED_TOTAL = int(os.environ.get("KARAONE_EXPECTED_TOTAL", "440"))
EXPECTED_VOWELS = int(os.environ.get("KARAONE_EXPECTED_VOWELS", "125"))
EXPECTED_CONSONANTS = int(os.environ.get("KARAONE_EXPECTED_CONSONANTS", "315"))
EVAL_MODE = os.environ.get("KARAONE_EVAL_MODE", "loso").lower().strip()
# `random` is a deliberately separate trial-level sensitivity analysis. It may
# place trials from the same subject in both train and test and must never be
# presented as cross-subject generalization. LOSO remains the default claim.
RANDOM_TEST_SIZE = float(os.environ.get("KARAONE_RANDOM_TEST_SIZE", "0.20"))
RANDOM_REPEATS = int(os.environ.get("KARAONE_RANDOM_REPEATS", "5"))
if EVAL_MODE not in {"loso", "within", "within_subject", "random", "stratified_random"}:
    raise ValueError("KARAONE_EVAL_MODE must be loso, random, or within_subject")
if not (0.05 <= RANDOM_TEST_SIZE < 0.50):
    raise ValueError("KARAONE_RANDOM_TEST_SIZE must be in [0.05, 0.50)")
if RANDOM_REPEATS < 1:
    raise ValueError("KARAONE_RANDOM_REPEATS must be >= 1")
INNER_SPLITS = int(os.environ.get("KARAONE_ADAPTIVE_INNER_SPLITS", "2" if FAST_MODE else "3"))
ADAPTIVE_TEMPERATURE = float(os.environ.get("KARAONE_ADAPTIVE_TEMPERATURE", "6"))
ADAPTIVE_MAX_BASES = int(os.environ.get("KARAONE_ADAPTIVE_MAX_BASES", "4"))
ADAPTIVE_GRID_STEP = int(os.environ.get("KARAONE_ADAPTIVE_GRID_STEP", "5" if FAST_MODE else "10"))
ADAPTIVE_THRESHOLD_STEP = float(os.environ.get("KARAONE_ADAPTIVE_THRESHOLD_STEP", "0.05" if FAST_MODE else "0.02"))

def _parse_positive_float_grid(raw: str, default: str) -> list[float]:
    """Parse a deterministic positive decimal grid for train-only adaptation."""
    text = str(raw if raw is not None else default).strip()
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Adaptive vowel weights must be finite and > 0; got {token!r}")
        values.append(value)
    if not values:
        raise ValueError("KARAONE_ADAPTIVE_VOWEL_WEIGHTS must contain at least one positive value")
    return sorted({round(float(value), 10) for value in values})

_default_vowel_weight_grid = "1.0,2.0" if FAST_MODE else "1.0,1.5,2.0,3.0,4.0"
ADAPTIVE_FINE_GRID = os.environ.get("KARAONE_FINE_ADAPTIVE_VOWEL_GRID", "0").lower() in {"1", "true", "yes"}
if ADAPTIVE_FINE_GRID and "KARAONE_ADAPTIVE_VOWEL_WEIGHTS" not in os.environ:
    _default_vowel_weight_grid = "1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.75,2.0"
ADAPTIVE_VOWEL_WEIGHTS = _parse_positive_float_grid(
    os.environ.get("KARAONE_ADAPTIVE_VOWEL_WEIGHTS", _default_vowel_weight_grid),
    _default_vowel_weight_grid,
)
ADAPTIVE_VOWEL_BA_FLOOR = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_BA_FLOOR", "0.50"))
ADAPTIVE_VOWEL_MIN_RECALL = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_MIN_RECALL", "0.40"))
ADAPTIVE_VOWEL_BA_TOLERANCE = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_BA_TOLERANCE", "0.02"))
# A non-unit vowel weight must show a meaningful inner vowel-recall gain and
# must not reduce inner BA/consonant recall beyond these fixed tolerances.
ADAPTIVE_VOWEL_MIN_GAIN = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_MIN_GAIN", "0.05" if FAST_MODE else "0.03"))
ADAPTIVE_VOWEL_MAX_BA_DROP = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_MAX_BA_DROP", "0.01"))
ADAPTIVE_VOWEL_MAX_CONSONANT_RECALL_DROP = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_MAX_CONSONANT_RECALL_DROP", "0.05"))
ADAPTIVE_VOWEL_THRESHOLD_STEP = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_THRESHOLD_STEP", "0.05" if FAST_MODE else "0.02"))
ADAPTIVE_VOWEL_THRESHOLD_MIN = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_THRESHOLD_MIN", "0.35" if FAST_MODE else "0.30"))
ADAPTIVE_VOWEL_THRESHOLD_MAX = float(os.environ.get("KARAONE_ADAPTIVE_VOWEL_THRESHOLD_MAX", "0.65" if FAST_MODE else "0.70"))
ADAPTIVE_VOWEL_INNER_SPLITS = int(os.environ.get("KARAONE_ADAPTIVE_VOWEL_INNER_SPLITS", str(INNER_SPLITS)))
ADAPTIVE_METRIC_AGGREGATION = os.environ.get("KARAONE_ADAPTIVE_METRIC_AGGREGATION", "subject_macro").lower().strip()
if ADAPTIVE_METRIC_AGGREGATION not in {"pooled", "subject_macro"}:
    raise ValueError("KARAONE_ADAPTIVE_METRIC_AGGREGATION must be pooled or subject_macro")
REQUESTED_ADAPTIVE_VOWEL_MODE = os.environ.get(
    "KARAONE_ADAPTIVE_VOWEL_MODE",
    "class_weight_fast" if FAST_MODE else "class_weight_full",
).lower().strip()
REQUIRE_ADAPTIVE_VOWEL_CLASS_WEIGHT = os.environ.get(
    "KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT", "1"
).lower() in {"1", "true", "yes"}
ALLOW_THRESHOLD_ONLY = os.environ.get(
    "KARAONE_ALLOW_THRESHOLD_ONLY", "0"
).lower() in {"1", "true", "yes"}
if REQUESTED_ADAPTIVE_VOWEL_MODE not in {"threshold_only_fast", "class_weight_fast", "class_weight_full"}:
    raise ValueError("KARAONE_ADAPTIVE_VOWEL_MODE must be threshold_only_fast, class_weight_fast, or class_weight_full")
if REQUESTED_ADAPTIVE_VOWEL_MODE == "threshold_only_fast" and not (
    ALLOW_THRESHOLD_ONLY and not REQUIRE_ADAPTIVE_VOWEL_CLASS_WEIGHT
):
    # A stale Kaggle environment variable must not silently disable the user's
    # requested vowel class-weight adaptation. Threshold-only adaptation is now
    # an explicit sensitivity analysis requiring two opt-in settings.
    ADAPTIVE_VOWEL_MODE = "class_weight_fast"
    ADAPTIVE_VOWEL_MODE_OVERRIDE = "threshold_only_fast->class_weight_fast"
else:
    ADAPTIVE_VOWEL_MODE = REQUESTED_ADAPTIVE_VOWEL_MODE
    ADAPTIVE_VOWEL_MODE_OVERRIDE = None
ENABLE_ADAPTIVE_FUSION = os.environ.get(
    "KARAONE_ENABLE_ADAPTIVE_FUSION", "1"
).lower() in {"1", "true", "yes"}
# This meta-branch selects one already-fitted requested classifier per outer
# fold using only its inner OOF metrics; it adds no model fits and no leakage.
ENABLE_ADAPTIVE_BEST_SINGLE = os.environ.get(
    "KARAONE_ENABLE_ADAPTIVE_BEST_SINGLE", "1"
).lower() in {"1", "true", "yes"}
ADAPTIVE_BEST_SINGLE_BA_TOLERANCE = float(os.environ.get(
    "KARAONE_ADAPTIVE_BEST_SINGLE_BA_TOLERANCE", "0.01"
))
STATIC_SVC_PROBABILITY = os.environ.get("KARAONE_STATIC_SVC_PROBABILITY", "0").lower() in {"1", "true", "yes"}
STATIC_FEATURE_K = int(os.environ.get("KARAONE_STATIC_FEATURE_K", "384" if FAST_MODE else "512"))
STATIC_SVC_C = float(os.environ.get("KARAONE_STATIC_SVC_C", "1.0"))
USE_SPECTRAL_SUMMARY = os.environ.get("KARAONE_USE_SPECTRAL_SUMMARY", "1").lower() in {"1", "true", "yes"}
USE_TEMPORAL_BINS = os.environ.get("KARAONE_USE_TEMPORAL_BINS", "1").lower() in {"1", "true", "yes"}
TEMPORAL_BINS = max(2, int(os.environ.get("KARAONE_TEMPORAL_BINS", "4")))
ROCKET_KERNELS = int(os.environ.get("KARAONE_ROCKET_KERNELS", "512" if FAST_MODE else "2000"))
HYDRA_KERNELS = int(os.environ.get("KARAONE_HYDRA_KERNELS", "4" if FAST_MODE else "8"))
HYDRA_GROUPS = int(os.environ.get("KARAONE_HYDRA_GROUPS", "8" if FAST_MODE else "64"))
N_JOBS = int(os.environ.get("KARAONE_N_JOBS", "2" if FAST_MODE else "-1"))
AUTO_INSTALL = os.environ.get("KARAONE_AUTO_INSTALL", "1").lower() in {"1", "true", "yes"}
REQUESTED = [x.strip().lower() for x in os.environ.get(
    "KARAONE_MODELS",
    "hydra,minirocket_hydra,multirocket,mr_hydra,catch22,summary,catch22_summary,catch22_summary_mr_hydra",
).split(",") if x.strip()]
# Optional training-only resampling: the model never receives prompt IDs as
# features. It only sees duplicated outer-training trials, with iy/uw balanced
# before the class-prior cap. Held-out prompts are never inspected.
VOWEL_PROMPT_BALANCE = os.environ.get("KARAONE_VOWEL_PROMPT_BALANCE", "1").lower() in {"1", "true", "yes"}
VOWEL_TARGET_RATIO = float(os.environ.get("KARAONE_VOWEL_TARGET_RATIO", "0.40"))
VOWEL_MAX_MULTIPLIER = float(os.environ.get("KARAONE_VOWEL_MAX_MULTIPLIER", "2.0"))
if not (0.20 <= VOWEL_TARGET_RATIO < 0.70):
    raise ValueError("KARAONE_VOWEL_TARGET_RATIO must be in [0.20, 0.70)")
if VOWEL_MAX_MULTIPLIER < 1.0:
    raise ValueError("KARAONE_VOWEL_MAX_MULTIPLIER must be >= 1.0")

MODEL_ALIASES = {
    "minirocket-hydra": "minirocket_hydra",
    "multirocket-hydra": "mr_hydra",
    "mrhydra": "mr_hydra",
    "catch22+summary": "catch22_summary",
    "catch22+summary+mrhydra": "catch22_summary_mr_hydra",
    "catch22_summary_mrhydra": "catch22_summary_mr_hydra",
}
REQUESTED = [MODEL_ALIASES.get(x, x) for x in REQUESTED]
VALID_MODELS = {
    "hydra", "minirocket_hydra", "multirocket", "mr_hydra",
    "catch22", "summary", "catch22_summary", "catch22_summary_mr_hydra",
}
UNKNOWN = sorted(set(REQUESTED) - VALID_MODELS)
if UNKNOWN:
    raise ValueError(f"Unknown KARAONE_MODELS entries: {UNKNOWN}; valid={sorted(VALID_MODELS)}")


def log(*items: Any) -> None:
    print(*items, flush=True)


def ensure_dependencies() -> None:
    """Install only missing optional Kaggle dependencies."""
    if not AUTO_INSTALL:
        return
    packages = {
        "mne": "mne",
        "meegkit": "meegkit",
        "aeon": "aeon",
    }
    missing = [pip_name for module, pip_name in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        log("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


# ---------------------------------------------------------------------------
# Metadata and raw extraction
# ---------------------------------------------------------------------------
def normalize_prompt(value: object) -> str:
    return re.sub(r"[^a-z]", "", str(value).strip().lower())


def find_subject_dir(subject: str) -> Path:
    direct = ROOT / subject
    if direct.exists():
        return direct
    matches = [p for p in ROOT.rglob(subject) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No directory for {subject} under {ROOT}")
    return sorted(matches, key=lambda p: (len(p.parts), str(p)))[0]


def first_file(directory: Path, pattern: str) -> Path | None:
    found = sorted(directory.rglob(pattern))
    return found[0] if found else None


def extract_pairs(value: object) -> list[tuple[int, int]]:
    """Convert MATLAB inclusive [start, stop] indices to Python [start, stop)."""
    arr = np.asarray(value)
    pairs: list[tuple[int, int]] = []
    if arr.dtype != object:
        a = np.asarray(arr).squeeze()
        if a.ndim == 1 and a.size == 2:
            a = a.reshape(1, 2)
        if a.ndim == 2:
            rows = a if a.shape[1] == 2 else a.T if a.shape[0] == 2 else np.empty((0, 2))
            for row in rows:
                if len(row) >= 2 and np.all(np.isfinite(row[:2])):
                    start = int(round(float(row[0]))) - 1
                    stop = int(round(float(row[1])))
                    if 0 <= start < stop:
                        pairs.append((start, stop))
            if pairs:
                return pairs
    for item in np.asarray(arr, dtype=object).reshape(-1):
        nested = np.asarray(item).squeeze()
        if nested.dtype == object and nested.size != 1:
            pairs.extend(extract_pairs(nested))
            continue
        try:
            numeric = np.asarray(nested, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if numeric.size >= 2 and np.all(np.isfinite(numeric[:2])):
            start = int(round(float(numeric[0]))) - 1
            stop = int(round(float(numeric[1])))
            if 0 <= start < stop:
                pairs.append((start, stop))
    return pairs


def load_pairs(path: Path, name: str) -> list[tuple[int, int]]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if name not in mat:
        raise KeyError(f"{path} has no {name}; available={[k for k in mat if not k.startswith('__')]}")
    pairs = extract_pairs(mat[name])
    if not pairs:
        raise ValueError(f"No valid {name} intervals in {path}")
    return pairs


def load_labels(path: Path) -> list[str]:
    return [normalize_prompt(x) for x in path.read_text(errors="ignore").splitlines() if normalize_prompt(x)]


def notch_epoch(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    if PREPROCESSING_MODE in {"official", "official_set", "set"}:
        return np.asarray(epoch, dtype=np.float32)
    if not (0 < LINE_NOISE_HZ < sfreq / 2.0) or epoch.shape[1] < 32:
        return np.asarray(epoch, dtype=np.float32)
    b, a = iirnotch(LINE_NOISE_HZ, LINE_NOISE_Q, fs=sfreq)
    return filtfilt(b, a, np.asarray(epoch, dtype=np.float32), axis=1).astype(np.float32)


def fallback_subspace_reconstruct(epochs: list[np.ndarray], calibration: list[np.ndarray]) -> tuple[list[np.ndarray], str]:
    """Unsupervised high-variance subspace reconstruction fallback.

    This is used only if MEEGkit ASR is unavailable or incompatible. The
    calibration covariance comes from clearing/rest intervals, not labels.
    """
    if not calibration:
        return epochs, "fallback_skipped_no_clearing_calibration"
    pool = np.concatenate(calibration[:ASR_CALIBRATION_TRIALS], axis=1)
    cov = np.cov(pool, rowvar=True) + 1e-12 * np.eye(pool.shape[0])
    vals, vecs = np.linalg.eigh(cov)
    cutoff = np.median(vals) * max(ASR_CUTOFF, 1.0) ** 2
    keep = vals <= cutoff
    # Always retain at least half of the dimensions to avoid an all-zero result.
    if int(keep.sum()) < max(2, pool.shape[0] // 2):
        keep[:] = True
        keep[np.argsort(vals)[-pool.shape[0] // 4:]] = False
    cleaned = []
    for epoch in epochs:
        coef = vecs.T @ epoch
        cleaned.append((vecs[:, keep] @ coef[keep]).astype(np.float32))
    return cleaned, f"fallback_eigen_subspace_removed={int((~keep).sum())}"


def asr_clean(epochs: list[np.ndarray], calibration: list[np.ndarray], sfreq: float) -> tuple[list[np.ndarray], dict]:
    if PREPROCESSING_MODE in {"official", "official_set", "set"} or not ASR_ENABLED:
        return epochs, {"asr_used": False, "reason": "official_mode_or_disabled"}
    if not calibration:
        return epochs, {"asr_used": False, "reason": "no_clearing_calibration"}
    try:
        from meegkit.asr import ASR
        cal = np.concatenate(calibration[:ASR_CALIBRATION_TRIALS], axis=1)
        asr = ASR(
            sfreq=sfreq,
            cutoff=ASR_CUTOFF,
            blocksize=100,
            win_len=0.5,
            win_overlap=0.66,
            max_bad_chans=ASR_MAX_BAD_CHANS,
            max_dropout_fraction=0.1,
            min_clean_fraction=0.25,
            method="euclid",
            estimator="scm",
        )
        _, mask = asr.fit(cal)
        cleaned = [np.asarray(asr.transform(epoch), dtype=np.float32) for epoch in epochs]
        changes = []
        for raw, clean in zip(epochs, cleaned):
            denom = np.sqrt(np.mean(raw * raw)) + 1e-12
            changes.append(float(np.sqrt(np.mean((clean - raw) ** 2)) / denom))
        return cleaned, {
            "asr_used": True,
            "asr_backend": "meegkit",
            "asr_cutoff": ASR_CUTOFF,
            "calibration_source": "clearing_inds",
            "calibration_trials": int(min(len(calibration), ASR_CALIBRATION_TRIALS)),
            "calibration_retained_fraction": float(np.mean(mask)),
            "relative_rms_change": changes,
            "changed_trial_count": int(sum(v > 0.05 for v in changes)),
        }
    except Exception as exc:
        cleaned, backend = fallback_subspace_reconstruct(epochs, calibration)
        return cleaned, {
            "asr_used": bool(cleaned is not epochs),
            "asr_backend": backend,
            "asr_error": f"{type(exc).__name__}: {exc}",
            "calibration_source": "clearing_inds",
        }


def _canonical_channel_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(name)).upper()
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    return aliases.get(text, text)


def _montage_positions(channel_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return channel positions and a validity mask from the standard 10-20 montage."""
    try:
        import mne
        montage = mne.channels.make_standard_montage("standard_1020")
        position_map = {
            _canonical_channel_name(name): np.asarray(position, dtype=float)
            for name, position in montage.get_positions()["ch_pos"].items()
        }
        coords = np.zeros((len(channel_names), 3), dtype=float)
        valid = np.zeros(len(channel_names), dtype=bool)
        for idx, name in enumerate(channel_names):
            key = _canonical_channel_name(name)
            if key in position_map and np.all(np.isfinite(position_map[key])):
                coords[idx] = position_map[key]
                valid[idx] = True
        return coords, valid
    except Exception:
        return np.zeros((len(channel_names), 3), dtype=float), np.zeros(len(channel_names), dtype=bool)


def apply_spatial_filter(epoch: np.ndarray, channel_names: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """Apply a small fixed-montage Laplacian without labels or test-fold data."""
    x = np.asarray(epoch, dtype=np.float32)
    if SPATIAL_FILTER != "laplacian" or channel_names is None or len(channel_names) != x.shape[0]:
        return x, {"spatial_filter": "none" if SPATIAL_FILTER == "none" else "laplacian_unavailable"}
    coords, valid = _montage_positions(channel_names)
    if int(valid.sum()) < 3:
        return x, {"spatial_filter": "laplacian_unavailable", "valid_montage_channels": int(valid.sum())}
    out = x.astype(np.float32, copy=True)
    valid_idx = np.flatnonzero(valid)
    for idx in valid_idx:
        others = valid_idx[valid_idx != idx]
        distances = np.linalg.norm(coords[others] - coords[idx], axis=1)
        neighbours = others[np.argsort(distances)[:min(LAPLACIAN_NEIGHBORS, len(others))]]
        if len(neighbours):
            local_mean = np.mean(x[neighbours], axis=0)
            lap = x[idx] - local_mean
            out[idx] = ((1.0 - LAPLACIAN_MIX) * x[idx] + LAPLACIAN_MIX * lap).astype(np.float32)
    return out, {
        "spatial_filter": "small_laplacian",
        "laplacian_neighbors": int(LAPLACIAN_NEIGHBORS),
        "laplacian_mix": float(LAPLACIAN_MIX),
        "valid_montage_channels": int(valid.sum()),
    }


def robust_normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    x = np.asarray(epoch, dtype=np.float32)
    if not ROBUST_NORMALIZE:
        return x
    med = np.median(x, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    scale = np.maximum(np.maximum(mad, sd), 1e-12)
    z = (x - med) / scale
    return np.clip(z, -ROBUST_CLIP, ROBUST_CLIP).astype(np.float32)


def preprocess_epoch(epoch: np.ndarray, sfreq: float, channel_names: list[str] | None = None) -> np.ndarray:
    x = np.nan_to_num(np.asarray(epoch, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if PREPROCESSING_MODE not in {"official", "official_set", "set"} and x.shape[1] > 32:
        high = min(50.0, 0.45 * float(sfreq))
        if 1.0 < high:
            sos = butter(4, [1.0, high], btype="bandpass", fs=float(sfreq), output="sos")
            x = sosfiltfilt(sos, x, axis=1).astype(np.float32)
    x = resample(x, TARGET_LEN, axis=1).astype(np.float32)
    if PREPROCESSING_MODE not in {"official", "official_set", "set"}:
        x = x - x.mean(axis=0, keepdims=True)  # CAR
        x, _ = apply_spatial_filter(x, channel_names)
    return robust_normalize_epoch(x)


def qc_fraction(epoch: np.ndarray, sfreq: float) -> float:
    """Robust fraction of overlapping windows with multi-signature artifacts."""
    x = np.nan_to_num(np.asarray(epoch, dtype=np.float64))
    win = max(32, int(round(QC_WINDOW_SECONDS * sfreq)))
    step = max(16, int(round(QC_STEP_SECONDS * sfreq)))
    starts = list(range(0, max(1, x.shape[1] - win + 1), step))
    if not starts or starts[-1] != max(0, x.shape[1] - win):
        starts.append(max(0, x.shape[1] - win))
    rec = []
    for st in starts:
        w = x[:, st:st + win]
        med = np.median(w, axis=1, keepdims=True)
        cen = w - med
        rms = np.sqrt(np.mean(cen * cen, axis=1) + 1e-12)
        ptp = np.percentile(cen, 99, axis=1) - np.percentile(cen, 1, axis=1)
        hf = np.sqrt(np.mean(np.diff(w, axis=1) ** 2, axis=1) + 1e-12)
        common = float(np.sqrt(np.mean(np.mean(cen, axis=0) ** 2) + 1e-12))
        flat = float(np.mean(np.mean(np.abs(np.diff(w, axis=1)) < 1e-8, axis=1) > 0.5))
        rec.append([float(np.median(rms)), float(np.median(ptp)), float(np.median(hf)), common, flat])
    a = np.asarray(rec)
    med = np.median(a, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(a - med), axis=0), 1e-12)
    z = np.abs(a - med) / mad
    bad = (np.any(z >= QC_Z, axis=1) | (z[:, 4] >= QC_Z)).mean()
    return float(bad)


def regress_eog_from_eeg(raw, eeg_picks: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove linear EOG components from continuous EEG, without labels."""
    if not EOG_REGRESSION:
        return data, {"eog_regression": False, "reason": "disabled"}
    try:
        import mne
        eog_picks = mne.pick_types(raw.info, eeg=False, eog=True, misc=False, exclude="bads")
        if len(eog_picks) == 0:
            name_pattern = re.compile(r"(?:EOG|VEOG|HEOG|VEO|HEO|EYE)", re.IGNORECASE)
            eog_picks = np.asarray([i for i, name in enumerate(raw.ch_names) if name_pattern.search(str(name))], dtype=int)
        if len(eog_picks) == 0:
            return data, {"eog_regression": False, "reason": "no_eog_channels"}
        eog = np.nan_to_num(raw.get_data(picks=eog_picks).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        eeg = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        eog_centered = eog - np.median(eog, axis=1, keepdims=True)
        eeg_centered = eeg - np.median(eeg, axis=1, keepdims=True)
        if eog_centered.shape[1] < max(100, 2 * eog_centered.shape[0]):
            return data, {"eog_regression": False, "reason": "insufficient_samples", "eog_channels": int(len(eog_picks))}
        coef, *_ = np.linalg.lstsq(eog_centered.T, eeg_centered.T, rcond=None)
        cleaned = eeg - (coef.T @ eog_centered)
        return cleaned.astype(np.float32), {
            "eog_regression": True,
            "eog_channels": int(len(eog_picks)),
            "eog_channel_names": [str(raw.ch_names[int(i)]) for i in eog_picks],
        }
    except Exception as exc:
        return data, {"eog_regression": False, "reason": f"{type(exc).__name__}: {exc}"}


def load_subject(subject: str) -> dict:
    import mne
    directory = find_subject_dir(subject)
    set_candidates = sorted(directory.glob("*.set")) or sorted(directory.glob("set_files/*.set")) or sorted(directory.rglob("*.set"))
    cnt_candidates = sorted(directory.glob("*.cnt")) or sorted(directory.rglob("*.cnt"))
    epoch_path = first_file(directory, "epoch_inds.mat")
    labels_path = first_file(directory, "labels.txt")
    if (not set_candidates and not cnt_candidates) or epoch_path is None or labels_path is None:
        raise FileNotFoundError(f"{subject}: requires *.set or *.cnt, epoch_inds.mat, and labels.txt under {directory}")
    recording_path = set_candidates[0] if set_candidates else cnt_candidates[0]
    log(f"{subject}: reading {recording_path.name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if recording_path.suffix.lower() == ".set":
            raw = mne.io.read_raw_eeglab(recording_path, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_cnt(recording_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True, eog=False, misc=False, exclude="bads")
    if len(picks) < TARGET_CHANNELS:
        raise ValueError(f"{subject}: only {len(picks)} EEG channels available")
    picks = np.asarray(picks[:TARGET_CHANNELS], dtype=int)
    data = raw.get_data(picks=picks).astype(np.float32)
    data, eog_info = regress_eog_from_eeg(raw, picks, data)
    channel_names = [raw.ch_names[int(i)] for i in picks]
    thinking = load_pairs(epoch_path, "thinking_inds")
    labels = load_labels(labels_path)
    if len(thinking) != len(labels):
        raise ValueError(f"{subject}: thinking_inds={len(thinking)} labels={len(labels)}")
    try:
        clearing = load_pairs(epoch_path, "clearing_inds")
    except Exception:
        clearing = []
    calibration = []
    for st, en in clearing[:ASR_CALIBRATION_TRIALS]:
        if 0 <= st < en <= data.shape[1]:
            calibration.append(notch_epoch(data[:, st:en], sfreq))
    raw_epochs, prompts, y = [], [], []
    excluded_words = 0
    for (st, en), prompt in zip(thinking, labels):
        if prompt in WORD_PROMPTS:
            excluded_words += 1
            continue
        if prompt not in KEEP_PROMPTS or not (0 <= st < en <= data.shape[1]):
            continue
        raw_epochs.append(notch_epoch(data[:, st:en], sfreq))
        prompts.append(prompt)
        y.append(0 if prompt in VOWELS else 1)
    if not raw_epochs:
        raise ValueError(f"{subject}: no phonemic trials")
    raw_qcs = [qc_fraction(e, sfreq) for e in raw_epochs]
    cleaned, asr_info = asr_clean(raw_epochs, calibration, sfreq)
    # The requested noise rule is applied after the requested ASR stage. This
    # avoids rejecting trials solely because the artifact detector measured
    # contamination before the artifact-reconstruction step.
    qcs = [qc_fraction(e, sfreq) for e in cleaned]
    keep = np.asarray([q <= QC_THRESHOLD for q in qcs], dtype=bool) if APPLY_QC else np.ones(len(raw_epochs), dtype=bool)
    y_arr = np.asarray(y, dtype=np.int64)
    y_keep = y_arr[keep]
    rejected_count = int((~keep).sum())
    rejection_fraction = float(rejected_count / max(len(raw_epochs), 1))
    class_coverage_ok = int(np.sum(y_keep == 0)) >= 5 and int(np.sum(y_keep == 1)) >= 5
    subject_rejected = bool(rejection_fraction > SUBJECT_REJECTION_FRACTION or not class_coverage_ok)
    rejection_reasons = []
    if rejection_fraction > SUBJECT_REJECTION_FRACTION:
        rejection_reasons.append(f"rejection_fraction>{SUBJECT_REJECTION_FRACTION:.3f}")
    if not class_coverage_ok:
        rejection_reasons.append("insufficient_post_QC_class_coverage")
    if np.any(keep):
        X = np.stack([preprocess_epoch(cleaned[i], sfreq, channel_names) for i in range(len(cleaned)) if keep[i]]).astype(np.float32)
    else:
        X = np.empty((0, TARGET_CHANNELS, TARGET_LEN), dtype=np.float32)
    prompts_keep = [p for p, k in zip(prompts, keep) if k]
    rec = {
        "subject": subject,
        "X": X,
        "y": y_keep,
        "prompts": prompts_keep,
        "raw_phonemic_count": len(raw_epochs),
        "kept_count": int(keep.sum()),
        "rejected_count": rejected_count,
        "rejection_fraction": rejection_fraction,
        "subject_rejected": subject_rejected,
        "subject_rejection_reason": ";".join(rejection_reasons) if rejection_reasons else None,
        "class_coverage_after_qc": {"vowel": int(np.sum(y_keep == 0)), "consonant": int(np.sum(y_keep == 1))},
        "excluded_words": int(excluded_words),
        "qc_stage": "post_asr_pre_resample_pre_car",
        "qc_fraction_min_median_max": [float(np.min(qcs)), float(np.median(qcs)), float(np.max(qcs))],
        "raw_qc_fraction_min_median_max": [float(np.min(raw_qcs)), float(np.median(raw_qcs)), float(np.max(raw_qcs))],
        "qc_threshold": QC_THRESHOLD,
        "asr_info": asr_info,
        "eog_info": eog_info,
        "spatial_filter": {"mode": SPATIAL_FILTER, "neighbors": LAPLACIAN_NEIGHBORS, "mix": LAPLACIAN_MIX},
        "sfreq": sfreq,
        "channel_names": channel_names,
        "epoch_metadata_source": str(epoch_path),
        "label_source": str(labels_path),
        "recording_path": str(recording_path),
        "recording_format": recording_path.suffix.lower().lstrip("."),
    }
    log(f"{subject}: raw_phonemic={len(raw_epochs)} kept={len(X)} rejected={rejected_count} reject_frac={rejection_fraction:.3f} subject_rejected={subject_rejected} V={int((y_keep==0).sum())} C={int((y_keep==1).sum())} QC[min/med/max]={min(qcs):.3f}/{np.median(qcs):.3f}/{max(qcs):.3f} EOG={eog_info.get('eog_regression')} ASR={asr_info.get('asr_backend', asr_info.get('reason', False))}")
    return rec


COHORT_SELECTION_AUDIT: dict[str, Any] = {}


def load_cohort() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load candidates, apply the fixed 10% noise rule, and rank by noise only."""
    global SUBJECTS, COHORT_SELECTION_AUDIT
    candidate_names = list(SUBJECTS)
    candidate_records: list[dict] = []
    missing_or_failed: list[dict] = []
    for subject in candidate_names:
        try:
            candidate_records.append(load_subject(subject))
        except Exception as exc:
            if SUBJECT_OVERRIDE:
                raise
            missing_or_failed.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            log(f"{subject}: skipped candidate ({type(exc).__name__}: {exc})")

    rankable = [r for r in candidate_records if r["kept_count"] > 0 and r["class_coverage_after_qc"]["vowel"] > 0 and r["class_coverage_after_qc"]["consonant"] > 0]
    rankable.sort(key=lambda r: (
        float(r["rejection_fraction"]),
        float(r["qc_fraction_min_median_max"][1]),
        float(r["qc_fraction_min_median_max"][2]),
        str(r["subject"]),
    ))
    strict_eligible = [r for r in rankable if not r["subject_rejected"]]
    selection_pool = strict_eligible if ENFORCE_SUBJECT_REJECTION else rankable
    selected = selection_pool[:MAX_SUBJECTS]
    selected_names = [r["subject"] for r in selected]
    rejected_subjects = [
        {"subject": r["subject"], "reason": r.get("subject_rejection_reason"), "rejection_fraction": r["rejection_fraction"], "qc_fraction_min_median_max": r["qc_fraction_min_median_max"]}
        for r in candidate_records if r["subject"] not in selected_names
    ]
    COHORT_SELECTION_AUDIT = {
        "candidate_subjects": candidate_names,
        "candidate_records": [{k: v for k, v in r.items() if k not in {"X", "y"}} for r in candidate_records],
        "missing_or_failed_candidates": missing_or_failed,
        "selection_rule": "remove each trial with post-ASR noise fraction >10%; rank subjects with usable post-QC data by subject rejection fraction, median trial noise, maximum trial noise, then subject ID; select the first eight; no model results used",
        "trial_noise_threshold": QC_THRESHOLD,
        "subject_rejection_fraction_threshold": SUBJECT_REJECTION_FRACTION,
        "enforce_subject_rejection": ENFORCE_SUBJECT_REJECTION,
        "strict_eligible_count": len(strict_eligible),
        "rankable_subject_count": len(rankable),
        "max_subjects": MAX_SUBJECTS,
        "selected_subjects": selected_names,
        "selected_subjects_flagged_above_subject_threshold": [r["subject"] for r in selected if r["subject_rejected"]],
        "not_selected_subjects": rejected_subjects,
    }
    if REQUIRE_BEST_EIGHT and len(selected) < MAX_SUBJECTS:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "cohort_selection_audit.json").write_text(json.dumps(COHORT_SELECTION_AUDIT, indent=2, default=str))
        log(f"Best-eight gate: {len(selected)}/{MAX_SUBJECTS} rankable subjects were available. Candidate audit saved to {OUT / 'cohort_selection_audit.json'}")
        raise RuntimeError(f"Only {len(selected)} rankable subjects were available; {MAX_SUBJECTS} are required. Set KARAONE_REQUIRE_BEST_EIGHT=0 only for an explicitly separate smaller-cohort run.")
    if len(selected) < 2:
        raise RuntimeError("At least two eligible subjects are required for subject-level evaluation.")
    SUBJECTS = selected_names
    X = np.concatenate([r["X"] for r in selected], axis=0)
    y = np.concatenate([r["y"] for r in selected], axis=0)
    groups = np.concatenate([np.repeat(r["subject"], len(r["y"])) for r in selected])
    prompts = np.asarray([p for r in selected for p in r["prompts"]], dtype=str)
    counts = np.bincount(y, minlength=2)
    log(f"SELECTED COHORT: X={X.shape} trials={len(y)} vowels={int(counts[0])} consonants={int(counts[1])} subjects={selected_names}")
    if REQUIRE_EXPECTED and (len(y) != EXPECTED_TOTAL or int(counts[0]) != EXPECTED_VOWELS or int(counts[1]) != EXPECTED_CONSONANTS):
        raise RuntimeError(f"Cohort integrity gate failed: got total/V/C={len(y)}/{counts[0]}/{counts[1]}, expected={EXPECTED_TOTAL}/{EXPECTED_VOWELS}/{EXPECTED_CONSONANTS}. Set KARAONE_REQUIRE_EXPECTED_COHORT=0 only for an explicitly separate cohort.")
    return X, y, groups, prompts, selected


# ---------------------------------------------------------------------------
# Feature branches and model wrappers
# ---------------------------------------------------------------------------
def safe_stat_features(X: np.ndarray) -> np.ndarray:
    """Compute label-independent temporal and spectral EEG features.

    Temporal statistics are followed, by default, with log band power and
    relative band power for theta, alpha, beta, and low-gamma bands. The
    spectral features are deterministic transforms of each epoch and are
    therefore safe to compute inside every training-only feature pipeline.
    They do not use labels or the held-out subject.
    """
    x = np.asarray(X, dtype=np.float32)
    cen = x - np.mean(x, axis=2, keepdims=True)
    sd = np.std(cen, axis=2) + 1e-6
    q25, q75 = np.percentile(x, [25, 75], axis=2)
    features = np.stack([
        np.mean(x, axis=2), np.mean(np.abs(x), axis=2), np.std(x, axis=2),
        np.sqrt(np.mean(x * x, axis=2) + 1e-12), np.min(x, axis=2), np.max(x, axis=2),
        q25, q75, q75 - q25, np.median(x, axis=2),
        np.mean(cen ** 3, axis=2) / (sd ** 3),
        np.mean(cen ** 4, axis=2) / (sd ** 4) - 3.0,
        np.mean(np.abs(np.diff(x, axis=2)), axis=2),
    ], axis=2)
    if USE_TEMPORAL_BINS and x.shape[2] >= TEMPORAL_BINS * 4:
        # Preserve coarse temporal evolution that whole-epoch statistics can
        # erase. Each bin contributes mean, standard deviation, and average
        # absolute temporal derivative per channel.
        edges = np.linspace(0, x.shape[2], TEMPORAL_BINS + 1, dtype=int)
        temporal_parts = []
        for start, stop in zip(edges[:-1], edges[1:]):
            segment = x[:, :, start:stop]
            temporal_parts.extend([
                np.mean(segment, axis=2),
                np.std(segment, axis=2),
                np.mean(np.abs(np.diff(segment, axis=2)), axis=2),
            ])
        features = np.concatenate([features, np.stack(temporal_parts, axis=2)], axis=2)
    if USE_SPECTRAL_SUMMARY and x.shape[2] >= 32:
        # The epochs are resampled to EFFECTIVE_FS before feature extraction.
        # Relative power is normalized by the 1--45 Hz power, avoiding the
        # 50-Hz notch edge and the DC component.
        freqs = np.fft.rfftfreq(x.shape[2], d=1.0 / EFFECTIVE_FS)
        fft = np.fft.rfft(x, axis=2)
        psd = (np.abs(fft) ** 2) / max(1, x.shape[2])
        valid = (freqs >= 1.0) & (freqs <= 45.0)
        total = np.sum(psd[:, :, valid], axis=2, keepdims=True) + 1e-8
        spectral_parts = []
        for low, high in ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)):
            band = (freqs >= low) & (freqs < high)
            power = np.mean(psd[:, :, band], axis=2)
            spectral_parts.extend([np.log1p(power), power / total[:, :, 0]])
        features = np.concatenate([features, np.stack(spectral_parts, axis=2)], axis=2)
    pooled = np.mean(features, axis=1)
    return np.nan_to_num(np.concatenate([features.reshape(len(x), -1), pooled], axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class StaticModel:
    def __init__(self, kind: str, seed: int = SEED, vowel_weight: float = 1.0, class_weight: dict | str | None = None):
        self.kind = kind
        self.seed = seed
        self.vowel_weight = float(vowel_weight)
        self.class_weight = class_weight if class_weight is not None else "balanced"
        self.catch = None
        self.scaler = None
        self.variance = None
        self.selector = None
        self.svc = None

    def make_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        parts = []
        if self.kind in {"catch22", "catch22_summary"}:
            from aeon.transformations.collection.feature_based import Catch22
            if fit:
                self.catch = Catch22(catch24=False, replace_nans=True, outlier_norm=True, n_jobs=N_JOBS)
                parts.append(self.catch.fit_transform(X))
            else:
                parts.append(self.catch.transform(X))
        if self.kind in {"summary", "catch22_summary"}:
            parts.append(safe_stat_features(X))
        return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        F = self.make_features(X, fit=True)
        self.scaler = StandardScaler().fit(F)
        scaled = self.scaler.transform(F)
        self.variance = VarianceThreshold(0.0).fit(scaled)
        reduced = self.variance.transform(scaled)
        k = min(STATIC_FEATURE_K, reduced.shape[1])
        self.selector = SelectKBest(f_classif, k=k).fit(reduced, y)
        selected = self.selector.transform(reduced)
        self.svc = SVC(C=STATIC_SVC_C, kernel="rbf", gamma="scale", class_weight=self.class_weight, probability=STATIC_SVC_PROBABILITY, random_state=self.seed)
        self.svc.fit(selected, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        F = self.make_features(X, fit=False)
        reduced = self.variance.transform(self.scaler.transform(F))
        selected = self.selector.transform(reduced)
        if getattr(self.svc, "probability", False):
            p = np.asarray(self.svc.predict_proba(selected), dtype=float)
            classes = np.asarray(self.svc.classes_)
            return p[:, int(np.where(classes == 1)[0][0])]
        decision = np.asarray(self.svc.decision_function(selected), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-np.clip(decision, -30.0, 30.0)))


class AeonModel:
    def __init__(self, kind: str, seed: int = SEED, vowel_weight: float = 1.0, class_weight: dict | str | None = None):
        self.kind = kind
        self.seed = seed
        self.vowel_weight = float(vowel_weight)
        self.class_weight = class_weight if class_weight is not None else "balanced"
        self.model = None

    def _new(self):
        from aeon.classification.convolution_based import (
            HydraClassifier, MiniRocketClassifier, MultiRocketClassifier, MultiRocketHydraClassifier,
        )
        rocket_kernels = max(128, int(ROCKET_KERNELS))
        if self.kind == "hydra":
            return HydraClassifier(n_kernels=HYDRA_KERNELS, n_groups=HYDRA_GROUPS, class_weight=self.class_weight, n_jobs=N_JOBS, random_state=self.seed)
        if self.kind == "minirocket":
            return MiniRocketClassifier(n_kernels=rocket_kernels, max_dilations_per_kernel=32, class_weight=self.class_weight, n_jobs=N_JOBS, random_state=self.seed)
        if self.kind == "multirocket":
            return MultiRocketClassifier(n_kernels=rocket_kernels, max_dilations_per_kernel=32, n_features_per_kernel=4, class_weight=self.class_weight, n_jobs=N_JOBS, random_state=self.seed)
        if self.kind == "mr_hydra":
            return MultiRocketHydraClassifier(n_kernels=HYDRA_KERNELS, n_groups=HYDRA_GROUPS, class_weight=self.class_weight, n_jobs=N_JOBS, random_state=self.seed)
        raise ValueError(self.kind)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = self._new()
        self.model.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            p = np.asarray(self.model.predict_proba(X), dtype=float)
            classes = np.asarray(getattr(self.model, "classes_", [0, 1]))
            if p.ndim == 2 and p.shape[1] > 1:
                return p[:, int(np.where(classes == 1)[0][0])]
        if hasattr(self.model, "decision_function"):
            d = np.asarray(self.model.decision_function(X), dtype=float).reshape(-1)
            return 1.0 / (1.0 + np.exp(-np.clip(d, -30, 30)))
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1)


class MiniRocketHydraFusion:
    """Portable MiniRocket+Hydra fallback for missing Aeon MiniRocketHydraClassifier."""
    def __init__(self, seed: int = SEED, vowel_weight: float = 1.0, class_weight: dict | str | None = None):
        self.vowel_weight = float(vowel_weight)
        self.class_weight = class_weight if class_weight is not None else "balanced"
        self.mini = AeonModel("minirocket", seed, vowel_weight=self.vowel_weight, class_weight=self.class_weight)
        self.hydra = AeonModel("hydra", seed, vowel_weight=self.vowel_weight, class_weight=self.class_weight)

    def fit(self, X, y):
        self.mini.fit(X, y); self.hydra.fit(X, y); return self

    def score(self, X):
        return 0.5 * self.mini.score(X) + 0.5 * self.hydra.score(X)


def make_model(name: str, vowel_weight: float = 1.0, class_weight: dict | str | None = None):
    if name in {"catch22", "summary", "catch22_summary"}:
        return StaticModel(name, vowel_weight=vowel_weight, class_weight=class_weight)
    if name == "minirocket_hydra":
        return MiniRocketHydraFusion(vowel_weight=vowel_weight, class_weight=class_weight)
    if name in {"hydra", "multirocket", "mr_hydra"}:
        return AeonModel(name, vowel_weight=vowel_weight, class_weight=class_weight)
    raise ValueError(name)


def _vowel_prompt_balanced_training_data(
    X: np.ndarray,
    y: np.ndarray,
    prompts: np.ndarray | list[str] | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resample only training trials to reduce vowel and iy/uw imbalance.

    This is deliberately performed inside every inner-training split and again
    on the complete outer-training set. It is not feature extraction and does
    not inspect outer-test labels or prompts. The cap prevents an aggressive
    class-prior shift from turning the classifier into an all-vowel predictor.
    """
    x = np.asarray(X)
    labels = np.asarray(y, dtype=int).reshape(-1)
    if not VOWEL_PROMPT_BALANCE or prompts is None or len(prompts) != len(labels):
        return x, labels, {"enabled": False, "n_before": int(len(labels)), "n_after": int(len(labels))}
    p = np.asarray([normalize_prompt(v) for v in prompts], dtype=object)
    rng = np.random.default_rng(int(seed))
    all_idx = np.arange(len(labels), dtype=int)
    vowel_idx = np.flatnonzero(labels == 0)
    consonant_idx = np.flatnonzero(labels == 1)
    if len(vowel_idx) == 0 or len(consonant_idx) == 0:
        return x, labels, {"enabled": False, "reason": "one_class_training_data", "n_before": int(len(labels)), "n_after": int(len(labels))}

    # First equalize iy and uw up to the larger observed training count. If one
    # prompt is absent, no synthetic prompt identity is invented.
    prompt_indices = {name: np.flatnonzero((labels == 0) & (p == name)) for name in ("iy", "uw")}
    available = [v for v in prompt_indices.values() if len(v) > 0]
    if len(available) >= 2:
        prompt_target = max(len(v) for v in available)
        vowel_parts = []
        for idx in available:
            if len(idx) < prompt_target:
                idx = np.concatenate([idx, rng.choice(idx, size=prompt_target - len(idx), replace=True)])
            vowel_parts.append(idx)
        vowel_balanced = np.concatenate(vowel_parts)
    else:
        vowel_balanced = vowel_idx.copy()

    desired_vowel_total = int(np.ceil(len(consonant_idx) * VOWEL_TARGET_RATIO / max(1e-8, 1.0 - VOWEL_TARGET_RATIO)))
    max_vowel_total = int(np.floor(len(vowel_idx) * VOWEL_MAX_MULTIPLIER))
    desired_vowel_total = min(max(desired_vowel_total, len(vowel_balanced)), max_vowel_total)
    if len(vowel_balanced) < desired_vowel_total:
        extra = rng.choice(vowel_balanced, size=desired_vowel_total - len(vowel_balanced), replace=True)
        vowel_balanced = np.concatenate([vowel_balanced, extra])
    elif len(vowel_balanced) > desired_vowel_total:
        vowel_balanced = rng.choice(vowel_balanced, size=desired_vowel_total, replace=False)

    selected = np.concatenate([consonant_idx, vowel_balanced])
    rng.shuffle(selected)
    out = {
        "enabled": True,
        "n_before": int(len(labels)),
        "n_after": int(len(selected)),
        "vowel_before": int(len(vowel_idx)),
        "vowel_after": int(np.sum(labels[selected] == 0)),
        "consonant_after": int(np.sum(labels[selected] == 1)),
        "iy_before": int(np.sum((labels == 0) & (p == "iy"))),
        "uw_before": int(np.sum((labels == 0) & (p == "uw"))),
        "target_vowel_ratio": float(VOWEL_TARGET_RATIO),
        "max_vowel_multiplier": float(VOWEL_MAX_MULTIPLIER),
    }
    return x[selected], labels[selected], out


# ---------------------------------------------------------------------------
# Metrics, adaptive fusion, and evaluation
# ---------------------------------------------------------------------------
def metrics(y: np.ndarray, score: np.ndarray, subject: str, model: str, fold: int, extra: dict | None = None, threshold: float = 0.5) -> dict:
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.5, posinf=1.0, neginf=0.0).reshape(-1)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    pred = (s >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {
        "model": model, "fold": int(fold), "test_subject": subject, "n_test": int(len(y)),
        "decision_threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "vowel_precision": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "vowel_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "consonant_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "consonant_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "confusion_matrix": cm.tolist(),
    }
    if extra:
        out.update(extra)
    return out


def summarize(rows: list[dict]) -> dict:
    result = {"n_folds": len(rows)}
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "vowel_precision", "vowel_recall", "consonant_precision", "consonant_recall", "roc_auc", "average_precision"]:
        a = np.asarray([r[key] for r in rows], dtype=float)
        result[key + "_mean"] = float(np.nanmean(a))
        result[key + "_std"] = float(np.nanstd(a, ddof=1)) if np.sum(np.isfinite(a)) > 1 else 0.0
    result["pooled_confusion_matrix"] = np.sum(np.asarray([r["confusion_matrix"] for r in rows], dtype=int), axis=0).tolist()
    return result


# Stable, flat exports for every successful outer fold. Nested adaptive metadata
# remains available in final_results.json; these columns make the key diagnostics
# readable in Kaggle without hiding low-performing folds.
_FOLD_TABLE_METRICS = [
    "accuracy", "balanced_accuracy", "macro_f1", "vowel_precision", "vowel_recall",
    "consonant_precision", "consonant_recall", "roc_auc", "average_precision",
]
_FOLD_TABLE_COLUMNS = [
    "fold", "test_subject", "model", "n_test", *_FOLD_TABLE_METRICS,
    "decision_threshold", "adaptive_vowel_weight", "adaptive_class_weight_vowel_0",
    "adaptive_class_weight_consonant_1", "adaptive_weights", "vowel_prompt_balance", "source_model",
    "fit_seconds", "inner_selected_balanced_accuracy", "inner_selected_vowel_recall",
    "inner_selected_consonant_recall", "confusion_matrix", "selection_rule",
]


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return str(value)


def _fold_table_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        class_weight = row.get("adaptive_class_weight") or {}
        source_class_weight = row.get("adaptive_single_source_class_weight") or {}
        class_weight_vowel = class_weight.get("vowel_0", source_class_weight.get("vowel_0", np.nan))
        class_weight_consonant = class_weight.get("consonant_1", source_class_weight.get("consonant_1", np.nan))
        source_model = row.get("adaptive_single_source_model", "")
        source_weight = row.get("adaptive_single_source_vowel_weight", np.nan)
        if "adaptive_vowel_weight" in row:
            source_weight = row.get("adaptive_vowel_weight")
        record = {key: row.get(key, np.nan) for key in _FOLD_TABLE_METRICS}
        record.update({
            "fold": int(row.get("fold", -1)),
            "test_subject": str(row.get("test_subject", "")),
            "model": str(row.get("model", "")),
            "n_test": int(row.get("n_test", 0)),
            "decision_threshold": row.get("decision_threshold", np.nan),
            "adaptive_vowel_weight": source_weight,
            "adaptive_class_weight_vowel_0": class_weight_vowel,
            "adaptive_class_weight_consonant_1": class_weight_consonant,
            "adaptive_weights": _json_cell(row.get("adaptive_weights", {})),
            "vowel_prompt_balance": _json_cell(row.get("vowel_prompt_balance", {})),
            "source_model": source_model,
            "fit_seconds": row.get("fit_seconds", np.nan),
            "inner_selected_balanced_accuracy": row.get("inner_selected_balanced_accuracy", np.nan),
            "inner_selected_vowel_recall": row.get("inner_selected_vowel_recall", np.nan),
            "inner_selected_consonant_recall": row.get("inner_selected_consonant_recall", np.nan),
            "confusion_matrix": _json_cell(row.get("confusion_matrix", [])),
            "selection_rule": row.get("adaptive_weight_selection_rule", row.get("fusion_selection_rule", row.get("adaptive_single_selection_rule", ""))),
        })
        records.append(record)
    frame = pd.DataFrame(records, columns=_FOLD_TABLE_COLUMNS)
    if len(frame):
        frame = frame.sort_values(["fold", "test_subject", "model"], kind="stable").reset_index(drop=True)
    return frame


_WEIGHT_TABLE_COLUMNS = [
    "fold", "test_subject", "model", "n_test", "candidate_stage", "candidate_weight",
    "candidate_selected", "candidate_usable", "candidate_feasible", "candidate_threshold",
    "inner_accuracy", "inner_balanced_accuracy", "inner_macro_f1", "inner_vowel_recall",
    "inner_consonant_recall", "selected_weight", "outer_accuracy", "outer_balanced_accuracy",
    "outer_macro_f1", "outer_vowel_recall", "outer_consonant_recall", "outer_threshold",
    "selection_rule", "adaptive_metric_aggregation", "adaptive_weights",
]


def _candidate_weight_table_frame(rows: list[dict]) -> pd.DataFrame:
    """Expand nested inner candidate records without adding outer-test fits.

    Candidate metrics are inner subject-disjoint validation metrics. The outer
    columns are repeated only as a join/reference to the final selected model
    result for that fold; they are never used to select a candidate.
    """
    records = []
    for row in rows:
        candidates = row.get("inner_vowel_weight_candidates")
        selected_weight = row.get("adaptive_vowel_weight", row.get("adaptive_single_source_vowel_weight", np.nan))
        if candidates:
            for candidate in candidates:
                candidate_weight = candidate.get("vowel_weight", np.nan)
                try:
                    is_selected = bool(np.isfinite(float(selected_weight)) and np.isfinite(float(candidate_weight)) and abs(float(selected_weight) - float(candidate_weight)) < 1e-9)
                except Exception:
                    is_selected = False
                records.append({
                    "fold": int(row.get("fold", -1)),
                    "test_subject": str(row.get("test_subject", "")),
                    "model": str(row.get("model", "")),
                    "n_test": int(row.get("n_test", 0)),
                    "candidate_stage": "inner_validation_candidate",
                    "candidate_weight": candidate_weight,
                    "candidate_selected": is_selected,
                    "candidate_usable": bool(candidate.get("usable", True)),
                    "candidate_feasible": candidate.get("feasible", np.nan),
                    "candidate_threshold": candidate.get("threshold", np.nan),
                    "inner_accuracy": candidate.get("accuracy", np.nan),
                    "inner_balanced_accuracy": candidate.get("balanced_accuracy", np.nan),
                    "inner_macro_f1": candidate.get("macro_f1", np.nan),
                    "inner_vowel_recall": candidate.get("vowel_recall", np.nan),
                    "inner_consonant_recall": candidate.get("consonant_recall", np.nan),
                    "selected_weight": selected_weight,
                    "outer_accuracy": row.get("accuracy", np.nan),
                    "outer_balanced_accuracy": row.get("balanced_accuracy", np.nan),
                    "outer_macro_f1": row.get("macro_f1", np.nan),
                    "outer_vowel_recall": row.get("vowel_recall", np.nan),
                    "outer_consonant_recall": row.get("consonant_recall", np.nan),
                    "outer_threshold": row.get("decision_threshold", np.nan),
                    "selection_rule": row.get("adaptive_weight_selection_rule", ""),
                    "adaptive_metric_aggregation": row.get("adaptive_metric_aggregation", ""),
                    "adaptive_weights": _json_cell(row.get("adaptive_weights", {})),
                })
        else:
            # Fusion/meta branches do not search the individual class-0 weight
            # grid. Keep them visible rather than silently dropping them.
            records.append({
                "fold": int(row.get("fold", -1)),
                "test_subject": str(row.get("test_subject", "")),
                "model": str(row.get("model", "")),
                "n_test": int(row.get("n_test", 0)),
                "candidate_stage": "not_applicable_fusion_or_meta_branch",
                "candidate_weight": np.nan,
                "candidate_selected": np.nan,
                "candidate_usable": np.nan,
                "candidate_feasible": np.nan,
                "candidate_threshold": np.nan,
                "inner_accuracy": np.nan,
                "inner_balanced_accuracy": row.get("inner_selected_balanced_accuracy", np.nan),
                "inner_macro_f1": row.get("inner_selected_macro_f1", np.nan),
                "inner_vowel_recall": row.get("inner_selected_vowel_recall", row.get("adaptive_single_inner_selected_vowel_recall", np.nan)),
                "inner_consonant_recall": row.get("inner_selected_consonant_recall", row.get("adaptive_single_inner_selected_consonant_recall", np.nan)),
                "selected_weight": selected_weight,
                "outer_accuracy": row.get("accuracy", np.nan),
                "outer_balanced_accuracy": row.get("balanced_accuracy", np.nan),
                "outer_macro_f1": row.get("macro_f1", np.nan),
                "outer_vowel_recall": row.get("vowel_recall", np.nan),
                "outer_consonant_recall": row.get("consonant_recall", np.nan),
                "outer_threshold": row.get("decision_threshold", np.nan),
                "selection_rule": row.get("fusion_selection_rule", row.get("adaptive_single_selection_rule", "")),
                "adaptive_metric_aggregation": row.get("fusion_metric_aggregation", row.get("adaptive_metric_aggregation", "")),
                "adaptive_weights": _json_cell(row.get("adaptive_weights", {})),
            })
    frame = pd.DataFrame(records, columns=_WEIGHT_TABLE_COLUMNS)
    if len(frame):
        frame = frame.sort_values(["fold", "test_subject", "model", "candidate_stage", "candidate_weight"], kind="stable", na_position="last").reset_index(drop=True)
    return frame


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def write_result_tables(rows: list[dict], summaries: dict, failures: list[dict], output_dir: Path, fold_subjects: list[tuple[int, str]]) -> dict:
    """Write complete per-fold, aggregate, and coverage tables without refitting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = _fold_table_frame(rows)
    fold_csv = output_dir / "final_fold_results_table.csv"
    fold_frame.to_csv(fold_csv, index=False)
    fold_frame.to_html(output_dir / "final_fold_results_table.html", index=False, na_rep="")

    display_columns = [
        "fold", "test_subject", "model", "n_test", *_FOLD_TABLE_METRICS,
        "decision_threshold", "adaptive_vowel_weight", "adaptive_class_weight_vowel_0",
        "adaptive_class_weight_consonant_1", "source_model", "vowel_prompt_balance", "confusion_matrix",
    ]
    display = fold_frame[display_columns].copy()
    display = display.rename(columns={
        "fold": "Fold", "test_subject": "Test subject", "model": "Model", "n_test": "N test",
        "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro-F1",
        "vowel_precision": "Vowel precision", "vowel_recall": "Vowel recall",
        "consonant_precision": "Consonant precision", "consonant_recall": "Consonant recall",
        "roc_auc": "ROC-AUC", "average_precision": "Average precision", "decision_threshold": "Threshold",
        "adaptive_vowel_weight": "Vowel weight", "adaptive_class_weight_vowel_0": "Class weight 0",
        "adaptive_class_weight_consonant_1": "Class weight 1", "source_model": "Source model",
        "vowel_prompt_balance": "Vowel prompt balance",
        "confusion_matrix": "Confusion matrix [actual rows 0/1, predicted cols 0/1]",
    })
    for col in [
        "Accuracy", "Balanced accuracy", "Macro-F1", "Vowel precision", "Vowel recall",
        "Consonant precision", "Consonant recall", "ROC-AUC", "Average precision",
    ]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{100.0 * float(value):.2f}%")
    for col in ["Threshold", "Vowel weight", "Class weight 0", "Class weight 1"]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    fold_md = output_dir / "final_fold_results_table.md"
    if EVAL_MODE == "loso":
        fold_title = "KaraOne strict-LOSO per-fold results"
        fold_description = "Each row is one requested model on one held-out subject."
    elif EVAL_MODE in {"random", "stratified_random"}:
        fold_title = "KaraOne random trial-split sensitivity results"
        fold_description = "Each row is one requested model on one stratified random trial split; subjects can occur in both train and test, so this is not cross-subject generalization."
    else:
        fold_title = "KaraOne within-subject per-fold results"
        fold_description = "Each row is one requested model on one within-subject trial split; this is not cross-subject generalization."
    fold_md.write_text(
        f"# {fold_title}\n\n"
        + fold_description
        + " Metrics are shown as percentages; `0` is vowel and `1` is consonant. The table is sorted by fold and test-split label, and no fold is filtered because of its performance. The machine-readable CSV preserves metric values on the 0–1 scale and includes adaptive/fusion diagnostics.\n\n"
        + _markdown_table(display) + "\n"
    )

    # Separate expansion of every class-0 vowel-weight candidate. Candidate
    # metrics are inner-validation values; outer metrics are repeated only as a
    # reference to the final selected result for the same fold/model.
    weight_frame = _candidate_weight_table_frame(rows)
    weight_csv = output_dir / "final_fold_vowel_weight_candidates.csv"
    weight_frame.to_csv(weight_csv, index=False)
    weight_frame.to_html(output_dir / "final_fold_vowel_weight_candidates.html", index=False, na_rep="")
    weight_display = weight_frame.copy()
    if len(weight_display):
        weight_display = weight_display.rename(columns={
            "fold": "Fold", "test_subject": "Test subject", "model": "Model", "n_test": "N test",
            "candidate_stage": "Candidate stage", "candidate_weight": "Candidate vowel weight",
            "candidate_selected": "Selected weight", "candidate_usable": "Usable", "candidate_feasible": "Feasible",
            "candidate_threshold": "Inner threshold", "inner_accuracy": "Inner accuracy",
            "inner_balanced_accuracy": "Inner balanced accuracy", "inner_macro_f1": "Inner macro-F1",
            "inner_vowel_recall": "Inner vowel recall", "inner_consonant_recall": "Inner consonant recall",
            "selected_weight": "Final selected weight", "outer_accuracy": "Outer accuracy",
            "outer_balanced_accuracy": "Outer balanced accuracy", "outer_macro_f1": "Outer macro-F1",
            "outer_vowel_recall": "Outer vowel recall", "outer_consonant_recall": "Outer consonant recall",
            "outer_threshold": "Outer threshold", "selection_rule": "Selection rule",
            "adaptive_metric_aggregation": "Metric aggregation", "adaptive_weights": "Fusion weights",
        })
        for col in [
            "Inner accuracy", "Inner balanced accuracy", "Inner macro-F1", "Inner vowel recall",
            "Inner consonant recall", "Outer accuracy", "Outer balanced accuracy", "Outer macro-F1",
            "Outer vowel recall", "Outer consonant recall",
        ]:
            if col in weight_display.columns:
                weight_display[col] = weight_display[col].map(lambda value: "" if pd.isna(value) else f"{100.0 * float(value):.2f}%")
        for col in ["Candidate vowel weight", "Inner threshold", "Final selected weight", "Outer threshold"]:
            if col in weight_display.columns:
                weight_display[col] = weight_display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    weight_title = "KaraOne per-fold adaptive vowel-weight candidates"
    weight_description = (
        "Each direct adaptive classifier contributes one row per configured vowel-weight candidate. "
        "Candidate metrics are selected from inner subject-disjoint validation only. Outer metrics are repeated "
        "for reference and correspond to the final selected weight; they were not used to select the candidate. "
        "Fusion and meta branches are retained with `not_applicable_fusion_or_meta_branch`.\n\n"
    )
    weight_md = output_dir / "final_fold_vowel_weight_candidates.md"
    weight_md.write_text(f"# {weight_title}\n\n" + weight_description + _markdown_table(weight_display) + "\n")

    summary_rows = [{"model": name, **summary} for name, summary in summaries.items()]
    summary_frame = pd.DataFrame(summary_rows)
    if len(summary_frame):
        summary_frame = summary_frame.sort_values("model", kind="stable").reset_index(drop=True)
    summary_frame.to_csv(output_dir / "final_summary_results_table.csv", index=False)
    summary_frame.to_html(output_dir / "final_summary_results_table.html", index=False, na_rep="")
    summary_display = summary_frame.copy()
    if len(summary_display):
        summary_display = summary_display.rename(columns={
            "model": "Model", "n_folds": "Folds", "accuracy_mean": "Accuracy mean", "accuracy_std": "Accuracy SD",
            "balanced_accuracy_mean": "BA mean", "balanced_accuracy_std": "BA SD", "macro_f1_mean": "Macro-F1 mean",
            "vowel_precision_mean": "Vowel precision mean", "vowel_recall_mean": "Vowel recall mean",
            "consonant_precision_mean": "Consonant precision mean", "consonant_recall_mean": "Consonant recall mean",
            "roc_auc_mean": "ROC-AUC mean", "average_precision_mean": "AP mean",
        })
        for col in [c for c in summary_display.columns if c.endswith("mean") or c.endswith("SD") or c.endswith("std") or c in {"Accuracy mean", "Accuracy SD", "BA mean", "BA SD", "Macro-F1 mean", "Vowel precision mean", "Vowel recall mean", "Consonant precision mean", "Consonant recall mean", "ROC-AUC mean", "AP mean"}]:
            if col in summary_display.columns and col != "Folds":
                summary_display[col] = summary_display[col].map(lambda value: "" if pd.isna(value) else f"{100.0 * float(value):.2f}%")
    if EVAL_MODE == "loso":
        summary_title = "KaraOne aggregate strict-LOSO results"
        summary_description = "Means and standard deviations are across held-out subjects."
    elif EVAL_MODE in {"random", "stratified_random"}:
        summary_title = "KaraOne aggregate random trial-split sensitivity results"
        summary_description = "Means and standard deviations are across repeated stratified random trial splits. Subjects may occur in both train and test, so these results are not cross-subject generalization."
    else:
        summary_title = "KaraOne aggregate within-subject results"
        summary_description = "Means and standard deviations are across within-subject trial splits. These results are not cross-subject generalization."
    (output_dir / "final_summary_results_table.md").write_text(
        f"# {summary_title}\n\n"
        + summary_description
        + " Metrics are percentages. This summary does not replace the complete per-fold table.\n\n"
        + (_markdown_table(summary_display) if len(summary_display) else "No successful model rows were produced.\n") + "\n"
    )

    expected_models = list(REQUESTED) + (["adaptive_best_single"] if ENABLE_ADAPTIVE_BEST_SINGLE else [])
    row_keys = {(int(r["fold"]), str(r["test_subject"]), str(r["model"])) for r in rows}
    error_map = {(int(f["fold"]), str(f["test_subject"]), str(f["model"])): f.get("error", "") for f in failures}
    coverage = []
    for fold, subject in fold_subjects:
        for model in expected_models:
            key = (int(fold), str(subject), model)
            status = "ok" if key in row_keys else ("failed" if key in error_map else "missing")
            coverage.append({"fold": int(fold), "test_subject": str(subject), "model": model, "status": status, "error": error_map.get(key, "")})
    coverage_frame = pd.DataFrame(coverage, columns=["fold", "test_subject", "model", "status", "error"])
    coverage_frame.to_csv(output_dir / "final_fold_coverage.csv", index=False)
    (output_dir / "fold_metrics_readme.md").write_text(
        "# Result-file guide\n\n"
        "`final_fold_results_table.csv` and `.md` contain one row for every successful model/fold result, including accuracy, balanced accuracy, macro-F1, per-class precision and recall, ROC-AUC, average precision, threshold, adaptive vowel weight, class weights, source model, and confusion matrix. `final_fold_vowel_weight_candidates.csv` and `.md` expand every configured candidate weight for each direct adaptive classifier; their inner columns are the metrics used for training-only selection, while their repeated outer columns are reference values for the final selected model. `final_fold_coverage.csv` explicitly lists every expected model/fold combination and marks it `ok`, `failed`, or `missing`; failure details are also in `final_results.json`. `final_summary_results_table.md` reports across-fold means and standard deviations. No outer test labels are used to select weights, thresholds, subjects, or models.\n"
    )
    return {
        "fold_csv": str(fold_csv),
        "fold_markdown": str(fold_md),
        "fold_html": str(output_dir / "final_fold_results_table.html"),
        "vowel_weight_candidates_csv": str(weight_csv),
        "vowel_weight_candidates_markdown": str(weight_md),
        "vowel_weight_candidates_html": str(output_dir / "final_fold_vowel_weight_candidates.html"),
        "summary_csv": str(output_dir / "final_summary_results_table.csv"),
        "summary_markdown": str(output_dir / "final_summary_results_table.md"),
        "coverage_csv": str(output_dir / "final_fold_coverage.csv"),
    }


def _candidate_simplex_weights(n: int, step: int) -> list[np.ndarray]:
    """Deterministic simplex grid; capped to keep Kaggle runtime practical."""
    step = max(1, int(step))
    if n == 1:
        return [np.ones(1, dtype=float)]
    if n == 2:
        vals = np.arange(0, step + 1, dtype=float) / step
        return [np.asarray([v, 1.0 - v], dtype=float) for v in vals]
    # For n>2, use one-hot, uniform, and pairwise mixtures. This is an
    # adaptive ensemble grid rather than a fixed equal-weight average.
    out = [np.ones(n, dtype=float) / n]
    for i in range(n):
        e = np.zeros(n, dtype=float); e[i] = 1.0; out.append(e)
    vals = np.arange(0, step + 1, dtype=float) / step
    for i in range(n):
        for j in range(i + 1, n):
            for v in vals:
                w = np.zeros(n, dtype=float); w[i] = v; w[j] = 1.0 - v; out.append(w)
    return out


def _threshold_grid(step: float) -> np.ndarray:
    step = max(0.01, float(step))
    return np.arange(0.30, 0.701, step, dtype=float)


def _adaptive_vowel_thresholds(step: float) -> np.ndarray:
    step = max(0.01, float(step))
    lo = min(0.90, max(0.05, float(ADAPTIVE_VOWEL_THRESHOLD_MIN)))
    hi = min(0.95, max(lo, float(ADAPTIVE_VOWEL_THRESHOLD_MAX)))
    values = np.arange(lo, hi + 0.5 * step, step, dtype=float)
    return np.unique(np.round(np.clip(values, 0.05, 0.95), 6))


def _make_adaptive_splits(y: np.ndarray, groups: np.ndarray | None, n_splits: int, seed: int):
    """Create leakage-safe inner splits, preferring held-out subjects."""
    y = np.asarray(y, dtype=int)
    n_splits = max(2, min(int(n_splits), int(np.min(np.bincount(y, minlength=2)))))
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        if len(unique_groups) >= n_splits:
            try:
                grouped = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                splits = list(grouped.split(np.zeros(len(y), dtype=np.float32), y, groups))
                valid = all(len(np.unique(y[a])) == 2 and len(np.unique(y[b])) == 2 for a, b in splits)
                if valid:
                    return splits, "subject_group"
            except Exception:
                pass
    trial = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(trial.split(np.zeros(len(y), dtype=np.float32), y)), "trial"


def _adaptation_metric_summary(y_true: np.ndarray, pred: np.ndarray, inner_splits: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    """Summarize an operating point for adaptive selection.

    ``subject_macro`` averages metrics equally across subject-disjoint inner
    validation folds. This is preferable for cross-subject adaptation because
    a subject with more trials cannot dominate the chosen weight. ``pooled`` is
    retained as a declared sensitivity option. Neither mode sees outer-test
    labels.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if ADAPTIVE_METRIC_AGGREGATION == "pooled" or not inner_splits:
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "vowel_recall": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
            "consonant_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        }
    parts = []
    for _, iva in inner_splits:
        iva = np.asarray(iva, dtype=int)
        if len(iva) == 0 or not np.all(np.isfinite(pred[iva])):
            continue
        if len(np.unique(y_true[iva])) < 2:
            continue
        parts.append({
            "accuracy": float(accuracy_score(y_true[iva], pred[iva])),
            "vowel_recall": float(recall_score(y_true[iva], pred[iva], pos_label=0, zero_division=0)),
            "consonant_recall": float(recall_score(y_true[iva], pred[iva], pos_label=1, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true[iva], pred[iva])),
            "macro_f1": float(f1_score(y_true[iva], pred[iva], average="macro", zero_division=0)),
        })
    if not parts:
        return _adaptation_metric_summary(y_true, pred, None)
    return {key: float(np.mean([part[key] for part in parts])) for key in parts[0]}


def fit_adaptive_vowel_model(Xtr: np.ndarray, ytr: np.ndarray, name: str, fold: int, groups_tr: np.ndarray | None = None, prompts_tr: np.ndarray | None = None):
    """Fit one classifier with a training-only vowel-recall adaptation.

    Class 0 is vowel and class 1 is consonant. In the default
    ``class_weight_fast`` mode, each complete inner split fits one model for
    every configured vowel class-weight candidate, selects the vowel-focused
    candidate and threshold on out-of-fold training scores, and refits the
    selected class weight on all outer-training data. ``threshold_only_fast``
    is an explicit sensitivity mode that keeps only the inverse-frequency
    weight and adapts the threshold. ``class_weight_full`` performs the larger
    configured class-weight search.
    No outer-test label is accessed in any mode.
    """
    ytr = np.asarray(ytr, dtype=int)
    counts = np.bincount(ytr, minlength=2)
    n_inner = max(2, min(int(ADAPTIVE_VOWEL_INNER_SPLITS), int(np.min(counts))))
    inner_splits, inner_split_unit = _make_adaptive_splits(ytr, groups_tr, n_inner, SEED + 1000 + fold)
    if ADAPTIVE_VOWEL_MODE == "threshold_only_fast":
        candidates = [1.0]
    else:
        candidates = sorted({max(0.1, float(w)) for w in ADAPTIVE_VOWEL_WEIGHTS})
    candidate_records = []
    oof_by_weight = {}
    class_counts = np.bincount(ytr, minlength=2).astype(float)
    base_class_weight = {
        0: float(len(ytr) / (2.0 * max(1.0, class_counts[0]))),
        1: float(len(ytr) / (2.0 * max(1.0, class_counts[1]))),
    }
    total_inner_fits = len(candidates) * n_inner
    log(f"    {name}: adaptive_mode={ADAPTIVE_VOWEL_MODE}, inner_unit={inner_split_unit}, inner_fits={total_inner_fits}, candidates={candidates}")
    for vowel_weight in candidates:
        oof = np.full(len(ytr), np.nan, dtype=float)
        failures = []
        for split_id, (itr, iva) in enumerate(inner_splits, 1):
            log(f"      {name}: inner fit {split_id}/{n_inner} weight={vowel_weight:g}")
            try:
                model = None
                X_inner, y_inner, _ = _vowel_prompt_balanced_training_data(
                    Xtr[itr], ytr[itr], prompts_tr[itr] if prompts_tr is not None else None,
                    SEED + 100000 + fold * 100 + split_id,
                )
                # If resampling is enabled, compute inverse-frequency weights
                # from the actual fit set. This avoids double-counting vowels
                # through both duplication and the original imbalanced prior.
                fit_counts = np.bincount(y_inner, minlength=2).astype(float)
                fit_base_weight = {
                    0: float(len(y_inner) / (2.0 * max(1.0, fit_counts[0]))),
                    1: float(len(y_inner) / (2.0 * max(1.0, fit_counts[1]))),
                }
                adaptive_class_weight = {
                    0: fit_base_weight[0] * float(vowel_weight),
                    1: fit_base_weight[1],
                }
                model = make_model(name, vowel_weight=vowel_weight, class_weight=adaptive_class_weight)
                model.fit(X_inner, y_inner)
                oof[iva] = model.score(Xtr[iva])
            except Exception as exc:
                failures.append({"split": split_id, "error": f"{type(exc).__name__}: {exc}"})
        if not np.all(np.isfinite(oof)):
            candidate_records.append({"vowel_weight": vowel_weight, "usable": False, "failures": failures})
            continue
        threshold_records = []
        for threshold in _adaptive_vowel_thresholds(ADAPTIVE_VOWEL_THRESHOLD_STEP):
            pred = (oof >= threshold).astype(int)
            operating = _adaptation_metric_summary(ytr, pred, inner_splits)
            vrec = operating["vowel_recall"]
            crec = operating["consonant_recall"]
            ba = operating["balanced_accuracy"]
            macro = operating["macro_f1"]
            feasible = (
                ba >= ADAPTIVE_VOWEL_BA_FLOOR
                and crec >= 0.25
                and vrec >= ADAPTIVE_VOWEL_MIN_RECALL
            )
            threshold_records.append({
                "vowel_weight": vowel_weight,
                "threshold": float(threshold),
                "accuracy": float(operating["accuracy"]),
                "vowel_recall": vrec,
                "consonant_recall": crec,
                "balanced_accuracy": ba,
                "macro_f1": macro,
                "feasible": feasible,
            })
        feasible_points = [r for r in threshold_records if r["feasible"]]
        if feasible_points:
            # First preserve balanced accuracy within a small tolerance of the
            # best feasible point, then maximize vowel recall. This prevents a
            # nearly all-vowel threshold from winning merely because it has a
            # high vowel recall.
            max_feasible_ba = max(r["balanced_accuracy"] for r in feasible_points)
            eligible = [
                r for r in feasible_points
                if r["balanced_accuracy"] >= max_feasible_ba - ADAPTIVE_VOWEL_BA_TOLERANCE
            ]
            selected_record = max(
                eligible,
                key=lambda r: (
                    r["vowel_recall"], r["balanced_accuracy"],
                    r["consonant_recall"], r["macro_f1"],
                    -abs(r["threshold"] - 0.5),
                ),
            )
        else:
            # If a candidate has no feasible threshold, retain its strongest
            # balanced-accuracy point for diagnostics, but it cannot outrank a
            # candidate with a feasible vowel/consonant operating point.
            selected_record = max(
                threshold_records,
                key=lambda r: (
                    r["balanced_accuracy"], r["vowel_recall"],
                    r["consonant_recall"], r["macro_f1"],
                    -abs(r["threshold"] - 0.5),
                ),
            )
        best_for_weight = (
            (
                int(bool(selected_record["feasible"])),
                float(selected_record["vowel_recall"]),
                float(selected_record["balanced_accuracy"]),
                float(selected_record["consonant_recall"]),
                float(selected_record["macro_f1"]),
                -abs(float(selected_record["threshold"]) - 0.5),
                -float(vowel_weight),
            ),
            selected_record,
        )
        candidate_records.append({"usable": True, **best_for_weight[1]})
        oof_by_weight[float(vowel_weight)] = oof.copy()
        log(f"      {name}: weight={vowel_weight:g} selected_inner_threshold={best_for_weight[1]['threshold']:.2f} inner_Vrec={best_for_weight[1]['vowel_recall']:.3f} inner_BA={best_for_weight[1]['balanced_accuracy']:.3f}")
    usable_records = [r for r in candidate_records if r.get("usable")]
    feasible_records = [r for r in usable_records if r.get("feasible")]
    unit_record = next(
        (r for r in usable_records if abs(float(r.get("vowel_weight", np.nan)) - 1.0) < 1e-9),
        None,
    )
    if feasible_records:
        max_feasible_ba = max(float(r["balanced_accuracy"]) for r in feasible_records)
        eligible_records = [
            r for r in feasible_records
            if float(r["balanced_accuracy"]) >= max_feasible_ba - ADAPTIVE_VOWEL_BA_TOLERANCE
        ]
        selected = max(
            eligible_records,
            key=lambda r: (
                float(r["vowel_recall"]), float(r["balanced_accuracy"]),
                float(r["consonant_recall"]), float(r["macro_f1"]),
                -float(r["vowel_weight"]),
            ),
        )
        adaptive_weight_selection_rule = "vowel_recall_priority_within_global_ba_tolerance"
    elif usable_records:
        selected = max(
            usable_records,
            key=lambda r: (
                float(r["balanced_accuracy"]), float(r["vowel_recall"]),
                float(r["consonant_recall"]), float(r["macro_f1"]),
                -float(r["vowel_weight"]),
            ),
        )
        adaptive_weight_selection_rule = "fallback_balanced_accuracy_no_feasible_weight"
    else:
        raise RuntimeError(f"Adaptive vowel weighting failed for {name}: {candidate_records}")

    # Stability guard: class weighting is retained only when it improves vowel
    # recall by a meaningful amount while preserving the primary BA objective
    # and consonant recognition. This is selected using inner scores only.
    if unit_record is not None and float(selected["vowel_weight"]) != 1.0:
        vowel_gain = float(selected["vowel_recall"]) - float(unit_record["vowel_recall"])
        ba_delta = float(selected["balanced_accuracy"]) - float(unit_record["balanced_accuracy"])
        consonant_delta = float(selected["consonant_recall"]) - float(unit_record["consonant_recall"])
        supported = (
            vowel_gain >= ADAPTIVE_VOWEL_MIN_GAIN
            and ba_delta >= -ADAPTIVE_VOWEL_MAX_BA_DROP
            and consonant_delta >= -ADAPTIVE_VOWEL_MAX_CONSONANT_RECALL_DROP
        )
        if not supported:
            selected = unit_record
            adaptive_weight_selection_rule += "_unit_baseline_guard"
        log(
            f"    {name}: weight_guard selected_nonunit={supported} "
            f"vowel_gain={vowel_gain:.3f} BA_delta={ba_delta:.3f} "
            f"Crec_delta={consonant_delta:.3f}"
        )
    best_oof = oof_by_weight[float(selected["vowel_weight"])]
    log(f"    {name}: selected vowel_weight={selected['vowel_weight']:g} by {adaptive_weight_selection_rule}")
    baseline_delta = None
    if unit_record is not None:
        baseline_delta = {
            "balanced_accuracy": float(selected["balanced_accuracy"] - unit_record["balanced_accuracy"]),
            "vowel_recall": float(selected["vowel_recall"] - unit_record["vowel_recall"]),
            "consonant_recall": float(selected["consonant_recall"] - unit_record["consonant_recall"]),
            "macro_f1": float(selected["macro_f1"] - unit_record["macro_f1"]),
        }
    X_final, y_final, final_prompt_balance = _vowel_prompt_balanced_training_data(
        Xtr, ytr, prompts_tr, SEED + 200000 + fold,
    )
    final_counts = np.bincount(y_final, minlength=2).astype(float)
    final_base_weight = {
        0: float(len(y_final) / (2.0 * max(1.0, final_counts[0]))),
        1: float(len(y_final) / (2.0 * max(1.0, final_counts[1]))),
    }
    final_weight = {0: final_base_weight[0] * float(selected["vowel_weight"]), 1: final_base_weight[1]}
    log(f"    {name}: final fit with vowel_weight={selected['vowel_weight']:g} class_weight0={final_weight[0]:.4f} class_weight1={final_weight[1]:.4f}")
    final_model = make_model(name, vowel_weight=selected["vowel_weight"], class_weight=final_weight)
    final_model.fit(X_final, y_final)
    info = {
        "adaptive_mode": ADAPTIVE_VOWEL_MODE,
        "adaptive_target": "vowel_recall_class_0",
        "adaptive_vowel_weight": float(selected["vowel_weight"]),
        "adaptive_vowel_min_gain": float(ADAPTIVE_VOWEL_MIN_GAIN),
        "adaptive_vowel_max_ba_drop": float(ADAPTIVE_VOWEL_MAX_BA_DROP),
        "adaptive_vowel_max_consonant_recall_drop": float(ADAPTIVE_VOWEL_MAX_CONSONANT_RECALL_DROP),
        "adaptive_class_weight": {"vowel_0": float(final_weight[0]), "consonant_1": float(final_weight[1])},
        "adaptive_vowel_threshold": float(selected["threshold"]),
        "inner_selected_vowel_recall": float(selected["vowel_recall"]),
        "inner_selected_consonant_recall": float(selected["consonant_recall"]),
        "inner_selected_balanced_accuracy": float(selected["balanced_accuracy"]),
        "inner_selected_macro_f1": float(selected["macro_f1"]),
        "inner_vowel_weight_candidates": candidate_records,
        "inner_unit_weight_baseline": unit_record,
        "inner_adaptive_minus_unit_weight": baseline_delta,
        "adaptive_weight_selection_rule": adaptive_weight_selection_rule,
        "inner_splits": n_inner,
        "inner_split_unit": inner_split_unit,
        "adaptive_metric_aggregation": ADAPTIVE_METRIC_AGGREGATION,
        "inner_model_fits": int(total_inner_fits),
        "final_model_fits": 1,
        "vowel_recall_objective_constraints": {"balanced_accuracy_floor": ADAPTIVE_VOWEL_BA_FLOOR, "minimum_vowel_recall": ADAPTIVE_VOWEL_MIN_RECALL, "minimum_consonant_recall": 0.25, "balanced_accuracy_tolerance_for_vowel_priority": ADAPTIVE_VOWEL_BA_TOLERANCE},
        "vowel_prompt_balance": final_prompt_balance,
        "_inner_oof_scores": best_oof,
    }
    return final_model, float(selected["threshold"]), info


def _fit_sigmoid_calibrator(scores: np.ndarray, y: np.ndarray):
    """Fit a train-only one-dimensional logistic calibrator when possible."""
    from sklearn.linear_model import LogisticRegression
    s = np.asarray(scores, dtype=float).reshape(-1, 1)
    if len(np.unique(y)) < 2 or np.std(s) < 1e-10:
        return None
    cal = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000)
    cal.fit(s, y)
    return cal


def adaptive_weights(Xtr: np.ndarray, ytr: np.ndarray, base_names: list[str], fold: int, precomputed_oof: dict[str, np.ndarray] | None = None, groups_tr: np.ndarray | None = None) -> tuple[dict, float, dict]:
    """Learn ensemble weights and threshold only from an inner training split.

    The outer test subject is never used here. Each base score is calibrated on
    inner training scores, then a deterministic grid searches weights and
    threshold with vowel recall as the primary objective. Balanced accuracy and
    consonant-recall floors prevent an all-vowel degenerate solution. This is a
    score-stream fusion safeguard; the same vowel-recall adaptive trainer is also
    applied separately to every individual classifier branch.
    """
    n_inner = max(2, min(int(INNER_SPLITS), int(np.min(np.bincount(ytr, minlength=2)))))
    oof = {name: np.full(len(ytr), np.nan, dtype=float) for name in base_names}
    failures = {}
    inner_parts = None
    if precomputed_oof is not None:
        inner_parts, grouped_unit = _make_adaptive_splits(ytr, groups_tr, n_inner, SEED + fold)
        inner_split_unit = f"precomputed_{grouped_unit}"
        for name in base_names:
            if name in precomputed_oof:
                candidate = np.asarray(precomputed_oof[name], dtype=float).reshape(-1)
                if len(candidate) == len(ytr):
                    oof[name] = candidate.copy()
                else:
                    failures[name] = f"precomputed score length {len(candidate)} != training length {len(ytr)}"
            else:
                failures[name] = "no precomputed inner score stream"
    else:
        inner_parts, inner_split_unit = _make_adaptive_splits(ytr, groups_tr, n_inner, SEED + fold)
        for split_id, (itr, iva) in enumerate(inner_parts):
            for name in base_names:
                try:
                    log(f"    fusion: legacy inner fit {split_id + 1}/{n_inner} model={name}")
                    m = make_model(name)
                    m.fit(Xtr[itr], ytr[itr])
                    oof[name][iva] = m.score(Xtr[iva])
                except Exception as exc:
                    failures[f"{name}_split{split_id + 1}"] = f"{type(exc).__name__}: {exc}"
    if precomputed_oof is None:
        inner_split_unit = inner_split_unit
    usable = [n for n in base_names if np.isfinite(oof[n]).sum() > 0]
    if not usable:
        raise RuntimeError(f"Adaptive ensemble could not fit any base model: {failures}")
    # Only streams with complete inner validation predictions participate in
    # the learned fusion; incomplete streams are reported as failures.
    usable = [n for n in usable if np.all(np.isfinite(oof[n]))]
    if not usable:
        raise RuntimeError(f"Adaptive ensemble has no complete inner score streams: {failures}")
    calibrated = {}
    cal_models = {}
    for name in usable:
        cal = _fit_sigmoid_calibrator(oof[name], ytr)
        cal_models[name] = cal
        if cal is None:
            calibrated[name] = np.clip(oof[name], 0.0, 1.0)
        else:
            calibrated[name] = cal.predict_proba(oof[name].reshape(-1, 1))[:, 1]
    # Rank streams by validation BA, then retain a small, interpretable set.
    inner_scores = {n: float(balanced_accuracy_score(ytr, calibrated[n] >= 0.5)) for n in usable}
    ranked = sorted(usable, key=lambda n: (-inner_scores[n], n))[:max(1, min(ADAPTIVE_MAX_BASES, len(usable)))]
    weight_grid = _candidate_simplex_weights(len(ranked), ADAPTIVE_GRID_STEP)
    fusion_candidates = []
    for w in weight_grid:
        fused = np.sum(np.vstack([w[i] * calibrated[n] for i, n in enumerate(ranked)]), axis=0)
        for threshold in _threshold_grid(ADAPTIVE_THRESHOLD_STEP):
            pred = (fused >= threshold).astype(int)
            operating = _adaptation_metric_summary(ytr, pred, inner_parts)
            ba = operating["balanced_accuracy"]
            vrec = operating["vowel_recall"]
            macro = operating["macro_f1"]
            crec = operating["consonant_recall"]
            feasible = (
                ba >= ADAPTIVE_VOWEL_BA_FLOOR
                and crec >= 0.25
                and vrec >= ADAPTIVE_VOWEL_MIN_RECALL
            )
            fusion_candidates.append({
                "weights": w.copy(), "threshold": float(threshold),
                "balanced_accuracy": ba, "vowel_recall": vrec,
                "consonant_recall": crec, "macro_f1": macro,
                "feasible": feasible,
            })
    feasible_fusion = [r for r in fusion_candidates if r["feasible"]]
    if feasible_fusion:
        max_ba = max(r["balanced_accuracy"] for r in feasible_fusion)
        eligible = [
            r for r in feasible_fusion
            if r["balanced_accuracy"] >= max_ba - ADAPTIVE_VOWEL_BA_TOLERANCE
        ]
        best = max(
            eligible,
            key=lambda r: (
                r["vowel_recall"], r["balanced_accuracy"],
                r["consonant_recall"], r["macro_f1"],
                -abs(r["threshold"] - 0.5),
            ),
        )
        fusion_selection_rule = "vowel_recall_priority_within_ba_tolerance"
    else:
        best = max(
            fusion_candidates,
            key=lambda r: (
                r["balanced_accuracy"], r["vowel_recall"],
                r["consonant_recall"], r["macro_f1"],
                -abs(r["threshold"] - 0.5),
            ),
        )
        fusion_selection_rule = "fallback_balanced_accuracy_no_feasible_point"
    best_w = np.asarray(best["weights"], dtype=float)
    best_threshold = float(best["threshold"])
    best_ba = float(best["balanced_accuracy"])
    best_vrec = float(best["vowel_recall"])
    best_macro = float(best["macro_f1"])
    weights = {n: float(best_w[i]) for i, n in enumerate(ranked)}
    # Explicitly calibrate the outer-test scores using calibration models fit on
    # inner data only. Returning models would expose objects in JSON, so use a
    # second helper in run_models to apply the stored affine/logistic parameters.
    calibration_params = {}
    for n in ranked:
        cal = cal_models[n]
        if cal is None:
            calibration_params[n] = {"kind": "identity"}
        else:
            calibration_params[n] = {"kind": "logistic", "coef": float(cal.coef_[0, 0]), "intercept": float(cal.intercept_[0])}
    info = {
        "adaptive_base_models": ranked,
        "inner_balanced_accuracy_after_calibration": inner_scores,
        "adaptive_weights": weights,
        "adaptive_threshold": float(best_threshold),
        "inner_selected_balanced_accuracy": best_ba,
        "inner_selected_vowel_recall": best_vrec,
        "inner_selected_macro_f1": best_macro,
        "inner_failures": failures,
        "calibration": calibration_params,
        "fusion_score_source": "reused_individual_inner_oof" if precomputed_oof is not None else "legacy_inner_refit",
        "fusion_selection_rule": fusion_selection_rule,
        "fusion_inner_candidate_count": len(fusion_candidates),
        "fusion_inner_split_unit": inner_split_unit,
        "fusion_metric_aggregation": ADAPTIVE_METRIC_AGGREGATION,
    }
    return weights, float(best_threshold), info


def _apply_calibration(score: np.ndarray, params: dict) -> np.ndarray:
    s = np.asarray(score, dtype=float).reshape(-1)
    if params.get("kind") != "logistic":
        return np.clip(s, 0.0, 1.0)
    z = float(params["coef"]) * s + float(params["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def evaluation_splits(y: np.ndarray, groups: np.ndarray):
    if EVAL_MODE == "loso":
        for i, subject in enumerate(SUBJECTS, 1):
            tr = groups != subject; te = groups == subject
            if np.sum(te) == 0:
                continue
            yield i, subject, np.where(tr)[0], np.where(te)[0]
    elif EVAL_MODE in {"within", "within_subject"}:
        fold = 0
        for subject in SUBJECTS:
            idx = np.where(groups == subject)[0]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for a, b in skf.split(idx, y[idx]):
                fold += 1
                yield fold, subject, idx[a], idx[b]
    elif EVAL_MODE in {"random", "stratified_random"}:
        splitter = StratifiedShuffleSplit(
            n_splits=RANDOM_REPEATS,
            test_size=RANDOM_TEST_SIZE,
            random_state=SEED,
        )
        # This is a trial-random sensitivity analysis: subject identity is not
        # held out. The explicit split label is carried into every result row.
        for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
            yield fold, f"random_trial_split_{fold}", tr, te
    else:
        raise ValueError("KARAONE_EVAL_MODE must be loso, random, or within_subject")


def run_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, prompts: np.ndarray | None = None) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    failures: list[dict] = []
    for fold, subject, tr, te in evaluation_splits(y, groups):
        log(f"\nFOLD {fold}: held-out={subject} train={len(tr)} test={len(te)}")
        fitted: dict[str, Any] = {}
        test_scores: dict[str, np.ndarray] = {}
        model_thresholds: dict[str, float] = {}
        model_adaptive_info: dict[str, dict] = {}
        inner_oof_streams: dict[str, np.ndarray] = {}
        for name in REQUESTED:
            if name == "catch22_summary_mr_hydra":
                continue
            try:
                t0 = time.time()
                m, model_threshold, adaptive_info = fit_adaptive_vowel_model(
                    X[tr], y[tr], name, fold, groups_tr=groups[tr],
                    prompts_tr=prompts[tr] if prompts is not None else None,
                )
                score = m.score(X[te])
                fitted[name] = m
                test_scores[name] = score
                model_thresholds[name] = float(model_threshold)
                model_adaptive_info[name] = dict(adaptive_info)
                private_oof = adaptive_info.pop("_inner_oof_scores", None)
                if private_oof is not None:
                    inner_oof_streams[name] = np.asarray(private_oof, dtype=float)
                r = metrics(y[te], score, subject, name, fold, {"fit_seconds": time.time() - t0, **adaptive_info}, threshold=model_threshold)
                rows.append(r)
                log(f"  {name}: vowel_weight={r.get('adaptive_vowel_weight', 1.0):.2f} class_weight0={r.get('adaptive_class_weight', {}).get('vowel_0', float('nan')):.3f} threshold={r.get('decision_threshold', 0.5):.2f} acc={r['accuracy']:.3f} BA={r['balanced_accuracy']:.3f} Vrec={r['vowel_recall']:.3f} Crec={r['consonant_recall']:.3f}")
            except Exception as exc:
                err = {"model": name, "fold": fold, "test_subject": subject, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(err); log("  FAILED", err)
        # Leakage-safe meta-selection: choose the strongest already-fitted
        # requested classifier using inner validation only. This adds no model
        # fits and lets the final report include an adaptive single-model branch
        # without selecting a model from the outer test labels.
        if ENABLE_ADAPTIVE_BEST_SINGLE:
            candidates_single = [
                n for n in REQUESTED
                if n in test_scores and n in model_adaptive_info
            ]
            if candidates_single:
                single_records = []
                for n in candidates_single:
                    a = model_adaptive_info[n]
                    single_records.append({
                        "model": n,
                        "balanced_accuracy": float(a.get("inner_selected_balanced_accuracy", 0.0)),
                        "vowel_recall": float(a.get("inner_selected_vowel_recall", 0.0)),
                        "consonant_recall": float(a.get("inner_selected_consonant_recall", 0.0)),
                        "macro_f1": float(a.get("inner_selected_macro_f1", 0.0)),
                        "threshold": float(model_thresholds[n]),
                    })
                feasible_single = [
                    r for r in single_records
                    if r["balanced_accuracy"] >= ADAPTIVE_VOWEL_BA_FLOOR
                    and r["vowel_recall"] >= ADAPTIVE_VOWEL_MIN_RECALL
                    and r["consonant_recall"] >= 0.25
                ]
                pool_single = feasible_single if feasible_single else single_records
                max_single_ba = max(r["balanced_accuracy"] for r in pool_single)
                eligible_single = [
                    r for r in pool_single
                    if r["balanced_accuracy"] >= max_single_ba - ADAPTIVE_BEST_SINGLE_BA_TOLERANCE
                ]
                selected_single = max(
                    eligible_single,
                    key=lambda r: (
                        r["vowel_recall"], r["balanced_accuracy"],
                        r["consonant_recall"], r["macro_f1"],
                        -abs(r["threshold"] - 0.5), r["model"],
                    ),
                )
                selected_name = selected_single["model"]
                source_info = model_adaptive_info[selected_name]
                single_extra = {
                    "adaptive_single_source_model": selected_name,
                    "adaptive_single_selection_rule": (
                        "inner_feasible_ba_tolerance_then_vowel_recall"
                        if feasible_single else
                        "inner_ba_tolerance_then_vowel_recall_no_feasible_model"
                    ),
                    "adaptive_single_inner_candidates": single_records,
                    "adaptive_single_inner_selected_balanced_accuracy": selected_single["balanced_accuracy"],
                    "adaptive_single_inner_selected_vowel_recall": selected_single["vowel_recall"],
                    "adaptive_single_inner_selected_consonant_recall": selected_single["consonant_recall"],
                    "adaptive_single_inner_selected_macro_f1": selected_single["macro_f1"],
                    "adaptive_single_source_vowel_weight": float(source_info.get("adaptive_vowel_weight", 1.0)),
                    "adaptive_single_source_class_weight": source_info.get("adaptive_class_weight", {}),
                    "adaptive_single_source_threshold": float(model_thresholds[selected_name]),
                    "adaptive_single_inner_split_unit": source_info.get("inner_split_unit"),
                }
                r = metrics(
                    y[te], test_scores[selected_name], subject,
                    "adaptive_best_single", fold, single_extra,
                    threshold=model_thresholds[selected_name],
                )
                rows.append(r)
                log(
                    f"  adaptive_best_single: source={selected_name} "
                    f"vowel_weight={single_extra['adaptive_single_source_vowel_weight']:.2f} "
                    f"threshold={r['decision_threshold']:.2f} acc={r['accuracy']:.3f} "
                    f"BA={r['balanced_accuracy']:.3f} Vrec={r['vowel_recall']:.3f} "
                    f"Crec={r['consonant_recall']:.3f}"
                )
        if "catch22_summary_mr_hydra" in REQUESTED and ENABLE_ADAPTIVE_FUSION:
            bases = [n for n in ["catch22_summary", "mr_hydra"] if n in test_scores]
            if len(bases) >= 2:
                try:
                    precomputed = {n: inner_oof_streams[n] for n in bases if n in inner_oof_streams}
                    weights, adaptive_threshold, inner = adaptive_weights(X[tr], y[tr], bases, fold, precomputed_oof=precomputed or None, groups_tr=groups[tr])
                    calibrated_test = {n: _apply_calibration(test_scores[n], inner["calibration"][n]) for n in inner["adaptive_base_models"]}
                    fused = sum(weights[n] * calibrated_test[n] for n in inner["adaptive_base_models"])
                    r = metrics(y[te], fused, subject, "catch22_summary_mr_hydra", fold, {"adaptive_weights": weights, **inner}, threshold=adaptive_threshold)
                    rows.append(r)
                    log(f"  catch22_summary_mr_hydra: bases={inner['adaptive_base_models']} weights={weights} threshold={adaptive_threshold:.2f} acc={r['accuracy']:.3f} BA={r['balanced_accuracy']:.3f} Vrec={r['vowel_recall']:.3f} Crec={r['consonant_recall']:.3f}")
                except Exception as exc:
                    failures.append({"model": "catch22_summary_mr_hydra", "fold": fold, "test_subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            else:
                failures.append({"model": "catch22_summary_mr_hydra", "fold": fold, "test_subject": subject, "error": f"needs catch22_summary and mr_hydra; available={bases}"})
        elif "catch22_summary_mr_hydra" in REQUESTED:
            failures.append({"model": "catch22_summary_mr_hydra", "fold": fold, "test_subject": subject, "error": "adaptive fusion disabled by KARAONE_ENABLE_ADAPTIVE_FUSION=0"})
    summaries = {name: summarize([r for r in rows if r["model"] == name]) for name in sorted(set(r["model"] for r in rows))}
    return rows, {"summaries": summaries, "failures": failures}


def main() -> None:
    ensure_dependencies()
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== KARAONE FINAL KAGGLE PIPELINE ===")
    log("KARAONE_ROOT:", ROOT)
    log("Subjects:", SUBJECTS)
    log("Models:", REQUESTED)
    log("Runtime:", "FAST_MODE=" + str(FAST_MODE), "adaptive_mode=" + ADAPTIVE_VOWEL_MODE, "requested_adaptive_mode=" + REQUESTED_ADAPTIVE_VOWEL_MODE, "mode_override=" + str(ADAPTIVE_VOWEL_MODE_OVERRIDE), "require_vowel_class_weight=" + str(REQUIRE_ADAPTIVE_VOWEL_CLASS_WEIGHT), "inner_splits=" + str(ADAPTIVE_VOWEL_INNER_SPLITS), "adaptive_metric_aggregation=" + ADAPTIVE_METRIC_AGGREGATION, "vowel_weights=" + str(ADAPTIVE_VOWEL_WEIGHTS), "adaptive_best_single=" + str(ENABLE_ADAPTIVE_BEST_SINGLE), "best_single_ba_tolerance=" + str(ADAPTIVE_BEST_SINGLE_BA_TOLERANCE), "fine_decimal_vowel_grid=" + str(ADAPTIVE_FINE_GRID), "spatial_filter=" + SPATIAL_FILTER, "laplacian_neighbors=" + str(LAPLACIAN_NEIGHBORS), "laplacian_mix=" + str(LAPLACIAN_MIX), "rocket_kernels=" + str(ROCKET_KERNELS), "hydra_kernels=" + str(HYDRA_KERNELS), "hydra_groups=" + str(HYDRA_GROUPS), "n_jobs=" + str(N_JOBS), "static_svc_probability=" + str(STATIC_SVC_PROBABILITY))
    log("Preprocessing:", PREPROCESSING_MODE, "ASR=", ASR_ENABLED, "notch=", LINE_NOISE_HZ, "CAR=", PREPROCESSING_MODE not in {"official", "official_set", "set"}, "robust_normalize=", ROBUST_NORMALIZE, "vowel_prompt_balance=", VOWEL_PROMPT_BALANCE, "vowel_target_ratio=", VOWEL_TARGET_RATIO)
    log("Evaluation:", EVAL_MODE, "strict LOSO unless explicitly overridden")
    if EVAL_MODE in {"random", "stratified_random"}:
        log("WARNING: random trial-split sensitivity mode; subjects can appear in both train and test; do not report as cross-subject generalization")
    X, y, groups, prompts, records = load_cohort()
    np.savez_compressed(OUT / "preprocessed_phonemic_cohort.npz", X=X, y=y, groups=groups, prompts=prompts)
    audit = {
        "root": str(ROOT), "subjects": SUBJECTS, "task": "iy/uw versus m/n/piy/tiy/diy; words excluded",
        "preprocessing": {"mode": PREPROCESSING_MODE, "asr_enabled": ASR_ENABLED, "eog_regression": EOG_REGRESSION, "notch_hz": LINE_NOISE_HZ, "bandpass_hz": [1.0, 50.0], "car": PREPROCESSING_MODE not in {"official", "official_set", "set"}, "spatial_filter": SPATIAL_FILTER, "laplacian_neighbors": LAPLACIAN_NEIGHBORS, "laplacian_mix": LAPLACIAN_MIX, "robust_trial_normalization": ROBUST_NORMALIZE, "vowel_prompt_balance": VOWEL_PROMPT_BALANCE, "vowel_target_ratio": VOWEL_TARGET_RATIO, "vowel_max_multiplier": VOWEL_MAX_MULTIPLIER, "target_len": TARGET_LEN, "effective_fs": EFFECTIVE_FS, "qc_applied": APPLY_QC, "qc_threshold": QC_THRESHOLD},
        "n_trials": int(len(y)), "vowels": int(np.sum(y == 0)), "consonants": int(np.sum(y == 1)),
        "class_names": {"0": "vowel", "1": "consonant"},
        "selection_audit": COHORT_SELECTION_AUDIT,
        "records": [{k: v for k, v in r.items() if k not in {"X", "y"}} for r in records],
    }
    (OUT / "cohort_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    fold_subjects = [(int(fold), str(subject)) for fold, subject, _, _ in evaluation_splits(y, groups)]
    rows, result = run_models(X, y, groups, prompts=prompts)
    majority = float(np.mean(y == 1))
    output = {
        "protocol": {"evaluation": EVAL_MODE, "subjects": SUBJECTS, "random_test_size": RANDOM_TEST_SIZE if EVAL_MODE in {"random", "stratified_random"} else None, "random_repeats": RANDOM_REPEATS if EVAL_MODE in {"random", "stratified_random"} else None, "generalization_note": ("strict subject-held-out LOSO; all supervised fitting, scaling, feature selection, and adaptive weights are training-fold-only" if EVAL_MODE == "loso" else "sensitivity analysis only; this evaluation does not hold subjects out and must not be reported as cross-subject generalization"), "adaptive_vowel_mode": ADAPTIVE_VOWEL_MODE, "adaptive_vowel_class_weight_required": REQUIRE_ADAPTIVE_VOWEL_CLASS_WEIGHT, "allow_threshold_only": ALLOW_THRESHOLD_ONLY, "adaptive_ensemble_definition": "each classifier receives inner-training vowel-recall adaptation; class_weight_fast/full search the class-0 vowel weight and then refit the selected class weight on all outer-training subjects; threshold_only_fast is permitted only when KARAONE_ALLOW_THRESHOLD_ONLY=1 and KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT=0; class-weight candidate selection prioritizes vowel recall only within the global inner balanced-accuracy tolerance; the optional fusion branch reuses individual inner score streams when available; no outer test labels",
 "no_drcif": True},
        "data": {"shape": list(X.shape), "n_trials": int(len(y)), "vowels": int(np.sum(y == 0)), "consonants": int(np.sum(y == 1)), "majority_accuracy_baseline": majority},
        "models": REQUESTED + (["adaptive_best_single"] if ENABLE_ADAPTIVE_BEST_SINGLE else []), "runtime": {"fast_mode": FAST_MODE, "adaptive_vowel_mode": ADAPTIVE_VOWEL_MODE, "requested_adaptive_vowel_mode": REQUESTED_ADAPTIVE_VOWEL_MODE, "adaptive_vowel_mode_override": ADAPTIVE_VOWEL_MODE_OVERRIDE, "require_vowel_class_weight": REQUIRE_ADAPTIVE_VOWEL_CLASS_WEIGHT, "adaptive_vowel_inner_splits": ADAPTIVE_VOWEL_INNER_SPLITS, "adaptive_vowel_weights": ADAPTIVE_VOWEL_WEIGHTS, "fine_decimal_vowel_grid": ADAPTIVE_FINE_GRID, "adaptive_vowel_min_recall": ADAPTIVE_VOWEL_MIN_RECALL, "adaptive_vowel_ba_tolerance": ADAPTIVE_VOWEL_BA_TOLERANCE, "adaptive_vowel_min_gain": ADAPTIVE_VOWEL_MIN_GAIN, "adaptive_vowel_max_ba_drop": ADAPTIVE_VOWEL_MAX_BA_DROP, "adaptive_vowel_max_consonant_recall_drop": ADAPTIVE_VOWEL_MAX_CONSONANT_RECALL_DROP, "adaptive_vowel_threshold_min": ADAPTIVE_VOWEL_THRESHOLD_MIN, "adaptive_vowel_threshold_max": ADAPTIVE_VOWEL_THRESHOLD_MAX, "adaptive_best_single_enabled": ENABLE_ADAPTIVE_BEST_SINGLE, "adaptive_best_single_ba_tolerance": ADAPTIVE_BEST_SINGLE_BA_TOLERANCE, "use_spectral_summary": USE_SPECTRAL_SUMMARY, "use_temporal_bins": USE_TEMPORAL_BINS, "temporal_bins": TEMPORAL_BINS, "static_svc_C": STATIC_SVC_C, "rocket_kernels": ROCKET_KERNELS,
 "hydra_kernels": HYDRA_KERNELS, "hydra_groups": HYDRA_GROUPS, "n_jobs": N_JOBS, "static_svc_probability": STATIC_SVC_PROBABILITY, "eog_regression": EOG_REGRESSION, "spatial_filter": SPATIAL_FILTER, "laplacian_neighbors": LAPLACIAN_NEIGHBORS, "laplacian_mix": LAPLACIAN_MIX, "random_test_size": RANDOM_TEST_SIZE, "random_repeats": RANDOM_REPEATS, "vowel_prompt_balance": VOWEL_PROMPT_BALANCE, "vowel_target_ratio": VOWEL_TARGET_RATIO, "vowel_max_multiplier": VOWEL_MAX_MULTIPLIER, "adaptive_fusion_enabled": ENABLE_ADAPTIVE_FUSION, "threshold_only_requires_explicit_opt_in": True},
 "summaries": result["summaries"], "failures": result["failures"], "fold_rows": rows,
        "selection_audit": COHORT_SELECTION_AUDIT,
    }
    table_paths = write_result_tables(rows, result["summaries"], result["failures"], OUT, fold_subjects)
    output["result_tables"] = table_paths
    (OUT / "final_results.json").write_text(json.dumps(output, indent=2, default=str))
    pd.DataFrame(rows).to_csv(OUT / "final_fold_metrics.csv", index=False)
    summary_rows = []
    for name, s in result["summaries"].items():
        summary_rows.append({"model": name, **s})
    pd.DataFrame(summary_rows).to_csv(OUT / "final_summary_metrics.csv", index=False)
    log("\n=== SUMMARY ===")
    log(f"Consonant-majority accuracy baseline: {majority:.4f} ({100*majority:.2f}%)")
    for name, s in result["summaries"].items():
        log(f"{name}: acc={s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f} BA={s['balanced_accuracy_mean']:.4f}±{s['balanced_accuracy_std']:.4f} macroF1={s['macro_f1_mean']:.4f} Vrec={s['vowel_recall_mean']:.4f} Crec={s['consonant_recall_mean']:.4f} AUC={s['roc_auc_mean']:.4f}")
    log("Wrote:", OUT / "final_results.json")
    log("Complete per-fold table:", table_paths["fold_markdown"])
    log("Machine-readable per-fold CSV:", table_paths["fold_csv"])
    log("Fold coverage audit:", table_paths["coverage_csv"])


if __name__ == "__main__":
    main()
