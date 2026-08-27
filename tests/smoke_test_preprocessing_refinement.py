import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("KARAONE_PREPROCESSING_MODE", "asr_notch_car")
os.environ.setdefault("KARAONE_AUTO_INSTALL", "0")
os.environ.setdefault("KARAONE_OUTPUT", "/tmp/karaone_preprocess_refinement_test")
sys.path.insert(0, str(ROOT))
import karaone_final_kaggle as pipe

import numpy as np
import mne

rng = np.random.default_rng(7)
x = rng.normal(size=(8, 256)).astype(np.float32)
names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
y, info = pipe.apply_spatial_filter(x, names)
assert y.shape == x.shape
assert np.all(np.isfinite(y))
assert info["spatial_filter"] in {"small_laplacian", "laplacian_unavailable"}

# A valid raw-like object with no EOG channels must be a safe no-op.
raw = mne.io.RawArray(
    rng.normal(size=(2, 256)).astype(np.float64),
    mne.create_info(["C3", "C4"], sfreq=256.0, ch_types=["eeg", "eeg"]),
    verbose="ERROR",
)
data = raw.get_data().astype(np.float32)
clean, eog_info = pipe.regress_eog_from_eeg(raw, np.array([0, 1]), data)
assert clean.shape == data.shape
assert np.allclose(clean, data)
assert eog_info["eog_regression"] is False
print("PREPROCESSING REFINEMENT SMOKE PASS", info, eog_info)
