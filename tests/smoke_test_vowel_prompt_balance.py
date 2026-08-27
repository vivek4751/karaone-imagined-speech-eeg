import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).with_name("karaone_final_kaggle.py")
spec = importlib.util.spec_from_file_location("karaone_vowel_balance_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

rng = np.random.default_rng(7)
X = rng.normal(size=(12, 2, 8)).astype(np.float32)
y = np.array([0, 0, 0, 0] + [1] * 8, dtype=int)
prompts = np.array(["iy", "iy", "uw", "uw", "m", "m", "n", "n", "piy", "tiy", "diy", "m"], dtype=object)
Xb, yb, info = module._vowel_prompt_balanced_training_data(X, y, prompts, seed=42)

assert info["enabled"] is True
assert len(Xb) == len(yb) == info["n_after"]
assert info["vowel_after"] == 6
assert info["consonant_after"] == 8
assert info["vowel_after"] / len(yb) <= module.VOWEL_TARGET_RATIO + 0.05
assert np.sum(yb == 0) == 6

# The helper must not mutate the source arrays.
assert X.shape == (12, 2, 8)
assert y.tolist() == [0, 0, 0, 0] + [1] * 8
print("vowel prompt balance smoke test passed", info)
