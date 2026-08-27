import importlib.util
import os
from pathlib import Path

import numpy as np

os.environ["KARAONE_EVAL_MODE"] = "random"
os.environ["KARAONE_RANDOM_TEST_SIZE"] = "0.20"
os.environ["KARAONE_RANDOM_REPEATS"] = "3"
os.environ["KARAONE_AUTO_INSTALL"] = "0"

path = Path(__file__).with_name("karaone_final_kaggle.py")
spec = importlib.util.spec_from_file_location("karaone_final_kaggle_random_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

y = np.array([0] * 15 + [1] * 15, dtype=int)
groups = np.array([f"S{i % 5:02d}" for i in range(len(y))], dtype=object)
splits = list(module.evaluation_splits(y, groups))
assert len(splits) == 3
for fold, label, tr, te in splits:
    assert fold in {1, 2, 3}
    assert label == f"random_trial_split_{fold}"
    assert len(tr) == 24 and len(te) == 6
    assert set(y[tr]) == {0, 1} and set(y[te]) == {0, 1}
    assert set(tr).isdisjoint(set(te))

print("random split smoke test passed")
print([(fold, label, len(tr), len(te)) for fold, label, tr, te in splits])
