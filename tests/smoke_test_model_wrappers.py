import importlib.util
import numpy as np
from pathlib import Path

path = Path('/home/ubuntu/karaone_work/karaone_final_kaggle.py')
spec = importlib.util.spec_from_file_location('karaone_final_kaggle', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rng = np.random.default_rng(123)
X = rng.normal(size=(20, 4, 128)).astype(np.float32)
y = np.array([0, 1] * 10, dtype=int)

for name in ['hydra', 'minirocket_hydra', 'multirocket', 'mr_hydra', 'catch22', 'summary', 'catch22_summary']:
    model = mod.make_model(name, vowel_weight=1.5, class_weight={0: 1.5, 1: 1.0})
    model.fit(X, y)
    score = np.asarray(model.score(X[:4]), dtype=float)
    assert score.shape == (4,), (name, score.shape)
    assert np.all(np.isfinite(score)), name
    print(name, 'PASS', score.shape, float(score.min()), float(score.max()))

print('MODEL WRAPPER SMOKE PASS')
