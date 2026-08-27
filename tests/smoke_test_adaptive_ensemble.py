import importlib.util
from pathlib import Path
import numpy as np

path = Path('/home/ubuntu/karaone_work/karaone_final_kaggle.py')
spec = importlib.util.spec_from_file_location('karaone_final_kaggle', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rng = np.random.default_rng(7)
n = 120
y = np.array([0, 1] * (n // 2), dtype=int)
# Synthetic inputs are used only to test control flow and invariants; no result
# from this test is a scientific benchmark.
X = np.zeros((n, 2, 32), dtype=np.float32)
latent = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 0.7, n)
X[:, 0, 0] = latent
X[:, 1, 0] = -0.35 * latent + rng.normal(0, 1.0, n)

class FakeModel:
    def __init__(self, kind, vowel_weight=1.0, class_weight=None):
        self.kind = kind
        self.vowel_weight = float(vowel_weight)
        self.class_weight = class_weight
    def fit(self, Xfit, yfit):
        self.fit_labels = np.asarray(yfit).copy()
        return self
    def score(self, Xscore):
        z = Xscore[:, 0, 0]
        if self.kind == 'good':
            return 1.0 / (1.0 + np.exp(-z))
        if self.kind == 'adaptive':
            # Increasing the vowel class weight shifts the consonant score down,
            # making vowel recovery better on this deliberately noisy fixture.
            z = z - 1.2 * np.log(max(self.vowel_weight, 1e-6))
            return 1.0 / (1.0 + np.exp(-z))
        return 1.0 / (1.0 + np.exp(-Xscore[:, 1, 0]))

mod.make_model = lambda name, **kwargs: FakeModel(name, **kwargs)
mod.INNER_SPLITS = 3
mod.ADAPTIVE_MAX_BASES = 2
mod.ADAPTIVE_GRID_STEP = 10
mod.ADAPTIVE_THRESHOLD_STEP = 0.02
weights, threshold, info = mod.adaptive_weights(X, y, ['good', 'weak'], fold=1)
assert set(weights) == {'good', 'weak'}, weights
assert abs(sum(weights.values()) - 1.0) < 1e-9, weights
assert all(v >= 0.0 for v in weights.values()), weights
assert 0.30 <= threshold <= 0.70, threshold
assert len(info['adaptive_base_models']) == 2, info
assert all(np.isfinite(v) for v in info['inner_balanced_accuracy_after_calibration'].values())
print('adaptive smoke test passed')
print('weights=', weights)
print('threshold=', threshold)
print('selected_inner_BA=', info['inner_selected_balanced_accuracy'])
print('selected_inner_vowel_recall=', info['inner_selected_vowel_recall'])
print('bases=', info['adaptive_base_models'])

# Also verify metrics honors a non-default threshold.
r = mod.metrics(y[:10], np.linspace(0.0, 1.0, 10), 'TEST', 'adaptive', 1, threshold=threshold)
assert r['decision_threshold'] == threshold
print('threshold metric audit passed')

# The fast fusion path must reuse already-computed inner score streams.
precomputed = {
    'good': 1.0 / (1.0 + np.exp(-X[:, 0, 0])),
    'weak': 1.0 / (1.0 + np.exp(-X[:, 1, 0])),
}
_, _, reused_info = mod.adaptive_weights(X, y, ['good', 'weak'], fold=3, precomputed_oof=precomputed)
assert reused_info['fusion_score_source'] == 'reused_individual_inner_oof'
print('precomputed fusion reuse audit passed')

# Verify the newly patched single-model trainer accepts keyword class weights,
# performs a complete inner split, and returns an auditable vowel target.
mod.ADAPTIVE_VOWEL_MODE = 'class_weight_fast'
mod.ADAPTIVE_VOWEL_WEIGHTS = [1.0, 3.0]
mod.ADAPTIVE_VOWEL_INNER_SPLITS = 2
mod.ADAPTIVE_VOWEL_THRESHOLD_STEP = 0.10
mod.ADAPTIVE_VOWEL_BA_FLOOR = 0.50
final_adaptive_model, selected_threshold, single_info = mod.fit_adaptive_vowel_model(X, y, 'adaptive', fold=2)
assert single_info['adaptive_target'] == 'vowel_recall_class_0'
assert single_info['adaptive_mode'] == 'class_weight_fast'
assert single_info['inner_splits'] == 2
assert single_info['inner_model_fits'] == 4
assert 0.20 <= selected_threshold <= 0.80
assert len(single_info['inner_vowel_weight_candidates']) == 2
assert single_info['adaptive_vowel_weight'] in {1.0, 3.0}
assert single_info['adaptive_class_weight']['vowel_0'] > 0
assert final_adaptive_model.vowel_weight == single_info['adaptive_vowel_weight']
assert final_adaptive_model.class_weight[0] == single_info['adaptive_class_weight']['vowel_0']
print('single-model vowel adaptation smoke test passed')
print('selected_vowel_weight=', single_info['adaptive_vowel_weight'])
print('selected_threshold=', selected_threshold)

# Explicit opt-out remains available for a threshold-only sensitivity run.
mod.ADAPTIVE_VOWEL_MODE = 'threshold_only_fast'
mod.ADAPTIVE_VOWEL_INNER_SPLITS = 2
_, fast_threshold, fast_info = mod.fit_adaptive_vowel_model(X, y, 'adaptive', fold=4)
assert fast_info['adaptive_mode'] == 'threshold_only_fast'
assert fast_info['inner_model_fits'] == 2
assert fast_info['final_model_fits'] == 1
assert fast_info['adaptive_vowel_weight'] == 1.0
assert 0.20 <= fast_threshold <= 0.80
print('threshold-only fast adaptation smoke test passed')

# Confirm grouped inner validation is selected when multiple training subjects
# are available, and no subject appears on both sides of a split.
group_ids = np.repeat(np.arange(6), 20)
group_splits, split_unit = mod._make_adaptive_splits(y, group_ids, 2, 99)
assert split_unit == 'subject_group', split_unit
for itr, iva in group_splits:
    assert set(group_ids[itr]).isdisjoint(set(group_ids[iva]))
print('subject-group inner split audit passed')

print('SUCCESS')
