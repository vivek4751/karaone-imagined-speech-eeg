from pathlib import Path
import importlib.util
import numpy as np

SCRIPT = Path('/home/ubuntu/karaone_work/karaone_final_kaggle.py')
spec = importlib.util.spec_from_file_location('karaone_final_kaggle', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Keep this test isolated from the full requested run.
mod.REQUESTED = ['hydra', 'multirocket']
mod.ENABLE_ADAPTIVE_BEST_SINGLE = True
mod.ENABLE_ADAPTIVE_FUSION = False
mod.ADAPTIVE_VOWEL_BA_FLOOR = 0.50
mod.ADAPTIVE_VOWEL_MIN_RECALL = 0.40
mod.ADAPTIVE_BEST_SINGLE_BA_TOLERANCE = 0.01

class FakeModel:
    def __init__(self, name):
        self.name = name

    def score(self, X):
        # Class-1 (consonant) score; the branch must reuse this exact stream.
        n = len(X)
        if self.name == 'hydra':
            return np.linspace(0.20, 0.80, n)
        return np.linspace(0.75, 0.25, n)

def fake_fit(X_train, y_train, name, fold, groups_tr=None, prompts_tr=None):
    if name == 'hydra':
        info = {
            'adaptive_vowel_weight': 1.0,
            'adaptive_class_weight': {'vowel_0': 1.8, 'consonant_1': 0.7},
            'inner_selected_balanced_accuracy': 0.55,
            'inner_selected_vowel_recall': 0.50,
            'inner_selected_consonant_recall': 0.60,
            'inner_selected_macro_f1': 0.52,
            'inner_split_unit': 'subject_group',
        }
    else:
        info = {
            'adaptive_vowel_weight': 1.3,
            'adaptive_class_weight': {'vowel_0': 2.34, 'consonant_1': 0.7},
            'inner_selected_balanced_accuracy': 0.56,
            'inner_selected_vowel_recall': 0.70,
            'inner_selected_consonant_recall': 0.42,
            'inner_selected_macro_f1': 0.55,
            'inner_split_unit': 'subject_group',
        }
    return FakeModel(name), 0.50, info

def fake_splits(y, groups):
    idx = np.arange(len(y))
    yield 1, 'TEST_SUBJECT', idx[:6], idx[6:]

mod.fit_adaptive_vowel_model = fake_fit
mod.evaluation_splits = fake_splits

X = np.zeros((10, 2, 8), dtype=float)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
groups = np.array(['A'] * 6 + ['B'] * 4)
rows, result = mod.run_models(X, y, groups)
meta = [r for r in rows if r['model'] == 'adaptive_best_single']
assert len(meta) == 1, meta
assert meta[0]['adaptive_single_source_model'] == 'multirocket', meta[0]
assert meta[0]['adaptive_single_source_vowel_weight'] == 1.3, meta[0]
assert meta[0]['adaptive_single_inner_split_unit'] == 'subject_group', meta[0]
assert 'adaptive_best_single' in result['summaries'], result['summaries'].keys()
print('ADAPTIVE BEST-SINGLE SMOKE PASS')
print('selected_source=', meta[0]['adaptive_single_source_model'])
print('outer_metrics=', {k: meta[0][k] for k in ('accuracy', 'balanced_accuracy', 'vowel_recall', 'consonant_recall')})
