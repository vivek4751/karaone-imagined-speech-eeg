from pathlib import Path
import importlib.util
import numpy as np

path = Path('/home/ubuntu/karaone_work/karaone_final_kaggle.py')
spec = importlib.util.spec_from_file_location('karaone_final_kaggle', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

records = {
    'A': {'subject': 'A', 'X': np.zeros((90, 62, 1280), np.float32), 'y': np.r_[np.zeros(30, int), np.ones(60, int)], 'prompts': ['iy'] * 30 + ['m'] * 60, 'raw_phonemic_count': 100, 'kept_count': 90, 'rejected_count': 10, 'rejection_fraction': 0.10, 'subject_rejected': False, 'subject_rejection_reason': None, 'class_coverage_after_qc': {'vowel': 30, 'consonant': 60}, 'excluded_words': 10, 'qc_fraction_min_median_max': [0.0, 0.04, 0.10], 'qc_threshold': 0.10, 'asr_info': {}, 'sfreq': 1000.0, 'channel_names': [f'EEG{i:02d}' for i in range(62)], 'epoch_metadata_source': '', 'label_source': '', 'recording_path': '', 'recording_format': 'set'},
    'B': {'subject': 'B', 'X': np.zeros((95, 62, 1280), np.float32), 'y': np.r_[np.zeros(35, int), np.ones(60, int)], 'prompts': ['iy'] * 35 + ['m'] * 60, 'raw_phonemic_count': 100, 'kept_count': 95, 'rejected_count': 5, 'rejection_fraction': 0.05, 'subject_rejected': False, 'subject_rejection_reason': None, 'class_coverage_after_qc': {'vowel': 35, 'consonant': 60}, 'excluded_words': 5, 'qc_fraction_min_median_max': [0.0, 0.02, 0.08], 'qc_threshold': 0.10, 'asr_info': {}, 'sfreq': 1000.0, 'channel_names': [f'EEG{i:02d}' for i in range(62)], 'epoch_metadata_source': '', 'label_source': '', 'recording_path': '', 'recording_format': 'set'},
    'C': {'subject': 'C', 'X': np.zeros((70, 62, 1280), np.float32), 'y': np.r_[np.zeros(20, int), np.ones(50, int)], 'prompts': ['iy'] * 20 + ['m'] * 50, 'raw_phonemic_count': 100, 'kept_count': 70, 'rejected_count': 30, 'rejection_fraction': 0.30, 'subject_rejected': True, 'subject_rejection_reason': 'rejection_fraction>0.100', 'class_coverage_after_qc': {'vowel': 20, 'consonant': 50}, 'excluded_words': 0, 'qc_fraction_min_median_max': [0.0, 0.20, 0.90], 'qc_threshold': 0.10, 'asr_info': {}, 'sfreq': 1000.0, 'channel_names': [f'EEG{i:02d}' for i in range(62)], 'epoch_metadata_source': '', 'label_source': '', 'recording_path': '', 'recording_format': 'set'},
}

mod.SUBJECTS = ['A', 'B', 'C']
mod.SUBJECT_OVERRIDE = []
mod.KNOWN_SUBJECTS = ['A', 'B', 'C']
mod.MAX_SUBJECTS = 2
mod.REQUIRE_BEST_EIGHT = False
mod.REQUIRE_EXPECTED = False
mod.load_subject = lambda subject: records[subject]
X, y, groups, prompts, selected = mod.load_cohort()
assert [r['subject'] for r in selected] == ['B', 'A'], [r['subject'] for r in selected]
assert 'C' in [x['subject'] for x in mod.COHORT_SELECTION_AUDIT['not_selected_subjects']]
assert X.shape[0] == 185 and int((y == 0).sum()) == 65 and int((y == 1).sum()) == 120
print('QC selection smoke test passed:', mod.COHORT_SELECTION_AUDIT['selected_subjects'])
print('Rejected subject audit:', mod.COHORT_SELECTION_AUDIT['not_selected_subjects'])

def main():
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

