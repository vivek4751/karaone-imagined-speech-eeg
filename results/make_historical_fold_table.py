from pathlib import Path
import re
import pandas as pd

SRC = Path('/home/ubuntu/upload/pasted_content_27.txt')
OUT = Path('/home/ubuntu/karaone_work')

fold_re = re.compile(r'^FOLD\s+(\d+):\s+held-out=([^\s]+)\s+train=(\d+)\s+test=(\d+)')
metric_re = re.compile(
    r'vowel_weight=(?P<weight>[0-9.]+).*?'
    r'(?:class_weight0=(?P<cw0>[0-9.]+)\s+)?'
    r'threshold=(?P<threshold>[0-9.]+)\s+'
    r'acc=(?P<accuracy>[0-9.]+)\s+'
    r'BA=(?P<ba>[0-9.]+)\s+'
    r'Vrec=(?P<vrec>[0-9.]+)\s+'
    r'Crec=(?P<crec>[0-9.]+)'
)
models = [
    'catch22_summary_mr_hydra', 'adaptive_best_single', 'minirocket_hydra',
    'multirocket', 'mr_hydra', 'catch22_summary', 'catch22', 'summary', 'hydra'
]
rows = []
current = None
for raw in SRC.read_text().splitlines():
    m = fold_re.match(raw)
    if m:
        current = {'fold': int(m.group(1)), 'test_subject': m.group(2), 'n_test': int(m.group(4))}
        continue
    if current is None or not raw.startswith('  '):
        continue
    stripped = raw.strip()
    model = next((name for name in models if stripped.startswith(name + ':')), None)
    if model is None:
        continue
    parsed = metric_re.search(stripped)
    if parsed:
        values = parsed.groupdict()
    else:
        fusion_metric = re.search(
            r'threshold=(?P<threshold>[0-9.]+)\s+'
            r'acc=(?P<accuracy>[0-9.]+)\s+'
            r'BA=(?P<ba>[0-9.]+)\s+'
            r'Vrec=(?P<vrec>[0-9.]+)\s+'
            r'Crec=(?P<crec>[0-9.]+)', stripped)
        if model != 'catch22_summary_mr_hydra' or not fusion_metric:
            continue
        values = {'weight': None, 'cw0': None, **fusion_metric.groupdict()}
    row = dict(current)
    row['model'] = model
    for key, value in values.items():
        row[key] = float(value) if value is not None else None
    source = re.search(r'\bsource=([^\s]+)', stripped)
    row['source_model'] = source.group(1) if source else ''
    weights = re.search(r'weights=(\{.*?\})', stripped)
    row['fusion_weights'] = weights.group(1) if weights else ''
    row['macro_f1'] = None
    row['vowel_precision'] = None
    row['consonant_precision'] = None
    row['roc_auc'] = None
    row['average_precision'] = None
    row['confusion_matrix'] = ''
    row['note'] = 'Parsed from completed Kaggle log; precision/F1/AUC/confusion fields were not printed in that log.'
    rows.append(row)

frame = pd.DataFrame(rows)
frame = frame.sort_values(['fold', 'test_subject', 'model'], kind='stable').reset_index(drop=True)
expected = 8 * 9
if len(frame) != expected:
    raise RuntimeError(f'Expected {expected} rows, parsed {len(frame)}')
if frame.groupby('fold').size().tolist() != [9] * 8:
    raise RuntimeError('Not all folds contain nine model rows')

cols = [
    'fold', 'test_subject', 'model', 'n_test', 'accuracy', 'balanced_accuracy',
    'vowel_recall', 'consonant_recall', 'adaptive_vowel_weight',
    'adaptive_class_weight_vowel_0', 'decision_threshold', 'source_model',
    'fusion_weights', 'macro_f1', 'vowel_precision', 'consonant_precision',
    'roc_auc', 'average_precision', 'confusion_matrix', 'note'
]
frame = frame.rename(columns={
    'ba': 'balanced_accuracy', 'vrec': 'vowel_recall', 'crec': 'consonant_recall',
    'weight': 'adaptive_vowel_weight', 'cw0': 'adaptive_class_weight_vowel_0',
    'threshold': 'decision_threshold'
})
frame = frame[cols]
frame.to_csv(OUT / 'historical_pasted_content_27_per_fold_table.csv', index=False)

display = frame.copy()
for col in ['accuracy', 'balanced_accuracy', 'vowel_recall', 'consonant_recall']:
    display[col] = display[col].map(lambda x: '' if pd.isna(x) else f'{100*x:.2f}%')
for col in ['adaptive_vowel_weight', 'adaptive_class_weight_vowel_0', 'decision_threshold']:
    display[col] = display[col].map(lambda x: '' if pd.isna(x) else f'{x:.4f}')
display = display.rename(columns={
    'fold': 'Fold', 'test_subject': 'Test subject', 'model': 'Model', 'n_test': 'N test',
    'accuracy': 'Accuracy', 'balanced_accuracy': 'Balanced accuracy',
    'vowel_recall': 'Vowel recall', 'consonant_recall': 'Consonant recall',
    'adaptive_vowel_weight': 'Vowel weight', 'adaptive_class_weight_vowel_0': 'Class weight 0',
    'decision_threshold': 'Threshold', 'source_model': 'Source model',
    'fusion_weights': 'Fusion weights', 'macro_f1': 'Macro-F1',
    'vowel_precision': 'Vowel precision', 'consonant_precision': 'Consonant precision',
    'roc_auc': 'ROC-AUC', 'average_precision': 'Average precision',
    'confusion_matrix': 'Confusion matrix', 'note': 'Note'
})
md = (
    '# Historical per-fold table from `pasted_content_27`\n\n'
    'This is a faithful parse of the completed Kaggle log available in the workspace. '
    'It used the eight-subject strict-LOSO cohort and the two-candidate `[1.0, 2.0]` adaptive grid. '
    'The log printed accuracy, balanced accuracy, vowel recall, consonant recall, threshold, and selected vowel weight. '
    'It did not print per-fold precision, macro-F1, ROC-AUC, average precision, or confusion matrices, so those cells are intentionally blank rather than reconstructed. '
    'The new pipeline writes those fields directly in `final_fold_results_table.md` on the next fresh Kaggle run.\n\n'
    + display.to_markdown(index=False) + '\n'
)
(OUT / 'historical_pasted_content_27_per_fold_table.md').write_text(md)
print(f'Wrote {len(frame)} rows')
print(frame.groupby('model').size().to_string())
