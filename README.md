# KaraOne Imagined-Speech EEG: Phonemic Vowel vs Consonant

This repository contains a single-file Kaggle pipeline for binary classification of **phonemic** KaraOne imagined-speech EEG trials. Class 0 contains the vowel prompts `iy` and `uw`; class 1 contains the consonant or syllabic prompts `m`, `n`, `piy`, `tiy`, and `diy`. The word prompts `gnaw`, `knew`, `pat`, and `pot` are excluded.

The main implementation is [`src/karaone_final_kaggle.py`](src/karaone_final_kaggle.py). It performs raw-file discovery, official trial-index and label alignment, ASR calibration from clearing intervals, 50 Hz notch filtering, 1–50 Hz filtering, CAR, robust trial normalization, post-ASR trial QC, noise-only best-eight subject selection, optional EOG regression, an official-style small spatial Laplacian, and the requested Hydra/MiniRocket-Hydra/MultiRocket/MR-Hydra/Catch22/Summary/Catch22+Summary/Catch22+Summary+MR-Hydra suite. DrCIF is not used.

## Evaluation protocol

The default benchmark is strict leave-one-subject-out evaluation. All supervised transformations, adaptive vowel-weight selection, threshold selection, fusion selection, and prompt-balanced resampling are performed inside the outer-training data. The held-out subject is used only once for final evaluation.

The pipeline also contains a separate random trial-split sensitivity mode. That mode is explicitly labeled in its outputs and must not be interpreted as cross-subject generalization because the same participant can appear in both training and test trials.

Ordinary accuracy is reported together with balanced accuracy, macro-F1, vowel precision/recall, consonant precision/recall, ROC-AUC, average precision, confusion matrices, and per-fold summaries. Because the cohort is consonant-majority, the consonant-majority baseline must be shown beside ordinary accuracy.

## Kaggle use

Upload `src/karaone_final_kaggle.py` to `/kaggle/working/` and run it with the KaraOne dataset mounted at `/kaggle/input/datasets/vivekranjan4751/karaone`. The exact setup, file discovery details, runtime controls, and output inventory are documented in [`docs/karaone_final_kaggle_guide.md`](docs/karaone_final_kaggle_guide.md).

A typical strict-LOSO configuration is:

```python
import os

os.environ.update({
    "KARAONE_ROOT": "/kaggle/input/datasets/vivekranjan4751/karaone",
    "KARAONE_OUTPUT": "/kaggle/working/karaone_results",
    "KARAONE_EVAL_MODE": "loso",
    "KARAONE_FAST_MODE": "1",
    "KARAONE_ADAPTIVE_VOWEL_MODE": "class_weight_fast",
    "KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT": "1",
    "KARAONE_ALLOW_THRESHOLD_ONLY": "0",
    "KARAONE_ADAPTIVE_METRIC_AGGREGATION": "subject_macro",
    "KARAONE_EOG_REGRESSION": "1",
    "KARAONE_SPATIAL_FILTER": "laplacian",
    "KARAONE_VOWEL_PROMPT_BALANCE": "1",
    "KARAONE_VOWEL_TARGET_RATIO": "0.40",
    "KARAONE_VOWEL_MAX_MULTIPLIER": "2.0",
    "KARAONE_ADAPTIVE_VOWEL_WEIGHTS": "1.0,1.1,1.2,1.3,1.4,1.6,2.0",
})

!python /kaggle/working/karaone_final_kaggle.py
```

## Per-fold and per-weight tables

Each run writes `final_fold_results_table.csv` and its Markdown/HTML versions for final outer-test rows. It also writes `final_fold_vowel_weight_candidates.csv`, `.md`, and `.html`. The candidate table contains every classifier/fold/candidate-weight row and the corresponding **inner subject-disjoint validation** accuracy, balanced accuracy, macro-F1, vowel recall, consonant recall, feasibility flag, and selected-weight flag. Outer metrics in candidate rows are reference values for the selected final fit; they are not used to tune the held-out fold.

## Repository results

The `results/` directory contains selected attached Kaggle logs and a historical per-fold table reconstructed from a completed run. These artifacts are included for provenance and interpretation. They are not a replacement for rerunning the current script after code changes. No raw EEG files are included.

The synthetic illustrative artifacts generated during development are intentionally excluded from this repository so they cannot be confused with measured results.

## Validation

The `tests/` directory contains deterministic smoke tests for the adaptive weight guard, adaptive fusion, adaptive best-single branch, model wrappers, preprocessing refinements, QC subject selection, random-split labeling, prompt-balanced training, and complete fold-table export. These tests use synthetic control fixtures only to test code paths; they are not scientific performance results.

## Scientific reporting note

Do not select a vowel weight or threshold from the outer held-out fold. Select it from subject-disjoint inner validation, refit on all outer-training subjects, and report the outer fold once. Report mean strict-LOSO metrics separately from the best individual fold and from any random or within-subject sensitivity analysis.
