# KaraOne Final Kaggle Pipeline

## Purpose

This single-file pipeline performs phonemic-only binary classification on the KaraOne imagined-speech EEG dataset. The vowel class is `iy` and `uw`; the consonant/syllabic class is `m`, `n`, `piy`, `tiy`, and `diy`. The word prompts `gnaw`, `knew`, `pat`, and `pot` are excluded before model fitting.

The default experiment discovers the known candidate subjects, removes noisy trials using the fixed quality policy, ranks all subjects with usable post-QC vowel and consonant trials by noise quality only, and retains the eight cleanest subjects. The default candidate list is `MM05`, `MM08`, `MM09`, `MM10`, `MM11`, `MM12`, `MM14`, `MM15`, `MM16`, `MM18`, `MM19`, `MM20`, `MM21`, and `P02`. Selection never uses classification performance. Subjects exceeding the 10% subject-level rejection fraction are flagged in the audit and remain in the default best-eight ranking so that the requested eight-subject cohort can be formed; set `KARAONE_ENFORCE_SUBJECT_REJECTION=1` only for a separate stricter sensitivity analysis.

## Kaggle setup

Add the full KaraOne raw dataset as a Kaggle input. In the notebook, set the root to the directory that contains the subject folders. For the previously used Kaggle layout:

```python
import os
os.environ["KARAONE_ROOT"] = "/kaggle/input/datasets/vivekranjan4751/karaone"
os.environ["KARAONE_OUTPUT"] = "/kaggle/working/karaone_final_results"
```

Upload `karaone_final_kaggle.py` to `/kaggle/working/`, then run:

```python
!python /kaggle/working/karaone_final_kaggle.py
```

The script can install missing optional packages automatically. If Kaggle blocks package installation, run this cell before the script:

```python
!pip -q install mne meegkit aeon
```

The input must contain, for each candidate subject, a raw EEGLAB `.set` or official `.cnt` file, `epoch_inds.mat`, and `kinect_data/labels.txt`. The code uses the official `thinking_inds` intervals and never invents a fixed five-second epoch. If only `all_features_simple.mat` or `all_features_ICA.mat` is available, this raw script is not the correct loader; use the separately documented feature-file branch.

## Default preprocessing

The default `KARAONE_PREPROCESSING_MODE=asr_notch_car` applies a 50 Hz notch, 1–50 Hz zero-phase band-pass, optional continuous EOG regression, MEEGkit ASR calibrated from unlabeled clearing/rest intervals when available, common-average reference, a small fixed-montage Laplacian, robust per-trial channel normalization, and resampling to 256 Hz with 1,280 samples. The retained EEG set is the first 62 EEG channels in the validated channel order; auxiliary EOG, ECG, EMG, VEO, HEO, and Trigger channels are not used as classifier inputs. EOG regression is applied before epoch extraction only when EOG channels are available and is label-independent. Noise QC is computed after ASR and before resampling/reference/spatial filtering; trials with noise fraction greater than 10% are removed.

ASR, EOG regression, the fixed-montage Laplacian, and per-trial robust normalization do not use prompt labels. The script records the ASR backend, EOG-cleaning status, calibration source, pre- and post-ASR QC statistics, rejected trials, subject rejection reasons, channel names, and preprocessing settings in `cohort_audit.json`. The Laplacian defaults to four nearest valid standard-10–20 neighbors with a 0.50 mix between the original channel and its local spatial difference; set `KARAONE_SPATIAL_FILTER=none` for a declared ablation. Set `KARAONE_EOG_REGRESSION=0` for an explicit no-EOG-regression control.
 Trial rejection is enabled by default with `KARAONE_NOISE_THRESHOLD=0.10`. A subject must have at least one retained vowel and one retained consonant trial to be rankable. The eight-subject ranking is based on rejection fraction, median noise, maximum noise, and subject ID only. Subjects above the 10% subject-level rejection fraction are visibly flagged but are not automatically removed in the default best-eight run. Set `KARAONE_ENFORCE_SUBJECT_REJECTION=1` for a separate strict subject-exclusion analysis. To run an explicitly separate sensitivity analysis without trial rejection, set `KARAONE_APPLY_QC_REJECTION=0`; do not call that the requested best-eight result.

Do not call an already processed official SET file “raw.” Run it as a separate control with `KARAONE_PREPROCESSING_MODE=official` and report that branch separately.

## Requested model suite

The default model list is:

```text
hydra,
minirocket_hydra,
multirocket,
mr_hydra,
catch22,
summary,
catch22_summary,
catch22_summary_mr_hydra
```

The available Aeon version may not expose a direct `MiniRocketHydraClassifier`. Therefore, `minirocket_hydra` is implemented transparently as a 50:50 probability fusion of independently trained MiniRocket and Hydra models. Every requested classifier branch has a vowel-recall adaptive mode. The default practical setting is `KARAONE_FAST_MODE=1`, which uses actual **vowel class-weight adaptation**: it fits inverse-frequency-balanced models multiplied by the configured vowel-weight candidates, complete two-fold inner out-of-fold scores, and an inner threshold grid constrained to 0.35–0.65 in fast mode (configurable). When more than one training subject is available, those inner folds are **subject-grouped**, so trials from a participant cannot appear in both an inner fit and its validation partition. The selector requires minimum inner vowel recall, minimum consonant recall, and a balanced-accuracy floor. By default, each candidate is scored by the **mean of the subject-disjoint inner-fold metrics**, so a larger participant cannot dominate the selection. Set `KARAONE_ADAPTIVE_METRIC_AGGREGATION=pooled` only for a declared pooled sensitivity analysis. It then gives vowel recall priority only among operating points within a small balanced-accuracy tolerance of the best feasible point, preventing an almost-all-vowel threshold from winning. The selected vowel weight and threshold are then refit/applied using all outer-training subjects before the unseen subject is predicted. The outer-fold result is never used to choose the weight.
 `class_weight_fast` is therefore the default; `class_weight_full` retains the slower larger candidate search. Threshold-only adaptation is an explicit sensitivity mode and is blocked by default unless both `KARAONE_ALLOW_THRESHOLD_ONLY=1` and `KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT=0` are set. In all modes, the outer test subject and its labels are never used for class weights, calibration, model weights, or threshold selection.

The `catch22_summary_mr_hydra` branch is enabled by default and reuses the individual branches' already-computed inner score streams when available. It therefore performs score calibration and a small vowel-recall-focused blend search without refitting Catch22 or MR-Hydra models a second time. Set `KARAONE_ENABLE_ADAPTIVE_FUSION=0` only when an explicitly documented omission is desired.

To run a smaller smoke or thesis comparison, set the list explicitly:

```python
os.environ["KARAONE_MODELS"] = "catch22,summary,catch22_summary,mr_hydra,catch22_summary_mr_hydra"
```

The default fast configuration is intended for a complete Kaggle run: 512 Rocket kernels, 4 Hydra kernels, 8 Hydra groups, two subject-grouped inner folds, subject-macro adaptive metric selection, vowel-weight candidates `[1.0, 2.0]` in `class_weight_fast`, four-neighbor small Laplacian, optional EOG regression, 384 retained static features, deterministic whole-epoch and four-bin temporal statistics plus log/relative theta-alpha-beta-low-gamma band-power features in the Summary path, and decision-score SVC output without the extra SVC probability-calibration cross-validation. `KARAONE_N_JOBS` defaults to 2 in fast mode. These settings preserve the requested classifier names and strict LOSO protocol but trade some representation capacity for runtime. The larger configuration can be requested with `KARAONE_FAST_MODE=0`; it uses 2,000 Rocket kernels, 8 Hydra kernels, 64 Hydra groups, three inner folds, the five default vowel-weight candidates, 512 static features, and the slower full adaptive search. A reproducibility audit of the selected settings is written to `final_results.json`. Decimal weights are supported: set `KARAONE_ADAPTIVE_VOWEL_WEIGHTS="1.0,1.1,1.2,1.3,1.4,1.6,2.0,2.5"` for an explicit grid, or set `KARAONE_FINE_ADAPTIVE_VOWEL_GRID=1` to use the built-in 1.0–2.0 fine grid. Values above 2.0 are allowed, but the stability guard can reject them if inner consonant recall or balanced accuracy deteriorates. Each extra candidate adds subject-grouped inner fits, so the fine grid is an optional slower sensitivity run, not a free accuracy setting.

The default run already uses `KARAONE_ADAPTIVE_VOWEL_MODE=class_weight_fast` with `KARAONE_FAST_MODE=1`. The adaptive safeguards default to `KARAONE_ADAPTIVE_VOWEL_MIN_RECALL=0.40` and `KARAONE_ADAPTIVE_VOWEL_BA_TOLERANCE=0.02`; both are recorded in the runtime audit. For the exhaustive setting, use `KARAONE_FAST_MODE=0` and `KARAONE_ADAPTIVE_VOWEL_MODE=class_weight_full`. To run the threshold-only sensitivity analysis, explicitly set `KARAONE_ADAPTIVE_VOWEL_MODE=threshold_only_fast`, `KARAONE_ALLOW_THRESHOLD_ONLY=1`, and `KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT=0`; that mode does not adapt the classifier’s vowel class weight.

## Evaluation policy

The default `KARAONE_EVAL_MODE=loso` holds out one complete selected subject at a time. All supervised operations are fitted on the outer training subjects only, including standardization, variance filtering, ANOVA feature selection, Catch22 fitting, classifier-specific class weighting where enabled, classifier-specific decision thresholds, score calibration, optional ensemble weights, and optional ensemble thresholds. Adaptive inner validation prefers subject-grouped splits; if a supplemental run has only one subject in its training data, the audit records the fallback to trial-stratified folds. Ordinary accuracy is secondary because consonants are the majority class. The script reports accuracy, balanced accuracy, macro-F1, vowel precision/recall, consonant precision/recall, ROC-AUC, average precision, pooled confusion matrices, and per-fold confusion matrices. Each adaptive fold also records the unit-vowel-weight inner baseline and the selected-minus-unit inner metric differences; this is a training-only diagnostic and requires no extra outer-test fit. The fusion selector uses the same vowel-recall minimum, consonant-recall minimum, balanced-accuracy floor, and balanced-accuracy tolerance safeguards. Runtime mode and fit counts are recorded in the output so a fast result is not confused with the exhaustive configuration.

Within-subject five-fold results can be generated only as a supplemental subject-dependent analysis:

```python
os.environ["KARAONE_EVAL_MODE"] = "within_subject"
```

A separate random trial-level train/test sensitivity analysis is also available. It uses repeated stratified random trial splits, so trials from the same participant can occur in both training and testing. This can measure how much easier the task becomes when participant identity is shared, but it is **not** a cross-subject generalization result and must never replace the LOSO benchmark:

```python
os.environ["KARAONE_EVAL_MODE"] = "random"
os.environ["KARAONE_RANDOM_TEST_SIZE"] = "0.20"
os.environ["KARAONE_RANDOM_REPEATS"] = "5"
```

The random mode writes the same fold and summary table filenames, but the Markdown headings and `test_subject` values explicitly identify random trial splits. Do not select subjects or models using outer test performance. For a quick sensitivity run, restrict `KARAONE_MODELS` to `mr_hydra` or `catch22_summary` before expanding to the full suite.

## Output files

The script writes the following files under `KARAONE_OUTPUT`:

| File | Contents |
|---|---|
| `cohort_audit.json` | Candidate and selected subjects, counts, prompts, preprocessing, ASR, pre/post-ASR QC, rejection flags, channels, and integrity information |
| `preprocessed_phonemic_cohort.npz` | The processed EEG trials, labels, subject groups, and prompts |
| `final_results.json` | Complete protocol, summaries, failures, fold rows, metrics, and confusion matrices |
| `final_fold_metrics.csv` | Raw one-row-per-model-and-held-out-fold metrics, including nested adaptive fields |
| `final_summary_metrics.csv` | Mean and standard deviation of each metric by model |
| `final_fold_results_table.csv` | Flat machine-readable per-fold table with metrics, thresholds, class weights, source model, and confusion matrix |
| `final_fold_results_table.md` | Human-readable table containing every successful model/fold row; metrics are shown as percentages |
| `final_fold_results_table.html` | HTML version of the complete per-fold table |
| `final_fold_vowel_weight_candidates.csv` | One row per classifier/fold/candidate vowel weight; inner validation metrics plus repeated final outer-test reference metrics |
| `final_fold_vowel_weight_candidates.md` | Human-readable candidate-weight table with percentages and selected-weight flags |
| `final_fold_vowel_weight_candidates.html` | HTML version of the candidate-weight table |
| `final_summary_results_table.csv` | Flat aggregate summary table |
| `final_summary_results_table.md` | Human-readable aggregate summary table |
| `final_fold_coverage.csv` | Explicit expected model/fold coverage with `ok`, `failed`, or `missing` status |
| `fold_metrics_readme.md` | Definitions and relationships among the result files |
| `smoke_test_model_wrappers.py` (local only) | Optional local validation helper for the installed Aeon/static wrappers; it is not needed in Kaggle |

## What to show in the report

Show the cohort audit first, then the full `final_summary_metrics.csv`, then the per-fold confusion matrices and class recalls. Include the selected-subject audit, the exact retained trial totals, the noise threshold, the subject rejection threshold, and the ranking rule before showing model metrics. For the earlier five-subject validation, the consonant-majority baseline was `315/440 = 71.59%`; the selected best-eight cohort has its own majority baseline and must be recalculated from `cohort_audit.json`. A model with high ordinary accuracy but near-zero vowel recall is not a successful binary classifier. The strict LOSO balanced accuracy is the primary result.

This script is intended to produce a reproducible and honest professor-facing benchmark. It does not guarantee 73–80% performance. The earlier validated five-subject studies showed that ordinary accuracy can be inflated by consonant-majority predictions while balanced accuracy remains near chance; the final script preserves that diagnostic rather than hiding it.


## Refinements after the threshold-only Kaggle log

The attached run reported `adaptive_mode=threshold_only_fast` and `candidates=[1.0]`. It therefore did not perform vowel class-weight search, even though the environment displayed the configured list `[1.0, 2.0]`. The current script prevents this stale-setting failure: `class_weight_fast` is the default, and threshold-only adaptation requires both `KARAONE_ALLOW_THRESHOLD_ONLY=1` and `KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT=0`.

The adaptive selector now compares all usable vowel-weight candidates globally using the mean inner metric across subject-disjoint validation folds. The output also expands every candidate into `final_fold_vowel_weight_candidates.csv`, where `inner_*` columns are the values used for selection and `outer_*` columns are repeated only to identify the final selected model result for that fold; outer values never select the weight. It first identifies the best feasible inner balanced accuracy, keeps candidates within `KARAONE_ADAPTIVE_VOWEL_BA_TOLERANCE` of that value, and only then prioritizes vowel recall. In fast mode, the adaptive threshold search is constrained to 0.35–0.65 by default to reduce all-vowel and all-consonant decisions under cross-subject score shifts. These are training-fold-only safeguards; no outer test labels are used.

Use the following fresh-kernel settings for the refined run:

```python
import os
os.environ.update({
    "KARAONE_ROOT": "/kaggle/input/datasets/vivekranjan4751/karaone",
    "KARAONE_OUTPUT": "/kaggle/working/karaone_refined_results",
    "KARAONE_FAST_MODE": "1",
    "KARAONE_ADAPTIVE_VOWEL_MODE": "class_weight_fast",
    "KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT": "1",
    "KARAONE_ALLOW_THRESHOLD_ONLY": "0",
    "KARAONE_ADAPTIVE_VOWEL_WEIGHTS": "1.0,2.0",
    "KARAONE_ADAPTIVE_VOWEL_INNER_SPLITS": "2",
    "KARAONE_ADAPTIVE_METRIC_AGGREGATION": "subject_macro",
    "KARAONE_EOG_REGRESSION": "1",
    "KARAONE_SPATIAL_FILTER": "laplacian",
    "KARAONE_LAPLACIAN_NEIGHBORS": "4",
    "KARAONE_LAPLACIAN_MIX": "0.50",
    "KARAONE_ADAPTIVE_VOWEL_THRESHOLD_MIN": "0.35",
    "KARAONE_ADAPTIVE_VOWEL_THRESHOLD_MAX": "0.65",
    "KARAONE_USE_SPECTRAL_SUMMARY": "1",
    "KARAONE_USE_TEMPORAL_BINS": "1",
    "KARAONE_ENABLE_ADAPTIVE_FUSION": "1",
    "KARAONE_AUTO_INSTALL": "0",
})
!python /kaggle/working/karaone_final_kaggle.py
```

The output must show `adaptive_mode=class_weight_fast`, `adaptive_metric_aggregation=subject_macro`, `spatial_filter=laplacian`, `candidates=[1.0, 2.0]`, and `final fit with vowel_weight=...`. If `threshold_only_fast` appears without an explicit opt-in, the wrong script copy is being executed. For a prespecified preprocessing ablation, rerun with `KARAONE_SPATIAL_FILTER=none` and/or `KARAONE_EOG_REGRESSION=0`; do not select whichever ablation gives the best outer result after looking at the test folds.

The 72–78% accuracies in the earlier log are not sufficient evidence of improvement because its mean balanced accuracy remained close to chance and several folds predicted almost entirely one class. The refined benchmark should be compared using mean balanced accuracy, pooled balanced accuracy, macro-F1, vowel recall, consonant recall, ROC-AUC, average precision, and the pooled confusion matrix. A preprocessing change is retained only as a prespecified sensitivity result unless it improves the subject-independent metrics consistently across folds.

## Correct rule for selecting the vowel weight

Do not randomly choose a vowel weight, and do not use the final outer-fold accuracy to choose it. In each LOSO fold, the outer test subject is kept untouched. The training subjects are divided into subject-disjoint inner folds. Every candidate, such as `1.0`, `1.1`, `1.2`, `1.4`, or `2.5`, is fitted and scored in those inner folds. The default selector averages the balanced accuracy, vowel recall, consonant recall, and macro-F1 **across the inner folds**, gives vowel recall priority only within the balanced-accuracy tolerance, and applies the unit-weight stability guard. After the weight and threshold are selected, one final model is refit on all outer-training subjects. The resulting outer-fold accuracy is used only for the final report.

Thus, use the **mean inner validation metrics per LOSO fold** for learning the weight. Use the **mean outer-fold metrics across all held-out subjects** only to compare the completed experiment against another completed experiment. Never tune a weight from the mean outer accuracy, because that uses the test subjects and invalidates the LOSO estimate. If you want one fixed weight for all folds instead of per-fold adaptation, that weight must be selected in a separate nested validation experiment and then frozen before the final LOSO run.

## End-to-end validation

The current local smoke tests pass syntax checking, stale-mode protection, adaptive single-model fitting, subject-group inner validation, adaptive fusion score reuse, and the requested model-wrapper paths.

## References

[1]: https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html "KaraOne imagined-speech EEG database"
[2]: https://scikit-learn.org/stable/modules/cross_validation.html "scikit-learn cross-validation documentation"
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html "scikit-learn StratifiedGroupKFold documentation"

References: [1] [2] [3]


## Adaptive vowel-weight stability guard

The current `class_weight_fast` selector searches the configured vowel class-weight candidates, but it no longer accepts a heavier vowel weight merely because it produces a high inner vowel recall. After the candidate is selected, it is compared with the unit-weight candidate using inner out-of-fold scores only. A non-unit weight is retained only when it improves inner vowel recall by at least `KARAONE_ADAPTIVE_VOWEL_MIN_GAIN` (default `0.05` in fast mode), does not lower inner balanced accuracy by more than `KARAONE_ADAPTIVE_VOWEL_MAX_BA_DROP` (default `0.01`), and does not lower inner consonant recall by more than `KARAONE_ADAPTIVE_VOWEL_MAX_CONSONANT_RECALL_DROP` (default `0.05`). Otherwise the unit-weight candidate is selected. This guard is intended to improve mean LOSO stability; it does not use outer-test labels.

For an auditable refined run, add the following settings:

```python
os.environ.update({
    "KARAONE_ADAPTIVE_VOWEL_MODE": "class_weight_fast",
    "KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT": "1",
    "KARAONE_ALLOW_THRESHOLD_ONLY": "0",
    "KARAONE_ADAPTIVE_VOWEL_WEIGHTS": "1.0,2.0",
    "KARAONE_ADAPTIVE_VOWEL_MIN_GAIN": "0.05",
    "KARAONE_ADAPTIVE_VOWEL_MAX_BA_DROP": "0.01",
    "KARAONE_ADAPTIVE_VOWEL_MAX_CONSONANT_RECALL_DROP": "0.05",
})
```

The log should show `weight_guard selected_nonunit=True` when weight 2 is retained, or `selected_nonunit=False` followed by `vowel_weight=1` when the heavier weight is unstable. Both outcomes are valid adaptive results. The runtime JSON records all three guard limits and the selected-minus-unit inner metric differences.


## Adaptive best-single-model branch

The script also emits an `adaptive_best_single` row when `KARAONE_ENABLE_ADAPTIVE_BEST_SINGLE=1` (the default). This is not a new classifier and it does not add model fits. In each outer fold it selects one of the already-fitted requested classifiers using only that classifier's inner subject-grouped metrics: feasible balanced accuracy first, then candidates within `KARAONE_ADAPTIVE_BEST_SINGLE_BA_TOLERANCE` of the best feasible value, followed by vowel recall, consonant recall, and macro-F1. The selected classifier's held-out scores and its training-only threshold are reused for the outer test subject. This branch is an auditable adaptive model-selection result; it must be compared with every fixed requested model and must never be selected from outer-test performance.

The branch can be disabled for a base-model-only table:

```python
os.environ["KARAONE_ENABLE_ADAPTIVE_BEST_SINGLE"] = "0"
```

The runtime audit records `adaptive_best_single_enabled`, `adaptive_best_single_ba_tolerance`, the selected source model per fold, its selected vowel weight, and all inner candidate metrics.

## Complete per-fold table and bounded sensitivity run

After the model loop, the script writes `final_fold_results_table.md`, `final_fold_results_table.csv`, and `final_fold_results_table.html`. Each successful row is one held-out subject and one requested model. The table includes accuracy, balanced accuracy, macro-F1, vowel/consonant precision and recall, ROC-AUC, average precision, decision threshold, selected vowel weight, final class weights, adaptive source model where applicable, and the two-by-two confusion matrix. The rows are sorted by fold, held-out subject, and model; low-performing folds are not removed. `final_fold_coverage.csv` lists every expected combination, including any failed or missing branch, so the table cannot be mistaken for a cherry-picked subset.

For the primary run, keep the default two-candidate grid so the complete eight-model suite remains practical. For a targeted sensitivity experiment on the most promising branches from earlier runs, use a fresh kernel and explicitly record the different model list and grid, for example:

```python
import os
os.environ.update({
    "KARAONE_ROOT": "/kaggle/input/datasets/vivekranjan4751/karaone",
    "KARAONE_OUTPUT": "/kaggle/working/karaone_mrhydra_decimal_sensitivity",
    "KARAONE_MODELS": "multirocket,mr_hydra,catch22_summary,catch22_summary_mr_hydra",
    "KARAONE_ADAPTIVE_VOWEL_MODE": "class_weight_fast",
    "KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT": "1",
    "KARAONE_ALLOW_THRESHOLD_ONLY": "0",
    "KARAONE_ADAPTIVE_VOWEL_WEIGHTS": "1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.75,2.0",
    "KARAONE_ADAPTIVE_VOWEL_INNER_SPLITS": "2",
    "KARAONE_ENABLE_ADAPTIVE_FUSION": "1",
    "KARAONE_ENABLE_ADAPTIVE_BEST_SINGLE": "1",
    "KARAONE_AUTO_INSTALL": "0",
})
!python /kaggle/working/karaone_final_kaggle.py
```

This is not an epoch-based neural-training loop. Hydra, MultiRocket, and MR-Hydra are fitted feature/classifier pipelines; increasing an `epoch` count is not a valid setting for them. The decimal sensitivity run adds inner model fits and can therefore be substantially slower. Its result should be compared with the primary run using mean and pooled balanced accuracy, vowel recall, macro-F1, ROC-AUC, and the complete per-fold table—not ordinary accuracy alone. A higher ordinary accuracy with zero vowel recall is not an improvement.


## New vowel-recall refinement: prompt-balanced training

The current refinement adds a training-only prompt-balanced resampling step. The classifier still receives only EEG signals and class labels; prompt names are not included as features. Inside every adaptive inner-training split, the two vowel prompts (`iy` and `uw`) are first equalized when both are present, and the total vowel training prior is then increased toward the declared `KARAONE_VOWEL_TARGET_RATIO` subject to `KARAONE_VOWEL_MAX_MULTIPLIER`. The same operation is applied to the complete outer-training set before the final refit. The held-out subject's EEG, labels, and prompts are never used in this step.

The default settings are:

```python
os.environ["KARAONE_VOWEL_PROMPT_BALANCE"] = "1"
os.environ["KARAONE_VOWEL_TARGET_RATIO"] = "0.40"
os.environ["KARAONE_VOWEL_MAX_MULTIPLIER"] = "2.0"
```

This is intended to address two separate failure modes visible in the attached run: the minority vowel class is underrepresented, and the two vowel prompts can be imbalanced even within the minority class. The cap is important because unrestricted oversampling can simply trade a consonant-heavy classifier for an all-vowel classifier. The output audit and every adaptive fold row record the before/after counts and the selected settings.

Run the refinement as a predeclared ablation rather than selecting whichever version gives the best outer result:

```python
# Main refined run
os.environ["KARAONE_VOWEL_PROMPT_BALANCE"] = "1"

# Matched ablation
# os.environ["KARAONE_VOWEL_PROMPT_BALANCE"] = "0"
```

Compare the two runs using mean outer balanced accuracy, mean vowel recall, mean consonant recall, macro-F1, and ordinary accuracy. Do not use the held-out outer folds to choose the settings after seeing the results. A higher vowel recall is not automatically an improvement if balanced accuracy and consonant recall collapse.


## Important correction after the latest Kaggle output

The attached run confirms that prompt balancing was enabled, but it still selected unit vowel weights in most folds because the heavier candidates reduced consonant recall. The script has now been corrected so inverse-frequency class weights are computed from the actual resampled fit set, rather than from the original imbalanced subset. This prevents accidental double-counting of the vowel class through both duplication and the original class prior. The latest attached Kaggle output should therefore be treated as the pre-correction prompt-balanced result; rerun the corrected file before drawing conclusions about this refinement.

The output also showed `EOG=False` for every participant, so the EOG-regression branch could not operate on that cohort. The Laplacian branch remains available. In the revised run, inspect `vowel_prompt_balance` and the final class weights in each fold table to verify that the intended training prior was applied.
