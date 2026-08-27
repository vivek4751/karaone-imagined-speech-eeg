# KaraOne Phonemic Vowel-versus-Consonant Classifier

## Submission summary

The submitted Kaggle script, `karaone_final_kaggle.py`, implements an auditable binary classifier for imagined-speech phonemic prompts. Vowels are `iy` and `uw`; consonant/syllabic prompts are `m`, `n`, `piy`, `tiy`, and `diy`. The word prompts `gnaw`, `knew`, `pat`, and `pot` are excluded before classification.

The default cohort is selected automatically from the known participants. Trials with post-ASR noise fraction greater than 10% are removed, and the eight subjects with the lowest noise-quality ranking are selected. The ranking uses rejection fraction, median noise, maximum noise, and subject ID only; it never uses classification results. Subjects exceeding the 10% subject-level rejection fraction are flagged in the audit. Balanced accuracy is the primary metric because ordinary accuracy can be high when a model predicts consonant for most trials.

## Preprocessing

The default raw-style branch applies a 50 Hz notch, zero-phase 1–50 Hz band-pass, MEEGkit ASR calibrated from unlabeled clearing/rest intervals, common-average reference, robust per-trial channel normalization, and resampling to 256 Hz with 1,280 samples. The code uses the official `thinking_inds` intervals rather than guessing an epoch duration. It excludes non-EEG auxiliary channels and retains the validated 62-channel EEG order.

The preprocessing audit records the candidate and selected subjects, selected channels, ASR backend and calibration, pre- and post-ASR QC statistics, rejected trials, subject flags, sampling information, and all subject counts. Trial QC rejection is enabled by default at the fixed 10% threshold. A stricter sensitivity analysis that automatically excludes subjects above the 10% subject-level rejection fraction can be enabled with `KARAONE_ENFORCE_SUBJECT_REJECTION=1`.

## Models

The script evaluates Hydra, MiniRocket-Hydra fusion, MultiRocket, MultiRocket-Hydra, Catch22, Summary, Catch22+Summary, and an adaptive Catch22+Summary+MultiRocket-Hydra fusion. Standardization, variance filtering, Catch22 fitting, feature selection, classifier fitting, score calibration, adaptive ensemble weights, and adaptive decision thresholds are all fitted only from the outer training subjects. The Summary path combines deterministic temporal statistics with log and relative theta, alpha, beta, and low-gamma band-power features. The default practical `class_weight_fast` mode fits each candidate classifier with inverse-frequency weights multiplied by candidate weights for class 0 (vowels), selects the vowel-focused class weight and decision threshold from complete two-fold inner out-of-fold scores, and refits the selected class weight on all outer-training subjects. The inner selector requires minimum vowel recall, minimum consonant recall, and a balanced-accuracy floor; it gives vowel recall priority only among points close to the best feasible balanced accuracy, preventing a nearly all-vowel operating point from being rewarded. When multiple training subjects are available, these inner folds are subject-disjoint. The optional `class_weight_full` mode performs the larger class-weight search, while `threshold_only_fast` is retained only as an explicitly disabled-by-default sensitivity analysis requiring both `KARAONE_ALLOW_THRESHOLD_ONLY=1` and `KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT=0`. In fast mode, threshold candidates default to 0.35–0.65 to reduce degenerate all-class predictions under cross-subject score shifts. The MiniRocket-Hydra branch is transparently implemented as a 50:50 probability fusion because the tested Aeon environment does not provide a portable direct `MiniRocketHydraClassifier` export. The final fusion branch reuses the individual models' inner score streams when available rather than refitting them.

## Evaluation

The default protocol is strict LOSO over the automatically selected eight-subject cohort. Each outer test subject is unseen during all supervised fitting stages. The earlier five-subject raw-CNT validation is a separate study and is not silently substituted for this cohort. The script reports accuracy, balanced accuracy, macro-F1, vowel precision and recall, consonant precision and recall, ROC-AUC, average precision, per-fold confusion matrices, pooled confusion matrices, model failures, runtime settings, and a complete cohort audit.

Within-subject five-fold evaluation is available only as an explicitly labeled supplemental mode. It must not be presented as evidence of cross-subject generalization. No subject, model, threshold, or preprocessing option is selected from outer test performance.

## Interpretation

The earlier validated KaraOne experiments showed that strict cross-subject balanced accuracy remains near chance even when ordinary accuracy appears higher. This is consistent with a consonant-majority bias and poor vowel recall. The purpose of the submitted script is therefore not to promise a target accuracy; it is to provide a reproducible comparison that makes class imbalance, subject shift, preprocessing effects, and model failures visible.

The appropriate headline result is the strict-LOSO balanced accuracy and its per-subject variability, not the best individual fold or ordinary accuracy alone. The runtime setting must accompany any reported number because the fast and exhaustive adaptive searches have different capacity and compute budgets. Any future 73–80% result should be treated as a separate claim requiring independent verification under the same subject-level split and complete class-wise reporting.


## Adaptive-weight stability refinement

The current selector includes a training-only unit-weight safeguard. A non-unit vowel weight is retained only if its inner out-of-fold results improve vowel recall by a predeclared minimum while preserving balanced accuracy and consonant recall within fixed tolerances. In the fast configuration these defaults are a minimum vowel-recall gain of 0.05, a maximum balanced-accuracy drop of 0.01, and a maximum consonant-recall drop of 0.05. If the heavier vowel weight fails these criteria, the final model uses the unit inverse-frequency weight for that outer fold. This prevents a high-vowel-recall but low-balanced-accuracy operating point from being promoted merely because it helps one fold. The selected weight, guard decision, and selected-minus-unit inner metrics are recorded in the audit. No outer-test label is used.


## Adaptive best-single-model selection

In addition to the fixed requested classifiers, the script can report an `adaptive_best_single` branch. For each outer LOSO fold, it chooses among the already-fitted requested classifiers using only inner subject-disjoint validation metrics. The selection first requires the balanced-accuracy and class-recall safeguards, then permits only models within a predeclared balanced-accuracy tolerance of the best feasible inner model before prioritizing vowel recall. The selected model's outer-test scores are reused without refitting or inspecting outer-test labels. This branch is reported separately from the base classifier suite and is not a justification for selecting a model from its best outer fold.


## Per-fold result tables and decimal-weight sensitivity

The script now writes `final_fold_results_table.csv`, `final_fold_results_table.md`, and `final_fold_results_table.html`. These contain every successful model/fold row, not only the strongest fold, with accuracy, balanced accuracy, macro-F1, class-wise precision and recall, ROC-AUC, average precision, decision threshold, selected vowel weight, class weights, adaptive source model, and the confusion matrix. `final_fold_coverage.csv` explicitly records every expected model/fold combination as `ok`, `failed`, or `missing`, while `final_results.json` retains the complete nested diagnostics.

The vowel class-weight candidates accept positive decimal values and values above 2.0. The default fast run remains `[1.0, 2.0]` for practical runtime. A predeclared sensitivity grid can be requested with `KARAONE_FINE_ADAPTIVE_VOWEL_GRID=1` or an explicit `KARAONE_ADAPTIVE_VOWEL_WEIGHTS` list. These candidates are evaluated with subject-grouped inner validation and the unit-weight stability guard; they are not tuned using outer-test labels. Because each extra candidate adds inner fits, a fine grid should be reported as a separate, slower sensitivity experiment.

Hydra, MultiRocket, and MR-Hydra are not epoch-trained neural networks. An epoch count is therefore not a valid MR-Hydra accuracy parameter. Any neural model with training epochs would be a separate classifier and should not replace the requested suite.

## Further vowel-recall refinement and correct weight-selection rule

The current refinement adds two label-independent preprocessing operations motivated by the official KaraOne description: continuous linear EOG regression when EOG channels are available, followed by a small fixed-montage Laplacian after common-average referencing. The Laplacian uses the nearest valid standard-10–20 neighbors and a declared mix coefficient. Both operations are applied without prompt labels and without access to the held-out LOSO subject. The pipeline records whether EOG channels were found and whether the spatial filter was applied, allowing explicit ablation runs with `KARAONE_EOG_REGRESSION=0` and `KARAONE_SPATIAL_FILTER=none`.

The adaptive vowel class weight is not randomly selected and is not selected from the final outer-fold accuracy. For each LOSO fold, candidate weights are evaluated in subject-disjoint inner validation folds. The default selection criterion averages balanced accuracy, vowel recall, consonant recall, and macro-F1 across those inner folds, which prevents a participant with more trials from dominating the choice. Vowel recall is prioritized only among candidates within the predeclared balanced-accuracy tolerance, and a unit-weight stability guard rejects non-unit weights that do not provide a meaningful inner vowel-recall gain or that damage balanced accuracy/consonant recall beyond the allowed limits. The selected candidate is then refit on all outer-training subjects, and the untouched outer subject is evaluated once. The outer result is used only for reporting and experiment comparison.

A fine decimal grid, including values above 2, is permitted as a declared sensitivity analysis, but it increases the number of inner fits. It must not be selected after inspecting outer-fold results. More MR-Hydra epochs are not applicable because MR-Hydra is a convolutional time-series classifier rather than an epoch-trained neural network.

The official KaraOne documentation states that the original processing included 1--50 Hz filtering, channel-mean subtraction, and a small Laplacian using adjacent-channel neighborhoods, and describes subject-independent leave-one-participant-out evaluation. See the saved source note at `reference/karaone_official_preprocessing_notes.md`.


## Additional vowel-recall refinement

The refined pipeline includes a prompt-balanced resampling step applied only within training data. Because the vowel class contains two prompts (`iy` and `uw`), each inner training split first equalizes those prompts when both are available. The total vowel training prior is then increased to a declared target ratio, capped by a maximum multiplier. This addresses minority-class and within-vowel-prompt imbalance without adding prompt identity as a classifier feature. The held-out subject is not inspected during resampling.

The default parameters are a 0.40 target vowel proportion and a 2.0 maximum multiplier. The method is evaluated against a matched no-resampling ablation. The main model-selection criterion remains the mean of subject-disjoint inner-fold metrics, with balanced accuracy protected by predeclared floors/tolerances and vowel recall used as the priority only within those constraints. Final outer-fold metrics remain untouched by tuning and must be reported for accuracy, balanced accuracy, macro-F1, vowel recall, and consonant recall.
