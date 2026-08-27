from pathlib import Path
import json
import tempfile
from pathlib import Path
import pandas as pd

import karaone_final_kaggle as pipe


def row(model, fold, subject, **extra):
    base = {
        "model": model,
        "fold": fold,
        "test_subject": subject,
        "n_test": 10,
        "decision_threshold": 0.50,
        "accuracy": 0.70,
        "balanced_accuracy": 0.65,
        "macro_f1": 0.62,
        "vowel_precision": 0.60,
        "vowel_recall": 0.50,
        "consonant_precision": 0.75,
        "consonant_recall": 0.80,
        "roc_auc": 0.68,
        "average_precision": 0.72,
        "confusion_matrix": [[2, 2], [1, 5]],
        "fit_seconds": 1.2,
        "adaptive_vowel_weight": 1.2,
        "adaptive_class_weight": {"vowel_0": 1.8, "consonant_1": 0.7},
        "adaptive_weight_selection_rule": "test_rule",
    }
    base.update(extra)
    return base


def main():
    rows = [
        row(
            "mr_hydra", 2, "MM11", adaptive_vowel_weight=1.3,
            inner_vowel_weight_candidates=[
                {"vowel_weight": 1.0, "usable": True, "feasible": True, "threshold": 0.50, "accuracy": 0.60, "balanced_accuracy": 0.58, "macro_f1": 0.55, "vowel_recall": 0.42, "consonant_recall": 0.74},
                {"vowel_weight": 1.3, "usable": True, "feasible": True, "threshold": 0.45, "accuracy": 0.68, "balanced_accuracy": 0.65, "macro_f1": 0.62, "vowel_recall": 0.50, "consonant_recall": 0.80},
            ],
            adaptive_metric_aggregation="subject_macro",
        ),
        row(
            "catch22_summary", 1, "MM09",
            inner_vowel_weight_candidates=[
                {"vowel_weight": 1.0, "usable": True, "feasible": False, "threshold": 0.50, "accuracy": 0.55, "balanced_accuracy": 0.50, "macro_f1": 0.48, "vowel_recall": 0.30, "consonant_recall": 0.70},
                {"vowel_weight": 1.2, "usable": True, "feasible": True, "threshold": 0.50, "accuracy": 0.64, "balanced_accuracy": 0.61, "macro_f1": 0.58, "vowel_recall": 0.46, "consonant_recall": 0.76},
            ],
            adaptive_metric_aggregation="subject_macro",
        ),
        row(
            "adaptive_best_single", 1, "MM09",
            adaptive_single_source_model="catch22_summary",
            adaptive_single_source_vowel_weight=1.4,
            adaptive_single_source_class_weight={"vowel_0": 2.1, "consonant_1": 0.7},
            adaptive_single_selection_rule="inner_rule",
            adaptive_vowel_weight=None,
        ),
    ]
    summaries = {
        "mr_hydra": {"n_folds": 1, "accuracy_mean": 0.70, "accuracy_std": 0.0, "balanced_accuracy_mean": 0.65, "balanced_accuracy_std": 0.0, "macro_f1_mean": 0.62, "vowel_precision_mean": 0.60, "vowel_recall_mean": 0.50, "consonant_precision_mean": 0.75, "consonant_recall_mean": 0.80, "roc_auc_mean": 0.68, "average_precision_mean": 0.72, "pooled_confusion_matrix": [[2, 2], [1, 5]]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        paths = pipe.write_result_tables(rows, summaries, [{"model": "hydra", "fold": 1, "test_subject": "MM09", "error": "synthetic failure"}], out, [(1, "MM09"), (2, "MM11")])
        assert all(Path(p).exists() for p in paths.values()), paths
        frame = pd.read_csv(out / "final_fold_results_table.csv")
        assert list(frame["model"]) == ["adaptive_best_single", "catch22_summary", "mr_hydra"]
        assert frame.loc[0, "source_model"] == "catch22_summary"
        assert frame.loc[0, "confusion_matrix"] == "[[2,2],[1,5]]"
        weight_frame = pd.read_csv(out / "final_fold_vowel_weight_candidates.csv")
        direct = weight_frame[weight_frame["candidate_stage"] == "inner_validation_candidate"]
        assert len(direct) == 4, len(direct)
        assert set(direct["candidate_weight"].round(1)) == {1.0, 1.2, 1.3}
        assert direct["inner_accuracy"].notna().all()
        assert direct["outer_accuracy"].notna().all()
        assert direct["candidate_selected"].sum() == 2
        coverage = pd.read_csv(out / "final_fold_coverage.csv")
        assert set(coverage["status"]) >= {"ok", "failed", "missing"}
        md = (out / "final_fold_results_table.md").read_text()
        assert "Balanced accuracy" in md and "Confusion matrix" in md
        print(json.dumps({"rows": len(frame), "coverage_rows": len(coverage), "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
