"""Phase 10 Execution Script: Error Analysis, SHAP, Locked Test Evaluation, and Notebook Generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure root directory on path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import (
    compute_classification_metrics,
    compute_official_style_score,
    compute_regression_metrics,
)
from src.load import load_config, load_parquet


def main():
    cfg = load_config()
    preds_df = load_parquet("data/processed/predictions.parquet")
    features_train = load_parquet("data/processed/features_train.parquet")
    targets_train = load_parquet("data/processed/targets_train.parquet")

    print(f"Loaded predictions: {preds_df.shape}")
    print(f"Splits in predictions: {preds_df['split'].value_counts().to_dict()}")

    # 1. EVALUATE LOCKED TEST SET
    test_preds = preds_df[preds_df["split"] == "test"].copy()
    print(f"\n--- TEST SET OVERVIEW ---")
    print(f"Total test events: {len(test_preds)}")
    print(f"Actual high risk count (y >= -6.0): {test_preds['actual_high_risk'].sum()} ({test_preds['actual_high_risk'].mean():.2%})")

    y_test_true = test_preds["actual_high_risk"].values
    y_test_risk_true = test_preds["actual_final_log_risk"].values
    y_test_prob_model = test_preds["high_risk_prob"].values
    y_test_pred_risk = test_preds["pred_final_log_risk"].values

    # Test Baseline A
    test_baseline_binary = (test_preds["baseline_recommendation"] == "MANOEUVRE").astype(int).values
    test_model_binary = (test_preds["recommendation"] == "MANOEUVRE").astype(int).values

    baseline_metrics = compute_classification_metrics(
        y_test_true,
        test_baseline_binary,
        y_score=test_preds["latest_log_risk"].values,
    )
    baseline_reg = compute_regression_metrics(y_test_risk_true, test_preds["latest_log_risk"].values, high_risk_threshold=-6.0)
    baseline_score = compute_official_style_score(baseline_reg["mse_high_risk"], baseline_metrics["f2"])

    model_metrics = compute_classification_metrics(
        y_test_true,
        test_model_binary,
        y_score=y_test_prob_model,
    )
    model_reg = compute_regression_metrics(y_test_risk_true, y_test_pred_risk, high_risk_threshold=-6.0)
    model_score = compute_official_style_score(model_reg["mse_high_risk"], model_metrics["f2"])

    print("\n=== LOCKED TEST BENCHMARK RESULTS ===")
    print(f"Baseline A  -> FN: {baseline_metrics['fn']}, FP: {baseline_metrics['fp']}, Recall: {baseline_metrics['recall']:.2%}, Precision: {baseline_metrics['precision']:.2%}, F2: {baseline_metrics['f2']:.4f}, PR-AUC: {baseline_metrics['pr_auc']:.4f}, Score: {baseline_score:.4f}")
    print(f"Model (Ours) -> FN: {model_metrics['fn']}, FP: {model_metrics['fp']}, Recall: {model_metrics['recall']:.2%}, Precision: {model_metrics['precision']:.2%}, F2: {model_metrics['f2']:.4f}, PR-AUC: {model_metrics['pr_auc']:.4f}, Score: {model_score:.4f}")

    # 2. ERROR ANALYSIS (FALSE NEGATIVES & FALSE POSITIVES)
    test_preds["is_fn_model"] = (test_preds["actual_high_risk"] == 1) & (test_preds["recommendation"] == "NO_MANOEUVRE")
    test_preds["is_fp_model"] = (test_preds["actual_high_risk"] == 0) & (test_preds["recommendation"] == "MANOEUVRE")
    test_preds["is_fn_base"] = (test_preds["actual_high_risk"] == 1) & (test_preds["baseline_recommendation"] == "NO_MANOEUVRE")
    test_preds["is_fp_base"] = (test_preds["actual_high_risk"] == 0) & (test_preds["baseline_recommendation"] == "MANOEUVRE")

    fn_events = test_preds[test_preds["is_fn_model"]].sort_values("actual_final_log_risk", ascending=False)
    fp_events = test_preds[test_preds["is_fp_model"]].sort_values("high_risk_prob", ascending=False)

    print(f"\nModel False Negatives count: {len(fn_events)}")
    if len(fn_events) > 0:
        print("Sample FN Events (Dangerous conjunctions missed by model):")
        print(fn_events[["event_id", "latest_log_risk", "actual_final_log_risk", "high_risk_prob", "recommendation"]].head(5))

    print(f"\nModel False Positives count: {len(fp_events)}")
    if len(fp_events) > 0:
        print("Sample FP Events (False alarms triggered by model):")
        print(fp_events[["event_id", "latest_log_risk", "actual_final_log_risk", "high_risk_prob", "recommendation"]].head(5))

    # Rescued events (Where baseline failed/missed, but model caught it!)
    rescued_events = test_preds[test_preds["is_fn_base"] & ~test_preds["is_fn_model"]]
    print(f"\nRescued Events (Baseline missed, Model caught): {len(rescued_events)}")
    if len(rescued_events) > 0:
        print(rescued_events[["event_id", "latest_log_risk", "actual_final_log_risk", "high_risk_prob", "recommendation"]].head(5))

    # Saved Burns (Where baseline alarmed unnecessarily, but model cleared safely)
    saved_burns = test_preds[test_preds["is_fp_base"] & ~test_preds["is_fp_model"]]
    print(f"\nAvoided Unnecessary Burns (Baseline FP, Model Safe): {len(saved_burns)}")

    # 3. GLOBAL SHAP IMPORTANCE CHART
    print("\nComputing Feature Importance & SHAP values...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    y_train = targets_train["y_high_risk"].values
    clf.fit(features_train, y_train)

    # Use TreeExplainer on a sample of 300 instances for clean, fast SHAP computation
    sample_idx = np.random.RandomState(42).choice(len(features_train), size=300, replace=False)
    X_sample = features_train.iloc[sample_idx]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # shap_values can be a list [class 0, class 1] or an array
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Plot and save
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    shap_png_path = figures_dir / "shap_importance.png"

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, plot_type="bar", max_display=15, show=False)
    plt.title("NearMiss — Top 15 Feature Importances (Tree SHAP Mean |SHAP|)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(shap_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP plot to {shap_png_path}")

    # Also build a detailed bee-swarm plot
    shap_beeswarm_path = figures_dir / "shap_summary_beeswarm.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, max_display=15, show=False)
    plt.title("NearMiss — SHAP Value Distribution & Impact on Collision Risk", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(shap_beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP beeswarm plot to {shap_beeswarm_path}")

    # 4. POPULATE NOTEBOOK 02_error_analysis.ipynb
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 02. Error Analysis & Model Interpretability\n",
                    "\n",
                    "**Project:** SW5 Satellite Collision Risk Predictor (NearMiss)  \n",
                    "**Goal:** Inspect false negatives (missed collisions), false positives (avoidable burns), and evaluate global SHAP explainability.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from src.load import load_config, load_parquet\n",
                    "\n",
                    "cfg = load_config()\n",
                    "preds = load_parquet('data/processed/predictions.parquet')\n",
                    "hist = load_parquet('data/processed/cdm_history.parquet')\n",
                    "test_preds = preds[preds['split'] == 'test'].copy()\n",
                    "print(f'Loaded {len(test_preds)} locked test events.')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Locked Test Set Performance Comparison\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.evaluate import compute_classification_metrics, compute_regression_metrics, compute_official_score\n",
                    "\n",
                    "y_true = test_preds['actual_high_risk'].values\n",
                    "y_risk = test_preds['actual_final_log_risk'].values\n",
                    "y_prob = test_preds['high_risk_prob'].values\n",
                    "y_pred_risk = test_preds['pred_final_log_risk'].values\n",
                    "\n",
                    "from src.evaluate import compute_classification_metrics, compute_regression_metrics, compute_official_style_score\n",
                    "\n",
                    "base_binary = (test_preds['baseline_recommendation'] == 'MANOEUVRE').astype(int).values\n",
                    "model_binary = (test_preds['recommendation'] == 'MANOEUVRE').astype(int).values\n",
                    "\n",
                    "base_cls = compute_classification_metrics(y_true, base_binary, y_score=test_preds['latest_log_risk'].values)\n",
                    "base_reg = compute_regression_metrics(y_risk, test_preds['latest_log_risk'].values, high_risk_threshold=-6.0)\n",
                    "base_score = compute_official_style_score(base_reg['mse_high_risk'], base_cls['f2'])\n",
                    "\n",
                    "model_cls = compute_classification_metrics(y_true, model_binary, y_score=y_prob)\n",
                    "model_reg = compute_regression_metrics(y_risk, y_pred_risk, high_risk_threshold=-6.0)\n",
                    "model_score = compute_official_style_score(model_reg['mse_high_risk'], model_cls['f2'])\n",
                    "\n",
                    "res_df = pd.DataFrame([\n",
                    "    {'Model': 'Baseline A (Persistence)', 'Recall': f\"{base_cls['recall']:.2%}\", 'Precision': f\"{base_cls['precision']:.2%}\", 'F2': base_cls['f2'], 'PR-AUC': base_cls['pr_auc'], 'Missed (FN)': base_cls['fn'], 'False Alarms (FP)': base_cls['fp'], 'Kelvins Score': base_score},\n",
                    "    {'Model': 'NearMiss Champion (RF+Platt)', 'Recall': f\"{model_cls['recall']:.2%}\", 'Precision': f\"{model_cls['precision']:.2%}\", 'F2': model_cls['f2'], 'PR-AUC': model_cls['pr_auc'], 'Missed (FN)': model_cls['fn'], 'False Alarms (FP)': model_cls['fp'], 'Kelvins Score': model_score},\n",
                    "])\n",
                    "res_df\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. False Negatives (Missed Dangerous Events)\n",
                    "Investigating events where ground truth is high risk ($y \\ge -6.0$), but the model recommended `NO_MANOEUVRE`.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fn_events = test_preds[(test_preds['actual_high_risk'] == 1) & (test_preds['recommendation'] == 'NO_MANOEUVRE')]\n",
                    "print(f'Total False Negatives on Locked Test: {len(fn_events)}')\n",
                    "fn_events[['event_id', 'latest_log_risk', 'actual_final_log_risk', 'high_risk_prob', 'recommendation', 'confidence']].head(10)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Avoided Unnecessary Burns (Saved Maneuvers)\n",
                    "Events where Baseline A raised a false alarm, but NearMiss correctly recommended `NO_MANOEUVRE`.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "saved = test_preds[(test_preds['baseline_recommendation'] == 'MANOEUVRE') & (test_preds['recommendation'] == 'NO_MANOEUVRE') & (test_preds['actual_high_risk'] == 0)]\n",
                    "print(f'Total Avoided Unnecessary Burns: {len(saved)}')\n",
                    "saved[['event_id', 'latest_log_risk', 'actual_final_log_risk', 'high_risk_prob', 'confidence']].head(10)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Global SHAP Feature Importances\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "display(Image(filename='reports/figures/shap_importance.png'))\n",
                    "display(Image(filename='reports/figures/shap_summary_beeswarm.png'))\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("notebooks/02_error_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    print("Updated notebooks/02_error_analysis.ipynb")

    return {
        "test_events": len(test_preds),
        "test_positives": int(y_test_true.sum()),
        "baseline": {
            "fn": int(baseline_metrics["fn"]),
            "fp": int(baseline_metrics["fp"]),
            "recall": float(baseline_metrics["recall"]),
            "precision": float(baseline_metrics["precision"]),
            "f2": float(baseline_metrics["f2"]),
            "pr_auc": float(baseline_metrics["pr_auc"]),
            "high_risk_mse": float(baseline_reg["mse_high_risk"]),
            "kelvins_score": float(baseline_score),
        },
        "model": {
            "fn": int(model_metrics["fn"]),
            "fp": int(model_metrics["fp"]),
            "recall": float(model_metrics["recall"]),
            "precision": float(model_metrics["precision"]),
            "f2": float(model_metrics["f2"]),
            "pr_auc": float(model_metrics["pr_auc"]),
            "high_risk_mse": float(model_reg["mse_high_risk"]),
            "kelvins_score": float(model_score),
        },
        "avoided_burns": int(len(saved_burns)),
        "rescued_events": int(len(rescued_events)),
    }


if __name__ == "__main__":
    results = main()
    with open("data/processed/phase_10_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nPhase 10 script executed successfully.")
