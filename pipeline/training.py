from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import re
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

from utils.data import dataloader
from utils.models import mlp_model
from utils.plots import plot_confusion
from utils.train import training, EarlyStopping, Inference, Accuracy
from utils.utils import min_max_norm_TPM

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_checkpoint_list(cfg: dict) -> list[str]:
    model_path = cfg["evaluation"]["model_save_path"]
    chk_list   = sorted(f for f in os.listdir(model_path) if f.endswith(".pth"))
    if not chk_list:
        raise FileNotFoundError(
            f"No .pth checkpoints found in '{model_path}'.\n"
            "Run training first (python run_pipeline.py) before evaluating."
        )
    logger.info("Found %d checkpoint(s) in %s", len(chk_list), model_path)
    return chk_list


def build_label_vector(
    cfg: dict,
    all_con_ids: list[str],
    samp_train: dict[str, list],
) -> list[int]:
    stress_types = cfg["data"]["stress_types"]
    label: list[int] = [0] * len(all_con_ids)
    for i, stress in enumerate(stress_types):
        label.extend([i + 1] * len(samp_train[stress]))

    dist = {0: len(all_con_ids)}
    for i, s in enumerate(stress_types):
        dist[i + 1] = len(samp_train[s])
    logger.info("Training label distribution: %s", dist)
    return label


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    cfg: dict,
    tpm_norm: pd.DataFrame,
    label: list[int],
    min_max_dic: dict,
) -> dict:
    tc = cfg["training"]
    os.makedirs(tc["model_save_path"], exist_ok=True)

    # dataloader() shuffles with its own fixed RNG (random_state=1 default)
    x_data, y_data = dataloader(tpm_norm.values.T, label)
    logger.info("Training tensors  x=%s  y=%s", x_data.shape, y_data.shape)

    skfold = StratifiedKFold(
        n_splits=tc["n_folds"],
        shuffle=True,
        random_state=cfg["random_seed"],
    )

    parameter_dic, cv_reports = training(
        x_data,
        y_data,
        min_max_dic,
        skfold,
        N_EPOCH             = tc["n_epochs"],
        early_stop_patience = tc["early_stop_patience"],
        min_epoch           = tc["min_epoch"],
        learning_rate       = tc["learning_rate"],
        batch_num           = tc["batch_size"],
        model_save_path     = tc["model_save_path"],
        cuda_vis_dev        = tc["cuda_device"],
        label_dim           = cfg["n_classes"],
        label_dic           = cfg["label_dic"],
    )

    logger.info("Training complete. Checkpoints in %s/", tc["model_save_path"])
    return parameter_dic, cv_reports


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    cfg: dict,
    testing_tpm: pd.DataFrame,
    testing_label: list[int],
    min_max_dic: dict,
    marker_genes: list[str],
) -> np.ndarray:
    ec        = cfg["evaluation"]
    n_classes = cfg["n_classes"]
    label_dic = cfg["label_dic"]
    input_dim = len(marker_genes)
    chk_list  = get_checkpoint_list(cfg)

    all_confusion    = np.zeros((n_classes, n_classes))
    ind_test_reports = []   # list of report dicts, one per checkpoint

    for chk_point in chk_list:
        logger.info("Evaluating: %s", chk_point)
        chk_file = os.path.join(ec["model_save_path"], chk_point)

        early_stopper = EarlyStopping(
            patience        = 0,
            save_path       = chk_file,
            norm_min_max_dic= min_max_dic,
            cuda_vis_dev    = cfg["training"]["cuda_device"],
        )

        model = mlp_model(dim1=input_dim, label_dim=n_classes)
        model, loaded_min_max_dic = early_stopper.load(model)

        # Use the normalisation params baked into this specific checkpoint
        test_norm, _ = min_max_norm_TPM(testing_tpm, loaded_min_max_dic)
        x_data = torch.FloatTensor(test_norm.values.T)
        y_data = torch.LongTensor(testing_label)

        test_output, test_loss = Inference(model, x_data, y_data)
        test_accuracy          = Accuracy(test_output, y_data, class_names=list(label_dic.values()))

        all_confusion += test_accuracy[0]
        ind_test_reports.append(test_accuracy[2])
        print(f"\n{'='*50}\nCheckpoint: {chk_point}")
        print("Confusion Matrix:")
        plot_confusion(test_accuracy[0], label_dic)
        print("\nClassification Report:")
        print(test_accuracy[1])

    logger.info("Evaluation complete over %d checkpoint(s).", len(chk_list))
    return all_confusion, ind_test_reports


def plot_averaged_confusion(
    cfg: dict,
    all_confusion: np.ndarray,
    n_folds: int,
) -> None:
    export_path = cfg["output"]["export_path"]
    fmt         = cfg["output"]["plot_format"]
    dpi         = cfg["output"]["plot_dpi"]
    label_dic   = cfg["label_dic"]
    os.makedirs(export_path, exist_ok=True)

    avg_cm = all_confusion / n_folds
    disp   = ConfusionMatrixDisplay(
        confusion_matrix=avg_cm,
        display_labels=list(label_dic.values()),
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Averaged Confusion Matrix (all CV folds)")
    out = os.path.join(export_path, f"confusion_matrix_avg.{fmt}")
    plt.savefig(out, bbox_inches="tight", dpi=dpi)
    plt.show()
    plt.close()
    logger.info("Averaged confusion matrix saved: %s", out)


# ---------------------------------------------------------------------------
# Performance export
# ---------------------------------------------------------------------------

def _reports_to_df(
    reports: list[dict],
    class_names: list[str],
    fold_labels: list,
) -> pd.DataFrame:
    rows = []
    keys = class_names + ["macro avg", "weighted avg"]
    metrics = ["precision", "recall", "f1-score", "support"]

    for label, rep in zip(fold_labels, reports):
        row = {"fold": str(label)}
        for cls in keys:
            if cls in rep:
                for m in metrics:
                    row[f"{cls}_{m}"] = rep[cls].get(m, float("nan"))
        row["accuracy"] = rep.get("accuracy", float("nan"))
        rows.append(row)

    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != "fold"]   # exclude fold — it's a label, not a metric
    support_cols = [c for c in numeric_cols if c.endswith("_support")]
    stat_cols    = [c for c in numeric_cols if c not in support_cols]

    mean_row = {"fold": "Mean"}
    std_row  = {"fold": "Std."}

    for c in stat_cols:
        mean_row[c] = round(df[c].mean(), 2)
        std_row[c]  = round(df[c].std(),  2)
    # Support columns are intentionally omitted — left as NaN in summary rows

    # Round all per-fold numeric values too
    df_rounded = df.copy()
    for c in stat_cols:
        df_rounded[c] = df_rounded[c].round(2)

    return pd.concat(
        [df_rounded, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )


def export_cv_performance(
    cfg: dict,
    cv_reports: list,
) -> pd.DataFrame:
    export_path = cfg["output"]["export_path"]
    class_names = list(cfg["label_dic"].values())
    os.makedirs(export_path, exist_ok=True)

    folds       = [r[0] for r in cv_reports]
    tst_reports = [r[2] for r in cv_reports]

    tst_df = _summarise(_reports_to_df(tst_reports, class_names, folds))

    tst_out = os.path.join(export_path, "performance_cv_internal_test.csv")
    tst_df.to_csv(tst_out, index=False)

    logger.info("CV internal test performance saved: %s", tst_out)
    return tst_df


def export_test_performance(
    cfg: dict,
    ind_test_reports: list[dict],
) -> pd.DataFrame:
    export_path = cfg["output"]["export_path"]
    class_names = list(cfg["label_dic"].values())
    n_folds     = len(ind_test_reports)
    os.makedirs(export_path, exist_ok=True)

    df  = _summarise(_reports_to_df(ind_test_reports, class_names,
                                    list(range(1, n_folds + 1))))
    out = os.path.join(export_path, "performance_independent_test.csv")
    df.to_csv(out, index=False)
    logger.info("Independent test performance saved: %s", out)
    return df
