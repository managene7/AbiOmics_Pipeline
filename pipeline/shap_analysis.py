from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from utils.models import mlp_model
from utils.train import EarlyStopping
from utils.utils import min_max_norm_TPM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shape normalisation
# ---------------------------------------------------------------------------

def normalise_shap(raw, n_features: int) -> list[np.ndarray]:
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        # Case (b): unpack last axis
        return [raw[:, :, c] for c in range(raw.shape[2])]

    # Case (a): list — check each array's orientation
    out: list[np.ndarray] = []
    for arr in raw:
        arr = np.array(arr)
        if arr.ndim == 2 and arr.shape[0] == n_features and arr.shape[1] != n_features:
            arr = arr.T   # (n_features, n_test) → (n_test, n_features)
        out.append(arr)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(
    cfg: dict,
    chk_path: str,
    input_dim: int,
    device: str,
) -> tuple[torch.nn.Module, dict]:
    es = EarlyStopping(
        patience         = 0,
        save_path        = chk_path,
        norm_min_max_dic = {},          # placeholder; load() extracts from file
        cuda_vis_dev     = cfg["training"]["cuda_device"],
    )
    model = mlp_model(dim1=input_dim, label_dim=cfg["n_classes"])
    model, loaded_min_max_dic = es.load(model)
    model = model.to(device)
    model.eval()
    return model, loaded_min_max_dic


def _make_background(
    cfg: dict,
    x_train_tensor: torch.Tensor,
    device: str,
) -> torch.Tensor:
    n = cfg["shap"]["n_background"]
    torch.manual_seed(cfg["random_seed"])
    idx = torch.randperm(x_train_tensor.shape[0])[:n]
    return x_train_tensor[idx].to(device)


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_single(
    cfg: dict,
    chk_path: str,
    background: torch.Tensor,
    x_test: torch.Tensor,
    input_dim: int,
) -> list[np.ndarray]:
    device = cfg["shap"]["device"]
    model, _ = _load_model(cfg, chk_path, input_dim, device)

    explainer = shap.GradientExplainer(model, background.to(device))
    raw        = explainer.shap_values(x_test.to(device))
    values     = normalise_shap(raw, input_dim)

    logger.info("  Single-checkpoint SHAP: %d classes  shape=%s",
                len(values), values[0].shape)
    return values


def compute_shap_cv_average(
    cfg: dict,
    chk_list: list[str],
    background: torch.Tensor,
    x_test: torch.Tensor,
    input_dim: int,
) -> list[np.ndarray]:
    device     = cfg["shap"]["device"]
    model_path = cfg["evaluation"]["model_save_path"]
    n_classes  = cfg["n_classes"]
    n_test     = x_test.shape[0]

    shap_sum = [np.zeros((n_test, input_dim)) for _ in range(n_classes)]

    for chk_point in chk_list:
        chk_path  = os.path.join(model_path, chk_point)
        model, _  = _load_model(cfg, chk_path, input_dim, device)

        explainer = shap.GradientExplainer(model, background.to(device))
        raw        = explainer.shap_values(x_test.to(device))
        fold_vals  = normalise_shap(raw, input_dim)

        for c in range(n_classes):
            shap_sum[c] += fold_vals[c]
        logger.info("  CV-SHAP done: %s", chk_point)

    shap_avg = [shap_sum[c] / len(chk_list) for c in range(n_classes)]
    logger.info("CV-averaged SHAP ready. Shape: %s", shap_avg[0].shape)
    return shap_avg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(
    cfg: dict,
    shap_values: list[np.ndarray],
    x_test_np: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    filename_prefix: str = "SHAP_summary",
) -> None:
    export_path = cfg["output"]["export_path"]
    fmt         = cfg["output"]["plot_format"]
    dpi         = cfg["output"]["plot_dpi"]
    top_n       = cfg["shap"]["top_n_genes"]
    os.makedirs(export_path, exist_ok=True)

    for cls_idx, cls_name in enumerate(class_names):
        shap.summary_plot(
            shap_values[cls_idx],
            features      = x_test_np,
            feature_names = feature_names,
            max_display   = top_n,
            show          = False,
            plot_type     = "dot",
        )
        plt.gcf().subplots_adjust(top=0.93, right=0.88)
        plt.title(f"SHAP Summary — {cls_name}", fontsize=13)
        out = os.path.join(export_path, f"{filename_prefix}_{cls_name}.{fmt}")
        plt.savefig(out, bbox_inches="tight", dpi=dpi)
        plt.show()
        plt.close()
        logger.info("Saved: %s", out)


def plot_shap_bar(
    cfg: dict,
    shap_values: list[np.ndarray],
    feature_names: list[str],
    class_names: list[str],
    filename: str = "SHAP_top_genes_all_classes",
) -> None:
    export_path = cfg["output"]["export_path"]
    fmt         = cfg["output"]["plot_format"]
    dpi         = cfg["output"]["plot_dpi"]
    top_n       = cfg["shap"]["top_n_genes"]
    n_cls       = len(class_names)
    os.makedirs(export_path, exist_ok=True)

    fig, axes = plt.subplots(1, n_cls, figsize=(6 * n_cls, 6), sharey=False)
    if n_cls == 1:
        axes = [axes]

    for cls_idx, (cls_name, ax) in enumerate(zip(class_names, axes)):
        mean_abs  = np.abs(shap_values[cls_idx]).mean(axis=0)
        top_idx   = np.argsort(mean_abs)[::-1][:top_n]
        top_genes = [feature_names[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]

        ax.barh(top_genes[::-1], top_vals[::-1], color="steelblue", edgecolor="white")
        ax.set_title(cls_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean |SHAP|", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle(
        f"Top {top_n} Important Genes per Stress Class (Mean |SHAP|)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(export_path, f"{filename}.{fmt}")
    plt.savefig(out, bbox_inches="tight", dpi=dpi)
    plt.show()
    plt.close()
    logger.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_shap_csv(
    cfg: dict,
    shap_values: list[np.ndarray],
    feature_names: list[str],
    class_names: list[str],
    sample_ids,
    filename_prefix: str = "SHAP_values",
) -> None:
    export_path = cfg["output"]["export_path"]
    os.makedirs(export_path, exist_ok=True)

    for cls_idx, cls_name in enumerate(class_names):
        df  = pd.DataFrame(shap_values[cls_idx],
                           index=sample_ids, columns=feature_names)
        out = os.path.join(export_path, f"{filename_prefix}_{cls_name}.csv")
        df.to_csv(out)
        logger.info("Saved: %s  shape=%s", out, df.shape)


def export_gene_ranking(
    cfg: dict,
    shap_values: list[np.ndarray],
    feature_names: list[str],
    class_names: list[str],
    filename: str = "SHAP_gene_ranking_all_classes",
) -> pd.DataFrame:
    export_path = cfg["output"]["export_path"]
    os.makedirs(export_path, exist_ok=True)

    rows = [
        {"gene": gene, "stress_class": cls_name,
         "mean_abs_shap": float(np.abs(shap_values[c]).mean(axis=0)[g])}
        for c, cls_name in enumerate(class_names)
        for g, gene in enumerate(feature_names)
    ]
    df = (pd.DataFrame(rows)
            .sort_values(["stress_class", "mean_abs_shap"],
                         ascending=[True, False])
            .reset_index(drop=True))
    out = os.path.join(export_path, f"{filename}.csv")
    df.to_csv(out, index=False)
    logger.info("Saved gene ranking: %s", out)
    return df


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_shap_analysis(
    cfg: dict,
    chk_list: list[str],
    tpm_train_norm: pd.DataFrame,
    testing_tpm: pd.DataFrame,
    min_max_dic: dict,
    marker_genes: list[str],
) -> None:
    device      = cfg["shap"]["device"]
    input_dim   = len(marker_genes)
    class_names = list(cfg["label_dic"].values())   # ["Control","Salt","Cold",…]
    model_path  = cfg["evaluation"]["model_save_path"]

    # ── Prepare tensors ──────────────────────────────────────────────────────
    x_train_tensor = torch.FloatTensor(tpm_train_norm.values.T)
    background     = _make_background(cfg, x_train_tensor, device)

    last_chk_path = os.path.join(model_path, chk_list[-1])
    es = EarlyStopping(patience=0, save_path=last_chk_path,
                       norm_min_max_dic={}, cuda_vis_dev=cfg["training"]["cuda_device"])
    _m = mlp_model(dim1=input_dim, label_dim=cfg["n_classes"])
    _, last_min_max = es.load(_m)

    test_norm, _  = min_max_norm_TPM(testing_tpm, last_min_max)
    x_test_tensor = torch.FloatTensor(test_norm.values.T).to(device)
    x_test_np     = x_test_tensor.cpu().numpy()
    sample_ids    = testing_tpm.columns.values

    logger.info("SHAP tensors: background=%d  test=%d  features=%d",
                background.shape[0], x_test_tensor.shape[0], input_dim)

    # ── Single-checkpoint SHAP ───────────────────────────────────────────────
    logger.info("Single-checkpoint SHAP (%s) …", chk_list[-1])
    shap_values = compute_shap_single(
        cfg, last_chk_path, background, x_test_tensor, input_dim
    )
    plot_shap_beeswarm(cfg, shap_values, x_test_np, marker_genes, class_names)
    plot_shap_bar(cfg, shap_values, marker_genes, class_names)
    export_shap_csv(cfg, shap_values, marker_genes, class_names, sample_ids)

    # ── CV-averaged SHAP ─────────────────────────────────────────────────────
    if cfg["shap"].get("run_cv_average", True):
        logger.info("CV-averaged SHAP across %d fold(s) …", len(chk_list))
        shap_avg = compute_shap_cv_average(
            cfg, chk_list, background, x_test_tensor, input_dim
        )
        plot_shap_beeswarm(cfg, shap_avg, x_test_np, marker_genes, class_names,
                           filename_prefix="SHAP_avg_summary")
        plot_shap_bar(cfg, shap_avg, marker_genes, class_names,
                      filename="SHAP_avg_top_genes_all_classes")
        export_shap_csv(cfg, shap_avg, marker_genes, class_names, sample_ids,
                        filename_prefix="SHAP_avg_values")
        export_gene_ranking(cfg, shap_avg, marker_genes, class_names)
