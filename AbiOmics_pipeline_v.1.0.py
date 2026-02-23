"""
AbiOmics_pipeline_v.1.0.py
===============
Main entry point for the Stress Classification Framework.

Pipeline stages
---------------
  1. Load Data       — TPM matrices + sample ID lists
  2. Split           — train / test split (reproducible)
  3. DESeq2          — per-stress differential expression + marker gene selection
  4. Train           — 5-fold cross-validation MLP training
  5. Evaluate        — inference on held-out test set, averaged confusion matrix
  6. SHAP            — GradientExplainer, beeswarm plots, CSV export

Usage
-----
  # Full pipeline
  python AbiOmics_pipeline_v.1.0.py

  # Skip DESeq2  (reuse marker_genes.txt from a previous run)
  python AbiOmics_pipeline_v.1.0.py --skip-deseq

  # Only evaluate + SHAP on existing checkpoints
  python AbiOmics_pipeline_v.1.0.py --skip-deseq --skip-training

  # SHAP only  (implies --skip-deseq and --skip-training)
  python AbiOmics_pipeline_v.1.0.py --shap-only

  # Load data and run DESeq2 but stop before training
  python AbiOmics_pipeline_v.1.0.py --dry-run

  # Custom config file
  python AbiOmics_pipeline_v.1.0.py --config experiments/rice/config.yaml

  # Write a log file in addition to stdout
  python AbiOmics_pipeline_v.1.0.py --log-file logs/run_001.log
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ── Pipeline modules ─────────────────────────────────────────────────────────
from pipeline.common import load_config, set_seed, setup_logging, ensure_dirs
from pipeline.data import (
    load_tpm_data,
    load_sample_ids,
    split_train_test,
    build_training_matrices,
    build_testing_matrix,
)
from pipeline.deseq import (
    run_deseq_analysis,
    select_marker_genes,
    export_deg_lists,
)
from pipeline.training import (
    build_label_vector,
    train_model,
    evaluate_model,
    plot_averaged_confusion,
    get_checkpoint_list,
    export_cv_performance,
    export_test_performance,
)
from pipeline.shap_analysis import run_shap_analysis

# ── Original utils (unchanged) ───────────────────────────────────────────────
from utils.data import txt_to_list
from utils.utils import min_max_norm_TPM, list_to_txt

logger = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plant Stress Classification Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",         default="config/config.yaml",
                   help="Path to the YAML config file")
    p.add_argument("--skip-deseq",     action="store_true",
                   help="Skip DESeq2; load marker_genes.txt from Export/ instead")
    p.add_argument("--skip-training",  action="store_true",
                   help="Skip training; use existing checkpoints")
    p.add_argument("--shap-only",      action="store_true",
                   help="Run SHAP only (implies --skip-deseq + --skip-training)")
    p.add_argument("--dry-run",        action="store_true",
                   help="Run stages 1–3 only (no training or SHAP)")
    p.add_argument("--log-file",       default=None,
                   help="Optional path for a persistent log file")
    return p.parse_args()


# =============================================================================
# Individual stages
# =============================================================================

def _stage_load(cfg: dict):
    logger.info("─" * 60)
    logger.info("STAGE 1 — Load Data")
    logger.info("─" * 60)
    df_tpm = load_tpm_data(cfg)
    all_control_ids, samples_list = load_sample_ids(cfg)
    return df_tpm, all_control_ids, samples_list


def _stage_split(cfg: dict, all_control_ids: list, samples_list: dict):
    logger.info("─" * 60)
    logger.info("STAGE 2 — Train / Test Split")
    logger.info("─" * 60)
    return split_train_test(cfg, all_control_ids, samples_list)


def _stage_deseq(cfg: dict, df_tpm: dict, ctrl_train: dict, samp_train: dict):
    logger.info("─" * 60)
    logger.info("STAGE 3 — DESeq2 Analysis & Marker Gene Selection")
    logger.info("─" * 60)

    all_control_tpm, all_tpm, all_con_ids, sam_ids, sample_tpm = \
        build_training_matrices(cfg, df_tpm, ctrl_train, samp_train)

    deg_up, deg_down, _ = run_deseq_analysis(
        cfg, all_control_tpm, sam_ids, sample_tpm, all_con_ids
    )

    all_gene_ids = all_tpm.index.values.tolist()

    all_unique_degs, deg_up_unique, deg_down_unique, deg_up_sub, deg_down_sub = \
        select_marker_genes(cfg, deg_up, deg_down, all_gene_ids)

    export_deg_lists(cfg, deg_up, deg_down,
                     deg_up_unique, deg_down_unique,
                     deg_up_sub, deg_down_sub)

    # Save the final marker gene list for --skip-deseq re-runs
    marker_path = os.path.join(cfg["output"]["export_path"], "marker_genes.txt")
    list_to_txt(all_unique_degs, marker_path)
    logger.info("Marker genes saved to %s", marker_path)

    return all_control_tpm, all_tpm, all_con_ids, sam_ids, sample_tpm, all_unique_degs


def _stage_deseq_from_file(cfg: dict, df_tpm: dict, ctrl_train: dict, samp_train: dict):
    """Load previously computed marker genes instead of running DESeq2."""
    logger.info("─" * 60)
    logger.info("STAGE 3 — Loading Marker Genes from File (DESeq2 skipped)")
    logger.info("─" * 60)

    marker_path = os.path.join(cfg["output"]["export_path"], "marker_genes.txt")
    if not os.path.exists(marker_path):
        raise FileNotFoundError(
            f"Marker gene file not found: {marker_path}\n"
            "Run once without --skip-deseq to generate it."
        )
    all_unique_degs = txt_to_list(marker_path)
    logger.info("Loaded %d marker genes from %s", len(all_unique_degs), marker_path)

    all_control_tpm, all_tpm, all_con_ids, sam_ids, sample_tpm = \
        build_training_matrices(cfg, df_tpm, ctrl_train, samp_train)

    return all_control_tpm, all_tpm, all_con_ids, sam_ids, sample_tpm, all_unique_degs


def _stage_train(cfg: dict, all_tpm, all_con_ids, samp_train, all_unique_degs):
    logger.info("─" * 60)
    logger.info("STAGE 4 — Model Training (%d-fold CV)", cfg["training"]["n_folds"])
    logger.info("─" * 60)

    tpm_filtered = all_tpm.loc[all_unique_degs]
    tpm_norm, min_max_dic = min_max_norm_TPM(tpm_filtered)
    min_max_dic["marker_list"] = all_unique_degs   # bake into checkpoint

    label = build_label_vector(cfg, all_con_ids, samp_train)
    _, cv_reports = train_model(cfg, tpm_norm, label, min_max_dic)
    export_cv_performance(cfg, cv_reports)

    return tpm_norm, min_max_dic


def _stage_evaluate(cfg: dict, df_tpm, ctrl_test, samp_test, min_max_dic, all_unique_degs):
    logger.info("─" * 60)
    logger.info("STAGE 5 — Evaluation on Held-Out Test Set")
    logger.info("─" * 60)

    testing_tpm, testing_label = build_testing_matrix(
        cfg, df_tpm, ctrl_test, samp_test, all_unique_degs
    )
    all_confusion, ind_test_reports = evaluate_model(
        cfg, testing_tpm, testing_label, min_max_dic, all_unique_degs
    )
    n_folds = len(get_checkpoint_list(cfg))
    plot_averaged_confusion(cfg, all_confusion, n_folds)
    export_test_performance(cfg, ind_test_reports)

    return testing_tpm, testing_label, all_confusion


def _stage_shap(cfg: dict, tpm_norm, testing_tpm, min_max_dic, all_unique_degs):
    logger.info("─" * 60)
    logger.info("STAGE 6 — SHAP Analysis")
    logger.info("─" * 60)
    chk_list = get_checkpoint_list(cfg)
    run_shap_analysis(
        cfg            = cfg,
        chk_list       = chk_list,
        tpm_train_norm = tpm_norm,
        testing_tpm    = testing_tpm,
        min_max_dic    = min_max_dic,
        marker_genes   = all_unique_degs,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    setup_logging(log_file=args.log_file)

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── Resolve flag shortcuts ────────────────────────────────────────────────
    if args.shap_only:
        args.skip_deseq   = True
        args.skip_training = True

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed(cfg["random_seed"])

    # ── Directories ───────────────────────────────────────────────────────────
    ensure_dirs(cfg)

    logger.info("=" * 60)
    logger.info("Plant Stress Classification Pipeline")
    logger.info("Config  : %s", args.config)
    logger.info("Seed    : %d", cfg["random_seed"])
    logger.info("Classes : %s", cfg["label_dic"])
    logger.info("=" * 60)

    # ── Stages 1–2 always run ─────────────────────────────────────────────────
    df_tpm, all_control_ids, samples_list = _stage_load(cfg)
    ctrl_train, ctrl_test, samp_train, samp_test = _stage_split(
        cfg, all_control_ids, samples_list
    )

    # ── Stage 3: DESeq2 or load from file ─────────────────────────────────────
    if args.skip_deseq:
        (all_control_tpm, all_tpm, all_con_ids,
         sam_ids, sample_tpm, all_unique_degs) = _stage_deseq_from_file(
            cfg, df_tpm, ctrl_train, samp_train
        )
    else:
        (all_control_tpm, all_tpm, all_con_ids,
         sam_ids, sample_tpm, all_unique_degs) = _stage_deseq(
            cfg, df_tpm, ctrl_train, samp_train
        )

    if args.dry_run:
        logger.info("Dry run complete — stopping before training.")
        sys.exit(0)

    # ── Stage 4: Training or reuse checkpoints ────────────────────────────────
    if args.skip_training:
        # Reconstruct tpm_norm and min_max_dic without re-training
        tpm_filtered      = all_tpm.loc[all_unique_degs]
        tpm_norm, min_max_dic = min_max_norm_TPM(tpm_filtered)
        min_max_dic["marker_list"] = all_unique_degs
        logger.info("Training skipped — using existing checkpoints.")
    else:
        tpm_norm, min_max_dic = _stage_train(
            cfg, all_tpm, all_con_ids, samp_train, all_unique_degs
        )

    # ── Stage 5: Evaluation ───────────────────────────────────────────────────
    if args.shap_only:
        # Still need testing_tpm for SHAP; skip full evaluation loop
        testing_tpm, _ = build_testing_matrix(
            cfg, df_tpm, ctrl_test, samp_test, all_unique_degs
        )
    else:
        testing_tpm, _, _ = _stage_evaluate(
            cfg, df_tpm, ctrl_test, samp_test, min_max_dic, all_unique_degs
        )

    # ── Stage 6: SHAP ─────────────────────────────────────────────────────────
    _stage_shap(cfg, tpm_norm, testing_tpm, min_max_dic, all_unique_degs)

    logger.info("=" * 60)
    logger.info("Pipeline complete.  Outputs in: %s/", cfg["output"]["export_path"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
