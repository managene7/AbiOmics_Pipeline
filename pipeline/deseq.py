from __future__ import annotations

import hashlib
import logging
import os
import random

import pandas as pd

from utils.utils import transpose_df, get_df_meta, do_DESeq_inference, list_to_txt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_key(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()


# ---------------------------------------------------------------------------
# DESeq2
# ---------------------------------------------------------------------------

def run_deseq_analysis(
    cfg: dict,
    all_control_tpm: pd.DataFrame,
    sam_ids: dict,
    sample_tpm: dict[str, pd.DataFrame],
    all_con_ids: list[str],
) -> tuple[dict, dict, dict]:
    dc   = cfg["data"]
    desc = cfg["deseq"]

    tpm_thresh       = dc["tpm_threshold"]
    stress_types = dc["stress_types"]
    export_path      = cfg["output"]["export_path"]
    os.makedirs(export_path, exist_ok=True)

    deg_up:   dict[str, list] = {}
    deg_down: dict[str, list] = {}
    df_meta:  dict[str, pd.DataFrame] = {}

    for stress in stress_types:
        logger.info("DESeq2: %s …", stress)

        # Merge control + stress, filter by TPM threshold, transpose for pydeseq2
        merged = pd.concat([all_control_tpm, sample_tpm[stress]], axis=1)
        merged_ov1  = merged[merged.max(axis=1) >= tpm_thresh]
        trans_merged = transpose_df(merged_ov1)   # adds empty-string first column

        # all_control_tpm.columns are the unique shared control IDs
        df_meta[stress] = get_df_meta(all_control_tpm.columns.tolist(), sam_ids[stress])

        _deg, deg_up[stress], deg_down[stress] = do_DESeq_inference(
            trans_merged,
            df_meta[stress],
            plot=True,
            stress=stress,
            log2fc=desc["log2fc_threshold"],
            padj=desc["padj_threshold"],
        )
        logger.info("  %-10s : %d up  %d down  %d total",
                    stress, len(deg_up[stress]), len(deg_down[stress]), len(_deg))

    return deg_up, deg_down, df_meta


# ---------------------------------------------------------------------------
# Marker gene selection
# ---------------------------------------------------------------------------

def select_marker_genes(
    cfg: dict,
    deg_up: dict[str, list],
    deg_down: dict[str, list],
    all_gene_ids: list[str],
) -> tuple[list[str], dict[str, list], dict[str, list], dict[str, list], dict[str, list]]:
    stress_types = cfg["data"]["stress_types"]
    n_per_dir    = cfg["deseq"]["n_deg_per_direction"]

    # Separate RNG so subsampling doesn't disturb the global random state
    rng = random.Random(cfg["random_seed"])

    deg_up_unique:   dict[str, list] = {}
    deg_down_unique: dict[str, list] = {}

    all_gene_set = set(all_gene_ids)

    for stress in stress_types:
        others = set(stress_types) - {stress}

        up_unique   = set(deg_up[stress])
        down_unique = set(deg_down[stress])

        # Remove genes that appear in any other stress
        for other in others:
            up_unique   -= set(deg_up[other])
            down_unique -= set(deg_down[other])

        # Restrict to genes present in the training matrix
        up_unique   &= all_gene_set
        down_unique &= all_gene_set

        # Deterministic sort for reproducibility
        deg_up_unique[stress]   = sorted(up_unique,   key=_stable_key)
        deg_down_unique[stress] = sorted(down_unique, key=_stable_key)

        logger.info("  Unique DEGs %-10s : %d up  %d down",
                    stress, len(deg_up_unique[stress]), len(deg_down_unique[stress]))

    # Subsample
    deg_up_sub:   dict[str, list] = {}
    deg_down_sub: dict[str, list] = {}

    for stress in stress_types:
        deg_down_sub[stress] = rng.sample(deg_down_unique[stress], k=n_per_dir)
        deg_up_sub[stress]   = rng.sample(deg_up_unique[stress],   k=n_per_dir)

    # Final gene list: all down-sub first, then all up-sub (original order)
    all_unique_degs: list[str] = []
    for stress in stress_types:
        all_unique_degs.extend(deg_down_sub[stress])
    for stress in stress_types:
        all_unique_degs.extend(deg_up_sub[stress])

    logger.info("Total marker genes selected: %d  (%d stresses × %d down + %d up)",
                len(all_unique_degs), len(stress_types), n_per_dir, n_per_dir)
    return all_unique_degs, deg_up_unique, deg_down_unique, deg_up_sub, deg_down_sub


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_deg_lists(
    cfg: dict,
    deg_up:        dict[str, list],
    deg_down:      dict[str, list],
    deg_up_unique: dict[str, list],
    deg_down_unique: dict[str, list],
    deg_up_sub:    dict[str, list] | None = None,
    deg_down_sub:  dict[str, list] | None = None,
) -> None:
    """Save all DEG lists to text files in the export directory."""
    export_path  = cfg["output"]["export_path"]
    stress_types = cfg["data"]["stress_types"]
    os.makedirs(export_path, exist_ok=True)

    for stress in stress_types:
        s = stress.lower()
        list_to_txt(deg_up[stress],          f"{export_path}/{s}_DEG_up.txt")
        list_to_txt(deg_down[stress],        f"{export_path}/{s}_DEG_down.txt")
        list_to_txt(deg_up_unique[stress],   f"{export_path}/{s}_DEG_up_unique.txt")
        list_to_txt(deg_down_unique[stress], f"{export_path}/{s}_DEG_down_unique.txt")
        if deg_up_sub:
            list_to_txt(deg_up_sub[stress],   f"{export_path}/{s}_DEG_up_sub.txt")
        if deg_down_sub:
            list_to_txt(deg_down_sub[stress], f"{export_path}/{s}_DEG_down_sub.txt")

    logger.info("DEG lists exported to %s/", export_path)
