from __future__ import annotations

import hashlib
import logging
import os
import random

import pandas as pd

from utils.data import get_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stable_key(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()


def _parse_id_file(cfg: dict) -> pd.DataFrame:
    dc      = cfg["data"]
    id_path = os.path.join(dc["data_path"], dc["id_file"])

    df = pd.read_csv(
        id_path,
        header=None,
        names=["sample_id", "label"],
        skipinitialspace=True,
    )
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["label"]     = df["label"].astype(str).str.strip()
    return df


def _validate(label_df: pd.DataFrame, tpm_columns: pd.Index, cfg: dict) -> None:
    dc            = cfg["data"]
    control_label = dc.get("control_label", "control")
    known_labels  = set(dc["stress_types"]) | {control_label}


    missing_samples = set(label_df["sample_id"]) - set(tpm_columns)
    if missing_samples:
        raise ValueError(
            f"{len(missing_samples)} sample ID(s) in the ID file are not "
            f"present as columns in the TPM matrix:"
            f"  {sorted(missing_samples)[:10]}"
            f"{'  ...' if len(missing_samples) > 10 else ''}"
        )


    unknown_labels = set(label_df["label"]) - known_labels
    if unknown_labels:
        raise ValueError(
            f"Unknown label(s) in ID file: {sorted(unknown_labels)}"
            f"Declared stress_types : {dc['stress_types']}"
            f"Control label         : '{control_label}'"
            f"Check that labels in the ID file exactly match config.yaml "
            f"(case-sensitive)."
        )


    duplicated = label_df[label_df.duplicated(subset="sample_id", keep=False)]
    if not duplicated.empty:
        dup_ids = duplicated["sample_id"].unique().tolist()
        detail  = "\n".join(
            f"  {sid!r}: {label_df.loc[label_df['sample_id']==sid, 'label'].tolist()}"
            for sid in dup_ids[:10]
        )
        raise ValueError(
            f"{len(dup_ids)} sample ID(s) appear under more than one label "
            f"in the ID file.  Each sample must have exactly one label:"
            f"{detail}"
            f"{'  ...' if len(dup_ids) > 10 else ''}"
        )


# ---------------------------------------------------------------------------
# Stage 1 — Load raw data
# ---------------------------------------------------------------------------

def load_tpm_data(cfg: dict) -> dict[str, pd.DataFrame]:
    dc            = cfg["data"]
    data_path     = dc["data_path"]
    control_label = dc.get("control_label", "control")

    tpm_path    = os.path.join(data_path, dc["tpm_file"])
    df_tpm_full = get_data(tpm_path, index_column=dc["index_column"])
    logger.info(
        "TPM matrix loaded: %d genes x %d samples  (%s)",
        df_tpm_full.shape[0], df_tpm_full.shape[1], dc["tpm_file"],
    )

    label_df = _parse_id_file(cfg)
    _validate(label_df, df_tpm_full.columns, cfg)

    df_tpm: dict[str, pd.DataFrame] = {}
    all_labels = [control_label] + list(dc["stress_types"])

    for label in all_labels:
        ids = label_df.loc[label_df["label"] == label, "sample_id"].tolist()
        if not ids:
            raise ValueError(
                f"No samples found for label '{label}' in the ID file. "
                f"Check spelling and case in config.yaml / ID file."
            )
        df_tpm[label] = df_tpm_full[ids]
        logger.info("  %-14s : %d samples", label, len(ids))

    return df_tpm


def load_sample_ids(cfg: dict) -> tuple[list[str], dict[str, list]]:
    dc            = cfg["data"]
    control_label = dc.get("control_label", "control")
    stress_types  = dc["stress_types"]

    label_df = _parse_id_file(cfg)

    all_control_ids: list[str] = label_df.loc[
        label_df["label"] == control_label, "sample_id"
    ].tolist()

    samples_list: dict[str, list] = {
        stress: label_df.loc[label_df["label"] == stress, "sample_id"].tolist()
        for stress in stress_types
    }

    logger.info("Control pool : %d samples", len(all_control_ids))
    for stress in stress_types:
        logger.info("  %-14s : %d stress samples", stress, len(samples_list[stress]))

    return all_control_ids, samples_list


# ---------------------------------------------------------------------------
# Stage 2 — Train / test split
# ---------------------------------------------------------------------------

def split_train_test(
    cfg: dict,
    all_control_ids: list[str],
    samples_list: dict[str, list],
) -> tuple[list[str], list[str], dict[str, list], dict[str, list]]:
    sc           = cfg["split"]
    stress_types = cfg["data"]["stress_types"]

    n_ct = sc["n_control_training"]
    n_cv = sc["n_control_testing"]
    n_st = sc["n_samples_training"]
    n_sv = sc["n_samples_testing"]

    # Controls: sampled once, shared across all stresses
    ctrl_train: list[str] = random.sample(all_control_ids, n_ct)
    ctrl_test_remainder   = sorted(
        set(all_control_ids) - set(ctrl_train),
        key=_stable_key,
    )
    ctrl_test: list[str] = random.sample(ctrl_test_remainder, n_cv)

    # Stress samples: sampled independently per stress
    samp_train: dict[str, list] = {}
    samp_test:  dict[str, list] = {}

    for stress in stress_types:
        samp_train[stress] = random.sample(samples_list[stress], n_st)
        remainder = sorted(
            set(samples_list[stress]) - set(samp_train[stress]),
            key=_stable_key,
        )
        samp_test[stress] = random.sample(remainder, n_sv)

    logger.info(
        "Split: %d ctrl-train (shared)  +  %d stress-train/class",
        n_ct, n_st,
    )
    logger.info(
        "       %d ctrl-test  (shared)  +  %d stress-test/class",
        n_cv, n_sv,
    )
    return ctrl_train, ctrl_test, samp_train, samp_test


# ---------------------------------------------------------------------------
# Stage 3 — Build training matrices
# ---------------------------------------------------------------------------

def build_training_matrices(
    cfg: dict,
    df_tpm: dict[str, pd.DataFrame],
    ctrl_train: list[str],
    samp_train: dict[str, list],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict, dict]:
    stress_types  = cfg["data"]["stress_types"]
    control_label = cfg["data"].get("control_label", "control")

    # One control matrix, shared across all stresses — no per-stress duplication
    all_control_tpm = df_tpm[control_label][ctrl_train]
    all_con_ids     = list(ctrl_train)

    sample_tpm: dict[str, pd.DataFrame] = {}
    sam_ids:    dict = {}

    for stress in stress_types:
        sample_tpm[stress] = df_tpm[stress][samp_train[stress]]
        sam_ids[stress]    = sample_tpm[stress].columns.values

    all_tpm = pd.concat(
        [all_control_tpm] + [sample_tpm[s] for s in stress_types], axis=1
    )

    logger.info(
        "Training matrix: %d genes x %d samples  "
        "(%d ctrl + %d stress x %d classes)",
        all_tpm.shape[0], all_tpm.shape[1],
        len(ctrl_train), len(samp_train[stress_types[0]]), len(stress_types),
    )
    return all_control_tpm, all_tpm, all_con_ids, sam_ids, sample_tpm


# ---------------------------------------------------------------------------
# Stage 5 — Build test matrix
# ---------------------------------------------------------------------------

def build_testing_matrix(
    cfg: dict,
    df_tpm: dict[str, pd.DataFrame],
    ctrl_test: list[str],
    samp_test: dict[str, list],
    marker_genes: list[str],
) -> tuple[pd.DataFrame, list[int]]:
    stress_types  = cfg["data"]["stress_types"]
    control_label = cfg["data"].get("control_label", "control")

    # One control test block — shared, not duplicated per stress
    all_control_test = df_tpm[control_label][ctrl_test]

    stress_test = pd.concat(
        [df_tpm[s][samp_test[s]] for s in stress_types], axis=1
    )
    all_test    = pd.concat([all_control_test, stress_test], axis=1)
    testing_tpm = all_test.loc[marker_genes]

    # Labels: 0 = control;  1..N follow stress_types order
    testing_label: list[int] = [0] * all_control_test.shape[1]
    for i, stress in enumerate(stress_types):
        testing_label.extend([i + 1] * len(samp_test[stress]))

    logger.info(
        "Test matrix: %d genes x %d samples  "
        "(%d ctrl + %d stress x %d classes)",
        testing_tpm.shape[0], testing_tpm.shape[1],
        all_control_test.shape[1], len(samp_test[stress_types[0]]), len(stress_types),
    )
    return testing_tpm, testing_label
