from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_SPLIT_ENV = "CATCH_PATIENT_SPLIT_CSV"


def default_split_csv(seed: int) -> str:
    return str(Path(__file__).resolve().parent / f"catch_patient_split_seed{seed}.csv")


def _unique_patients(values: Iterable[object]) -> np.ndarray:
    patients = pd.Series(values).dropna().astype(str).unique()
    return np.array(sorted(patients), dtype=object)


def _patient_universe(
    df: pd.DataFrame,
    patient_col: str,
    source_csv_path: Optional[str],
) -> np.ndarray:
    if source_csv_path and os.path.exists(source_csv_path):
        source_df = pd.read_csv(source_csv_path, usecols=[patient_col])
        return _unique_patients(source_df[patient_col])
    return _unique_patients(df[patient_col])


def _create_split_table(
    patients: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    assert 0.0 < train_ratio < 1.0
    assert 0.0 <= val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0

    rng = np.random.RandomState(seed)
    patients = np.array(patients, dtype=object).copy()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    split = np.empty(n, dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_val] = "val"
    split[n_train + n_val:] = "test"

    return pd.DataFrame({"patient_id": patients.astype(str), "split": split})


def shared_patient_level_split_3way(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 3407,
    split_csv: Optional[str] = None,
    source_csv_path: Optional[str] = None,
):
    """Apply one persistent CATCH patient-level split across all downstream tasks."""
    split_csv = split_csv or os.environ.get(DEFAULT_SPLIT_ENV) or default_split_csv(seed)
    split_path = Path(split_csv)

    if split_path.exists():
        split_df = pd.read_csv(split_path)
        required = {"patient_id", "split"}
        missing = required.difference(split_df.columns)
        if missing:
            raise ValueError(f"Split file {split_path} is missing columns: {sorted(missing)}")
    else:
        patients = _patient_universe(df, patient_col, source_csv_path)
        split_df = _create_split_table(patients, train_ratio, val_ratio, seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(split_path, index=False)

    split_map = dict(zip(split_df["patient_id"].astype(str), split_df["split"].astype(str)))
    split_values = df[patient_col].astype(str).map(split_map)
    missing_patients = sorted(df.loc[split_values.isna(), patient_col].astype(str).unique())
    if missing_patients:
        preview = ", ".join(missing_patients[:10])
        raise ValueError(
            f"{len(missing_patients)} patients are absent from shared split {split_path}: {preview}"
        )

    df_train = df[split_values == "train"].reset_index(drop=True)
    df_val = df[split_values == "val"].reset_index(drop=True)
    df_test = df[split_values == "test"].reset_index(drop=True)
    return df_train, df_val, df_test
