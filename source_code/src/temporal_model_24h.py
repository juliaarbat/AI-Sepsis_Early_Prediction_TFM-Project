"""Sequential models for predicting sepsis on the following day.

This module keeps a single main public entry point,
`train_and_evaluate_temporal_model_24h`, which prepares the data, creates the split,
trains a sequential model, and saves metrics/predictions. It also
exposes two small functions shared with the classic models:
`add_admission_diagnosis_features` and `audit_operational_microbiology`.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEEP_LEARNING_OUTPUTS_DIR
from .predictive_model_24h import (
    ELIGIBILITY_COL,
    ID_COL,
    PATIENT_COL,
    TARGET,
    _is_sofa_or_availability_derived_column,
    fit_preprocessor,
    calculate_top_risk_capture,
    calculate_metrics,
    calculate_episode_metrics,
    prepare_model_dataset_24h,
    summarize_cohort,
    summarize_splits,
    select_minimum_sensitivity_threshold,
    select_youden_threshold,
    transform_features,
)
from .real_policies import (
    REAL_ALL_2026,
    REAL_NEW_2026,
    REAL_READMITTED_2026,
    normalize_real_policy,
    select_excluded_real_units,
    select_real_units,
)
from .output_contracts import (
    LEVEL_EPISODE,
    LEVEL_NEXT_DAY,
    SPLIT_REAL,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    TOP_RISK_LEVEL_DAY,
    TOP_RISK_LEVEL_EPISODE,
    TRAIN_VALID_TEST_SPLITS,
    deep_output_paths,
    deep_tuning_filename,
)
from .split_utils import (
    normalize_split_unit as _normalize_split_unit,
    split_unit as _shared_split_unit,
    validate_exclusive_patients_between_splits,
)

# Temporal variables: one row is one available day in the episode.
TEMPORAL_COLUMNS = [
    "SBP",
    "DBP",
    "TAM",
    "HR",
    "RESP",
    "O2SAT",
    "TEMP",
    "FIO2",
    "DIURESIS",
    "GLASGOW",
    "porta_o2",
    "dispositius_invasius_previs",
    "ph_arterial",
    "pao2_arterial",
    "paco2_arterial",
    "bicarbonat_arterial",
    "exc_base_arterial",
    "lactat_arterial",
    "ph_venos",
    "pao2_venos",
    "paco2_venos",
    "bicarbonat_venos",
    "exc_base_venos",
    "lactat_venos",
    "hematocrit",
    "hemoglobina",
    "leucocits",
    "pct_neutrofils",
    "granulocits_immadurs",
    "plaquetes",
    "fibrinogen",
    "temps_protrombina_pct",
    "pcr",
    "procalcitonina",
    "glucosa",
    "urea",
    "creatinina",
    "bilirubina_total",
    "got_ast",
    "albumina",
    "proteines_totals",
    "troponina",
    "hemocultiu_positiu",
    "hemocultiu_germen",
    "hemocultiu_temps_positivitat_h",
    "urocultiu_resultat",
    "aspirat_traqueal_germen",
    "broncoaspirat_germen",
    "bal_germen",
    "ag_pneumococ",
    "ag_legionella",
    "colonitzacio_previa_blee",
    "colonitzacio_previa_cre",
    "colonitzacio_previa_mrsa",
    "colonitzacio_previa_vre",
    "cultiu_positiu_previ_90d",
    "vasopressor_dobutamina",
    "vasopressor_dopamina",
    "vasopressor_noradrenalina",
    "vasopressor_adrenalina",
    "antibiotic",
    "atb_duracio",
    "antibiotics_previs_90d",
    "vasopressor_qualsevol",
    "vasopressor_multiple",
    "dia_relatiu",
    "dies_des_ingres",
]

# Subset of temporal variables that can be excluded in robustness analyses.
MICROBIOLOGY_COLUMNS = {
    "hemocultiu_positiu",
    "hemocultiu_germen",
    "hemocultiu_temps_positivitat_h",
    "urocultiu_resultat",
    "aspirat_traqueal_germen",
    "broncoaspirat_germen",
    "bal_germen",
    "ag_pneumococ",
    "ag_legionella",
    "colonitzacio_previa_blee",
    "colonitzacio_previa_cre",
    "colonitzacio_previa_mrsa",
    "colonitzacio_previa_vre",
    "cultiu_positiu_previ_90d",
}

# Simple admission-diagnosis derivatives reused as static variables.
DIAGNOSTIC_INGRES_DERIVED_COLUMNS = [
    "diagnostic_ingres_codi",
    "diagnostic_ingres_prefix3",
    "grup_diagnostic_ingres",
]

# Variables describing the baseline admission context.
STATIC_COLUMNS = [
    "edat",
    "sexe",
    "font_admissio",
    "centre_origen",
    "codi_servei_admissor",
    "diagnostic_ingres_codi",
    "diagnostic_ingres_prefix3",
    "grup_diagnostic_ingres",
    "hospitalitzacio_recent_90d",
    "reingres_30d",
    "cirurgia",
    "urgencia_cirurgia",
    "temps_cirurgia",
    "COMORB_DIABETES_MELLITUS",
    "COMORB_NEOPLASIA_SOLIDA",
    "COMORB_NEOPLASIA_HEMATOLOGICA",
    "COMORB_ENOLISME_SEVER",
    "COMORB_CIRROSI_HEPATICA",
    "COMORB_VIH_SIDA",
    "COMORB_TRASPLANT_ORGAN_SOLID",
    "COMORB_TRASPLANT_MOLL_OS",
    "COMORB_AGAMMAGLOBULINEMIA",
    "COMORB_HIPOGAMMAGLOBULINEMIA",
    "COMORB_MALABSORTIVES",
    "COMORB_MALNUTRICIO_SEVERA",
    "COMORB_ASPLENIA",
    "COMORB_ESPLENECTOMIA",
    "COMORB_IRC_DIALISI",
    "COMORB_NEUTROPENIA_GREU",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_evaluate_temporal_model_24h(
    df_sofa: pd.DataFrame,
    output_dir: Path = DEEP_LEARNING_OUTPUTS_DIR / "transformer" / "simple_execution",
    lookback_days: int | None = 10,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.15,
    model_type: str = "transformer",
    recurrent_hidden_size: int | None = None,
    recurrent_bidirectional: bool = False,
    max_missing_ratio: float = 0.80,
    imbalance_strategy: str = "pos_weight",
    exclude_microbiology: bool = False,
    train_parts: int = 1,
    split_proportions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    split_unit: str = "patient",
    real_start_date: str | pd.Timestamp | None = "2026-01-01",
    real_overlap_policy: str = REAL_ALL_2026,
    evaluate_real_from_real_start: bool = False,
    early_stopping_patience: int | None = 4,
    early_stopping_min_delta: float = 0.0,
    optuna_trials: int | None = None,
    tune_lookback_days: bool = True,
    output_prefix: str = "transformer_24h",
    seed: int = 42,
    verbose: bool = True,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Train, evaluate, and optionally write outputs for one 24h sequential model.

    By default this creates a multimodal Transformer: a temporal branch with the
    last `lookback_days` days and a static branch with demographics,
    comorbidities, and admission diagnosis. The same pipeline accepts `model_type`
    set to `lstm` or `rnn` so `recurrent_24h_main.py` can reuse
    preprocessing, splits, and metrics.
    """
    model_type = model_type.lower().strip()
    if model_type not in {"transformer", "lstm", "rnn"}:
        raise ValueError("model_type must be 'transformer', 'lstm', or 'rnn'.")
    if model_type == "transformer" and d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if optuna_trials is not None and optuna_trials > 0:
        return _train_and_evaluate_sequence_24h_with_optuna(
            df_sofa=df_sofa,
            output_dir=output_dir,
            lookback_days=lookback_days,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            model_type=model_type,
            recurrent_hidden_size=recurrent_hidden_size,
            recurrent_bidirectional=recurrent_bidirectional,
            max_missing_ratio=max_missing_ratio,
            imbalance_strategy=imbalance_strategy,
            exclude_microbiology=exclude_microbiology,
            train_parts=train_parts,
            split_proportions=split_proportions,
            split_unit=split_unit,
            real_start_date=real_start_date,
            real_overlap_policy=real_overlap_policy,
            evaluate_real_from_real_start=evaluate_real_from_real_start,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            optuna_trials=int(optuna_trials),
            tune_lookback_days=tune_lookback_days,
            output_prefix=output_prefix,
            seed=seed,
            verbose=verbose,
        )

    torch, nn, DataLoader, WeightedRandomSampler = _import_torch()
    _set_seed(seed, torch)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare the cohort, temporal split, and candidate variables.
    # Everything learned afterwards (preprocessors, thresholds, and weights) is computed
    # without touching the real split.
    df_model = add_admission_diagnosis_features(prepare_model_dataset_24h(df_sofa))
    microbiology_info = audit_operational_microbiology(df_model)
    split_map, split_temporal_info = _create_final_temporal_split(
        df_model,
        proportions=split_proportions,
        split_unit=split_unit,
        real_start_date=real_start_date,
        real_overlap_policy=real_overlap_policy,
    )
    df_model["split"] = df_model[ID_COL].map(split_map)
    df_model = df_model.loc[df_model["split"].notna()].copy()
    df_model, real_filter_info = _filter_real_from_start_date(
        df_model,
        real_start_date=real_start_date,
        enabled=evaluate_real_from_real_start,
    )
    validate_exclusive_patients_between_splits(df_model)
    df_model = df_model.sort_values(["split", ID_COL, "data_index"], kind="stable").reset_index(drop=True)
    max_sequence_len = _resolve_max_sequence_len(df_model, lookback_days)

    train = df_model.loc[df_model["split"] == SPLIT_TRAIN].copy().reset_index(drop=True)
    valid = df_model.loc[df_model["split"] == SPLIT_VALID].copy().reset_index(drop=True)
    test = df_model.loc[df_model["split"] == SPLIT_TEST].copy().reset_index(drop=True)
    real = df_model.loc[df_model["split"] == SPLIT_REAL].copy().reset_index(drop=True)

    # 2. Separate preprocessing for the temporal and static branches.
    temporal_candidates = _filter_microbiology_columns(
        TEMPORAL_COLUMNS,
        exclude_microbiology=exclude_microbiology,
    )
    temporal_columns = _available_columns(train, temporal_candidates)
    static_columns = _available_columns(train, STATIC_COLUMNS)
    diagnostic_static_columns = [
        col
        for col in DIAGNOSTIC_INGRES_DERIVED_COLUMNS
        if col in static_columns
    ]
    temporal_preprocessor = fit_preprocessor(
        _preprocessor_frame(train, temporal_columns),
        max_missing_ratio=max_missing_ratio,
    )
    static_preprocessor = fit_preprocessor(
        _preprocessor_frame(train, static_columns),
        max_missing_ratio=max_missing_ratio,
        force_categorical_columns=set(diagnostic_static_columns),
    )

    x_train_temporal = transform_features(
        _preprocessor_frame(train, temporal_columns),
        temporal_preprocessor,
    ).astype("float32")
    x_valid_temporal = transform_features(
        _preprocessor_frame(valid, temporal_columns),
        temporal_preprocessor,
    ).astype("float32")
    x_test_temporal = transform_features(
        _preprocessor_frame(test, temporal_columns),
        temporal_preprocessor,
    ).astype("float32")
    x_real_temporal = (
        transform_features(
            _preprocessor_frame(real, temporal_columns),
            temporal_preprocessor,
        ).astype("float32")
        if not real.empty
        else None
    )
    x_train_static = transform_features(
        _preprocessor_frame(train, static_columns),
        static_preprocessor,
    ).astype("float32")
    x_valid_static = transform_features(
        _preprocessor_frame(valid, static_columns),
        static_preprocessor,
    ).astype("float32")
    x_test_static = transform_features(
        _preprocessor_frame(test, static_columns),
        static_preprocessor,
    ).astype("float32")
    x_real_static = (
        transform_features(
            _preprocessor_frame(real, static_columns),
            static_preprocessor,
        ).astype("float32")
        if not real.empty
        else None
    )

    # 3. Sequential datasets and DataLoaders. Padding is computed by episode.
    train_ds = MultiModalSequenceDataset(
        train,
        x_train_temporal,
        x_train_static,
        max_sequence_len,
        torch,
    )
    valid_ds = MultiModalSequenceDataset(
        valid,
        x_valid_temporal,
        x_valid_static,
        max_sequence_len,
        torch,
    )
    test_ds = MultiModalSequenceDataset(
        test,
        x_test_temporal,
        x_test_static,
        max_sequence_len,
        torch,
    )
    real_ds = (
        MultiModalSequenceDataset(
            real,
            x_real_temporal,
            x_real_static,
            max_sequence_len,
            torch,
        )
        if x_real_temporal is not None and x_real_static is not None
        else None
    )

    train_part_indices = _create_train_partitions_by_episode(train, train_parts, seed=seed)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    real_loader = (
        DataLoader(real_ds, batch_size=batch_size, shuffle=False)
        if real_ds is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 4. PyTorch architecture: Transformer by default, or RNN/LSTM on request.
    model = _create_sequence_model_24h(
        model_type=model_type,
        n_temporal_features=x_train_temporal.shape[1],
        n_static_features=x_train_static.shape[1],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_len=max_sequence_len,
        recurrent_hidden_size=recurrent_hidden_size,
        recurrent_bidirectional=recurrent_bidirectional,
        nn=nn,
        torch=torch,
    ).to(device)

    y_train = train[TARGET].astype(int).to_numpy()
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    criterion = _create_criterion(
        strategy=imbalance_strategy,
        pos=pos,
        neg=neg,
        nn=nn,
        torch=torch,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 5. Training with early stopping on validation AUPRC.
    history: list[dict[str, float | int]] = []
    best_state = None
    best_valid_auprc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch: int | None = None

    if verbose:
        print(
            f"[{model_type}] Dataset prepared: "
            f"{len(train)} train, {len(valid)} valid, {len(test)} test, "
            f"{len(real)} real, "
            f"lookback_days={lookback_days}, max_sequence_len={max_sequence_len}, "
            f"parts_train={len(train_part_indices)}, "
            f"device={device}.",
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        part_idx = (epoch - 1) % len(train_part_indices)
        indices_part = train_part_indices[part_idx]
        train_cycle_completed = part_idx == len(train_part_indices) - 1 or epoch == epochs
        train_loader = _create_train_loader(
            train_ds=train_ds,
            train=train,
            indices_part=indices_part,
            batch_size=batch_size,
            imbalance_strategy=imbalance_strategy,
            torch=torch,
            DataLoader=DataLoader,
            WeightedRandomSampler=WeightedRandomSampler,
        )
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device, torch)
        pred_valid = _predict_loader(model, valid_loader, device, torch)
        y_valid = valid[TARGET].astype(int).to_numpy()
        valid_auroc = _float(calculate_metrics(y_valid, pred_valid, 0.5)["auroc"])
        valid_auprc = _float(calculate_metrics(y_valid, pred_valid, 0.5)["auprc"])
        history.append(
            {
                "epoch": epoch,
                "train_part": part_idx + 1,
                "train_parts": len(train_part_indices),
                "train_rows_part": int(len(indices_part)),
                "train_episodes_part": int(train.iloc[indices_part][ID_COL].nunique()),
                "train_loss": train_loss,
                "valid_auroc": valid_auroc,
                "valid_auprc": valid_auprc,
                "train_cycle_completed": train_cycle_completed,
            }
        )
        # When training is partitioned, wait for a full pass across all parts
        # before deciding whether the model improved or early stopping should activate.
        if train_cycle_completed:
            improved = valid_auprc > best_valid_auprc + early_stopping_min_delta
            if improved:
                best_valid_auprc = valid_auprc
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

        if verbose:
            print(
                f"[{model_type}] "
                f"Epoch {epoch}/{epochs} | part {part_idx + 1}/{len(train_part_indices)} "
                f"({len(indices_part)} files) | loss={train_loss:.4f} | "
                f"valid AUPRC={valid_auprc:.4f} | valid AUROC={valid_auroc:.4f}",
                flush=True,
            )

        if (
            early_stopping_patience is not None
            and train_cycle_completed
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_epoch = epoch
            if verbose:
                patience_unit = (
                    "full train cycles"
                    if len(train_part_indices) > 1
                    else "epochs"
                )
                print(
                    f"[{model_type}] Early stopping: "
                    f"no improvement for {early_stopping_patience} {patience_unit}.",
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 6. Final predictions and thresholds selected only with validation.
    pred_train = _predict_loader(model, train_eval_loader, device, torch)
    pred_valid = _predict_loader(model, valid_loader, device, torch)
    pred_test = _predict_loader(model, test_loader, device, torch)
    pred_real = (
        _predict_loader(model, real_loader, device, torch)
        if real_loader is not None
        else np.array([], dtype=float)
    )

    y_train = train[TARGET].astype(int).to_numpy()
    y_valid = valid[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()
    y_real = real[TARGET].astype(int).to_numpy() if not real.empty else None

    threshold_youden = select_youden_threshold(y_valid, pred_valid)
    threshold_sens80 = select_minimum_sensitivity_threshold(y_valid, pred_valid, sensitivity=0.80)

    metrics = {
        SPLIT_TRAIN: calculate_metrics(y_train, pred_train, threshold_youden),
        SPLIT_VALID: calculate_metrics(y_valid, pred_valid, threshold_youden),
        SPLIT_TEST: calculate_metrics(y_test, pred_test, threshold_youden),
        "test_sensitivity_80_threshold": calculate_metrics(y_test, pred_test, threshold_sens80),
        "train_episode": calculate_episode_metrics(train, pred_train, threshold_youden),
        "valid_episode": calculate_episode_metrics(valid, pred_valid, threshold_youden),
        "test_episode": calculate_episode_metrics(test, pred_test, threshold_youden),
        "test_episode_sensitivity_80_threshold": calculate_episode_metrics(
            test,
            pred_test,
            threshold_sens80,
        ),
    }
    if y_real is not None:
        metrics.update(
            {
                SPLIT_REAL: calculate_metrics(y_real, pred_real, threshold_youden),
                "real_sensitivity_80_threshold": calculate_metrics(y_real, pred_real, threshold_sens80),
                "real_episode": calculate_episode_metrics(real, pred_real, threshold_youden),
                "real_episode_sensitivity_80_threshold": calculate_episode_metrics(
                    real,
                    pred_real,
                    threshold_sens80,
                ),
            }
        )
    top_risk = {
        "test_day": calculate_top_risk_capture(test, pred_test, level=TOP_RISK_LEVEL_DAY),
        "test_episode": calculate_top_risk_capture(test, pred_test, level=TOP_RISK_LEVEL_EPISODE),
    }
    if y_real is not None:
        top_risk.update(
            {
                "real_day": calculate_top_risk_capture(real, pred_real, level=TOP_RISK_LEVEL_DAY),
                "real_episode": calculate_top_risk_capture(real, pred_real, level=TOP_RISK_LEVEL_EPISODE),
            }
        )

    # 7. Auditable summary: save architecture, splits, metrics, and decisions.
    output_paths = deep_output_paths(output_dir, output_prefix)
    summary = {
        "objective": "Temporal deep-learning prediction of SOFA-positive sepsis on the following day",
        "granularity": (
            "1 temporal step = 1 day. Using data available through day D, "
            "the model predicts whether observed day D+1 will have sepsis."
        ),
        "target": TARGET,
        "model": _sequence_model_name(model_type),
        "model_type": model_type,
        "architecture": _describe_sequence_architecture(model_type),
        "lookback_days": lookback_days,
        "tune_lookback_days": tune_lookback_days,
        "max_sequence_len": max_sequence_len,
        "train_parts": len(train_part_indices),
        "train_parts_detail": [
            {
                "part": i + 1,
                "n_rows": int(len(indices)),
                "n_episodes": int(train.iloc[indices][ID_COL].nunique()),
                "n_positives": int(train.iloc[indices][TARGET].sum()),
            }
            for i, indices in enumerate(train_part_indices)
        ],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "recurrent_hidden_size": recurrent_hidden_size,
        "recurrent_bidirectional": recurrent_bidirectional,
        "imbalance_strategy": imbalance_strategy,
        "exclude_microbiology": exclude_microbiology,
        "operational_microbiology": microbiology_info,
        "split_temporal": split_temporal_info,
        "filter_audit": real_filter_info,
        "real_start_date": None if real_start_date is None else pd.Timestamp(real_start_date).isoformat(),
        "real_overlap_policy": real_overlap_policy,
        "evaluate_real_from_real_start": evaluate_real_from_real_start,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "best_epoch": best_epoch,
        "best_valid_auprc": best_valid_auprc if np.isfinite(best_valid_auprc) else None,
        "stopped_epoch": stopped_epoch,
        "sliding_windows": (
            _describe_temporal_window(lookback_days, max_sequence_len)
        ),
        "partitioned_training": _describe_training_partitions(len(train_part_indices)),
        "training_partitions": _describe_training_partitions(len(train_part_indices)),
        "missing_value_handling": (
            "Variables with more than 80% missingness in train are excluded; numeric variables are imputed with "
            "train median + missing indicator; categoricals use the __MISSING__ category."
        ),
        "imbalance_handling": _describe_imbalance_strategy(imbalance_strategy),
        "pos_weight_train": neg / pos if imbalance_strategy in {"pos_weight", "focal_loss"} else None,
        "max_missing_ratio_features": max_missing_ratio,
        "n_original_temporal_variables": len(temporal_columns),
        "original_temporal_variables": temporal_columns,
        "n_original_static_variables": len(static_columns),
        "original_static_variables": static_columns,
        "static_admission_diagnosis_variables": diagnostic_static_columns,
        "n_static_admission_diagnosis_features": len(
            [
                feature
                for feature in static_preprocessor.feature_names
                if any(
                    feature == col or feature.startswith(f"{col}__")
                    for col in diagnostic_static_columns
                )
            ]
        ),
        "n_temporal_variables_excluded_over_80_pct_missing": len(
            temporal_preprocessor.excluded_high_missing_columns
        ),
        "temporal_variables_excluded_over_80_pct_missing": (
            temporal_preprocessor.excluded_high_missing_columns
        ),
        "n_static_variables_excluded_over_80_pct_missing": len(
            static_preprocessor.excluded_high_missing_columns
        ),
        "static_variables_excluded_over_80_pct_missing": (
            static_preprocessor.excluded_high_missing_columns
        ),
        "n_temporal_variables_excluded_for_sofa_leakage": len(
            temporal_preprocessor.excluded_leakage_columns
        ),
        "temporal_variables_excluded_for_sofa_leakage": (
            temporal_preprocessor.excluded_leakage_columns
        ),
        "n_static_variables_excluded_for_sofa_leakage": len(
            static_preprocessor.excluded_leakage_columns
        ),
        "static_variables_excluded_for_sofa_leakage": (
            static_preprocessor.excluded_leakage_columns
        ),
        "n_sofa_derived_variables_excluded": len(
            _sofa_or_availability_columns(train.columns)
        ),
        "sofa_derived_variables_excluded": _sofa_or_availability_columns(train.columns),
        "device": str(device),
        "seed": seed,
        "n_temporal_features_per_day": len(temporal_preprocessor.feature_names),
        "n_static_features": len(static_preprocessor.feature_names),
        "n_features_per_day": len(temporal_preprocessor.feature_names),
        "threshold_youden_valid": threshold_youden,
        "threshold_sensitivity_80_valid": threshold_sens80,
        "cohort": summarize_cohort(df_model),
        "splits": summarize_splits(df_model),
        "split_unit": split_unit,
        "history": history,
        "metrics": metrics,
        "top_risk": top_risk,
        "outputs": {
            "summary": str(output_paths["summary"]),
            "predictions": str(output_paths["predictions"]),
            "comparable_metrics": str(output_paths["comparable_metrics"]),
            "excluded_missing_variables": str(output_paths["excluded_missing_variables"]),
            "model": str(output_paths["model"]),
        },
    }

    if save_outputs:
        _save_temporal_model_outputs(
            output_dir=output_dir,
            summary=summary,
            model=model,
            temporal_preprocessor=temporal_preprocessor,
            static_preprocessor=static_preprocessor,
            train=train,
            valid=valid,
            test=test,
            real=real,
            pred_train=pred_train,
            pred_valid=pred_valid,
            pred_test=pred_test,
            pred_real=pred_real,
            torch=torch,
            output_prefix=output_prefix,
        )
    return summary


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def _train_and_evaluate_sequence_24h_with_optuna(
    df_sofa: pd.DataFrame,
    output_dir: Path,
    lookback_days: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    model_type: str,
    recurrent_hidden_size: int | None,
    recurrent_bidirectional: bool,
    max_missing_ratio: float,
    imbalance_strategy: str,
    exclude_microbiology: bool,
    train_parts: int,
    split_proportions: tuple[float, float, float],
    split_unit: str,
    real_start_date: str | pd.Timestamp | None,
    real_overlap_policy: str,
    evaluate_real_from_real_start: bool,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    optuna_trials: int,
    tune_lookback_days: bool,
    output_prefix: str,
    seed: int,
    verbose: bool,
) -> dict[str, object]:
    """Select hyperparameters with Optuna and train the final model."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is not installed. Add it with `pip install optuna` "
            "or run with optuna_trials=None."
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_rows: list[dict[str, object]] = []

    if verbose:
        print(
            f"[{model_type}] Optuna: {optuna_trials} trials on validation AUPRC",
            flush=True,
        )

    def objective(trial) -> float:
        started = time.perf_counter()
        params = _suggest_sequence_hyperparameters(
            trial=trial,
            model_type=model_type,
            current_lookback_days=lookback_days,
            current_batch_size=batch_size,
            current_learning_rate=learning_rate,
            current_d_model=d_model,
            current_n_heads=n_heads,
            current_n_layers=n_layers,
            current_dropout=dropout,
            current_hidden_size=recurrent_hidden_size,
            current_bidirectional=recurrent_bidirectional,
            current_imbalance_strategy=imbalance_strategy,
            tune_lookback_days=tune_lookback_days,
        )
        row = {
            "model_type": model_type,
            "trial": int(trial.number),
            "hyperparameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "status": "ok",
            "valid_auprc": np.nan,
            "valid_auroc": np.nan,
            "valid_episode_auprc": np.nan,
            "valid_episode_auroc": np.nan,
            "seconds": np.nan,
            "best": False,
        }
        try:
            summary = train_and_evaluate_temporal_model_24h(
                df_sofa=df_sofa,
                output_dir=output_dir,
                lookback_days=params["lookback_days"],
                epochs=epochs,
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate"],
                d_model=params["d_model"],
                n_heads=params["n_heads"],
                n_layers=params["n_layers"],
                dropout=params["dropout"],
                model_type=model_type,
                recurrent_hidden_size=params["recurrent_hidden_size"],
                recurrent_bidirectional=params["recurrent_bidirectional"],
                max_missing_ratio=max_missing_ratio,
                imbalance_strategy=params["imbalance_strategy"],
                exclude_microbiology=exclude_microbiology,
                train_parts=train_parts,
                split_proportions=split_proportions,
                split_unit=split_unit,
                real_start_date=real_start_date,
                real_overlap_policy=real_overlap_policy,
                evaluate_real_from_real_start=evaluate_real_from_real_start,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                optuna_trials=None,
                tune_lookback_days=tune_lookback_days,
                output_prefix=output_prefix,
                seed=seed + trial.number,
                verbose=False,
                save_outputs=False,
            )
            valid_metrics = summary["metrics"]["valid"]
            valid_episode_metrics = summary["metrics"]["valid_episode"]
            valid_auprc = _float(valid_metrics["auprc"])
            valid_auroc = _float(valid_metrics["auroc"])
            valid_episode_auprc = _float(valid_episode_metrics["auprc"])
            valid_episode_auroc = _float(valid_episode_metrics["auroc"])
            row.update(
                {
                    "valid_auprc": valid_auprc,
                    "valid_auroc": valid_auroc,
                    "valid_episode_auprc": valid_episode_auprc,
                    "valid_episode_auroc": valid_episode_auroc,
                    "seconds": float(time.perf_counter() - started),
                }
            )
            tuning_rows.append(row)
            if verbose:
                print(
                    f"[{model_type}] Trial {trial.number + 1}/{optuna_trials}: "
                    f"valid AUPRC={valid_auprc:.4f}, AUROC={valid_auroc:.4f}, "
                    f"episode AUPRC={valid_episode_auprc:.4f}, "
                    f"episode AUROC={valid_episode_auroc:.4f}, "
                    f"params={params}",
                    flush=True,
                )
            return valid_auprc if np.isfinite(valid_auprc) else 0.0
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "seconds": float(time.perf_counter() - started),
                }
            )
            tuning_rows.append(row)
            raise

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False, catch=(Exception,))

    completed_rows = [row for row in tuning_rows if row["status"] == "ok"]
    if completed_rows:
        best_trial_number = int(study.best_trial.number)
        for row in tuning_rows:
            row["best"] = row["status"] == "ok" and int(row["trial"]) == best_trial_number
        best_row = next(row for row in tuning_rows if row["best"])
        best_params = json.loads(str(best_row["hyperparameters"]))
    else:
        best_params = {}

    pd.DataFrame(tuning_rows).to_csv(output_dir / deep_tuning_filename(output_prefix), index=False)

    final_params = {
        "lookback_days": lookback_days,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "recurrent_hidden_size": recurrent_hidden_size,
        "recurrent_bidirectional": recurrent_bidirectional,
        "imbalance_strategy": imbalance_strategy,
        "tune_lookback_days": tune_lookback_days,
    }
    final_params.update(best_params)

    if verbose:
        print(
            f"[{model_type}] Hyperparameters selected by Optuna: {final_params}",
            flush=True,
        )

    summary = train_and_evaluate_temporal_model_24h(
        df_sofa=df_sofa,
        output_dir=output_dir,
        lookback_days=final_params["lookback_days"],
        epochs=epochs,
        batch_size=final_params["batch_size"],
        learning_rate=final_params["learning_rate"],
        d_model=final_params["d_model"],
        n_heads=final_params["n_heads"],
        n_layers=final_params["n_layers"],
        dropout=final_params["dropout"],
        model_type=model_type,
        recurrent_hidden_size=final_params["recurrent_hidden_size"],
        recurrent_bidirectional=final_params["recurrent_bidirectional"],
        max_missing_ratio=max_missing_ratio,
        imbalance_strategy=final_params["imbalance_strategy"],
        exclude_microbiology=exclude_microbiology,
        train_parts=train_parts,
        split_proportions=split_proportions,
        split_unit=split_unit,
        real_start_date=real_start_date,
        real_overlap_policy=real_overlap_policy,
        evaluate_real_from_real_start=evaluate_real_from_real_start,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        optuna_trials=None,
        tune_lookback_days=tune_lookback_days,
        output_prefix=output_prefix,
        seed=seed,
        verbose=verbose,
        save_outputs=True,
    )
    summary["hyperparameter_tuning"] = {
        "method": "optuna",
        "criterion": "Best validation AUPRC at D+1/row level",
        "saved_secondary_metrics": [
            "valid_auroc",
            "valid_episode_auprc",
            "valid_episode_auroc",
        ],
        "optuna_trials": optuna_trials,
        "tune_lookback_days": tune_lookback_days,
        "trials_ok": len(completed_rows),
        "path": str(output_dir / deep_tuning_filename(output_prefix)),
    }
    summary["best_hyperparameters"] = final_params
    with open(deep_output_paths(output_dir, output_prefix)["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _suggest_sequence_hyperparameters(
    trial,
    model_type: str,
    current_lookback_days: int | None,
    current_batch_size: int,
    current_learning_rate: float,
    current_d_model: int,
    current_n_heads: int,
    current_n_layers: int,
    current_dropout: float,
    current_hidden_size: int | None,
    current_bidirectional: bool,
    current_imbalance_strategy: str,
    tune_lookback_days: bool = True,
) -> dict[str, object]:
    """Espai de cerca conservador per a Transformer, LSTM i RNN."""
    if tune_lookback_days:
        lookback = trial.suggest_categorical(
            "lookback_days",
            _options_with_current([5, 7, 10, 14], current_lookback_days),
        )
    else:
        if current_lookback_days is None or int(current_lookback_days) < 1:
            raise ValueError("A fixed previous-day-only Optuna run requires lookback_days >= 1.")
        lookback = int(current_lookback_days)
    batch_size = trial.suggest_categorical(
        "batch_size",
            _options_with_current([64, 128, 256], current_batch_size),
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.05, 0.35)
    imbalance = trial.suggest_categorical(
        "imbalance_strategy",
            _options_with_current(["pos_weight", "focal_loss", "weighted_sampler"], current_imbalance_strategy),
    )

    if model_type == "transformer":
        parelles = []
        for model_dim in _options_with_current([32, 48, 64, 96, 128], current_d_model):
            for heads in _options_with_current([2, 4, 8], current_n_heads):
                if int(model_dim) % int(heads) == 0:
                    parelles.append(f"{int(model_dim)}x{int(heads)}")
        parella = trial.suggest_categorical("d_model_n_heads", sorted(set(parelles)))
        d_model_text, n_heads_text = str(parella).split("x", maxsplit=1)
        d_model = int(d_model_text)
        n_heads = int(n_heads_text)
        hidden_size = current_hidden_size
        bidirectional = current_bidirectional
    else:
        hidden_size = trial.suggest_categorical(
            "recurrent_hidden_size",
            _options_with_current([32, 64, 96, 128], current_hidden_size or current_d_model),
        )
        d_model = hidden_size
        n_heads = current_n_heads
        bidirectional = trial.suggest_categorical(
            "recurrent_bidirectional",
            _options_with_current([False, True], current_bidirectional),
        )

    return {
        "lookback_days": lookback,
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "n_layers": int(n_layers),
        "dropout": float(dropout),
        "recurrent_hidden_size": None if hidden_size is None else int(hidden_size),
        "recurrent_bidirectional": bool(bidirectional),
        "imbalance_strategy": str(imbalance),
    }


def _options_with_current(options: list[object], current: object) -> list[object]:
    """Include the current configured value in the Optuna search space."""
    result = list(options)
    if current not in result:
        result.append(current)
    return result


# ---------------------------------------------------------------------------
# Sequential dataset
# ---------------------------------------------------------------------------


class MultiModalSequenceDataset:
    """Build temporal windows for each eligible day.

    Each item returns the temporal sequence through day D, the padding mask,
    the static variables from day D, and the `next_day_sepsis` label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_temporal: np.ndarray,
        x_static: np.ndarray,
        lookback: int,
        torch,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.x_temporal = torch.tensor(x_temporal, dtype=torch.float32)
        self.x_static = torch.tensor(x_static, dtype=torch.float32)
        self.y = torch.tensor(self.df[TARGET].astype(int).to_numpy(), dtype=torch.float32)
        self.lookback = lookback
        self.torch = torch
        self.starts = _calculate_episode_start(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        start = max(int(self.starts[idx]), idx - self.lookback + 1)
        seq = self.x_temporal[start : idx + 1]
        pad = self.lookback - seq.shape[0]
        if pad > 0:
            # Left padding keeps the last window step aligned with the real day D.
            padding = self.torch.zeros(
                (pad, self.x_temporal.shape[1]),
                dtype=self.torch.float32,
            )
            seq = self.torch.cat([padding, seq], dim=0)
        padding_mask = self.torch.zeros(self.lookback, dtype=self.torch.bool)
        if pad > 0:
            # True marks synthetic positions that the Transformer must ignore.
            padding_mask[:pad] = True
        return seq, padding_mask, self.x_static[idx], self.y[idx], idx


# ---------------------------------------------------------------------------
# Temporal splitting and windows
# ---------------------------------------------------------------------------


def _resolve_max_sequence_len(df_model: pd.DataFrame, lookback_days: int | None) -> int:
    """Return the temporal window size used by the dataset."""
    if lookback_days is not None:
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1 or None to use the full episode.")
        return int(lookback_days)

    max_len = int(df_model.groupby("Episodi").size().max())
    if max_len < 1:
        raise ValueError("There are not enough rows to build temporal sequences.")
    return max_len


def _filter_real_from_start_date(
    df_model: pd.DataFrame,
    real_start_date: str | pd.Timestamp | None,
    enabled: bool,
    date_col: str = "data_index",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """When requested, keep only rows after the cutoff in the real split."""
    if not enabled or real_start_date is None or SPLIT_REAL not in set(df_model["split"]):
        return df_model, {"filter_real_from_start_date": False, "removed_rows": 0}
    if date_col not in df_model.columns:
        raise ValueError(f"The temporal column '{date_col}' is required to filter the real split.")

    dates = pd.to_datetime(df_model[date_col], errors="coerce")
    cutoff = pd.Timestamp(real_start_date).normalize()
    remove_mask = (df_model["split"] == SPLIT_REAL) & (dates < cutoff)
    filtered = df_model.loc[~remove_mask].copy()
    return filtered, {
        "filter_real_from_start_date": True,
        "real_start_date": cutoff.isoformat(),
        "removed_rows": int(remove_mask.sum()),
    }


def _create_final_temporal_split(
    df_model: pd.DataFrame,
    proportions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    date_col: str = "data_index",
    split_unit: str = "patient",
    real_start_date: str | pd.Timestamp | None = "2026-01-01",
    real_overlap_policy: str = REAL_ALL_2026,
) -> tuple[dict[object, str], dict[str, object]]:
    """Create train, valid, test, and real without splitting patients or episodes.

    Units spanning two development periods are purged to
    reduce temporal contamination. The real split can reserve readmissions only,
    new 2026 patients only, or both groups together.
    """
    split_unit = _normalize_split_unit(split_unit)
    real_overlap_policy = normalize_real_policy(real_overlap_policy)
    if date_col not in df_model.columns:
        raise ValueError(f"The temporal column '{date_col}' is required to create the temporal split.")
    if len(proportions) != 3 or any(value <= 0 for value in proportions):
        raise ValueError("Train, validation, and test proportions must be three positive values.")
    if not np.isclose(sum(proportions), 1.0):
        raise ValueError("Train, validation, and test proportions must sum to 1.")

    dates = pd.to_datetime(df_model[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(
            f"Column '{date_col}' contains {int(dates.isna().sum())} missing or invalid dates."
        )

    split_unit_value = _temporal_split_unit(df_model, split_unit)
    unit_label = split_unit if split_unit == "episode" or PATIENT_COL in df_model.columns else "episode"
    df_dates = df_model.assign(_data_split=dates, _split_unit=split_unit_value)
    units = (
        df_dates
        .groupby("_split_unit")["_data_split"]
        .agg(start_date="min", end_date="max")
        .sort_values(["start_date", "end_date"], kind="stable")
    )

    real_start = (
        pd.Timestamp(real_start_date).normalize()
        if real_start_date is not None
        else None
    )
    if real_start is not None and pd.isna(real_start):
        raise ValueError("real_start_date is not a valid date.")

    real_units: set[object] = set()
    excluded_units: set[object] = set()
    if real_start is not None:
        real_mask = select_real_units(
            units,
            policy=real_overlap_policy,
            real_start=real_start,
            start_date_col="start_date",
            end_date_col="end_date",
        )
        real_units = set(units.index[real_mask].tolist())
        excluded_mask = select_excluded_real_units(
            units,
            policy=real_overlap_policy,
            real_start=real_start,
            start_date_col="start_date",
            end_date_col="end_date",
        )
        excluded_units = set(units.index[excluded_mask].tolist()) - real_units

    dev_units = units.loc[~units.index.isin(real_units | excluded_units)].copy()
    unique_start_dates = dev_units["start_date"].drop_duplicates().sort_values().tolist()
    if len(unique_start_dates) < 3:
        raise ValueError(
            f"At least three distinct {unit_label} start dates are required for "
            "train, validation, and test after reserving the real split."
        )

    cumulative_proportions = np.cumsum(proportions[:-1])
    cutoff_indices = [
        min(max(int(np.floor(len(unique_start_dates) * value)), 1), len(unique_start_dates) - 1)
        for value in cumulative_proportions
    ]
    if len(set(cutoff_indices)) != 2:
        raise ValueError("There are not enough distinct dates to build two temporal cutoffs.")
    tall_valid, tall_test = [
        pd.Timestamp(unique_start_dates[index]) for index in cutoff_indices
    ]

    unit_split_map: dict[object, str] = {unit: SPLIT_REAL for unit in real_units}
    temporally_purged_units: list[object] = []
    for unit, row in dev_units.iterrows():
        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"])
        if real_overlap_policy == REAL_NEW_2026:
            if start < tall_valid:
                unit_split_map[unit] = SPLIT_TRAIN
            elif start < tall_test:
                unit_split_map[unit] = SPLIT_VALID
            else:
                unit_split_map[unit] = SPLIT_TEST
        elif end < tall_valid:
            unit_split_map[unit] = SPLIT_TRAIN
        elif start >= tall_valid and end < tall_test:
            unit_split_map[unit] = SPLIT_VALID
        elif start >= tall_test:
            unit_split_map[unit] = SPLIT_TEST
        else:
            temporally_purged_units.append(unit)

    episodes_per_unit = df_dates.groupby("_split_unit")[ID_COL].unique().to_dict()
    split_map: dict[object, str] = {}
    for unit, split in unit_split_map.items():
        for episode in episodes_per_unit[unit]:
            split_map[episode] = split

    counts = pd.Series(split_map).value_counts()
    empty_splits = [split for split in TRAIN_VALID_TEST_SPLITS if int(counts.get(split, 0)) == 0]
    if real_start is not None and int(counts.get(SPLIT_REAL, 0)) == 0:
        empty_splits.append(SPLIT_REAL)
    if empty_splits:
        raise ValueError(
            "Temporal cutoffs left empty splits: "
            f"{', '.join(empty_splits)}. Review real_start_date or the split proportions."
        )

    purged_episodes = [
        episode
        for unit in temporally_purged_units
        for episode in episodes_per_unit[unit]
    ]
    return split_map, {
        "strategy": (
            f"Chronological order by {unit_label}; units that cross "
            "train/valid/test are purged when the real split reserves all "
            "patients with real activity or only readmissions. With "
            f"{REAL_NEW_2026}, the real split contains only new units."
        ),
        "split_unit": unit_label,
        "real_overlap_policy": real_overlap_policy,
        "real_start_date": None if real_start is None else real_start.isoformat(),
        "target_proportions": dict(zip(TRAIN_VALID_TEST_SPLITS, proportions)),
        "cutoff_dates": {
            "validation_start": tall_valid.isoformat(),
            "test_start": tall_test.isoformat(),
        },
        "n_original_units": int(len(units)),
        "n_real_units": int(len(real_units)),
        "n_excluded_units": int(len(excluded_units)),
        "unit_exclusion_reason": (
            f"New 2026 patients excluded under the {REAL_READMITTED_2026} policy"
            if excluded_units
            else None
        ),
        "n_real_readmitted_units": int(
            (units.loc[list(real_units), "start_date"] < real_start).sum()
            if real_units and real_start is not None
            else 0
        ),
        "n_real_new_units": int(
            (units.loc[list(real_units), "start_date"] >= real_start).sum()
            if real_units and real_start is not None
            else 0
        ),
        "n_development_units_with_real_activity": int(
            (dev_units["end_date"] >= real_start).sum()
            if real_start is not None
            else 0
        ),
        "n_development_units": int(len(dev_units)),
        "excluded_units": [str(value) for value in excluded_units],
        "n_temporally_purged_units": int(len(temporally_purged_units)),
        "temporally_purged_units": [str(value) for value in temporally_purged_units],
        "n_original_episodes": int(df_model[ID_COL].nunique()),
        "n_included_episodes": int(len(split_map)),
        "n_temporally_purged_episodes": int(len(purged_episodes)),
        "temporally_purged_episodes": [str(value) for value in purged_episodes],
    }


def _temporal_split_unit(df_model: pd.DataFrame, split_unit: str) -> pd.Series:
    """Return the unit that cannot be split across train/valid/test/real."""
    return _shared_split_unit(
        df_model,
        split_unit,
        missing_patient_label="episode_without_patient",
    )


def _describe_temporal_window(lookback_days: int | None, max_sequence_len: int) -> str:
    """Method text saved to the summary JSON."""
    if lookback_days is None:
        return (
            "For each eligible patient and day, a window is created with all "
            "available previous days from the same episode through the current day; "
            f"shorter sequences are padded up to max_sequence_len={max_sequence_len}."
        )
    return (
        "For each eligible patient and day, a window is created with the last "
        f"{lookback_days} available days; short windows are padded."
    )


def _describe_training_partitions(train_parts: int) -> str:
    """Method text about internal train partitioning."""
    if train_parts <= 1:
        return (
            "Each epoch iterates over the full train set. Internal splitting happens "
            "only through DataLoader minibatches."
        )
    return (
        "Train is split into disjoint episode partitions. Each epoch trains one "
        "partition and epochs rotate over partitions; validation and test are always "
        "computed on the full split. This keeps quick exploratory runs cheaper."
    )


def _create_train_partitions_by_episode(
    train: pd.DataFrame,
    train_parts: int,
    seed: int,
) -> list[np.ndarray]:
    """Split train by episode when partitioned training is requested."""
    if train_parts < 1:
        raise ValueError("train_parts must be >= 1.")

    train_parts = min(int(train_parts), int(train[ID_COL].nunique()))
    if train_parts <= 1:
        return [np.arange(len(train), dtype=int)]

    rng = np.random.default_rng(seed)
    episodes = train[ID_COL].dropna().drop_duplicates().to_numpy()
    rng.shuffle(episodes)

    episode_parts = np.array_split(episodes, train_parts)
    parts_indices: list[np.ndarray] = []
    for partition_episodes in episode_parts:
        mask = train[ID_COL].isin(partition_episodes).to_numpy()
        indices = np.flatnonzero(mask).astype(int)
        if len(indices) > 0:
            parts_indices.append(indices)

    if not parts_indices:
        raise ValueError("Train partitions could not be created.")
    return parts_indices


def _available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Keep candidate variables that are available and not SOFA-derived."""
    return [
        col
        for col in candidates
        if col in df.columns and not _is_sofa_or_availability_derived_column(col)
    ]


def _sofa_or_availability_columns(columns: pd.Index | list[str]) -> list[str]:
    """Return auxiliary SOFA/availability columns detected in the dataframe."""
    return sorted(
        str(col)
        for col in columns
        if _is_sofa_or_availability_derived_column(str(col))
    )


# ---------------------------------------------------------------------------
# Features shared with classic models
# ---------------------------------------------------------------------------


def audit_operational_microbiology(df_model: pd.DataFrame) -> dict[str, object]:
    """Summarize operational microbiology without modifying the dataframe.

    This function only audits temporal microbiology availability. It does not
    re-anchor dates or remove rows, because that decision is made earlier in SQL.
    """
    info: dict[str, object] = {
        "criterion": (
            "Temporal microbiology availability is built in SQL. "
            "Python does not re-anchor or exclude microbiology by default; it only audits "
            "whether positive blood cultures exist before the availability date."
        ),
        "python_handling": "unchanged",
        "present_microbiology_variables": sorted(
            col for col in MICROBIOLOGY_COLUMNS if col in df_model.columns
        ),
        "default_excluded_microbiology_variables": [],
    }
    required = {
        ID_COL,
        "data_index",
        "hemocultiu_positiu",
        "hemocultiu_positiu_data_extraccio",
        "hemocultiu_temps_positivitat_h",
    }
    missing = sorted(required - set(df_model.columns))
    if missing:
        info["blood_culture_availability_audit"] = "not_available"
        info["blood_culture_audit_reason"] = (
            "Missing columns for the availability audit: " + ", ".join(missing)
        )
        return info

    positive_rows = pd.to_numeric(
        df_model["hemocultiu_positiu"],
        errors="coerce",
    ).fillna(0).eq(1)
    extraction_dates = pd.to_datetime(
        df_model["hemocultiu_positiu_data_extraccio"],
        errors="coerce",
    )
    time_to_positive = pd.to_numeric(
        df_model["hemocultiu_temps_positivitat_h"],
        errors="coerce",
    )
    row_dates = pd.to_datetime(df_model["data_index"], errors="coerce").dt.normalize()
    availability_dates = (
        extraction_dates
        + pd.to_timedelta(time_to_positive.where(time_to_positive >= 0), unit="h")
    ).dt.normalize()
    anchored_positive_rows = (
        positive_rows
        & extraction_dates.notna()
        & time_to_positive.notna()
        & (time_to_positive >= 0)
        & availability_dates.notna()
    )
    positive_before_availability = (
        anchored_positive_rows
        & row_dates.notna()
        & (row_dates < availability_dates)
    )
    positive_without_anchor = positive_rows & ~anchored_positive_rows

    info["blood_culture_availability_audit"] = "calculated"
    info["positive_blood_culture_rows"] = int(positive_rows.sum())
    info["positive_blood_culture_rows_before_availability"] = int(
        positive_before_availability.sum()
    )
    info["positive_blood_culture_rows_without_anchor"] = int(
        positive_without_anchor.sum()
    )
    return info


def _filter_microbiology_columns(
    columns: list[str],
    exclude_microbiology: bool,
) -> list[str]:
    """Exclude microbiology variables when a no-microbiology analysis is requested."""
    if not exclude_microbiology:
        return columns
    return [col for col in columns if col not in MICROBIOLOGY_COLUMNS]


def add_admission_diagnosis_features(df_model: pd.DataFrame) -> pd.DataFrame:
    """Add categorical representations of the diagnosis available at admission.

    Classic models also use it to keep the same diagnosis coding
    across pipelines.
    """
    if "diagnostic_ingres" not in df_model.columns:
        return df_model

    df_model = df_model.copy()
    codi = _normalize_icd_diagnosis(df_model["diagnostic_ingres"])
    df_model["diagnostic_ingres_codi"] = codi
    df_model["diagnostic_ingres_prefix3"] = codi.map(_diagnosis_icd_prefix3)
    df_model["grup_diagnostic_ingres"] = codi.map(_classify_icd_diagnosis)
    return df_model


def _normalize_icd_diagnosis(series: pd.Series) -> pd.Series:
    """Clean ICD/CIE codes stored as text."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA, "<NA>": pd.NA})
    )


def _diagnosis_icd_prefix3(codi: str | pd.NA) -> str | pd.NA:
    """Extract the three-character diagnosis prefix."""
    if pd.isna(codi):
        return pd.NA
    net = "".join(ch for ch in str(codi).upper() if ch.isalnum())
    if len(net) < 3:
        return pd.NA
    return net[:3]


def _classify_icd_diagnosis(codi: str | pd.NA) -> str | pd.NA:
    """Group ICD/CIE codes into broad clinical families."""
    if pd.isna(codi):
        return pd.NA
    codi_net = "".join(ch for ch in str(codi).upper() if ch.isalnum())
    if not codi_net:
        return pd.NA

    prefix = codi_net[:1]
    if prefix in {"A", "B"}:
        return "Infeccioses"
    if prefix == "C" or codi_net.startswith(("D0", "D1", "D2", "D3", "D4")):
        return "Oncologiques"
    if prefix == "D":
        return "Hematologiques i immunologiques"
    if prefix == "E":
        return "Endocrines i metaboliques"
    if prefix == "F":
        return "Salut mental"
    if prefix == "G":
        return "Neurologiques"
    if prefix == "H":
        return "Oftalmologiques i ORL"
    if prefix == "I":
        return "Cardiovasculars"
    if prefix == "J":
        return "Respiratories"
    if prefix == "K":
        return "Digestives"
    if prefix == "L":
        return "Dermatologiques"
    if prefix == "M":
        return "Musculoesqueletiques"
    if prefix == "N":
        return "Genitourinaries"
    if prefix == "O":
        return "Obstetriques"
    if prefix == "P":
        return "Perinatals"
    if prefix == "Q":
        return "Congenites"
    if prefix == "R":
        return "Symptoms and signs"
    if prefix in {"S", "T"}:
        return "Lesions i traumatismes"
    if prefix in {"V", "W", "X", "Y"}:
        return "Causes externes"
    if prefix == "Z":
        return "Factors sanitaris i seguiment"
    return "Altres"


def _preprocessor_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep only identifiers, target, and variables entering the preprocessor."""
    base = ["Episodi", TARGET, ELIGIBILITY_COL]
    if "split" in df.columns:
        base.append("split")
    selected = [col for col in base + columns if col in df.columns]
    return df.loc[:, selected].copy()


def _create_sequence_model_24h(
    model_type: str,
    n_temporal_features: int,
    n_static_features: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    max_len: int,
    recurrent_hidden_size: int | None,
    recurrent_bidirectional: bool,
    nn,
    torch,
):
    """Create the requested PyTorch network with a shared interface."""
    if model_type == "transformer":
        return TransformerSepsis24h(
            n_temporal_features=n_temporal_features,
            n_static_features=n_static_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len,
            nn=nn,
            torch=torch,
        )
    return RecurrentSepsis24h(
        model_type=model_type,
        n_temporal_features=n_temporal_features,
        n_static_features=n_static_features,
        hidden_size=recurrent_hidden_size or d_model,
        n_layers=n_layers,
        dropout=dropout,
        bidirectional=recurrent_bidirectional,
        nn=nn,
        torch=torch,
    )


def _sequence_model_name(model_type: str) -> str:
    """Human-readable model name for saved summaries."""
    if model_type == "lstm":
        return "LSTM multimodal PyTorch"
    if model_type == "rnn":
        return "RNN multimodal PyTorch"
    return "TransformerEncoder multimodal PyTorch"


def _describe_sequence_architecture(model_type: str) -> str:
    """Short architecture description for the summary JSON."""
    if model_type == "lstm":
        return (
            "Temporal LSTM branch for vitals, laboratory values, treatments, "
            "and daily microbiology; dense static branch for age, sex, "
            "comorbidities, admission diagnosis, and admission type; concatenation fusion."
        )
    if model_type == "rnn":
        return (
            "Simple temporal RNN branch for vitals, laboratory values, treatments, "
            "and daily microbiology; dense static branch for age, sex, "
            "comorbidities, admission diagnosis, and admission type; concatenation fusion."
        )
    return (
        "Temporal Transformer branch for vitals, laboratory values, treatments "
        "and daily microbiology; dense static branch for age, sex, "
        "comorbidities, admission diagnosis, and admission type; concatenation fusion."
    )


# ---------------------------------------------------------------------------
# PyTorch architectures
# ---------------------------------------------------------------------------


class TransformerSepsis24h:
    """Factory for a temporal + static multimodal Transformer."""

    def __new__(
        cls,
        n_temporal_features,
        n_static_features,
        d_model,
        n_heads,
        n_layers,
        dropout,
        max_len,
        nn,
        torch,
    ):
        class _TransformerSepsis24h(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                static_hidden = max(d_model // 2, 8)
                self.input_projection = nn.Linear(n_temporal_features, d_model)
                self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
                nn.init.trunc_normal_(self.positional_embedding, std=0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.static_encoder = nn.Sequential(
                    nn.LayerNorm(n_static_features),
                    nn.Linear(n_static_features, static_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(static_hidden, d_model),
                    nn.GELU(),
                )
                self.classifier = nn.Sequential(
                    nn.LayerNorm(d_model * 2),
                    nn.Linear(d_model * 2, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )

            def forward(self, x, padding_mask, static_x):
                # Learnable temporal positions distinguish older days from the
                # current day even when padding fills the start of the window.
                h = self.input_projection(x) + self.positional_embedding
                h = self.encoder(h, src_key_padding_mask=padding_mask)
                # Because padding is on the left, h[:, -1, :] represents day D.
                current_day = h[:, -1, :]
                static_repr = self.static_encoder(static_x)
                # Multimodal fusion: recent temporal summary + static admission context.
                combined = torch.cat([current_day, static_repr], dim=1)
                return self.classifier(combined).squeeze(-1)

        return _TransformerSepsis24h()


class RecurrentSepsis24h:
    """Shared factory for RNN and LSTM models."""

    def __new__(
        cls,
        model_type,
        n_temporal_features,
        n_static_features,
        hidden_size,
        n_layers,
        dropout,
        bidirectional,
        nn,
        torch,
    ):
        class _RecurrentSepsis24h(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                recurrent_dropout = dropout if n_layers > 1 else 0.0
                recurrent_cls = nn.LSTM if model_type == "lstm" else nn.RNN
                recurrent_kwargs = {}
                if model_type == "rnn":
                    recurrent_kwargs["nonlinearity"] = "tanh"
                self.recurrent = recurrent_cls(
                    input_size=n_temporal_features,
                    hidden_size=hidden_size,
                    num_layers=n_layers,
                    dropout=recurrent_dropout,
                    batch_first=True,
                    bidirectional=bidirectional,
                    **recurrent_kwargs,
                )
                recurrent_out = hidden_size * (2 if bidirectional else 1)
                static_hidden = max(hidden_size // 2, 8)
                self.static_encoder = nn.Sequential(
                    nn.LayerNorm(n_static_features),
                    nn.Linear(n_static_features, static_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(static_hidden, recurrent_out),
                    nn.GELU(),
                )
                self.classifier = nn.Sequential(
                    nn.LayerNorm(recurrent_out * 2),
                    nn.Linear(recurrent_out * 2, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )

            def forward(self, x, padding_mask, static_x):
                # The dataset left-pads so the Transformer can
                # read the last position directly. RNN/LSTM models, instead,
                # must not process those zeros as if they were real days.
                x_packed = self._pack_left_padded_sequences(x, padding_mask)
                _, hidden = self.recurrent(x_packed)
                current_day = self._last_hidden_state(hidden)
                static_repr = self.static_encoder(static_x)
                combined = torch.cat([current_day, static_repr], dim=1)
                return self.classifier(combined).squeeze(-1)

            def _pack_left_padded_sequences(self, x, padding_mask):
                lengths = (~padding_mask).sum(dim=1).clamp(min=1)
                sequences = [
                    x_i[mask_i.logical_not()]
                    for x_i, mask_i in zip(x, padding_mask)
                ]
                padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
                packed = nn.utils.rnn.pack_padded_sequence(
                    padded,
                    lengths.detach().cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                return packed

            def _last_hidden_state(self, hidden):
                h_n = hidden[0] if model_type == "lstm" else hidden
                if bidirectional:
                    forward_last = h_n[-2]
                    backward_last = h_n[-1]
                    return torch.cat([forward_last, backward_last], dim=1)
                return h_n[-1]

        return _RecurrentSepsis24h()


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------


def _train_epoch(model, loader, criterion, optimizer, device, torch) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    losses: list[float] = []
    for seq, padding_mask, static_x, y, _ in loader:
        seq = seq.to(device)
        padding_mask = padding_mask.to(device)
        static_x = static_x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(seq, padding_mask, static_x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _create_train_loader(
    train_ds,
    train: pd.DataFrame,
    indices_part: np.ndarray,
    batch_size: int,
    imbalance_strategy: str,
    torch,
    DataLoader,
    WeightedRandomSampler,
):
    """Build the train DataLoader, with a sampler when class balancing is needed."""
    subset = torch.utils.data.Subset(train_ds, indices_part.tolist())
    y_part = train.iloc[indices_part][TARGET].astype(int).to_numpy()
    sampler = _create_train_sampler(
        y_train=y_part,
        strategy=imbalance_strategy,
        torch=torch,
        WeightedRandomSampler=WeightedRandomSampler,
    )
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )


def _create_train_sampler(
    y_train: np.ndarray,
    strategy: str,
    torch,
    WeightedRandomSampler,
):
    """Create a balanced sampler only when `strategy` requests it."""
    if strategy != "weighted_sampler":
        return None

    n_pos = max(float(y_train.sum()), 1.0)
    n_neg = max(float(len(y_train) - y_train.sum()), 1.0)
    weights = np.where(y_train == 1, 1.0 / n_pos, 1.0 / n_neg)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def _create_criterion(strategy: str, pos: float, neg: float, nn, torch, device):
    """Return the loss function for the imbalance strategy."""
    if strategy == "none":
        return nn.BCEWithLogitsLoss()
    if strategy == "pos_weight":
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if strategy == "weighted_sampler":
        return nn.BCEWithLogitsLoss()
    if strategy == "focal_loss":
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        return FocalLossWithLogits(pos_weight=pos_weight, gamma=2.0, torch=torch)
    raise ValueError(
        "imbalance_strategy must be: 'none', 'pos_weight', "
        "'weighted_sampler', or 'focal_loss'"
    )


def _describe_imbalance_strategy(strategy: str) -> str:
    """Method text describing how class imbalance was handled."""
    if strategy == "none":
        return "No balancing; plain BCEWithLogitsLoss."
    if strategy == "pos_weight":
        return "BCEWithLogitsLoss with pos_weight = negatives / positives computed only on train."
    if strategy == "weighted_sampler":
        return "WeightedRandomSampler only on train; validation and test keep the real prevalence."
    if strategy == "focal_loss":
        return "Focal loss with pos_weight = negatives / positives and gamma=2 computed only on train."
    return strategy


class FocalLossWithLogits:
    """Binary focal loss for cohorts with a minority positive class."""

    def __init__(self, pos_weight, gamma: float, torch) -> None:
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.torch = torch

    def __call__(self, logits, targets):
        torch = self.torch
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        focal = (1 - pt).pow(self.gamma) * bce
        return focal.mean()


def _predict_loader(model, loader, device, torch) -> np.ndarray:
    """Compute risk probabilities for a DataLoader without updating weights."""
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for seq, padding_mask, static_x, _, _ in loader:
            seq = seq.to(device)
            padding_mask = padding_mask.to(device)
            static_x = static_x.to(device)
            logits = model(seq, padding_mask, static_x)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            preds.append(prob)
    return np.concatenate(preds) if preds else np.array([])


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _save_temporal_model_outputs(
    output_dir: Path,
    summary: dict[str, object],
    model,
    temporal_preprocessor,
    static_preprocessor,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    real: pd.DataFrame,
    pred_train: np.ndarray,
    pred_valid: np.ndarray,
    pred_test: np.ndarray,
    pred_real: np.ndarray,
    torch,
    output_prefix: str,
) -> None:
    """Save summary, predictions, comparable metrics, and model weights.

    The output folder represents one concrete run (model + policy).
    This keeps file names simple and avoids scattered outputs at the root level.
    """
    paths = deep_output_paths(output_dir, output_prefix)
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    predictions = pd.concat(
        [
            _prediction_rows(train, pred_train, output_prefix),
            _prediction_rows(valid, pred_valid, output_prefix),
            _prediction_rows(test, pred_test, output_prefix),
        ]
        + (
            [_prediction_rows(real, pred_real, output_prefix)]
            if not real.empty
            else []
        ),
        ignore_index=True,
    )
    predictions.to_csv(paths["predictions"], index=False)

    pd.DataFrame(_comparable_metric_rows(summary["metrics"])).to_csv(
        paths["comparable_metrics"],
        index=False,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "temporal_feature_names": temporal_preprocessor.feature_names,
            "static_feature_names": static_preprocessor.feature_names,
            "temporal_preprocessor": {
                "numeric_columns": temporal_preprocessor.numeric_columns,
                "categorical_levels": temporal_preprocessor.categorical_levels,
                "medians": temporal_preprocessor.medians,
                "means": temporal_preprocessor.means,
                "stds": temporal_preprocessor.stds,
                "max_missing_ratio": temporal_preprocessor.max_missing_ratio,
                "excluded_high_missing_columns": temporal_preprocessor.excluded_high_missing_columns,
                "missing_ratios": temporal_preprocessor.missing_ratios,
            },
            "static_preprocessor": {
                "numeric_columns": static_preprocessor.numeric_columns,
                "categorical_levels": static_preprocessor.categorical_levels,
                "medians": static_preprocessor.medians,
                "means": static_preprocessor.means,
                "stds": static_preprocessor.stds,
                "max_missing_ratio": static_preprocessor.max_missing_ratio,
                "excluded_high_missing_columns": static_preprocessor.excluded_high_missing_columns,
                "missing_ratios": static_preprocessor.missing_ratios,
            },
            "summary": summary,
        },
        paths["model"],
    )

    pd.DataFrame(
        {
            "branch": ["temporal"] * len(temporal_preprocessor.excluded_high_missing_columns)
            + ["static"] * len(static_preprocessor.excluded_high_missing_columns),
            "variable": temporal_preprocessor.excluded_high_missing_columns
            + static_preprocessor.excluded_high_missing_columns,
            "train_missing_pct": [
                round(100 * temporal_preprocessor.missing_ratios[col], 3)
                for col in temporal_preprocessor.excluded_high_missing_columns
            ]
            + [
                round(100 * static_preprocessor.missing_ratios[col], 3)
                for col in static_preprocessor.excluded_high_missing_columns
            ],
        }
    ).sort_values("train_missing_pct", ascending=False).to_csv(
        paths["excluded_missing_variables"],
        index=False,
    )


def _prediction_rows(df_split: pd.DataFrame, pred: np.ndarray, output_prefix: str) -> pd.DataFrame:
    """Build prediction CSV rows for one split."""
    cols = [
        col
        for col in ["Episodi", "data_index", "dia_relatiu", TARGET, "split"]
        if col in df_split.columns
    ]
    out = df_split[cols].copy()
    out = out.rename(columns={"dia_relatiu": "relative_day"})
    out[_prediction_column(output_prefix)] = pred
    return out


def _prediction_column(output_prefix: str) -> str:
    """Stable risk-column name based on the output prefix."""
    if output_prefix.startswith("transformer"):
        return "sepsis_risk_24h_transformer"
    safe_prefix = "".join(ch if ch.isalnum() else "_" for ch in output_prefix).strip("_")
    return f"sepsis_risk_24h_{safe_prefix}"


def _comparable_metric_rows(metrics: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """Flatten JSON metrics into tabular rows."""
    rows: list[dict[str, object]] = []
    for level, suffix in (
        (LEVEL_NEXT_DAY, ""),
        (LEVEL_EPISODE, "_episode"),
    ):
        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST, SPLIT_REAL):
            key = f"{split}{suffix}"
            if key not in metrics:
                continue
            values = metrics[key]
            rows.append(
                {
                    "level": level,
                    "split": split,
                    "n": values["n"],
                    "positives": values["positives"],
                    "prevalence": values["prevalence"],
                    "auroc": values["auroc"],
                    "auprc": values["auprc"],
                    "auprc_lift": values.get("auprc_lift"),
                    "threshold": values["threshold"],
                    "accuracy": values["accuracy"],
                    "sensitivity": values["sensitivity"],
                    "specificity": values["specificity"],
                    "ppv": values["ppv"],
                    "npv": values["npv"],
                    "f1": values["f1"],
                    "tp": values["tp"],
                    "tn": values["tn"],
                    "fp": values["fp"],
                    "fn": values["fn"],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Utilitats petites
# ---------------------------------------------------------------------------


def _calculate_episode_start(df: pd.DataFrame) -> np.ndarray:
    """Mark the starting index of each episode in a sorted dataframe."""
    starts = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("Episodi", sort=False).indices.items():
        pos = np.asarray(idx)
        starts[pos] = int(pos.min())
    return starts


def _import_torch():
    """Import PyTorch only when a neural model must actually be trained."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install it with: "
            "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch, nn, DataLoader, WeightedRandomSampler


def _set_seed(seed: int, torch) -> None:
    """Set NumPy and PyTorch seeds to improve reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _float(value) -> float:
    """Convert possibly null values to float or NaN."""
    return float(value) if value is not None and not math.isnan(float(value)) else float("nan")




