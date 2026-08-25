"""Predicción del MOID Orbital (Regresión y Clasificación Binaria).

Este script evalúa la capacidad de predecir el MOID (Minimum Orbit Intersection Distance)
y clasificar la condición geométrica MOID <= 0.05 au a partir de características
observacionales tempranas de aproximaciones cercanas (post_discovery == 1).

Uso (desde la raíz del repo):
    python scripts/predict_moid.py
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Asegurar codificación utf-8 en Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(RAIZ, "data")
DIR_FIG = os.path.join(RAIZ, "results", "figures")
DIR_TAB = os.path.join(RAIZ, "results", "tables")

os.makedirs(DIR_FIG, exist_ok=True)
os.makedirs(DIR_TAB, exist_ok=True)

SEED = 42
UMBRAL_MOID = 0.05  # au


def resolver_ruta_cad():
    ruta_std = os.path.join(DIR_DATA, "close_approaches.csv")
    if os.path.exists(ruta_std):
        return ruta_std
    snaps = sorted(glob.glob(os.path.join(DIR_DATA, "close_approaches_v*.csv")))
    if snaps:
        return snaps[-1]
    sys.exit("No se encontró el dataset close_approaches.csv ni ninguna versión snapshot en data/")


def cargar_y_agregar():
    ruta_csv = resolver_ruta_cad()
    print(f"Cargando dataset desde: {os.path.basename(ruta_csv)}")
    df = pd.read_csv(ruta_csv)

    if "post_discovery" not in df.columns:
        sys.exit("El CSV no contiene la columna 'post_discovery'.")

    # Filtrar solo eventos realmente observados
    obs = df[df["post_discovery"] == 1].copy()

    # Calcular incertidumbre observacional
    obs["dist_unc"] = obs["CA DistanceNominal (au)"] - obs["CA DistanceMinimum (au)"]

    # Agregar por objeto
    obj = obs.groupby("Object").agg(
        distnom_min=("CA DistanceNominal (au)", "min"),
        distmin_min=("CA DistanceMinimum (au)", "min"),
        vinf_max=("V infinity(km/s)", "max"),
        vinf_med=("V infinity(km/s)", "median"),
        H_obs=("H(mag)", "min"),
        n_appro=("Object", "size"),
        dist_unc_med=("dist_unc", "median"),
        moid=("MOID (au)", "max"),
        pha=("PHA_official", "max"),
        first_obs_year=("first_obs_year", "max"),
    ).reset_index()

    # Limpiar missing valores en objetivo y características clave
    cols_check = ["moid", "distnom_min", "vinf_max", "H_obs"]
    obj = obj.dropna(subset=cols_check).copy()
    obj["is_moid_hazardous"] = (obj["moid"] <= UMBRAL_MOID).astype(int)

    print(f"Objetos procesados: {len(obj):,} | MOID <= 0.05 au: {obj['is_moid_hazardous'].sum():,} ({100*obj['is_moid_hazardous'].mean():.1f}%)", flush=True)
    return obj


def evaluar_regresion_moid(obj):
    print("\n" + "=" * 70)
    print("1. REGRESIÓN CONTINUA DEL MOID (Target: 'moid' en UA)")
    print("=" * 70)

    features = ["distnom_min", "vinf_max", "H_obs", "n_appro", "dist_unc_med"]
    X = obj[features].values
    y = obj["moid"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # Baseline ingenuo: Usar distnom_min directamente como predicción de MOID
    r2_naive = r2_score(y, obj["distnom_min"].values)
    mae_naive = mean_absolute_error(y, obj["distnom_min"].values)
    rmse_naive = np.sqrt(mean_squared_error(y, obj["distnom_min"].values))

    print(f"Baseline ingenuo (distnom_min directo): R2 = {r2_naive:.4f} | MAE = {mae_naive:.4f} au | RMSE = {rmse_naive:.4f} au")

    if HAS_XGB:
        model_reg = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=SEED, n_jobs=-1)
        model_cls = xgb.XGBClassifier(n_estimators=50, scale_pos_weight=(len(y)-y.sum())/y.sum(), learning_rate=0.1, max_depth=5, random_state=SEED, n_jobs=-1)
        gb_name = "XGBoost"
    else:
        model_reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=SEED)
        model_cls = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=SEED)
        gb_name = "GradientBoosting"

    modelos = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1),
        gb_name: model_reg,
    }

    resultados = {}
    preds_oof = {}

    for nombre, model in modelos.items():
        r2_list, mae_list, rmse_list = [], [], []
        oof_preds = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            oof_preds[val_idx] = preds

            r2_list.append(r2_score(y_va, preds))
            mae_list.append(mean_absolute_error(y_va, preds))
            rmse_list.append(np.sqrt(mean_squared_error(y_va, preds)))

        resultados[nombre] = {
            "R2_mean": float(np.mean(r2_list)),
            "MAE_mean": float(np.mean(mae_list)),
            "RMSE_mean": float(np.mean(rmse_list)),
        }
        preds_oof[nombre] = oof_preds
        print(f"Modelo {nombre:<15}: R2 = {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f} | MAE = {np.mean(mae_list):.4f} au | RMSE = {np.mean(rmse_list):.4f} au")

    # Gráfica de Scatter: MOID Real vs MOID Predicho (GBDT)
    plt.figure(figsize=(7, 6))
    plt.scatter(y, preds_oof[gb_name], alpha=0.3, s=12, color="#2b5c8f", label=f"Predicciones {gb_name} OOF")
    plt.plot([0, y.max()], [0, y.max()], 'r--', label="Línea Perfecta y=x")
    plt.axvline(UMBRAL_MOID, color='orange', linestyle=':', label="Umbral 0.05 au")
    plt.axhline(UMBRAL_MOID, color='orange', linestyle=':')
    plt.xlabel("MOID Orbital Real (au)")
    plt.ylabel("MOID Predicho (au)")
    plt.title(f"Regresión del MOID Orbital (R² = {resultados[gb_name]['R2_mean']:.3f}, MAE = {resultados[gb_name]['MAE_mean']:.4f} au)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(DIR_FIG, "pred_vs_actual_moid.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Figura de regresión guardada en: {fig_path}")

    return resultados, preds_oof


def evaluar_clasificacion_moid(obj):
    print("\n" + "=" * 70)
    print("2. CLASIFICACIÓN BINARIA DEL UMBRAL PELIGROSO (Target: MOID <= 0.05 au)")
    print("=" * 70)

    features = ["distnom_min", "vinf_max", "H_obs", "n_appro", "dist_unc_med"]
    X = obj[features].values
    y = obj["is_moid_hazardous"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Proxy directo: distnom_min <= 0.05
    proxy_pred = (obj["distnom_min"] <= UMBRAL_MOID).astype(int).values
    prec_proxy = precision_score(y, proxy_pred, zero_division=0)
    rec_proxy = recall_score(y, proxy_pred)
    f2_proxy = fbeta_score(y, proxy_pred, beta=2)

    print(f"Proxy observacional directo (distnom_min <= 0.05): Precisión = {prec_proxy:.4f} | Recall = {rec_proxy:.4f} | F2 = {f2_proxy:.4f}")

    if HAS_XGB:
        model_cls = xgb.XGBClassifier(n_estimators=50, scale_pos_weight=(len(y)-y.sum())/y.sum(), learning_rate=0.1, max_depth=5, random_state=SEED, n_jobs=-1)
        gb_name = "XGBoost"
    else:
        model_cls = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=SEED)
        gb_name = "GradientBoosting"

    modelos = {
        "Regresión Logística": LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=SEED, n_jobs=-1),
        gb_name: model_cls,
    }

    resultados = {}
    probs_oof = {}

    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for nombre, model in modelos.items():
        f2_list, prec_list, rec_list, roc_list, pr_list = [], [], [], [], []
        oof_probs = np.zeros(len(y))

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = probs

            # Selección de umbral óptimo F2
            preds = (probs >= 0.35).astype(int)

            f2_list.append(fbeta_score(y_va, preds, beta=2))
            prec_list.append(precision_score(y_va, preds, zero_division=0))
            rec_list.append(recall_score(y_va, preds))
            roc_list.append(roc_auc_score(y_va, probs))

        resultados[nombre] = {
            "F2_mean": float(np.mean(f2_list)),
            "Precision_mean": float(np.mean(prec_list)),
            "Recall_mean": float(np.mean(rec_list)),
            "ROC_AUC_mean": float(np.mean(roc_list)),
        }
        probs_oof[nombre] = oof_probs
        print(f"Modelo {nombre:<20}: F2 = {np.mean(f2_list):.4f} | Prec = {np.mean(prec_list):.4f} | Recall = {np.mean(rec_list):.4f} | ROC-AUC = {np.mean(roc_list):.4f}")

        # Curva ROC
        fpr, tpr, _ = roc_curve(y, oof_probs)
        ax1.plot(fpr, tpr, label=f"{nombre} (AUC = {resultados[nombre]['ROC_AUC_mean']:.3f})")

        # Curva Precision-Recall
        prec_curve, rec_curve, _ = precision_recall_curve(y, oof_probs)
        ax2.plot(rec_curve, prec_curve, label=f"{nombre}")

    ax1.plot([0, 1], [0, 1], 'k--', label="Azar")
    ax1.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax1.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    ax1.set_title("Curva ROC — Clasificación MOID <= 0.05 au")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(rec_proxy, prec_proxy, 'ro', markersize=8, label="Proxy observacional directo")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precisión")
    ax2.set_title("Curva Precision-Recall — Clasificación MOID <= 0.05 au")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(DIR_FIG, "moid_classification_roc.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Figura de clasificación guardada en: {fig_path}")

    return resultados


def evaluar_holdout_temporal(obj):
    print("\n" + "=" * 70)
    print("3. VALIDACIÓN TEMPORAL (Hold-out por fecha de descubrimiento)")
    print("=" * 70)

    train_mask = (obj["first_obs_year"] <= 2014)
    test_mask = (obj["first_obs_year"] >= 2015)

    df_tr = obj[train_mask].copy()
    df_te = obj[test_mask].copy()

    print(f"Entrenamiento (descubiertos <= 2014): {len(df_tr):,} objetos (MOID<=0.05: {100*df_tr['is_moid_hazardous'].mean():.1f}%)")
    print(f"Evaluación (descubiertos >= 2015)   : {len(df_te):,} objetos (MOID<=0.05: {100*df_te['is_moid_hazardous'].mean():.1f}%)")

    features = ["distnom_min", "vinf_max", "H_obs", "n_appro", "dist_unc_med"]
    X_tr, y_tr_reg, y_tr_cls = df_tr[features].values, df_tr["moid"].values, df_tr["is_moid_hazardous"].values
    X_te, y_te_reg, y_te_cls = df_te[features].values, df_te["moid"].values, df_te["is_moid_hazardous"].values

    if HAS_XGB:
        gb_reg = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=SEED, n_jobs=-1)
        gb_cls = xgb.XGBClassifier(n_estimators=50, scale_pos_weight=(len(y_tr_cls)-y_tr_cls.sum())/y_tr_cls.sum(), learning_rate=0.1, max_depth=5, random_state=SEED, n_jobs=-1)
        gb_name = "XGBoost"
    else:
        gb_reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=SEED)
        gb_cls = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=SEED)
        gb_name = "GradientBoosting"

    gb_reg.fit(X_tr, y_tr_reg)
    preds_reg = gb_reg.predict(X_te)

    r2_temp = r2_score(y_te_reg, preds_reg)
    mae_temp = mean_absolute_error(y_te_reg, preds_reg)
    print(f"Regresión {gb_name} en Hold-out Temporal (>=2015): R2 = {r2_temp:.4f} | MAE = {mae_temp:.4f} au")

    gb_cls.fit(X_tr, y_tr_cls)
    probs_cls = gb_cls.predict_proba(X_te)[:, 1]
    preds_cls = (probs_cls >= 0.35).astype(int)

    f2_temp = fbeta_score(y_te_cls, preds_cls, beta=2)
    prec_temp = precision_score(y_te_cls, preds_cls, zero_division=0)
    rec_temp = recall_score(y_te_cls, preds_cls)

    print(f"Clasificación {gb_name} en Hold-out Temporal (>=2015): F2 = {f2_temp:.4f} | Prec = {prec_temp:.4f} | Recall = {rec_temp:.4f}")

    return {
        "Regression_Holdout_R2": float(r2_temp),
        "Regression_Holdout_MAE": float(mae_temp),
        "Classification_Holdout_F2": float(f2_temp),
        "Classification_Holdout_Precision": float(prec_temp),
        "Classification_Holdout_Recall": float(rec_temp),
    }


def main():
    obj = cargar_y_agregar()
    res_reg, preds_reg = evaluar_regresion_moid(obj)
    res_cls = evaluar_clasificacion_moid(obj)
    res_temp = evaluar_holdout_temporal(obj)

    resumen = {
        "Regression": res_reg,
        "Classification": res_cls,
        "TemporalHoldout": res_temp,
    }

    ruta_json = os.path.join(DIR_TAB, "moid_prediction_summary.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"✔ Resultados guardados exitosamente en {ruta_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
