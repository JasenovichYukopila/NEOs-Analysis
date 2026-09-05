"""Validación física de las predicciones del MOID Orbital.

Este script evalúa si el modelo de regresión para estimar el MOID respeta la
restricción geométrica fundamental: MOID <= distnom_min. Además, propone un
modelo restringido y evalúa la importancia de las variables (SHAP / Impureza)
para entender si el modelo se apoya en características cinemáticas reales o 
en sesgos observacionales (tamaño H).
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost no está instalado. Instálalo con 'pip install xgboost'.")
    sys.exit(1)

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
        H_obs=("H(mag)", "min"),
        n_appro=("Object", "size"),
        dist_unc_med=("dist_unc", "median"),
        moid=("MOID (au)", "max"),
        pha=("PHA_official", "max"),
    ).reset_index()

    cols_check = ["moid", "distnom_min", "vinf_max", "H_obs"]
    obj = obj.dropna(subset=cols_check).copy()

    print(f"Objetos procesados para validación física: {len(obj):,}")
    return obj


def validacion_fisica(obj):
    print("\n" + "=" * 70)
    print("VALIDACIÓN FÍSICA Y RESTRICCIÓN GEOMÉTRICA DEL MOID")
    print("=" * 70)

    features = ["distnom_min", "vinf_max", "H_obs", "n_appro", "dist_unc_med"]
    X = obj[features].values
    y = obj["moid"].values
    
    # Restricción física fundamental: MOID <= distnom_min
    distnom_min_vals = obj["distnom_min"].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=SEED, n_jobs=-1)
    
    oof_preds = np.zeros(len(y))
    
    print("Entrenando XGBoost y obteniendo predicciones Out-of-Fold...")
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_va)

    # 1. Analizar violaciones geométricas
    # Una predicción es físicamente imposible si predice un MOID mayor que la distancia a la que lo observamos
    violaciones_mask = oof_preds > distnom_min_vals
    n_violaciones = violaciones_mask.sum()
    pct_violaciones = n_violaciones / len(y) * 100
    
    print(f"\n[Test Geométrico] Violaciones (Predicción > distnom_min): {n_violaciones:,} ({pct_violaciones:.2f}%)")
    
    # 2. Modelo Restringido Matemáticamente
    # Forzamos que la predicción nunca sea mayor a la distancia nominal observada mínima
    oof_preds_restringido = np.minimum(oof_preds, distnom_min_vals)
    
    # 3. Comparar Métricas
    r2_unconstrained = r2_score(y, oof_preds)
    mae_unconstrained = mean_absolute_error(y, oof_preds)
    
    r2_constrained = r2_score(y, oof_preds_restringido)
    mae_constrained = mean_absolute_error(y, oof_preds_restringido)
    
    print("\n[Métricas de Rendimiento]")
    print(f"Modelo Libre:       R2 = {r2_unconstrained:.4f} | MAE = {mae_unconstrained:.4f} au")
    print(f"Modelo Restringido: R2 = {r2_constrained:.4f} | MAE = {mae_constrained:.4f} au")
    
    mejora_mae = (mae_unconstrained - mae_constrained) / mae_unconstrained * 100
    print(f"Mejora en el error (MAE) al forzar la física: {mejora_mae:.2f}%")

    # 4. Importancia de las Características
    model.fit(X, y)
    importances = model.feature_importances_
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Scatter plot de violaciones
    plt.subplot(1, 2, 1)
    # Puntos que cumplen la física
    plt.scatter(distnom_min_vals[~violaciones_mask], oof_preds[~violaciones_mask], 
                alpha=0.3, s=8, color="#2b5c8f", label="Físicamente posible")
    # Puntos que violan la física
    plt.scatter(distnom_min_vals[violaciones_mask], oof_preds[violaciones_mask], 
                alpha=0.4, s=8, color="#d9534f", label="Violación geométrica")
    
    # Línea límite físico (Predicción = distnom_min)
    max_val = min(0.3, max(distnom_min_vals.max(), oof_preds.max()))
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label="Límite físico (y=x)")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel("Mínima Distancia Observada (distnom_min, au)")
    plt.ylabel("MOID Predicho Libre (au)")
    plt.title(f"Test de Restricción Geométrica\n({pct_violaciones:.1f}% violaciones)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Importancia de variables
    plt.subplot(1, 2, 2)
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color="#5cb85c", align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Importancia Relativa (Gain)")
    plt.title("¿Qué usa el modelo para predecir el MOID?")
    
    plt.tight_layout()
    fig_path = os.path.join(DIR_FIG, "moid_physics_validation.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nFigura guardada en: {fig_path}")

    # Exportar resultados
    resultados = {
        "Geometric_Violations_Count": int(n_violaciones),
        "Geometric_Violations_Pct": float(pct_violaciones),
        "Metrics_Unconstrained": {
            "R2": float(r2_unconstrained),
            "MAE": float(mae_unconstrained)
        },
        "Metrics_Constrained": {
            "R2": float(r2_constrained),
            "MAE": float(mae_constrained)
        },
        "Feature_Importances": {features[i]: float(importances[i]) for i in range(len(features))}
    }
    
    ruta_json = os.path.join(DIR_TAB, "moid_physics_validation.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
        
    print(f"Resumen guardado en: {ruta_json}")


def main():
    obj = cargar_y_agregar()
    validacion_fisica(obj)


if __name__ == "__main__":
    main()
