# 06 — Predicción del MOID Orbital

[← Índice](README.md)

---

## Contexto Astronómico y Motivación

La definición oficial de la IAU y CNEOS (JPL/NASA) para un Asteroide Potencialmente Peligroso (PHA) requiere evaluar dos condiciones:

$$\text{PHA} \iff \text{MOID} \le 0.05\text{ au} \quad \wedge \quad H \le 22\text{ mag}$$

Mientras que la magnitud absoluta $H$ (o su estimador $H_{\text{obs}}$) puede derivarse con relativa rapidez tras las primeras fotometrías, el **MOID** (*Minimum Orbit Intersection Distance*) **no es un parámetro directamente observable** en un pasaje cercano. 

El MOID es una constante geométrica calculada a partir de los seis elementos orbitales completos de la órbita heliocéntrica del asteroide ($a, e, i, \Omega, \omega, M$). Obtener una solución orbital precisa exige acumular múltiples observaciones astrométricas distribuidas a lo largo de semanas, meses o años (un arco orbital `data_arc` prolongado).

Para un asteroide recién descubierto o con arcos observacionales cortos:
1. El **MOID oficial aún no está disponible** o presenta un alto margen de incertidumbre.
2. La distancia mínima observada (`distnom_min`) suele ser **significativamente mayor que el MOID** (mediana $3.8\times$ mayor entre los PHA reales), por lo que usar `distnom_min \le 0.05` como regla directa tiene un *recall* muy bajo ($\sim 29.5\%$).

Por ello, la **predicción del MOID mediante Machine Learning** busca modelar y estimar la distancia orbital teórica mínima a partir del conjunto de variables cinemáticas y observacionales disponibles tras los primeros avistamientos.

---

## Formulación de las Tareas de Machine Learning

El módulo de predicción (`scripts/predict_moid.py`) aborda dos problemas complementarios:

### 1. Regresión Continua del MOID
- **Variable Objetivo ($y$):** `MOID (au)` en escala continua.
- **Métricas:** Coeficiente de determinación ($R^2$), Error Absoluto Medio ($\text{MAE}$ en UA) y Raíz del Error Cuadrático Medio ($\text{RMSE}$ en UA).
- **Línea de Base (Baseline Naïve):** Asumir directamente que $\widehat{\text{MOID}} = \text{distnom\_min}$.

### 2. Clasificación Binaria del Umbral Peligroso
- **Variable Objetivo ($y$):** `is_moid_hazardous` $= 1$ si $\text{MOID} \le 0.05\text{ au}$, y $0$ en caso contrario.
- **Métricas:** $F_2\text{-score}$ (priorizando minimizar falsos negativos), Precisión, Recall, ROC-AUC y PR-AUC.
- **Línea de Base:** Regla directa $\text{distnom\_min} \le 0.05\text{ au}$.

---

## Set de Características Utilizadas (Features)

Para evitar fugas de información y colinealidades triviales, se utilizan únicamente variables observadas agregadas por objeto sobre eventos posteriores al descubrimiento (`post_discovery == 1`):

| Feature | Descripción | Rol |
| :--- | :--- | :--- |
| `distnom_min` | Mínima distancia nominal observada (UA) | Proxy observacional de proximidad |
| `vinf_max` | Máxima velocidad relativa en el infinito ($\text{km/s}$) | Parámetro cinemático |
| `H_obs` | Mínima magnitud absoluta observada (mag) | Estimación de tamaño intrínseco |
| `n_appro` | Número de aproximaciones cercanas observadas | Exposición observacional |
| `dist_unc_med` | Incertidumbre observacional mediana ($\text{dist}_{\text{nom}} - \text{dist}_{\text{min}}$) | Indicador de calidad orbital |

---

## Resultados Empíricos

### 1. Regresión del MOID
Evaluado mediante **Validación Cruzada de 5 Folds (5-Fold CV)** sobre los objetos con aproximaciones observadas:

- **Baseline Naïve (`distnom_min` directo):**
  - $R^2 = 0.7481$
  - $\text{MAE} = 0.0573\text{ au}$
  - $\text{RMSE} = 0.0898\text{ au}$

- **Modelos de Machine Learning (Set `kin+size`):**
  - **Ridge Regression:** $R^2 \approx 0.7512$, $\text{MAE} \approx 0.0565\text{ au}$
  - **Random Forest Regressor:** $R^2 \approx 0.8145$, $\text{MAE} \approx 0.0468\text{ au}$
  - **XGBoost Regressor:** **$R^2 = 0.8423$**, **$\text{MAE} = 0.0431\text{ au}$**, $\text{RMSE} = 0.0708\text{ au}$

El modelo **XGBoost Regressor** reduce el error absoluto medio ($\text{MAE}$) en más de un $24.8\%$ respecto a la cota observada directa, corrigiendo en gran medida el sesgo donde $\text{distnom\_min} > \text{MOID}$.

### 2. Clasificación Binaria del Umbral $\text{MOID} \le 0.05\text{ au}$

- **Proxy directo ($\text{distnom\_min} \le 0.05$):**
  - Precisión: $0.985$ | Recall: $0.295$ | $F_2\text{-score}$: $0.343$
- **Regresión Logística:** $F_2 = 0.698$ | $\text{ROC-AUC} = 0.884$
- **Random Forest Classifier:** $F_2 = 0.742$ | $\text{ROC-AUC} = 0.912$
- **XGBoost Classifier:** **$F_2 = 0.768$** | Precisión: $0.712$ | Recall: $0.784$ | **$\text{ROC-AUC} = 0.925$**

ML logra elevar el *recall* del umbral orbital desde el $29.5\%$ (del proxy ingenuo) hasta más del **$78.4\%$**, identificando una porción sustancial de los objetos cuyo MOID es peligroso aunque no hayan sido capturados en un acercamiento extremadamente próximo.

### 3. Validación Temporal (Hold-out por Año de Descubrimiento)
Para evaluar la generalización sobre asteroides de reciente descubrimiento:
- **Entrenamiento:** Objetos descubiertos $\le 2014$.
- **Test:** Objetos descubiertos $\ge 2015$.

- **Resultados en Test:**
  - Regresión XGBoost: $R^2 = 0.7921$, $\text{MAE} = 0.0482\text{ au}$.
  - Clasificación XGBoost: $F_2 = 0.712$, $\text{ROC-AUC} = 0.895$.

Aunque se evidencia un ligero descenso atribuible al sesgo de selección observacional de los sondeos modernos, el modelo mantiene una alta capacidad de estimación.

---

## Conclusiones Metodológicas

1. **Factibilidad de la Predicción:** Es posible estimar razonablemente el `MOID` orbital ($R^2 > 0.84$) usando solo características cinemáticas y observacionales tempranas, cerrando la brecha antes de contar con un arco orbital prolongado.
2. **Superación del Proxy Ingenuo:** El modelo ML supera ampliamente a la distancia observada mínima como estimador del MOID, al aprender patrones entre la velocidad en el infinito, la cantidad de aproximaciones y la incertidumbre del catálogo.
3. **Reproducibilidad:** El script `scripts/predict_moid.py` permite regenerar las tablas de métricas y los gráficos en `results/figures/pred_vs_actual_moid.png` y `results/figures/moid_classification_roc.png`.

---

[← Volver al Índice](README.md)
