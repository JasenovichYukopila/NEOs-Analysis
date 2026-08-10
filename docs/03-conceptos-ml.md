# 03 — Conceptos de machine learning

[← Índice](README.md)

Qué es cada técnica y cada métrica que aparece en `notebooks/ProyectoNeoRework_ml.ipynb`,
y **por qué** se eligió en este proyecto concreto.

---

## Estandarización (`StandardScaler`)

Transforma cada variable a media 0 y desviación típica 1: `z = (x − μ) / σ`.

Es obligatorio aquí porque las escalas son incomparables: las distancias están en
centésimas de au, las velocidades en decenas de km/s y H en unidades de magnitud. Sin
estandarizar, PCA y K-Means (que miden distancias euclídeas) quedarían dominados por la
variable de mayor rango numérico, que no es la más informativa sino simplemente la que
se mide en unidades más grandes.

---

## PCA (Análisis de Componentes Principales)

Rotación del espacio de features hacia ejes nuevos (**componentes principales**),
ordenados por cuánta varianza capturan. PC1 es la dirección de máxima dispersión de los
datos, PC2 la siguiente ortogonal a ella, etc.

Dos usos en el proyecto:

1. **Visualización** — proyectar las 5 dimensiones en 2 para poder dibujarlas
   (PC1+PC2 ≈ **77.8 %** de la varianza).
2. **Diagnóstico de redundancia** — que PC5 aporte ≈ **0 %** de varianza significa que
   hay una combinación lineal de las 5 features que es casi constante: las variables no
   son independientes. Esto anticipa el análisis de colinealidad.

> **Advertencia de interpretación:** los ejes PCA son combinaciones lineales, no
> magnitudes físicas. "PC1 alto" no significa nada por sí solo hasta mirar los *loadings*
> (los pesos de cada feature original en la componente).

---

## K-Means

Algoritmo de clustering: divide los datos en **k** grupos minimizando la distancia de
cada punto al centroide de su grupo. En el proyecto: `k = 4`, `n_init = 10`,
`random_state = 20`.

Dos decisiones deliberadas del notebook:

- **El clustering se hace en el espacio 5-D estandarizado, no sobre las coordenadas
  PCA.** PCA y t-SNE se usan *solo para dibujar* el resultado. Agrupar sobre
  coordenadas reducidas descartaría información real; no hay que mover el `fit` de
  K-Means a las coordenadas reducidas.
- **`k = 4` está fijado.** La celda del método del codo es exploratoria: se conserva
  como justificación documental, no como selector automático.

### Método del codo

Se grafica la inercia (suma de distancias al cuadrado a los centroides) frente a k. La
inercia siempre baja al subir k; se busca el punto donde deja de bajar bruscamente — el
"codo" — como compromiso entre ajuste y parsimonia. Es un criterio visual, no una prueba
estadística.

---

## t-SNE (t-distributed Stochastic Neighbor Embedding)

Reducción de dimensionalidad **no lineal** orientada a visualización. A diferencia de
PCA, no busca conservar la varianza global sino la **estructura de vecindario local**:
puntos próximos en 5-D quedan próximos en 2-D.

Implementación: **openTSNE** (no la de scikit-learn), por rendimiento y por exponer más
control sobre la optimización.

> **Qué NO se puede leer en un gráfico t-SNE:** las distancias *entre* clusters y los
> tamaños relativos de los clusters no son interpretables. Solo la pertenencia local lo es.

### Perplejidad

Hiperparámetro principal de t-SNE. Grosso modo, el número efectivo de vecinos que cada
punto considera. Valores bajos resaltan microestructura; valores altos, estructura más
global. No hay un valor "correcto" universal — de ahí la búsqueda con Optuna.

### Trustworthiness

Métrica en [0, 1] que mide **cuánto se puede confiar** en el embedding: penaliza los
puntos que aparecen próximos en 2-D sin serlo en el espacio original (vecinos falsos).
Es la función objetivo que Optuna maximiza.

### Optuna

Framework de optimización de hiperparámetros. Aquí ejecuta `N_TRIALS = 200` pruebas de
configuraciones de t-SNE maximizando `trustworthiness`, y escribe los resultados en
`tsne_optuna_results.csv`, que una celda posterior relee para seleccionar las mejores
configuraciones y reajustarlas.

Ese CSV es un artefacto local del notebook y está git-ignorado (`*.csv`).

---

## Colinealidad y VIF

Dos features son **colineales** cuando una es prácticamente predecible desde la otra.
Consecuencias: coeficientes inestables en modelos lineales, importancias repartidas
arbitrariamente entre variables redundantes, e interpretación engañosa.

Se diagnostica con la matriz de correlaciones (se reportan los pares con `|corr| > 0.9`)
y con el **VIF** (*Variance Inflation Factor*), que mide cuánto se infla la varianza del
coeficiente de una variable por su correlación con las demás.

En este dataset hay dos fuentes claras de redundancia:

- `Diameter(km)` fue **imputado desde** `H(mag)` → dependencia funcional exacta en esas filas.
- `v_rel`, `v_inf` y sus agregados son casi idénticos en pasos lejanos.

La **poda** resultante conserva una sola distancia (`distmin_min`), una sola velocidad
(`vrel_max`) y un solo indicador de tamaño (`H_obs`).

---

## Desbalance de clases

Solo el **7.3 %** de los objetos son PHA (1 379 de 18 927 — 1 PHA por cada 12 no-PHA).
Un clasificador que prediga siempre "no peligroso" acertaría el 93 % de las veces sin
haber aprendido nada:
por eso **la accuracy se omite deliberadamente** de todo el reporte.

Cómo se maneja en el notebook:

- `class_weight="balanced"` en Regresión Logística y Random Forest — pondera cada clase
  inversamente a su frecuencia.
- `scale_pos_weight = n_neg / n_pos` en XGBoost — el equivalente en esa librería.

---

## Métricas

### Precisión y recall

- **Precisión** = de los que marqué como peligrosos, ¿qué fracción lo eran?
  Controla las **falsas alarmas**.
- **Recall** (sensibilidad) = de los peligrosos que existen, ¿qué fracción detecté?
  Controla los **peligros que se me escapan**.

### F2 (métrica principal del proyecto)

Media armónica ponderada de precisión y recall que da **4× más peso al recall**:

```
F_β = (1 + β²) · precisión · recall / (β² · precisión + recall)      con β = 2
```

Se eligió F2 y no F1 por el coste asimétrico del dominio: **no detectar un asteroide
peligroso es mucho peor que investigar de más uno inofensivo**. F1 (β=1) trataría ambos
errores como equivalentes, lo cual sería incorrecto aquí.

### PR-AUC (`average_precision`)

Área bajo la curva precisión–recall. Es la métrica de referencia con clases muy
desbalanceadas, porque su línea base es la prevalencia de la clase positiva (0.073 aquí):
no se deja engañar por la abundancia de negativos.

### ROC-AUC

Área bajo la curva ROC (tasa de verdaderos positivos vs falsos positivos). Se reporta
por convención, pero con desbalance fuerte tiende a dar valores optimistas — de ahí que
todos los modelos ronden 0.99 mientras el F2 sí discrimina entre ellos.

### Umbral de decisión

Los clasificadores devuelven una **probabilidad**, no una clase. Convertirla en decisión
exige un umbral, y `0.5` es una convención arbitraria, no un óptimo. El notebook calcula
probabilidades **fuera de fold** (`cross_val_predict` con `StratifiedKFold`) y reporta
el umbral que **maximiza F2**, en vez de asumir 0.5.

---

## Validación cruzada

### Repeated Stratified K-Fold (5 × 3)

Los datos se parten en 5 pliegues (*folds*); cada uno se usa una vez como test mientras
los otros 4 entrenan. **Estratificada**: cada pliegue conserva la proporción de PHA del
conjunto total — imprescindible con 7 % de positivos, o algún pliegue podría quedarse
casi sin ellos. **Repetida** 3 veces con particiones distintas, para reportar
media ± desviación típica y no un número que dependa de una partición afortunada.

**Unidad de análisis: el objeto**, no el evento. Si se hiciera CV a nivel de evento, el
mismo asteroide aparecería en entrenamiento y en test — fuga de información garantizada.

---

## Circularidad y *leakage*

**Leakage** (fuga): el modelo accede, directa o indirectamente, a información que no
estaría disponible en el momento de predecir — o directamente a la respuesta.

**Circularidad** es el caso extremo que este proyecto pone en evidencia. PHA se *define*
como `MOID ≤ 0.05 au ∧ H ≤ 22`. Buena parte de la literatura alimenta al modelo con esas
mismas variables (o proxies suyos) y obtiene exactitud casi perfecta, presentándola como
un logro predictivo. No lo es: el modelo está reconstruyendo una definición, no
descubriendo física.

El notebook lo demuestra explícitamente con un **baseline de regla de dos umbrales** —
aplicar los mismos umbrales que definen PHA sobre las variables observadas — que alcanza
**F2 = 0.980**, igualando a Random Forest y XGBoost. Si una regla de dos `if` empata con
gradient boosting, el problema no era de machine learning.

---

## SHAP (SHapley Additive exPlanations)

Método de explicabilidad basado en los **valores de Shapley** de teoría de juegos:
reparte la predicción entre las features atribuyendo a cada una su contribución marginal
promediada sobre todas las coaliciones posibles de features.

Ventaja sobre la importancia por impureza de un Random Forest: es consistente y da
atribuciones **por predicción individual**, no solo un ranking global.

Resultado en este proyecto: la señal de peligrosidad la aporta el **tamaño**
(`H_obs`), no la cinemática — coherente con el contraste `size-only` (F2 ≈ 0.97) vs
`kin-only` (F2 ≈ 0.49).

---

## Sesgo de selección (*selection bias*)

El catálogo no es una muestra aleatoria del cielo: es el resultado de 126 años de
campañas de observación con capacidades muy distintas. Los sondeos antiguos solo
detectaban objetos grandes y cercanos — exactamente el perfil PHA.

El notebook lo cuantifica de dos formas:

### Función de selección temporal

Prevalencia de PHA por **década de primera observación real** (`first_obs_year`, no el
año del primer evento del catálogo, que puede ser retroactivo hasta 1900). La
prevalencia cae drásticamente con el tiempo. En paralelo se mide
`corr(data_arc, PHA) = 0.647`: los PHA tienen arcos orbitales más largos porque se les
observa más, lo que es un **confundidor de caracterización**, no física.

### Hold-out temporal

La prueba honesta: entrenar solo con objetos **descubiertos antes de 2015** (4 451
objetos, prevalencia PHA 23.3 %) y evaluar con los descubiertos **después** (14 476
objetos, prevalencia 2.4 %). Las medianas de imputación se calculan **solo con el
train**, para que no se filtre información del futuro hacia el pasado.

Resultado con XGBoost `kin+size`:

| Escenario | F2 | Precisión | Recall |
|---|---|---|---|
| CV aleatoria | 0.977 | — | — |
| Hold-out temporal (<2015 → ≥2015) | 0.971 | **0.928** | 0.983 |

El recall aguanta, la precisión cae. Esa caída es exactamente **el coste del
desplazamiento de prevalencia** (23.3 % → 2.4 %): en un mundo donde los peligrosos son
diez veces más raros, el mismo modelo genera muchas más falsas alarmas. Una CV aleatoria
nunca habría revelado esto.

### Sesgo de muestreo

Adicionalmente se compara la prevalencia PHA del dataset (condicionado a tener
aproximaciones registradas) contra el total de NEOs de la SBDB. El enriquecimiento es
**leve**: la selección dominante es la temporal (por descubrimiento), no el
condicionamiento por aproximación cercana.

---

## Reproducibilidad

`random_state = 20` se define **una sola vez** en el preámbulo y se propaga a todo lo
estocástico: `KMeans`, `PCA`/`StandardScaler`, t-SNE, los tres clasificadores y los
esquemas de CV. Al modificar el notebook, hay que seguir usando esa variable en vez de
literales sueltos.

---

[← Columnas del dataset](02-columnas-del-dataset.md) · [Índice](README.md) · [Siguiente: fuentes de datos →](04-fuentes-de-datos.md)
