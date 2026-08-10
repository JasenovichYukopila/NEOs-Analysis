# Conceptos del proyecto

Glosario de referencia para entender qué significa cada término, cada columna y cada
técnica que aparece en los notebooks y en el paper. Pensado para leerse **antes** de
tocar el código, o como consulta puntual cuando aparezca una sigla desconocida.

| Documento | Contenido |
|---|---|
| [01 — Conceptos astronómicos](01-conceptos-astronomicos.md) | NEO, PHA, MOID, magnitud absoluta H, albedo, unidad astronómica, velocidades relativa e infinita, aproximación cercana |
| [02 — Columnas del dataset](02-columnas-del-dataset.md) | Qué es cada columna del CSV, sus unidades, de qué API viene y si se usa como predictor o no |
| [03 — Conceptos de machine learning](03-conceptos-ml.md) | PCA, K-Means, t-SNE, Optuna, F2, PR-AUC, SHAP, colinealidad, circularidad, sesgo de selección |
| [04 — Fuentes de datos (APIs de JPL)](04-fuentes-de-datos.md) | CAD API vs SBDB API: qué devuelve cada una y por qué hacen falta las dos |

## La idea del proyecto en un párrafo

Un NEO se etiqueta oficialmente como **PHA** (potencialmente peligroso) según dos
propiedades de su **órbita**: su distancia mínima orbital a la Tierra (`MOID ≤ 0.05 au`)
y su tamaño (`H ≤ 22 mag`). Este proyecto pregunta si esa peligrosidad puede inferirse
usando **solo lo que se ha observado** en sus pasos cercanos (distancias, velocidades,
brillo medido), sin los elementos orbitales que *definen* la etiqueta — y cuánto
distorsiona esa inferencia el hecho de que el catálogo tenga 126 años de sesgos de
descubrimiento.
