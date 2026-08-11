# Conceptos del proyecto

Glosario de referencia para entender qué significa cada término, cada columna y cada
técnica que aparece en los notebooks y en el paper. Pensado para leerse **antes** de
tocar el código, o como consulta puntual cuando aparezca una sigla desconocida.

| Documento | Contenido |
|---|---|
| [01 — Conceptos astronómicos](01-conceptos-astronomicos.md) | NEO, PHA, MOID, magnitud absoluta H, albedo, unidad astronómica, velocidades relativa e infinita, aproximación cercana |
| [02 — Columnas del dataset](02-columnas-del-dataset.md) | Qué es cada columna del CSV, sus unidades, de qué API viene y si se usa como predictor o no |
| [03 — Conceptos de machine learning](03-conceptos-ml.md) | PCA, K-Means, F2, PR-AUC, SHAP, colinealidad, circularidad, sesgo de selección |
| [04 — Fuentes de datos (APIs de JPL)](04-fuentes-de-datos.md) | CAD API vs SBDB API: qué devuelve cada una y por qué hacen falta las dos |
| [05 — Discrepancias físicas y teóricas](05-discrepancias.md) | Auditoría contra la documentación de JPL, la literatura y otros repos: qué no cuadra, qué sí, y por qué se corrigió |

> El documento 05 registra tanto los hallazgos **como su corrección**: el dataset ya no
> está censurado y las conclusiones del notebook reflejan los datos corregidos. Su
> evidencia es reproducible con `python scripts/verificar_discrepancias.py`.

## La idea del proyecto en un párrafo

Un NEO se etiqueta oficialmente como **PHA** (potencialmente peligroso) según dos
propiedades de su **órbita**: su distancia mínima orbital a la Tierra (`MOID ≤ 0.05 au`)
y su tamaño (`H ≤ 22 mag`). Como el MOID no se observa directamente, este proyecto mide
**cuánto se pierde al sustituirlo por la distancia mínima efectivamente observada** en los
pasos cercanos, y cómo distorsiona esa sustitución el hecho de que el catálogo tenga
126 años de sesgos de descubrimiento.
