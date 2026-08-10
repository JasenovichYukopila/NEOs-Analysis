# 02 — Columnas del dataset

[← Índice](README.md)

Referencia de cada columna de `data/close_approaches.csv`: nombre crudo en la API de
JPL, nombre renombrado en el proyecto, significado, unidades y **rol** (¿es predictor o
no?).

El CSV tiene **una fila por evento de aproximación**. Las columnas que son propiedades
del *objeto* (etiquetas, MOID, metadatos de selección) están **desnormalizadas**: se
repiten idénticas en todas las filas del mismo `Object`.

---

## Bloque 1 — Aproximaciones cercanas (CAD API)

Descargadas de `https://ssd-api.jpl.nasa.gov/cad.api` y renombradas en el notebook de
datos.

| Nombre crudo | Nombre en el proyecto | Unidad | Qué es | Rol |
|---|---|---|---|---|
| `des` | `Object` | — | Designación del objeto (p. ej. `433`, `2011 AG5`). **Clave de agrupación** | Identificador |
| `cd` | `Close-Approach (CA) Date` | fecha | Fecha y hora del máximo acercamiento (UTC) | Temporal |
| `dist` | `CA DistanceNominal (au)` | au | Distancia **nominal** del acercamiento: el valor más probable según la solución orbital | Feature |
| `dist_min` | `CA DistanceMinimum (au)` | au | Distancia **mínima** dentro de la incertidumbre (borde inferior del intervalo). Es el escenario pesimista y la base de `PHA_proxy` | Feature |
| `v_rel` | `V relative(km/s)` | km/s | Velocidad relativa en el instante de máxima aproximación | Feature |
| `v_inf` | `V infinity(km/s)` | km/s | Velocidad relativa descontando la atracción terrestre | Feature |
| `h` | `H(mag)` | mag | Magnitud absoluta. **Menor H = objeto más grande** | Feature |
| `diameter` | `Diameter(km)` | km | Diámetro. Medido si existe; si no, **imputado desde H** | Feature (con caveat) |
| `diameter_sigma` | `Std Diameter(km)` | km | Incertidumbre del diámetro. Si falta, se fija en `0.35 · Diameter` | Diagnóstico |

> **Caveat de `Diameter(km)`:** para las filas sin diámetro medido se calcula como
> `3552 · 10^(−0.2·H)`, es decir, una función determinista de `H(mag)`. Ambas columnas
> **no son independientes**; el análisis de colinealidad descarta el diámetro y conserva
> H. Ver [albedo y estimación del diámetro](01-conceptos-astronomicos.md#albedo-y-estimación-del-diámetro).

### La `limpiar_fecha`

El campo `cd` de JPL puede traer un sufijo de incertidumbre (`±...`). La función
`limpiar_fecha` lo elimina antes de parsear la fecha.

---

## Bloque 2 — Catálogo de objetos (SBDB API)

Descargadas de `https://ssd-api.jpl.nasa.gov/sbdb_query.api` (grupo `neo`) y cruzadas
contra `Object` por la designación `pdes`. **Ninguna de estas columnas es predictor.**

| Nombre crudo | Nombre en el proyecto | Unidad | Qué es | Rol |
|---|---|---|---|---|
| `pha` | `PHA_official` | 0/1 | Flag oficial PHA de JPL (`Y`/`N` → 1/0) | **Target (ground truth)** |
| `moid` | `MOID (au)` | au | Distancia mínima de intersección orbital | Validación ⚠ **nunca feature** |
| `H` | `H_SBDB(mag)` | mag | Magnitud absoluta del catálogo (frente a la observada en cada evento) | Validación |
| `first_obs` | `first_obs_year` | año | Año de la primera observación **real** del objeto | Metadato de selección |
| `data_arc` | `data_arc(d)` | días | Arco orbital: span entre primera y última observación | Metadato de selección |
| `n_obs_used` | `n_obs_used` | recuento | Número de observaciones astrométricas usadas en la solución orbital | Metadato de selección |

> **Por qué `MOID (au)` no puede ser feature:** PHA se *define* como `MOID ≤ 0.05 au` y
> `H ≤ 22`. Meter el MOID en el modelo sería darle literalmente la mitad de la regla de
> decisión. Ver [circularidad](03-conceptos-ml.md#circularidad-y-leakage).

---

## Bloque 3 — Etiquetas derivadas

Calculadas en el notebook de datos, a nivel de **objeto** y desnormalizadas en cada fila.

| Columna | Definición | Rol |
|---|---|---|
| `PHA_official` | Flag `pha` de la SBDB | Target |
| `PHA_proxy` | `min(H(mag)) ≤ 22` **y** `min(CA DistanceMinimum (au)) ≤ 0.05`, por objeto | Target alternativo / sub-resultado |

La comparación `PHA_proxy` vs `PHA_official` es un resultado del paper por sí misma:
mide si la distancia mínima **observada** puede sustituir al MOID **orbital**.

---

## Bloque 4 — Features agregadas por objeto (notebook de ML)

El notebook de ML colapsa los eventos a un registro por objeto con
`df.groupby("Object").agg(...)`. Estas son las columnas con las que realmente entrena el
clasificador:

| Feature | Agregación | Interpretación |
|---|---|---|
| `distmin_min` | `min(CA DistanceMinimum (au))` | El paso más cercano jamás registrado — el proxy del MOID |
| `distnom_min` | `min(CA DistanceNominal (au))` | Ídem sobre la distancia nominal |
| `vrel_max` | `max(V relative(km/s))` | Encuentro más rápido (peor caso energético) |
| `vrel_med` | `median(V relative(km/s))` | Velocidad típica, robusta a outliers |
| `vinf_med` | `median(V infinity(km/s))` | Velocidad típica sin focalización gravitatoria |
| `H_obs` | `min(H(mag))` | Estimación de mayor tamaño observada |
| `diam_max` | `max(Diameter(km))` | Diámetro máximo (descartado por colinealidad con H) |
| `n_appro` | `size` | **Nº de eventos registrados: exposición observacional, no física** |
| `first_obs_year` | `max` | Metadato de selección |
| `data_arc` | `max` | Metadato de selección |
| `pha` | `max(PHA_official)` | Etiqueta del objeto |

### `n_appro` merece una nota

No es una propiedad física del asteroide. Un objeto tiene muchas aproximaciones
registradas porque **se le ha seguido bien**, no porque sea intrínsecamente más
peligroso. Es una variable de *exposición observacional* y, como tal, un vector de
sesgo de selección; se incluye deliberadamente y su efecto se examina en el análisis
SHAP y en el hold-out temporal.

---

## Conjuntos de features usados

Tras la poda por colinealidad, el notebook compara tres conjuntos:

| Conjunto | Columnas | Pregunta que responde |
|---|---|---|
| `kin+size` | `distmin_min`, `vrel_max`, `H_obs`, `n_appro` | ¿Cuánto se puede predecir con todo lo observable? |
| `kin-only` | `distmin_min`, `vrel_max`, `n_appro` | ¿Basta la **cinemática** sola, sin tamaño? |
| `size-only` | `H_obs` | ¿Cuánto explica el **tamaño** por sí solo? |

El contraste entre `kin-only` (F2 ≈ 0.49) y `size-only` (F2 ≈ 0.97) es el hallazgo
central: la señal de peligrosidad la aporta casi entera el tamaño, no el movimiento.

---

## Conjunto canónico de features del análisis no supervisado

PCA, K-Means y t-SNE trabajan sobre las **5 columnas a nivel de evento**, definidas una
sola vez en la celda de preámbulo del notebook de ML:

```python
features = ["CA DistanceNominal (au)", "V relative(km/s)",
            "V infinity(km/s)", "H(mag)", "Diameter(km)"]
```

Las filas con NaN en cualquiera de ellas se eliminan (`df.dropna(subset=features)`).

---

[← Conceptos astronómicos](01-conceptos-astronomicos.md) · [Índice](README.md) · [Siguiente: conceptos de ML →](03-conceptos-ml.md)
