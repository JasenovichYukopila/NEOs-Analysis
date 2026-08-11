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
| `dist_min` | `CA DistanceMinimum (au)` | au | Distancia mínima **a 3-sigma**: borde inferior del intervalo de confianza que induce la incertidumbre orbital. Base de `PHA_proxy` ⚠ [no es una distancia física](05-discrepancias.md#e--dist_min-es-una-cota-3-sigma-no-una-distancia-física) | Feature |
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
| *(derivada)* | `post_discovery` | 0/1 | 1 si el evento es **posterior** a la primera observación del objeto; 0 si la CAD API lo calculó por integración hacia atrás. Solo el 21.9 % de los eventos son observados | **Filtro del análisis principal** |

> **Por qué `MOID (au)` no puede ser feature:** PHA se *define* como `MOID ≤ 0.05 au` y
> `H ≤ 22`. Meter el MOID en el modelo sería darle literalmente la mitad de la regla de
> decisión. Ver [circularidad](03-conceptos-ml.md#circularidad-y-leakage).

---

## Bloque 3 — Etiquetas derivadas

Calculadas en el notebook de datos, a nivel de **objeto** y desnormalizadas en cada fila.

| Columna | Definición | Rol |
|---|---|---|
| `PHA_official` | Flag `pha` de la SBDB | Target |
| `PHA_proxy` | `min(H(mag)) ≤ 22` **y** `min(CA DistanceMinimum (au)) ≤ 0.05`, por objeto y **solo sobre eventos con `post_discovery == 1`** | Sustituto observacional |

La comparación `PHA_proxy` vs `PHA_official` es **el resultado central** del trabajo: mide
si la distancia mínima *observada* puede sustituir al MOID *orbital*. Restringirla a los
eventos realmente observados no es un detalle: incluir los calculados retroactivamente
infla el recall del proxy de 0.30 a 0.57.

---

## Bloque 4 — Features agregadas por objeto (notebook de ML)

El notebook de ML colapsa los eventos a un registro por objeto con
`df.groupby("Object").agg(...)`. Estas son las columnas con las que realmente entrena el
clasificador:

Se calculan **solo sobre eventos con `post_discovery == 1`**.

| Feature | Agregación | Interpretación |
|---|---|---|
| `distnom_min` | `min(CA DistanceNominal (au))` | Paso observado más cercano — el proxy del MOID **usado como feature** |
| `distmin_min` | `min(CA DistanceMinimum (au))` | Ídem sobre la cota 3σ; se conserva para evaluar el proxy, no como feature |
| `dist_unc_med` | `median(dist − dist_min)` | Separación nominal-3σ: **calidad orbital**, separada de la proximidad |
| `vrel_max` | `max(V relative(km/s))` | Se conserva solo para el diagnóstico de colinealidad |
| `vinf_med` / `vinf_max` | `median` / `max(V infinity(km/s))` | Velocidad sin focalización gravitatoria |
| `H_obs` | `min(H(mag))` | Estimación de mayor tamaño observada |
| `diam_max` | `max(Diameter(km))` | Descartado por ser función determinista de H |
| `n_appro` | `size` | **Nº de aproximaciones observadas: exposición observacional, no física** |
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

| Conjunto | Columnas | Qué mide |
|---|---|---|
| `kin+size` | `distnom_min`, `vinf_max`, `H_obs`, `n_appro` | Todo lo observable |
| `kin-only` | `distnom_min`, `vinf_max`, `n_appro` | El proxy de la mitad **MOID** de la definición |
| `size-only` | `H_obs` | La mitad **tamaño** de la definición |

Se leen como una **descomposición de la definición PHA**, no como una comparación de
modelos predictivos: `H_obs` *es* una de las dos variables que definen la etiqueta, así
que su F2 alto es definicional. Ver [discrepancia B](05-discrepancias.md#b--hmag-de-la-cad-api-es-literalmente-la-variable-que-define-la-etiqueta).

---

## Conjunto canónico de features del análisis no supervisado

PCA y K-Means trabajan sobre **tres cantidades independientes** a nivel de evento,
definidas una sola vez en la celda de preámbulo del notebook de ML:

```python
features = ["CA DistanceNominal (au)", "V infinity(km/s)", "H(mag)"]
```

`Diameter(km)` y `V relative(km/s)` **se excluyen a propósito**: son funciones
deterministas de las otras (el diámetro se imputa desde H; `v_rel` se deriva de `v_inf` y
la distancia), y su inclusión degeneraba el espectro del PCA produciendo una componente
de varianza ≈ 0 por aritmética, no por física. Ver
[discrepancia C](05-discrepancias.md#c--el-diámetro-no-es-una-variable-el-979--es-h-reescalado).

Las filas con NaN en cualquiera de ellas se eliminan (`df.dropna(subset=features)`).

---

[← Conceptos astronómicos](01-conceptos-astronomicos.md) · [Índice](README.md) · [Siguiente: conceptos de ML →](03-conceptos-ml.md)
