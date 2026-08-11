# 04 — Fuentes de datos (APIs de JPL)

[← Índice](README.md)

El proyecto consume **dos** APIs distintas del *Solar System Dynamics* de JPL/NASA. Cada
una responde a una pregunta diferente, y hacen falta las dos.

---

## CAD API — Close Approach Data

```
https://ssd-api.jpl.nasa.gov/cad.api?date-min=<a>-01-01&date-max=<b>-01-01
                                    &dist-max=0.5&diameter=true
```

**Unidad de respuesta: el evento.** Devuelve una fila por cada aproximación cercana
calculada, con la fecha, la distancia y las velocidades del encuentro.

- `dist-max=0.5` — **explícito y obligatorio**. El valor por defecto de la API es
  `0.05` au, que es exactamente el umbral de distancia de la definición PHA: dejarlo
  implícito censura la muestra en el umbral de la propia etiqueta. 0.5 au es el límite
  superior que sirve la base de datos de JPL (pedir 1 au devuelve lo mismo).
  Ver [discrepancia A](05-discrepancias.md#a--el-dataset-está-censurado-en-el-umbral-que-define-la-etiqueta).
- `date-min` / `date-max` — la consulta se **trocea por décadas** (`descargar_cad`, en el
  notebook de datos). Una petición única de 340 000 eventos tarda ~6 min y falla de forma
  intermitente con `ChunkedEncodingError`; troceada tarda ~2 min y es reproducible.
  Historia completa desde 1900, con `drop_duplicates` en las fronteras por seguridad.
- `diameter=true` — pide los campos `diameter` y `diameter_sigma`, ausentes por defecto.

> ⚠ La cobertura desde 1900 incluye aproximaciones **calculadas retroactivamente** para
> objetos descubiertos hace pocos años: el **78.1 % de los eventos son anteriores al
> descubrimiento** de su objeto. Por eso el CSV lleva la columna `post_discovery`, y el
> análisis principal se restringe a los eventos realmente observados.

Campos usados: `des`, `cd`, `dist`, `dist_min`, `v_rel`, `v_inf`, `h`, `diameter`,
`diameter_sigma`.

La respuesta JSON trae los nombres de columna en `fields` y las filas en `data`; el
notebook las ensambla con `pd.DataFrame(data=datos, columns=columnas)`.

Documentación: <https://ssd-api.jpl.nasa.gov/doc/cad.html>

---

## SBDB Query API — Small-Body Database

```
https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=<lista>&sb-group=neo
```

**Unidad de respuesta: el objeto.** Devuelve el catálogo de cuerpos menores con sus
propiedades orbitales y de catalogación. `sb-group=neo` lo restringe a los NEOs.

Campos usados: `pdes`, `full_name`, `pha`, `neo`, `moid`, `H`, `diameter`, `first_obs`,
`data_arc`, `n_obs_used`.

El cruce con las aproximaciones se hace por designación: `Object` (de `des`) contra
`pdes`, previo `drop_duplicates("pdes")`.

Documentación: <https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html>

---

## Por qué hacen falta las dos

| Pregunta | API |
|---|---|
| ¿Qué se **observó** de este objeto? (distancias, velocidades, brillo por paso) | **CAD** |
| ¿Es oficialmente PHA? ¿Cuál es su MOID? ¿Cuándo se descubrió? | **SBDB** |

La CAD API **no** trae el flag `pha` ni el `moid`. Sin la SBDB no habría etiqueta de
verdad (`PHA_official`) ni forma de validar `PHA_proxy` ni de estudiar la función de
selección. Y esa es precisamente la separación limpia que el diseño del proyecto
necesita: **la CAD aporta las features, la SBDB aporta el target y los metadatos —
nunca al revés**.

---

## Caché en disco

El notebook de datos evita redescargar en cada ejecución:

- Escribe `close_approaches.csv` en su **propio directorio** (`carpeta_destino = "."`),
  es decir `data/close_approaches.csv`. Hay que ejecutarlo con `data/` como directorio
  de trabajo.
- `archivo_es_reciente(..., dias_maximos=30)` — solo redescarga si el archivo falta o
  tiene más de **30 días**.
- El CSV de la SBDB (`sbdb_neo.csv`) se cachea aparte, y se revalida comprobando que
  contenga todos los campos esperados (`set(sbdb_fields).issubset(columns)`); si el
  esquema cambió, se redescarga.

`*.csv` está en `.gitignore`, así que estos artefactos nunca se commitean: cada clon
regenera sus datos desde las APIs.

---

## Orden de ejecución

```
data/ProyectoNeoRework_data.ipynb        (cwd = data/)
        ↓  data/close_approaches.csv
notebooks/ProyectoNeoRework_ml.ipynb     (cwd = notebooks/, lee ../data/close_approaches.csv)
```

El notebook de ML lanza `FileNotFoundError` si el CSV no existe. Para regenerarlo todo
sin abrir Jupyter:

```bash
python .claude/skills/run-neos-analysis/driver.py
```

---

[← Conceptos de ML](03-conceptos-ml.md) · [Índice](README.md) · [Siguiente: discrepancias →](05-discrepancias.md)
