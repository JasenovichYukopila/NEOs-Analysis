# NEO Analysis: Near-Earth Objects Classification & Visualization

##  Proyecto Colaborativo de Análisis de Objetos Cercanos a la Tierra

Este repositorio contiene un análisis exhaustivo de objetos cercanos a la Tierra (NEOs - Near-Earth Objects) utilizando datos de la API de JPL/NASA. El proyecto explora la clasificación, visualización y análisis de rareza de estos objetos astronómicos mediante técnicas de machine learning y visualización de datos.

## Colaboradores

Este es un trabajo colaborativo realizado por:

- **Jasen yukoplia** - 
- **Dariem Garcia** - 
- **Carlos Toro** -


## Características Principales

### 1. Obtención y Procesamiento de Datos
- Descarga automática de datos desde la API de JPL/NASA
- Cache local de archivo CSV con validación de antigüedad (30 días)
- Limpieza y normalización de datos astronómicos

### 2. Métricas de Rareza
- Cálculo del índice de rareza basado en:
  - Distancia de aproximación (AU)
  - Magnitud absoluta (H)
  - Velocidad relativa
  - Tasa de impacto estimada

### 3. Análisis de Clustering
- **K-Means** con 6 clusters para clasificación automática
- **PCA** para reducción de dimensionalidad (73.27% varianza explicada)
- **t-SNE** para visualización de alta calidad
- **UMAP** para análisis de estructura topológica

### 4. Visualizaciones
- Mapas de densidad 2D con contornos KDE
- Scatter plots con colores por rareza (escala progresiva)
- Visualización de clusters con paleta personalizada
- Métricas de confianza (Trustworthiness & Continuity)

## Requisitos

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=flat&logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)

### Instalación

Clona el repositorio e instala las dependencias:
```bash
git clone https://github.com/JasenovichYukopila/NEOs-Analysis.git
cd NEOs-Analysis
pip install -r requirements.txt
```

 Se recomienda usar un entorno virtual:
> ```bash
> python -m venv .venv
> source .venv/bin/activate  # En Windows: .venv\Scripts\activate
> pip install -r requirements.txt
> ```

### Dependencias principales

| Paquete | Versión | Uso |
|---|---|---|
| `numpy` | ≥1.24 | Operaciones numéricas |
| `pandas` | ≥2.0 | Manipulación de datos |
| `matplotlib` | ≥3.7 | Visualización base |
| `seaborn` | ≥0.12 | Visualización estadística |
| `scikit-learn` | ≥1.3 | Machine learning |
| `scipy` | ≥1.11 | Cómputo científico |
| `umap-learn` | ≥0.5 | Reducción de dimensiones (UMAP) |
| `openTSNE` | ≥1.0 | Reducción de dimensiones (t-SNE) |
| `scienceplots` | ≥2.0 | Estilos de gráficas científicas |
| `requests` | ≥2.31 | Peticiones HTTP |
