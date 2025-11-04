from pathlib import Path
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Aseguramos acceso al directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cargamos los datos ambientales y microbianos
from utils.load_data import load_env_micro
df_env, df_micro = load_env_micro()


# 1. Fusión de los datos (por número de suelo)

df = pd.merge(df_micro, df_env, on="soil_number", how="inner")


# 2. Selección de variables numéricas para el PCA

vars_num = [
    "bacterial_growth_rate", "fungal_growth_rate", "respiration_rate",
    "pH", "aridity_index", "carbon_availability", "incubation_temperature",
    "soil_moisture_at_the_end_of_drying",
    "soil_moisture_in_the_moist_control",
    "soil_moisture_after_rewetting",
    "soil_moisture_increment_at_rewetting",
    "carbon_use_efficiency_in_the_moist_control",
    "fungal_to_bacterial_dominance_in_the_moist_control",
]
vars_num = [c for c in vars_num if c in df.columns]

# Creamos una matriz limpia solo con valores numéricos
X = df[vars_num].apply(pd.to_numeric, errors="coerce").dropna(how="any").copy()
# Guardamos las etiquetas de cobertura del suelo para colorear los puntos
meta = df.loc[X.index, ["land_cover"]].copy()
meta["land_cover"] = meta["land_cover"].astype("category")


# 3. Renombramos variables para mostrar nombres legibles en el gráfico

ren = {
    "bacterial_growth_rate": "Crec. bacteriano",
    "fungal_growth_rate": "Crec. fúngico",
    "respiration_rate": "Respiración",
    "pH": "pH",
    "aridity_index": "Aridez",
    "carbon_availability": "C disp.",
    "incubation_temperature": "Temp incub.",
    "soil_moisture_at_the_end_of_drying": "Hum. fin sec.",
    "soil_moisture_in_the_moist_control": "Hum. control",
    "soil_moisture_after_rewetting": "Hum. post-reh.",
    "soil_moisture_increment_at_rewetting": "ΔHum. reh.",
    "carbon_use_efficiency_in_the_moist_control": "CUE (ctrl)",
    "fungal_to_bacterial_dominance_in_the_moist_control": "F:B (ctrl)",
}
var_labels = [ren.get(c, c) for c in X.columns]


# 4. Estandarización y cálculo del PCA

scaler = StandardScaler()
Xz = scaler.fit_transform(X)  # Estandarizamos para que todas las variables tengan igual peso

# Ejecutamos el PCA con dos componentes principales
pca = PCA(n_components=2, random_state=42)
scores = pca.fit_transform(Xz)        # Coordenadas de las muestras
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # Cargas (vectores de las variables)

# Obtenemos la varianza explicada por cada componente
exp1, exp2 = pca.explained_variance_ratio_[:2] * 100

# Ajustamos la escala de las flechas para que quepan dentro del gráfico
sx = scores[:, 0]; sy = scores[:, 1]
rx = (sx.max() - sx.min()); ry = (sy.max() - sy.min())
scale = 0.9 * min(rx, ry) / np.max(np.sqrt((loadings**2).sum(axis=1)))


# 5. Visualización: biplot con puntos (muestras) y flechas (variables)

sns.set_theme(style="whitegrid", context="notebook")
fig, ax = plt.subplots(figsize=(10, 7))

# Dibujamos los puntos de las muestras, coloreados por tipo de cobertura
sns.scatterplot(x=scores[:, 0], y=scores[:, 1],
                hue=meta["land_cover"], s=35, alpha=0.8, ax=ax)

# Dibujamos flechas que representan las variables y su dirección de influencia
idx_vars = np.argsort(np.linalg.norm(loadings, axis=1))[::-1][:10]  # seleccionamos las más influyentes
for i in idx_vars:
    vx, vy = loadings[i, 0] * scale, loadings[i, 1] * scale
    ax.arrow(0, 0, vx, vy, width=0.01, head_width=0.15, head_length=0.2,
             length_includes_head=True, color="black", alpha=0.7)
    ax.text(vx * 1.05, vy * 1.05, var_labels[i],
            fontsize=8, ha="center", va="center")

# Añadimos líneas de referencia y etiquetas
ax.axhline(0, color="gray", lw=0.6)
ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel(f"PC1 ({exp1:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({exp2:.1f}%)", fontsize=11)
ax.set_title("PCA Biplot: estructura multivariante", fontsize=13, pad=10)
ax.legend(title="Cobertura", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title_fontsize=9)

# Guardamos el gráfico final
plt.tight_layout()
out = PROJECT_ROOT / "outputs" / "grafico4_pca_biplot.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300)
plt.show()
