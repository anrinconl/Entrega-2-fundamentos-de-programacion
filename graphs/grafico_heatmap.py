from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Aseguramos el acceso al directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importamos la función que carga los datos ambientales y microbianos
from utils.load_data import load_env_micro

# 1. Carga y combinación de los conjuntos de datos
df_env, df_micro = load_env_micro()
# Unimos ambas tablas a través del número de suelo (soil_number)
df = pd.merge(df_micro, df_env, on="soil_number", how="inner")

# 2. Selección de variables relevantes
vars_candidatas = [
    "bacterial_growth_rate", "fungal_growth_rate", "respiration_rate",
    "pH", "aridity_index", "carbon_availability", "incubation_temperature",
    "soil_moisture_at_the_end_of_drying",
    "soil_moisture_in_the_moist_control",
    "soil_moisture_after_rewetting",
    "soil_moisture_increment_at_rewetting",
    "carbon_use_efficiency_in_the_moist_control",
    "fungal_to_bacterial_dominance_in_the_moist_control",
]

# Nos aseguramos de incluir solo las columnas que existen en el DataFrame
cols = [c for c in vars_candidatas if c in df.columns]
df_sel = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")

# 3. Renombramos las columnas para etiquetas más legibles en el gráfico

ren = {
    "bacterial_growth_rate": "Crec. bacteriano",
    "fungal_growth_rate": "Crec. fúngico",
    "respiration_rate": "Respiración",
    "pH": "pH",
    "aridity_index": "Aridez",
    "carbon_availability": "C disponible",
    "incubation_temperature": "Temp incub.",
    "soil_moisture_at_the_end_of_drying": "Humedad fin secado",
    "soil_moisture_in_the_moist_control": "Humedad control",
    "soil_moisture_after_rewetting": "Humedad post-rehum.",
    "soil_moisture_increment_at_rewetting": "ΔHumedad rehum.",
    "carbon_use_efficiency_in_the_moist_control": "CUE (control)",
    "fungal_to_bacterial_dominance_in_the_moist_control": "F:B (control)",
}
df_plot = df_sel.rename(columns=ren)


# 4. Cálculo de correlaciones

# Usamos el método de Spearman (adecuado para relaciones no lineales)
corr = df_plot.corr(method="spearman")


# 5. Creación del mapa de calor (heatmap)

sns.set_theme(style="white", context="notebook")
fig, ax = plt.subplots(figsize=(10, 8))

# Dibujamos el mapa de correlación con anotaciones y escala de color
sns.heatmap(
    corr,
    cmap="RdBu_r", vmin=-1, vmax=1, center=0,  # paleta centrada en 0
    annot=True, fmt=".2f",                    # mostramos valores numéricos
    annot_kws={"size": 8},                   # tamaño del texto
    cbar_kws={"shrink": 0.8, "label": "Coef. de correlación"},
    linewidths=0.6, linecolor="gray", square=True, ax=ax
)

# Ajustes de formato de ejes y título
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_title(
    "Mapa de calor de correlaciones\nVariables ambientales vs tasas microbianas",
    fontsize=12, pad=12
)

# Guardamos el gráfico en la carpeta outputs
plt.tight_layout()
out = PROJECT_ROOT / "outputs" / "grafico3_heatmap_correlaciones_v2.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300)
plt.show()
