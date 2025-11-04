from pathlib import Path
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from numpy.polynomial import Polynomial

#  Cargamos el loader
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.load_data import load_env_micro  # CSV


# Traemos tablas ambiental y microbiana
df_env, df_micro = load_env_micro()

# Unimos por suelo y nos quedamos con respiración + crecimientos
df = pd.merge(
    df_micro[["soil_number", "respiration_rate", "bacterial_growth_rate", "fungal_growth_rate"]],
    df_env[["soil_number", "land_cover"]],
    on="soil_number", how="inner"
).dropna(subset=["respiration_rate", "bacterial_growth_rate", "fungal_growth_rate"])

# Pasamos a formato largo (comparar bacteria/hongo en la misma figura)
long = df.melt(
    id_vars=["soil_number", "land_cover", "respiration_rate"],
    value_vars=["bacterial_growth_rate", "fungal_growth_rate"],
    var_name="type", value_name="growth"
)
# Nombres limpios y solo tasas positivas
long["type"] = long["type"].str.replace("_growth_rate", "", regex=False).str.capitalize()
long = long[long["growth"] > 0].copy()

# Usamos log10 del crecimiento para comprimir escala (evita outliers gigantes)
long["growth_log10"] = np.log10(long["growth"])

# 2) Estilo y paletas
sns.set_theme(style="whitegrid", context="notebook")

# Definimos colores consistentes para cada panel
PALETAS = {
    "Bacterial": {"base": "#3366CC", "contorno": "#003366", "relleno": "#66A3FF"},
    "Fungal": {"base": "#CC3311", "contorno": "#660000", "relleno": "#FF7043"},
}

# 3) Figura 1x2: bacterias vs hongos
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
pairs = [("Bacterial", "Bacteriano"), ("Fungal", "Fúngico")]

for ax, (key, label) in zip(axes, pairs):
    d = long[long["type"] == key]
    if d.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=10)
        continue

    pal = PALETAS[key]

    # 3.1) Ponemos la nube de puntos (dispersión)
    sns.scatterplot(
        data=d, x="respiration_rate", y="growth_log10",
        s=20, alpha=0.4, color=pal["base"], ax=ax
    )

    # 3.2) Añadimos densidades KDE: relleno + contornos (morfología de la nube)
    sns.kdeplot(
        data=d, x="respiration_rate", y="growth_log10",
        fill=True, levels=25, thresh=0.02, alpha=0.45,
        cmap=sns.light_palette(pal["relleno"], as_cmap=True), ax=ax
    )
    sns.kdeplot(
        data=d, x="respiration_rate", y="growth_log10",
        levels=7, color=pal["contorno"], linewidths=0.8, ax=ax
    )

    # 3.3) Trazamos una tendencia polinómica suave (grado 3)
    x = d["respiration_rate"].to_numpy()
    y = d["growth_log10"].to_numpy()
    if len(x) > 5:
        x_fit = np.linspace(x.min(), x.max(), 200)
        p = Polynomial.fit(x, y, deg=3)  # suaviza sin imponer linealidad
        ax.plot(x_fit, p(x_fit), color=pal["contorno"], lw=2.5, label="Tendencia")

    # 3.4) Mostramos la correlación de Spearman (ρ) en el título del panel
    rho, _ = spearmanr(x, y)
    ax.set_title(f"Respiración vs Crecimiento {label}\nρ = {rho:.2f}",
                 fontsize=11, pad=8, color=pal["contorno"])

# Ejes y formato generales
axes[0].set_xlabel("Tasa de respiración")
axes[1].set_xlabel("Tasa de respiración")
axes[0].set_ylabel("Crecimiento (log10)")
for ax in axes:
    ax.grid(True, alpha=0.4, lw=0.6)
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(colors="#333333", labelsize=9)

# Título general y ajuste final
fig.suptitle("Densidad conjunta: Respiración ↔ Crecimiento microbiano", y=0.98,
             fontsize=13, color="#222222")
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Guardamos la figura
out = PROJECT_ROOT / "outputs" / "grafico5_joint_density.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300)
plt.show()
