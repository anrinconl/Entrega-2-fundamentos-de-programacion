from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Aseguramos acceso al directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cargamos la función que importa los archivos CSV
from utils.load_data import load_env_micro


# 1. Función para preparar los datos en formato largo

def preparar_long(df_env, df_micro) -> pd.DataFrame:
    # Unimos los datos ambientales y microbianos según el número de suelo
    df = pd.merge(
        df_micro[["soil_number", "time", "bacterial_growth_rate", "fungal_growth_rate"]],
        df_env[["soil_number", "land_cover"]],
        on="soil_number", how="inner",
    ).dropna(subset=["time", "bacterial_growth_rate", "fungal_growth_rate", "land_cover"])

    # Filtramos tiempos válidos y definimos el tipo de cobertura como variable categórica
    df = df[df["time"] >= 0].copy()
    df["land_cover"] = df["land_cover"].astype("category")

    # Reestructuramos la tabla: de formato ancho a largo (para comparar bacteria/hongo)
    long = df.melt(
        id_vars=["soil_number", "time", "land_cover"],
        value_vars=["bacterial_growth_rate", "fungal_growth_rate"],
        var_name="type", value_name="growth_rate"
    ).dropna(subset=["growth_rate"])

    # Simplificamos los nombres de tipo ("Bacterial", "Fungal")
    long["type"] = long["type"].str.replace("_growth_rate", "", regex=False).str.capitalize()

    # Devolvemos solo valores positivos de tasa de crecimiento
    return long[long["growth_rate"] > 0].copy()


# 2. Función para resumir los datos por intervalos de tiempo

def resumen_bins(df_cover: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    # Agrupamos el tiempo en 20 intervalos y calculamos los cuantiles (25%, 50%, 75%)
    tb = pd.cut(df_cover["time"], bins=n_bins)
    q = (df_cover.groupby(["type", tb], observed=True)["growth_rate"]
         .quantile([0.25, 0.5, 0.75]).unstack().reset_index()
         .rename(columns={0.25: "q25", 0.5: "q50", 0.75: "q75"}))
    # Calculamos el punto medio de cada intervalo para graficar
    q["t"] = q["time"].apply(lambda iv: (iv.left + iv.right) / 2)
    return q.dropna(subset=["q50"])



# 3. Función para ajustar los límites del eje Y (escala logarítmica)

def set_limites_log(ax, q: pd.DataFrame):
    ax.set_yscale("log")  # Usamos escala logarítmica para cubrir varios órdenes de magnitud
    if q.empty:
        ax.set_ylim(1e-4, 2e-1); return
    y_lo, y_hi = q["q25"].min(), q["q75"].max()
    lower = max(1e-5, y_lo / 1.6); upper = min(1.0, y_hi * 1.6)
    if not (upper > lower): lower, upper = 1e-4, 2e-1
    ax.set_ylim(lower, upper); ax.margins(x=0.02)


# 4. Cargamos y preparamos los datos

df_env, df_micro = load_env_micro()
long = preparar_long(df_env, df_micro)

# Obtenemos las coberturas del suelo presentes
covers = list(long["land_cover"].cat.categories) if hasattr(long["land_cover"], "cat") else sorted(long["land_cover"].unique())

# Calculamos los resúmenes estadísticos por cobertura
resumen = {c: resumen_bins(long[long["land_cover"] == c]) for c in covers}


# 5. Graficamos subplots por tipo de cobertura

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.ravel()

for ax, cover in zip(axes, covers):
    q = resumen[cover]
    # Graficamos la mediana (línea) y el rango intercuartílico (sombreado)
    for typ in ("Bacterial", "Fungal"):
        d = q[q["type"] == typ].sort_values("t")
        if d.empty: continue
        ax.fill_between(d["t"], d["q25"], d["q75"], alpha=0.25)  # Sombra (variabilidad)
        ax.plot(d["t"], d["q50"], lw=2, label=typ)              # Línea (mediana)
    ax.set_title(str(cover))
    ax.grid(True, which="major", alpha=0.6)
    set_limites_log(ax, q)  # Ajustamos límites logarítmicos


# Etiquetas y leyenda general
axes[2].set_xlabel("Tiempo")
axes[3].set_xlabel("Tiempo")
axes[0].set_ylabel("Tasa de crecimiento (log)")
axes[2].set_ylabel("Tasa de crecimiento (log)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", title="Tipo microbiano")

# Título general y formato final
fig.suptitle("Crecimiento bacteriano vs fúngico en el tiempo ", y=0.98)
fig.tight_layout(rect=[0, 0, 0.98, 0.96])

# Guardamos el gráfico en la carpeta outputs
out = PROJECT_ROOT / "outputs" / "grafico2_lineas_subplots.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300)
plt.show()
