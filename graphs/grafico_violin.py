import argparse
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Aseguramos que el script pueda importar los módulos del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importamos funciones para cargar y preparar los datos
from utils.load_data import load_env_micro, prep_graph1  # noqa: E402


def main(args):
    # Cargamos los datos ambientales y microbianos
    df_env, df_micro = load_env_micro()

    # Combinamos los dataframes y filtramos categorías con pocas observaciones
    df, order = prep_graph1(df_env, df_micro, min_n=args.min_n)

    # Definimos la variable dependiente: tasa de respiración microbiana
    y = "respiration_rate"

    # Si se activa la opción log, transformamos los valores a log10
    if args.log:
        df = df[df[y] > 0].copy()
        df["resp_log10"] = np.log10(df[y])
        yplot = "resp_log10"
    else:
        yplot = y

    # Definimos el estilo general del gráfico
    sns.set_theme(context="talk", style="whitegrid")

    # Creamos la figura base
    fig, ax = plt.subplots(figsize=(10, 5 + 0.2 * max(len(order), 1)))

    # Dibujamos un gráfico de violín para mostrar la distribución de las tasas por tipo de cobertura del suelo
    sns.violinplot(
        data=df, x="land_cover", y=yplot, order=order,
        inner="quartile", cut=0, linewidth=1.2, ax=ax
    )

    # Añadimos los puntos individuales sobre el violín (distribución de los datos reales)
    sns.stripplot(
        data=df, x="land_cover", y=yplot, order=order,
        jitter=0.15, size=3.5, alpha=0.7,
        color="Orange",
        zorder=3,
        ax=ax
    )

    # Colocamos los títulos, etiquetas y rotamos los nombres del eje X
    ax.set_xlabel("Cobertura del suelo (land_cover)")
    ax.set_ylabel("Tasa de respiración (log10)" if args.log else "Tasa de respiración")
    ax.set_title("Distribución de tasas de respiración por tipo de cobertura")
    ax.tick_params(axis="x", rotation=35)

    # Trazamos líneas de referencia: media y mediana global
    vals = df[yplot]
    ax.axhline(vals.mean(), ls="--", lw=1.2, label="Media global")
    ax.axhline(vals.median(), ls=":", lw=1.2, label="Mediana global")
    ax.legend(loc="upper right")

    # Creamos la carpeta de salida y guardamos la figura generada
    outdir = PROJECT_ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "_log" if args.log else ""
    if args.boxen: suffix += "_boxen"
    if args.strip: suffix += "_strip"
    outfile = outdir / f"grafico1_violin{suffix}.png"

    # Ajustamos el diseño final y guardamos el gráfico en formato PNG
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()


if __name__ == "__main__":
    # Configuramos los argumentos del script (permite cambiar opciones al ejecutarlo)
    parser = argparse.ArgumentParser(description="Gráfico 1: Violin–Boxen–Point por land_cover")
    parser.add_argument("--min-n", type=int, default=5, help="mínimo de observaciones por categoría")
    parser.add_argument("--log", action="store_true", help="usar escala log10 de la tasa")
    parser.add_argument("--boxen", action="store_true", help="añadir capa boxen")
    parser.add_argument("--strip", action="store_true", help="añadir capa strip (puntos)")
    args = parser.parse_args()
    main(args)
