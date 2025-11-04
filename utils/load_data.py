# utils/load_data.py
from pathlib import Path
import pandas as pd

# Directorio raíz del proyecto (sube un nivel desde /utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _read_csv_smart(path: Path) -> pd.DataFrame:
    """Lector robusto para CSV (coma/; y utf-8/latin-1)."""
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")

def load_env_micro():
    """Carga los dos CSV ubicados en la raíz del proyecto."""
    env_path = PROJECT_ROOT / "data/environmental-data.csv"
    micro_path = PROJECT_ROOT / "data/microbial-responses.csv"

    # Comprobación temprana (mensajes claros)
    for p in (env_path, micro_path):
        if not p.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {p}")

    df_env = _read_csv_smart(env_path)
    df_micro = _read_csv_smart(micro_path)
    return df_env, df_micro

def prep_graph1(df_env: pd.DataFrame, df_micro: pd.DataFrame, min_n: int = 5):
    """
    Prepara el DataFrame para el Gráfico 1:
      - Une env (land_cover) con micro (respiration_rate) por soil_number.
      - Limpia tipos/nulos.
      - Filtra categorías con menos de min_n observaciones.
      - Devuelve (df_limpio, orden_categorias_por_mediana).
    """
    env_cols = ["soil_number", "land_cover"]
    micro_cols = ["soil_number", "respiration_rate"]

    df = pd.merge(
        df_micro[micro_cols],
        df_env[env_cols],
        on="soil_number",
        how="inner"
    )

    df["land_cover"] = df["land_cover"].astype("category", errors="ignore")
    df["respiration_rate"] = pd.to_numeric(df["respiration_rate"], errors="coerce")
    df = df.dropna(subset=["land_cover", "respiration_rate"]).copy()

    if min_n and min_n > 0:
        counts = df["land_cover"].value_counts()
        valid_levels = counts[counts >= min_n].index
        df = df[df["land_cover"].isin(valid_levels)].copy()
        if hasattr(df["land_cover"], "cat"):
            df["land_cover"] = df["land_cover"].cat.remove_unused_categories()

    order_by_median = (
        df.groupby("land_cover", observed=True)["respiration_rate"]
          .median()
          .sort_values(ascending=False)
          .index
          .tolist()
    )
    return df, order_by_median
