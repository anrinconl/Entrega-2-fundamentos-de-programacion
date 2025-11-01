# Entrega#2.py
import pandas as pd
from pathlib import Path

# (1) Rutas robustas relativas a este archivo .py
BASE = Path(__file__).resolve().parent   # carpeta donde está este script
ENV_PATH = BASE / "environmental-data.csv"
MICRO_PATH = BASE / "microbial-responses.csv"

# (2) Comprobación temprana de existencia (errores claros)
for p in (ENV_PATH, MICRO_PATH):
    if not p.exists():
        raise FileNotFoundError(f"No encuentro el archivo: {p}")

# (3) Lector "inteligente" que intenta separador/encoding comunes
def read_csv_smart(path: Path) -> pd.DataFrame:
    try:
        # Intento directo (rápido) con coma y utf-8
        return pd.read_csv(path)
    except Exception:
        # Si falla, intento detectar separador y usar engine='python'
        # También pruebo un encoding alternativo
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")

# (4) Leer archivos
df_env = read_csv_smart(ENV_PATH)
df_micro = read_csv_smart(MICRO_PATH)

# (5) Inspección rápida (puedes comentar si no lo quieres en consola)
print("\n[environmental-data.csv]")
print(df_env.shape)
print(df_env.head(3))
print(df_env.info())

print("\n[microbial-responses.csv]")
print(df_micro.shape)
print(df_micro.head(3))
print(df_micro.info())