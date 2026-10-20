"""Actividad: árbol de decisión para regresión sobre cualquier CSV.

Uso:
    python actividad.py                                  # usa el CSV que está junto al script
    python actividad.py seguros_gasto_medico.csv
    python actividad.py mis_datos.csv --objetivo precio  # otra columna como objetivo

Por defecto la última columna es lo que se predice (debe ser numérica).
El script solo calcula y muestra resultados; la interpretación es tuya.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

SEED = 42
BASE = Path(__file__).parent
LINEA = "─" * 72
MAX_CATEGORIAS = 30  # texto con más valores distintos suele ser un id, nombre o fecha
MAX_FILAS = 50_000  # arriba de esto la búsqueda tarda demasiado para una clase


def paso(n, titulo):
    print(f"\n{LINEA}\n PASO {n}. {titulo}\n{LINEA}")


def salir(msg):
    sys.exit(f"\n  ✖ {msg}\n")


def elegir_archivo():
    archivos = sorted(BASE.glob("*.csv")) or sorted((BASE / "datasets").glob("*.csv"))
    if not archivos:
        salir("No hay ningún CSV junto al script. Pasa la ruta: python actividad.py archivo.csv")
    if len(archivos) == 1:
        return archivos[0]
    for i, f in enumerate(archivos, 1):
        print(f"  [{i}] {f.name}")
    while True:
        op = input(f"Elige un dataset (1-{len(archivos)}): ").strip()
        if op.isdigit() and 1 <= int(op) <= len(archivos):
            return archivos[int(op) - 1]
        print("  Opción no válida, escribe solo el número.")


def leer_csv(archivo):
    # sep=None detecta "," o ";" (CSV exportados de Excel en español)
    try:
        return pd.read_csv(archivo, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(archivo, sep=None, engine="python", encoding="latin-1")


def cargar_y_limpiar(archivo, objetivo):
    df = leer_csv(archivo)
    objetivo = objetivo or df.columns[-1]
    if objetivo not in df.columns:
        salir(f"No existe la columna '{objetivo}'. Columnas: {list(df.columns)}")
    print(f"  Archivo: {archivo.name} | {len(df)} filas | {df.shape[1] - 1} variables de entrada")
    print(f"  Objetivo: '{objetivo}'\n")
    print(df.head().to_string(index=False))

    df[objetivo] = pd.to_numeric(df[objetivo], errors="coerce")
    if df[objetivo].notna().sum() == 0:
        salir(f"'{objetivo}' no es numérica: esto sería clasificación, no regresión.")

    vacias = [c for c in df.columns if c != objetivo and df[c].isna().mean() > 0.5]
    if vacias:
        df = df.drop(columns=vacias)
        print(f"\n  - Eliminé columnas con más de la mitad vacía: {vacias}")
    antes = len(df)
    df = df.dropna()
    print(f"  - Filas con datos faltantes eliminadas: {antes - len(df)} (quedan {len(df)})")
    if len(df) > MAX_FILAS:
        df = df.sample(MAX_FILAS, random_state=SEED)
        print(f"  - Usé una muestra aleatoria de {MAX_FILAS} filas")
    if len(df) < 30:
        salir(f"Solo quedan {len(df)} filas; se necesitan al menos 30.")

    X, y = df.drop(columns=objetivo), df[objetivo]
    texto = X.select_dtypes(exclude="number").columns
    muchas = [c for c in texto if X[c].nunique() > MAX_CATEGORIAS]
    if muchas:
        X = X.drop(columns=muchas)
        print(f"  - Eliminé columnas de texto con más de {MAX_CATEGORIAS} valores distintos: {muchas}")
    texto = [c for c in texto if c not in muchas]
    if texto:
        X = pd.get_dummies(X, columns=texto, dtype=int)
        print(f"  - Columnas de texto convertidas a 0/1 (one-hot): {texto}")
    if X.shape[1] == 0:
        salir("No quedaron variables de entrada.")
    print(f"  - Variables finales: {X.shape[1]}")
    print(f"\n  '{objetivo}': mínimo {y.min():.2f} | máximo {y.max():.2f}"
          f" | promedio {y.mean():.2f} | desviación estándar {y.std():.2f}")
    return X, y, objetivo


def evaluar(nombre, modelo, X_tr, y_tr, X_te, y_te):
    p_te = modelo.predict(X_te)
    r2_tr = r2_score(y_tr, modelo.predict(X_tr))
    print(f"  {nombre:<22} {r2_tr:>9.3f} {r2_score(y_te, p_te):>9.3f}"
          f" {mean_absolute_error(y_te, p_te):>11.3f} {np.sqrt(mean_squared_error(y_te, p_te)):>11.3f}"
          f" {modelo.get_depth():>6} {modelo.get_n_leaves():>6}")
    return p_te


def main():
    ap = argparse.ArgumentParser(description="Árbol de decisión para regresión sobre un CSV.")
    ap.add_argument("archivo", nargs="?", help="ruta del CSV (si se omite, usa el que está junto al script)")
    ap.add_argument("--objetivo", help="columna a predecir (por defecto la última)")
    args = ap.parse_args()
    archivo = Path(args.archivo) if args.archivo else elegir_archivo()
    if not archivo.is_file():
        salir(f"No encuentro el archivo {archivo}")
    out = BASE / "salidas" / archivo.stem
    out.mkdir(parents=True, exist_ok=True)

    paso(1, "Cargar y preparar los datos")
    X, y, objetivo = cargar_y_limpiar(archivo, args.objetivo)

    paso(2, "Separar entrenamiento (80 %) y prueba (20 %)")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"  Train: {len(X_tr)} filas | Test: {len(X_te)} filas | semilla {SEED}")

    paso(3, "Entrenar tres árboles")
    profundo = DecisionTreeRegressor(random_state=SEED).fit(X_tr, y_tr)
    simple = DecisionTreeRegressor(max_depth=3, random_state=SEED).fit(X_tr, y_tr)
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=SEED),
        {"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 5, 20, 50]},
        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
    ).fit(X_tr, y_tr)
    mejor = grid.best_estimator_
    print(f"  GridSearchCV (5 folds) eligió: {grid.best_params_}"
          f" | RMSE promedio en validación: {-grid.best_score_:.3f}")

    paso(4, "Métricas")
    print(f"  {'Modelo':<22} {'R² train':>9} {'R² test':>9} {'MAE test':>11} {'RMSE test':>11}"
          f" {'prof.':>6} {'hojas':>6}")
    evaluar("Sin límites", profundo, X_tr, y_tr, X_te, y_te)
    evaluar("max_depth=3", simple, X_tr, y_tr, X_te, y_te)
    pred = evaluar("Ajustado (GridSearch)", mejor, X_tr, y_tr, X_te, y_te)

    paso(5, "Importancia de variables (árbol ajustado)")
    imp = pd.Series(mejor.feature_importances_, index=X.columns).sort_values(ascending=False)
    for col, v in imp.items():
        print(f"  {col:<32} {v:5.3f} {'█' * round(v * 40)}")

    paso(6, "Reglas del árbol max_depth=3")
    print(export_text(simple, feature_names=list(X.columns), decimals=2))

    paso(7, "Caso para predecir a mano (primera fila del test)")
    caso = X_te.iloc[0]
    for col, v in caso.items():
        print(f"  {col:<32} {v}")
    print(f"\n  Valor real de '{objetivo}': {y_te.iloc[0]:.2f}")

    paso(8, "R² según la profundidad máxima")
    print(f"  {'max_depth':>9} {'R² train':>9} {'R² test':>9}")
    prof = range(1, 21)
    tr, te = [], []
    for d in prof:
        a = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_tr, y_tr)
        tr.append(r2_score(y_tr, a.predict(X_tr)))
        te.append(r2_score(y_te, a.predict(X_te)))
        print(f"  {d:>9} {tr[-1]:>9.3f} {te[-1]:>9.3f}")

    paso(9, "Gráficas")
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(simple, feature_names=X.columns, filled=True, rounded=True, fontsize=8, ax=ax)
    fig.savefig(out / "arbol_depth3.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te, pred, s=6, alpha=0.4)
    lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    ax.plot(lim, lim, "r--")
    ax.set(xlabel=f"{objetivo} real", ylabel=f"{objetivo} predicho",
           title="Real vs predicho (test, árbol ajustado)")
    fig.savefig(out / "real_vs_predicho.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prof, tr, marker=".", label="train")
    ax.plot(prof, te, marker=".", label="test")
    ax.set(xlabel="max_depth", ylabel="R²", title="R² vs profundidad", xticks=list(prof))
    ax.legend()
    fig.savefig(out / "curva_profundidad.png", dpi=120, bbox_inches="tight")
    print(f"  Guardadas en {out}: arbol_depth3.png, real_vs_predicho.png, curva_profundidad.png")


if __name__ == "__main__":
    main()
