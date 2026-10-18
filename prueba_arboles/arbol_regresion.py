"""Árbol de decisión para regresión (scikit-learn).

Al iniciar pide elegir uno de los datasets de la carpeta `datasets/`
(la última columna de cada CSV es el valor a predecir) y narra en consola
cada paso: carga, limpieza, entrenamiento, evaluación, interpretación y gráficas.
"""
import json
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
DATOS = BASE / "datasets"
LINEA = "─" * 72


def paso(n, titulo, explicacion=""):
    print(f"\n{LINEA}\n PASO {n}. {titulo}\n{LINEA}")
    if explicacion:
        print(f"  ℹ {explicacion}\n")


def elegir_dataset():
    catalogo = json.loads((DATOS / "catalogo.json").read_text(encoding="utf-8"))
    archivos = sorted(DATOS.glob("*.csv"))
    print(f"\n{LINEA}\n ÁRBOLES DE DECISIÓN PARA REGRESIÓN\n{LINEA}")
    print("Datasets disponibles:\n")
    for i, f in enumerate(archivos, 1):
        info = catalogo.get(f.name, {})
        print(f"  [{i}] {info.get('tema', f.stem):<20} {f.name}")
        if info:
            print(f"      {info['descripcion']}\n      Se predice: {info['objetivo']}\n")
    while True:
        op = input(f"Elige un dataset (1-{len(archivos)}): ").strip()
        if op.isdigit() and 1 <= int(op) <= len(archivos):
            f = archivos[int(op) - 1]
            return f, catalogo.get(f.name, {})
        print("  Opción no válida, escribe solo el número.")


def cargar_y_limpiar(archivo):
    df = pd.read_csv(archivo)
    objetivo = df.columns[-1]
    print(f"  ✔ Leí {archivo.name}: {len(df)} filas y {df.shape[1] - 1} variables de entrada.")
    print(f"  ✔ Variable objetivo (lo que vamos a predecir): '{objetivo}'")
    print("\n  Primeras filas:\n")
    print(df.head().to_string(index=False))

    nulos = int(df.isna().any(axis=1).sum())
    if nulos:
        df = df.dropna()
        print(f"\n  ⚠ Había {nulos} filas con datos faltantes; las eliminé (quedan {len(df)}).")
    else:
        print("\n  ✔ No hay datos faltantes.")

    X, y = df.drop(columns=objetivo), df[objetivo]
    texto = X.select_dtypes(exclude="number").columns.tolist()
    if texto:
        # sklearn solo acepta números: cada categoría se vuelve una columna 0/1
        X = pd.get_dummies(X, columns=texto, dtype=int)
        print(f"  ⚠ Las columnas {texto} son texto. Las convertí a columnas 0/1 (one-hot)."
              f" Ahora hay {X.shape[1]} variables.")
    else:
        print("  ✔ Todas las variables ya son numéricas.")

    print(f"\n  El objetivo '{objetivo}' va de {y.min():.2f} a {y.max():.2f}"
          f" (promedio {y.mean():.2f}, desviación estándar {y.std():.2f}).")
    return X, y, objetivo


def evaluar(nombre, modelo, X_tr, y_tr, X_te, y_te):
    p_tr, p_te = modelo.predict(X_tr), modelo.predict(X_te)
    r2_tr, r2_te = r2_score(y_tr, p_tr), r2_score(y_te, p_te)
    mae, rmse = mean_absolute_error(y_te, p_te), np.sqrt(mean_squared_error(y_te, p_te))
    print(f"  {nombre:<22} | R² train={r2_tr:6.3f}  R² test={r2_te:6.3f}"
          f" | MAE test={mae:8.3f} | RMSE test={rmse:8.3f}"
          f" | hojas={modelo.get_n_leaves()}")
    return dict(nombre=nombre, r2_tr=r2_tr, r2_te=r2_te, mae=mae, pred=p_te)


def explicar_un_caso(arbol, X_te, y_te, objetivo):
    """Recorre el árbol pequeño para un caso del test, como se haría a mano."""
    fila = X_te.iloc[[0]]
    t = arbol.tree_
    print(f"  Tomo el primer caso del test (valor real de '{objetivo}' = {y_te.iloc[0]:.2f}):\n")
    for nodo in arbol.decision_path(fila).indices:
        if t.children_left[nodo] == -1:  # hoja
            print(f"   → Llegué a una hoja con {t.n_node_samples[nodo]} casos de entrenamiento."
                  f" Su promedio es la predicción: {t.value[nodo][0][0]:.2f}")
            break
        col, umbral = X_te.columns[t.feature[nodo]], t.threshold[nodo]
        valor = fila.iloc[0, t.feature[nodo]]
        lado = "sí → izquierda" if valor <= umbral else "no → derecha"
        print(f"   ¿{col} <= {umbral:.2f}?  (este caso tiene {valor:.2f})  {lado}")


def main():
    archivo, info = elegir_dataset()
    out = BASE / "salidas" / archivo.stem
    out.mkdir(parents=True, exist_ok=True)

    paso(1, f"Cargar y preparar los datos — {info.get('tema', archivo.stem)}",
         "Un árbol necesita una tabla numérica sin huecos: X = variables de entrada, y = objetivo.")
    X, y, objetivo = cargar_y_limpiar(archivo)

    paso(2, "Separar entrenamiento y prueba",
         "El modelo aprende con el 80 % de los datos (train) y se califica con el 20 % que nunca vio"
         " (test). Así medimos si generaliza o solo memoriza.")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"  ✔ Train: {len(X_tr)} filas | Test: {len(X_te)} filas (semilla {SEED}, reproducible).")

    paso(3, "Entrenar tres árboles para compararlos",
         "En cada nodo el árbol prueba cortes 'variable <= umbral' y se queda con el que más reduce"
         " el error cuadrático. Cada hoja predice el promedio de sus casos.")
    print("  • Árbol sin límites: crece hasta que cada hoja es casi un solo caso...")
    profundo = DecisionTreeRegressor(random_state=SEED).fit(X_tr, y_tr)
    print(f"    listo: profundidad {profundo.get_depth()}, {profundo.get_n_leaves()} hojas.")
    print("  • Árbol con max_depth=3: máximo 3 preguntas, fácil de leer...")
    simple = DecisionTreeRegressor(max_depth=3, random_state=SEED).fit(X_tr, y_tr)
    print(f"    listo: {simple.get_n_leaves()} hojas.")
    print("  • Árbol ajustado con GridSearchCV: pruebo 20 combinaciones de max_depth y"
          " min_samples_leaf,\n    cada una con validación cruzada de 5 partes (100 entrenamientos)...")
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=SEED),
        {"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 5, 20, 50]},
        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
    ).fit(X_tr, y_tr)
    mejor = grid.best_estimator_
    print(f"    listo: la mejor combinación fue {grid.best_params_}"
          f" (RMSE promedio en validación = {-grid.best_score_:.3f}).")

    paso(4, "Evaluar los modelos con datos que no vieron",
         "R² = qué fracción de la variación explica el modelo (1 es perfecto, 0 es igual a adivinar"
         " el promedio). MAE y RMSE = error típico, en las mismas unidades del objetivo.")
    res = [evaluar("Sin límites", profundo, X_tr, y_tr, X_te, y_te),
           evaluar("max_depth=3", simple, X_tr, y_tr, X_te, y_te),
           evaluar("Ajustado (GridSearch)", mejor, X_tr, y_tr, X_te, y_te)]
    sin_lim, _, ajustado = res
    print(f"\n  Lectura:")
    print(f"  • Sin límites: R² train {sin_lim['r2_tr']:.2f} vs test {sin_lim['r2_te']:.2f}."
          f" La brecha de {sin_lim['r2_tr'] - sin_lim['r2_te']:.2f} indica SOBREAJUSTE (memorizó).")
    print(f"  • max_depth=3: es simple; si su R² es bajo en train y test, está SUBAJUSTADO.")
    print(f"  • Ajustado: en promedio se equivoca por ±{ajustado['mae']:.2f} en '{objetivo}'.")
    print(f"  • El mejor en test es: {max(res, key=lambda r: r['r2_te'])['nombre']}.")

    paso(5, "¿Qué variables usa más el modelo ajustado?",
         "La importancia mide cuánto redujo el error cada variable, sumando todos sus cortes. Suma 1.")
    imp = pd.Series(mejor.feature_importances_, index=X.columns).sort_values(ascending=False)
    for col, v in imp.head(8).items():
        print(f"  {col:<28} {v:5.3f} {'█' * round(v * 40)}")

    paso(6, "Leer el árbol pequeño como reglas if/else",
         "Esto es lo que hace interpretable a un árbol: cada camino de la raíz a una hoja es una regla.")
    print(export_text(simple, feature_names=list(X.columns), decimals=2))
    explicar_un_caso(simple, X_te, y_te, objetivo)

    paso(7, "Generar gráficas", f"Se guardan en {out.relative_to(BASE)}/")
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(simple, feature_names=X.columns, filled=True, rounded=True, fontsize=8, ax=ax)
    fig.savefig(out / "arbol_depth3.png", dpi=120, bbox_inches="tight")
    print("  ✔ arbol_depth3.png      → el diagrama del árbol de 3 niveles (color = valor predicho).")

    pred = ajustado["pred"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te, pred, s=6, alpha=0.4)
    lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    ax.plot(lim, lim, "r--")
    ax.set(xlabel=f"{objetivo} real", ylabel=f"{objetivo} predicho", title="Real vs predicho (test)")
    fig.savefig(out / "real_vs_predicho.png", dpi=120, bbox_inches="tight")
    print("  ✔ real_vs_predicho.png  → cuanto más cerca de la diagonal roja, mejor la predicción.")

    print("  … entrenando árboles de profundidad 1 a 20 para la curva de sobreajuste...")
    prof = range(1, 21)
    arboles = [DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_tr, y_tr) for d in prof]
    tr = [r2_score(y_tr, a.predict(X_tr)) for a in arboles]
    te = [r2_score(y_te, a.predict(X_te)) for a in arboles]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prof, tr, label="train"); ax.plot(prof, te, label="test")
    ax.set(xlabel="max_depth", ylabel="R²", title="Sobreajuste vs profundidad"); ax.legend()
    fig.savefig(out / "curva_profundidad.png", dpi=120, bbox_inches="tight")
    mejor_d = prof[int(np.argmax(te))]
    print(f"  ✔ curva_profundidad.png → train sube siempre; test toca su máximo en"
          f" max_depth={mejor_d} (R²={max(te):.3f}) y después deja de mejorar.")

    print(f"\n{LINEA}\n ✅ Listo. Revisa las gráficas en: {out}\n{LINEA}")


if __name__ == "__main__":
    main()
