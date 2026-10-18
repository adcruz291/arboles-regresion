# Notas del orador — Árboles de decisión para regresión

> Notas de apoyo para la exposición, no un guion para leer. Los números citados salen de correr
> `arbol_regresion.py` con scikit-learn 1.9.1 y semilla 42; con otra versión pueden variar un poco.

---

## 1. Cómo se generó el script: lo que le pedimos al modelo

El código fue escrito por un modelo de IA, **Claude Opus 5.5 (modo de razonamiento alto) dentro de
Claude Code**, el asistente de programación que trabaja en la terminal sobre la carpeta del proyecto.
Nosotros definimos en los prompts qué tenía que hacer el programa y cómo debía comportarse. El modelo
lo escribió, lo ejecutó para comprobar que funcionara y propuso algunos detalles de implementación.

**Requerimientos de la primera versión**
- Un árbol de decisión para **regresión** con scikit-learn (`DecisionTreeRegressor`).
- Comparar un árbol sin límites, uno pequeño e interpretable y uno con hiperparámetros elegidos por
  búsqueda con validación cruzada.
- Métricas de evaluación, importancia de variables, reglas del árbol en texto y gráficas guardadas
  en una carpeta.

**Requerimientos de la segunda versión (la que presentamos)**
- Preparar **varios datasets de diferentes temáticas** y guardarlos en una carpeta.
- Que el script, **al iniciar, pida elegir el dataset** que se quiere usar.
- Que **narre en consola lo que va haciendo**, para que la salida sea más legible y amena que una
  lista de números.
- Estas notas del orador.

**Detalles que propuso el modelo al implementarlo**
- Traducir las columnas al español y convertir unidades: km/L en lugar de millas por galón, kg en
  lugar de libras.
- Una convención simple: en cada CSV, **la última columna es lo que se predice**. Un `catalogo.json`
  guarda el tema y la descripción que aparecen en el menú.
- Dejar a propósito un dataset con datos faltantes (autos) y dos con columnas de texto (autos y
  bicicletas), para que se vea la etapa de limpieza.
- Recorrer paso a paso un caso del conjunto de prueba dentro del árbol pequeño, mostrando cada
  pregunta y hacia dónde se va, como se haría a mano.
- Guardar las gráficas en una subcarpeta por dataset (`salidas/<dataset>/`) para que no se
  sobrescriban entre corridas.

---

## 2. Diversidad de datasets

La idea es mostrar que **el mismo algoritmo, sin cambiar una línea**, se aplica a problemas de áreas
muy distintas, y que su desempeño depende mucho de los datos. Todos son datasets públicos y conocidos
(UCI / OpenML / scikit-learn).

| # | Tema | Archivo | Filas | Se predice | R² test (ajustado) |
|---|---|---|---|---|---|
| 1 | Automotriz | `autos_rendimiento.csv` | 398 | rendimiento en km/L | 0.85 |
| 2 | Movilidad urbana | `bicicletas_demanda.csv` | 17 379 | bicis rentadas por hora | 0.91 |
| 3 | Ingeniería civil | `concreto_resistencia.csv` | 1 030 | resistencia del concreto (MPa) | 0.83 |
| 4 | Salud | `diabetes_progresion.csv` | 442 | avance de la diabetes en un año | 0.38 |
| 5 | Alimentos y química | `vino_calidad.csv` | 1 599 | calidad del vino (0-10) | 0.33 |
| 6 | Bienes raíces | `viviendas_california.csv` | 20 640 | valor de la vivienda (100k USD) | 0.73 |

Qué enseña cada uno:

- **Bicicletas**: el mejor para la demo en vivo. El árbol pequeño sale casi intuitivo: la primera
  pregunta es la *hora* (de madrugada casi nadie renta), luego la *temperatura* y el *año* (el
  servicio creció en 2012). La variable `hora` concentra el 63 % de la importancia. También muestra
  la codificación de texto (`estacion`, `clima`).
- **Autos**: el caso de "datos sucios". Tiene 6 autos sin dato de caballos de fuerza y una columna
  de texto (`origen`). Con pocos datos (392 filas) el árbol ajustado sigue siendo bueno.
- **Concreto**: relación casi física y con poco ruido. Aquí el árbol **sin límites** resulta ser el
  mejor (GridSearch lo elige). Sirve para decir que limitar el árbol no siempre ayuda: depende de
  cuánto ruido tengan los datos.
- **Diabetes y vino**: los casos difíciles. El árbol sin límites tiene R² = 1.00 en entrenamiento
  y **0.07-0.08 en prueba**, que es el ejemplo más dramático de sobreajuste. Incluso el mejor árbol
  explica poco (R² ≈ 0.35): hay mucho ruido o faltan variables, y un solo árbol tiene un techo. Es
  un buen momento para mencionar Random Forest y Gradient Boosting.
- **Viviendas**: el clásico de los libros, de tamaño grande. Sobreajuste claro (1.00 contra 0.62)
  que el ajuste corrige (0.73). La variable más importante es el ingreso medio del distrito.

---

## 3. Conceptos de los árboles de regresión

**Qué es.** Un modelo de aprendizaje supervisado que predice un **número** (regresión) haciendo una
serie de preguntas de sí o no sobre las variables, como un diagrama de flujo. A diferencia de un
árbol de clasificación, la hoja no da una clase, sino un valor.

**Cómo decide cada corte.** En cada nodo prueba todas las variables y todos los umbrales posibles
(`hora <= 6.5`, `temperatura <= 20.1`, …). Se queda con el corte que más reduce el **error
cuadrático medio (MSE)** de los dos grupos resultantes, es decir, el que deja grupos lo más
homogéneos posible. Luego repite el proceso dentro de cada grupo. A esto se le llama partición
recursiva y es un algoritmo *greedy*: elige el mejor corte en ese momento, sin mirar hacia adelante.

**Qué predice una hoja.** El **promedio** del objetivo de los casos de entrenamiento que cayeron
ahí. Por eso la predicción es una función escalonada: el árbol divide el espacio en rectángulos y
asigna un valor constante a cada uno.

**Profundidad y sobreajuste.** Un árbol sin límites sigue cortando hasta que casi cada hoja tiene
un solo caso: memoriza el entrenamiento (R² = 1) pero falla con datos nuevos (**sobreajuste**, alta
varianza). Un árbol muy corto no captura el patrón (**subajuste**, alto sesgo). El punto bueno está
en medio.

**Hiperparámetros que lo controlan.**
- `max_depth`: número máximo de preguntas encadenadas.
- `min_samples_leaf`: mínimo de casos por hoja. Evita hojas basadas en 1 o 2 ejemplos y suaviza
  las predicciones.
- Otros que no usamos: `min_samples_split` y `ccp_alpha` (poda por complejidad).

**Validación cruzada.** Para elegir hiperparámetros sin "hacer trampa" con el test, se divide el
entrenamiento en 5 partes. Se entrena con 4 y se valida con la quinta, rotando. Se elige la
combinación con menor error promedio. El test se usa solo al final.

**Métricas.**
- **R²**: qué fracción de la variación del objetivo explica el modelo. 1 es perfecto y 0 equivale
  a predecir siempre el promedio.
- **MAE**: error absoluto promedio, en las unidades del objetivo. Por ejemplo, "se equivoca por ±31
  bicicletas".
- **RMSE**: parecido al MAE, pero castiga más los errores grandes.

**Importancia de variables.** Cuánto redujo el error cada variable, sumando todos los cortes donde
se usó. Suma 1. Dice qué usa el modelo, no qué *causa* el resultado.

**Ventajas.** Es interpretable (se lee como reglas), no necesita escalar ni normalizar las
variables, captura relaciones no lineales e interacciones, y es rápido.

**Limitaciones y errores comunes.**
- Es inestable: pequeños cambios en los datos cambian mucho el árbol.
- Sobreajusta si no se limita.
- No extrapola: nunca predice fuera del rango visto en el entrenamiento.
- Errores comunes: evaluar con los mismos datos de entrenamiento, olvidar codificar las variables
  de texto, y confundir importancia con causalidad.

---

## 4. Dónde aparece cada concepto en el script

La consola está organizada en 7 pasos numerados. Cada uno empieza con una línea `ℹ` que explica el
porqué.

| Paso | Qué hace el script | Concepto que ilustra |
|---|---|---|
| Menú | Lista los 6 datasets con tema y objetivo; valida la opción elegida | Mismo modelo, distintos dominios |
| 1 | Carga el CSV, elimina filas con nulos y aplica one-hot a columnas de texto | Preparación de datos: `X` numérica, `y` objetivo |
| 2 | `train_test_split` 80/20 con semilla 42 | Evaluar con datos no vistos; reproducibilidad |
| 3 | Entrena el árbol sin límites, el de `max_depth=3` y `GridSearchCV` (20 combinaciones × 5 folds) | Profundidad, hiperparámetros, validación cruzada |
| 4 | Tabla de R² train/test, MAE, RMSE y número de hojas, más una lectura automática | Sobreajuste (brecha train-test), subajuste, métricas |
| 5 | Barras de importancia de variables | Qué variables reducen más el error |
| 6 | Reglas `if/else` del árbol pequeño y recorrido de un caso real hasta su hoja | Interpretabilidad; la hoja predice un promedio |
| 7 | Tres gráficas en `salidas/<dataset>/` | Ver abajo |

**Las gráficas**
- `arbol_depth3.png`: el diagrama del árbol pequeño. El color indica el valor predicho. Se puede
  comparar con las reglas impresas en el paso 6.
- `real_vs_predicho.png`: cada punto es un caso del test. Cuanto más cerca de la diagonal roja,
  mejor. Se notan **franjas horizontales**: son las hojas, que predicen el mismo valor para
  muchos casos.
- `curva_profundidad.png`: R² de entrenamiento y de prueba para profundidades de 1 a 20. La curva
  de entrenamiento sube siempre y la de prueba se estanca o baja. La separación entre ambas es el
  sobreajuste dibujado.

**Momento sugerido para la demo.** Ejecutar con **bicicletas** y detenerse en el paso 6. El
recorrido del caso es muy claro: "¿hora ≤ 6.5? No (19 h) → ¿temperatura ≤ 20 °C? No (32.8 °C) →
¿año ≤ 2011? No (2012) → hoja con 2 816 casos, promedio 382.7". El valor real era 425. Después,
correr **diabetes** para mostrar el contraste: R² = 1.00 en entrenamiento contra 0.08 en prueba.

**Cómo ejecutarlo**
```bash
cd prueba_arboles
source prueba_arboles/bin/activate
python arbol_regresion.py
```
