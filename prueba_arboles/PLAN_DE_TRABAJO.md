# Árboles de decisión para regresión — explicación y plan de trabajo

## Cómo ejecutar
```bash
cd "/home/bbcoke/7MO SEMESTRE/Ciencia de Datos/Actividad 3/prueba_arboles"
source prueba_arboles/bin/activate
python arbol_regresion.py
```

## Cómo funciona el script (`arbol_regresion.py`)
1. **Pide elegir** un dataset de `datasets/` (6 temáticas; la última columna del CSV es el objetivo), elimina filas con nulos y convierte columnas de texto a 0/1. La narración de cada paso se imprime en consola.
2. **Divide** 80 % train / 20 % test (`random_state=42`).
3. **Entrena tres `DecisionTreeRegressor`:**
   - *Sin límites*: crece hasta memorizar el train (sobreajuste).
   - *`max_depth=3`*: pequeño e interpretable (subajuste).
   - *Ajustado*: `GridSearchCV` (5 folds) busca `max_depth` y `min_samples_leaf`.
4. **Evalúa** con R², MAE y RMSE en train y test.
5. **Interpreta**: importancia de variables y reglas `if/else` del árbol pequeño.
6. **Grafica** en `salidas/<dataset>/`: árbol, real vs predicho y curva R² vs profundidad.

### Idea del algoritmo
En cada nodo el árbol prueba umbrales de cada variable (`MedInc <= 5.09`) y elige el que más reduce el error cuadrático (MSE) de los dos grupos resultantes. Repite recursivamente. Cada hoja predice el **promedio** del objetivo de las muestras que caen en ella → la predicción es una función escalonada.

## Modelo a entrenar
`sklearn.tree.DecisionTreeRegressor` (criterio `squared_error`). Hiperparámetros clave: `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha` (poda).

## Output esperado (resultado real de la corrida)
| Modelo | R² train | R² test | MAE test | RMSE test |
|---|---|---|---|---|
| Sin límites | 1.000 | 0.623 | 0.454 | 0.703 |
| max_depth=3 | 0.538 | 0.510 | 0.598 | 0.802 |
| Ajustado (`max_depth=None`, `min_samples_leaf=20`) | 0.809 | 0.727 | 0.404 | 0.598 |

- Variable más importante: `MedInc` (~0.63), luego `AveOccup`, `Latitude`, `Longitude`.
- Lectura: el árbol sin límites tiene R² train = 1 y test 0.62 → sobreajuste. El ajustado generaliza mejor (0.73). Resultados pueden variar levemente con otra versión de sklearn.

## Plan de trabajo
| # | Paso | Qué hacer | Qué debes entender |
|---|---|---|---|
| 1 | Ejecutar | Correr el script y revisar la consola y `salidas/` | Qué métricas se imprimen |
| 2 | Leer el código | Seguir `main()` línea a línea | Split, fit, predict, métricas |
| 3 | Interpretar el árbol | Abrir `arbol_depth3.png` y comparar con las reglas impresas | Cómo se llega a una hoja y qué valor predice |
| 4 | Sobreajuste | Ver `curva_profundidad.png` | Train sube siempre; test se estanca y baja |
| 5 | Experimentar | Cambiar `max_depth`, `min_samples_leaf`, `random_state`, `test_size` | Sensibilidad del árbol (alta varianza) |
| 6 | Poda | Probar `ccp_alpha` (0.001, 0.01) | Alternativa a limitar profundidad |
| 7 | Comparar | Agregar `RandomForestRegressor` y `LinearRegression` | Por qué los ensambles mejoran un árbol solo |
| 8 | Datos propios | Copiar un CSV a `datasets/` con el objetivo como última columna | Preparar X, y; manejo de nulos/categóricas |
| 9 | Reporte | Resumir hallazgos: métricas, variables clave, conclusión | Justificar el modelo elegido |

## Criterios de éxito
- Script corre sin errores y genera 3 gráficas.
- Puedes explicar por qué el árbol sin límites sobreajusta.
- Puedes leer una regla del árbol y predecir a mano un caso.
