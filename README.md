# Regresión lineal con NumPy

El notebook [main.ipynb](main.ipynb) explora `Student_Performance.csv` e implementa
regresión lineal con actualización manual de los parámetros:
`beta = beta - alpha * (X.T @ (X @ beta - y) / len(y))`.

Compara dos experimentos: **A: Batch Gradient Descent** y **B: Mini Batch Gradient
Descent**, con minilotes de 64, tasa de aprendizaje 0.05, 200 épocas y semilla 42.
Ambos usan la misma división 70% entrenamiento, 15% validación y 15% prueba,
inicialización en ceros y estandarización calculada solo con entrenamiento.
Incluye métricas MSE, RMSE, MAE y R², curvas de costo, tiempos, cantidad de
actualizaciones y coeficientes en unidades originales. El costo optimizado es MSE/2.

El MSE se calcula manualmente como la suma de los errores al cuadrado dividida
por el número de observaciones. Se registra sobre todo entrenamiento y toda
validación después de **cada actualización de pesos**, incluidos los minilotes.
Las curvas superpuestas permiten comparar ambos conjuntos y buscar señales de
sobreajuste, con una vista completa y otra del último 20% de actualizaciones.
La figura se guarda también en `resultados/mse_actualizaciones.png` al ejecutar
el notebook. Una tabla muestra la brecha final y la tendencia reciente de MSE.
Los tiempos incluyen el cálculo de estas métricas por actualización.

Desde la raíz del repositorio, en PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyterlab main.ipynb
```

Seleccionar el entorno `.venv` y ejecutar todas las celdas en orden. El notebook
contiene toda la implementación; no requiere `scikit-learn`.

Para verificar el gradiente, la convergencia y el manejo de minilotes:

```powershell
python -m unittest discover -s tests -v
```
