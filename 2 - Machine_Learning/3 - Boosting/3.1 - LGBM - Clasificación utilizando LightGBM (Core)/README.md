# LGBMcore

* author: Guillermo Cabrera
* date: 'Nov 2024'
* title: Clasificación utilizando LightGBM (Core)
* course: Fundamentos de Machine Learning
* format: '.ipynb', '.py', 'csv'


Ejecutar el archivo .ipynb dentro de la carpeta 'code' para visualizar los resultados.

Propósito del Proyecto

El propósito de este proyecto es implementar un pipeline completo de machine learning para resolver un problema de clasificación utilizando LightGBM (LGBM). El enfoque principal estará en realizar un Análisis Exploratorio de Datos (EDA), preparar los datos mediante un preprocesamiento adecuado, entrenar el modelo y optimizar sus hiperparámetros para maximizar su rendimiento.

Técnicas y Modelos Utilizados

Se implementó y evaluó el modelo de clasificación utilizando LightGBM (LGBM), un algoritmo eficiente basado en boosting, diseñado para manejar grandes volúmenes de datos y proporcionar resultados precisos en problemas de clasificación.

Hiperparámetros básicos: El modelo fue inicialmente entrenado con parámetros por defecto para establecer una línea base de rendimiento.

Optimización avanzada: Se utilizó GridSearchCV para ajustar hiperparámetros clave como la profundidad de los árboles, el número de hojas (num_leaves), el aprendizaje (learning_rate), y el número de iteraciones (n_estimators). Este proceso maximizó métricas como la precisión, recall y F1-score, asegurando un modelo más robusto y eficiente.