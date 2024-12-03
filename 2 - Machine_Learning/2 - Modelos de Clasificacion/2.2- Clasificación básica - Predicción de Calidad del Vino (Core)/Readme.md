---
author: Guillermo Cabrera
date: 'Nov 2024'
Title: Clasificación básica / Predicción de Calidad del Vino (Core)
Course: Fundamentos de Machine Learning
format: '.ipynb', '.py', 'csv', 'pdf'
---

Ejecutar el archivo .ipynb de la carpeta Code para visualizar los resultados.

Las conclusiones finales se encuentran en la carpeta PDF.


Propósito del Proyecto

Este proyecto tiene como objetivo principal desarrollar modelos de clasificación que predigan la calidad del vino, clasificada en una escala de 0 a 10. El dataset utilizado contiene diferentes atributos fisicoquímicos de los vinos y su calidad, evaluada por expertos. El fin es identificar las mejores técnicas y modelos que ofrezcan un rendimiento óptimo en la predicción de la calidad del vino.


Técnicas y Modelos Utilizados

Se implementaron y evaluaron diferentes modelos de clasificación, incluyendo:

Regresión Logística: Para evaluar su capacidad de manejo de problemas de clasificación binaria y multiclase en datos linealmente separables.

Random Forest: Modelo de ensamble basado en múltiples árboles de decisión para capturar relaciones complejas y reducir el sobreajuste.

K-Nearest Neighbors (KNN): Utilizado para comparar resultados, aunque sus limitaciones de escalabilidad y sensibilidad al ruido lo hicieron menos eficiente en comparación con los modelos anteriores.