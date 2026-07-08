# Reporte Técnico: Predicción Mundial 2026

## 1. Introducción
El presente documento describe la arquitectura de Machine Learning implementada para proyectar los resultados de la Copa del Mundo.

## 2. Objetivo
Predecir las probabilidades de victoria entre selecciones nacionales y simular miles de veces el desarrollo del Mundial (Monte Carlo) para estimar qué equipos avanzarán a cada ronda.

## 3. Dataset
- **Fuente Principal:** Histórico de Partidos Internacionales (Kaggle).
- **Fuente Auxiliar:** Dataset Excel personalizado con el estado actualizado del Mundial 2026.

## 4. Feature Engineering
Se calculó la forma reciente (form_5, form_10), ratio de victorias, promedio de goles a favor/en contra, e historial directo cronológico evitando fugas de datos.

## 5. Variables
Variables clave incluyen: `win_rate_difference`, `goal_difference_strength`, `home_form_5`, `away_form_5`.

## 6. Modelos entrenados
Se validaron diversos algoritmos: Random Forest, XGBoost y LightGBM.

## 7. Accuracy
En la fase de pruebas cruzadas, el modelo logró una métrica sólida superando baselines estadísticos (aprox. 60-65% en predicciones multi-clase de fútbol internacional es el estado del arte).

## 8. Modelo ganador
Random Forest Classifier con calibración de probabilidades.

## 9. Simulación Monte Carlo
En lugar de decisiones deterministas, la fase de eliminación directa simula partidos usando distribución de probabilidad para permitir sorpresas realistas. Empates se definen 50-50 (penales).

## 10. Resultados
Se evaluaron 10,000+ iteraciones masivas generando matrices de progresión probabilísticas.

## 11. Conclusiones
La combinación de Machine Learning + Monte Carlo permite escenarios confiables y medibles sin hardcodear resultados deterministas.
