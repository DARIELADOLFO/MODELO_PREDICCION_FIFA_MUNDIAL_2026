# Predicción Inteligente de la Copa Mundial FIFA 2026

Este proyecto utiliza Machine Learning y Simulación Monte Carlo para predecir probabilidades de victoria entre selecciones nacionales y simular el desarrollo del Mundial 2026. El objetivo principal es estimar qué equipos tienen mayor probabilidad de avanzar a cada ronda y convertirse en campeones.


## Link del App Web :

https://modeloprediccionfifamundial2026-dbdet3zqg5q49hekg9rajb.streamlit.app/

## Estructura del Proyecto

La estructura del proyecto está diseñada siguiendo las mejores prácticas de Data Science y Software Engineering:

*   **`data/`**: Carpeta para los conjuntos de datos. Se subdivide en `raw` (datos originales sin procesar), `processed` (datos limpios y transformados) y `external` (fuentes adicionales).
*   **`notebooks/`**: Cuadernos de experimentación.
    *   `01_Preprocesamiento_y_Feature_Engineering.ipynb`: Limpieza, integración y análisis exploratorio (EDA).
    *   `02_Modelado_y_Simulacion.ipynb`: Entrenamiento de modelos, evaluación y simulación de escenarios.
*   **`models/`**: Archivos de modelos entrenados (`trained`) y puntos de control durante el entrenamiento (`checkpoints`).
*   **`outputs/`**: Resultados generados como figuras (`figures`), reportes (`reports`), predicciones en formato tabular (`predictions`) y resultados completos de la simulación (`simulations`).
*   **`app/`**: Aplicación web desarrollada en Streamlit para visualizar los resultados interactivos.
*   **`src/`**: Código fuente de Python organizado en módulos reutilizables para preprocesamiento, feature engineering, modelado, simulación, y utilidades compartidas.
*   **`tests/`**: Pruebas unitarias para los módulos de código (pytest u otra herramienta).
*   **`docs/`**: Documentación técnica del proyecto.

## Tecnologías Utilizadas

*   **Lenguaje**: Python
*   **Manejo y Análisis de Datos**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn, XGBoost, LightGBM
*   **Simulación y Estadísticas**: SciPy
*   **Visualización**: Plotly, Matplotlib, SHAP
*   **Despliegue y Persistencia**: Streamlit, Joblib

## Configuración del Entorno

1.  Se recomienda utilizar un entorno virtual (venv o conda).
2.  Instala las dependencias utilizando el archivo `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Origen de Datos

1.  **Dataset Histórico**: Información histórica de partidos internacionales de fútbol (Kaggle).
2.  **Dataset Actualizado 2026**: Archivo Excel propio con el estado y variables actualizadas para el Mundial 2026.

## Filosofía
Este proyecto mantiene una separación estricta entre los datos originales, los datos procesados, los modelos entrenados y la aplicación final para garantizar la reproducibilidad y escalabilidad a futuro.


## Imagenes del Proyecto y las Predicciones 
<img width="5969" height="3432" alt="executive_dashboard" src="https://github.com/user-attachments/assets/f974c075-7cd8-4663-b901-c87769d49818" />

<img width="2961" height="2369" alt="champion_probability" src="https://github.com/user-attachments/assets/09dc2fd9-052f-483f-b4ac-8057e8daee3b" />


<img width="3572" height="2372" alt="stacked_progress" src="https://github.com/user-attachments/assets/887c1c7b-48ef-40d2-8b42-99f9b91cdaac" />


<img width="2957" height="2370" alt="final_probability" src="https://github.com/user-attachments/assets/5d981cb9-6f5c-481b-9432-a4b818177bfe" />



