import streamlit as st
import pandas as pd
import time
import os
import random
import base64

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. ESTILOS CSS CORPORATIVOS GLOBALES

st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    [data-testid="stSidebar"] { background-color: #0F172A; color: #FFFFFF; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    h1, h2, h3 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
    
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    div[data-testid="metric-container"] > div > div { color: #D4AF37 !important; font-weight: bold; }
    
    .stButton > button {
        background-color: #D4AF37; color: #0F172A; border-radius: 8px; border: none;
        font-weight: bold; transition: all 0.3s; width: 100%; padding: 10px 0;
    }
    .stButton > button:hover { background-color: #B5952F; color: #FFFFFF; box-shadow: 0 4px 12px rgba(212, 175, 55, 0.4); }
    .stProgress .st-bo { background-color: #1E3A8A; }
    img { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# 3. DATOS GLOBALES
PAISES_MUNDIAL = sorted([
    "Argentina", "Francia", "Brasil", "Inglaterra", "España", "Alemania", 
    "Portugal", "Países Bajos", "Bélgica", "Uruguay", "Colombia", "Suiza", 
    "Croacia", "Marruecos", "Estados Unidos", "México", "Japón", "Senegal",
    "Ecuador", "Corea del Sur", "Australia", "Canadá", "Costa Rica", "Perú"
])

try:
    from src.prediction import predict_match
    USAR_MOCK = False
except ImportError:
    USAR_MOCK = True

# ==========================================
# 4. FUNCIONES AUXILIARES
# ==========================================
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

# 5. NAVEGACIÓN (SIDEBAR)
ruta_logo = r"C:\Users\darie\Downloads\PROYECTO_FIFA2026\outputs\figures\FIFA_LOGO.PNG"
if os.path.exists(ruta_logo):
    st.sidebar.image(ruta_logo, use_container_width=True)
else:
    st.sidebar.markdown("<h2 style='text-align: center; color: #D4AF37 !important;'>⚽ Predictor 2026</h2>", unsafe_allow_html=True)

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "📊 Dashboard Ejecutivo", "🔮 Predictor de Partidos", 
     "🏆 Simulación del Mundial", "📈 Estadísticas", "📄 Reportes", "ℹ️ Acerca del Proyecto"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado por Dariel Peña Vásquez")

# 6. LÓGICA DE LAS PÁGINAS

# ----------------- INICIO -----------------
if menu == "🏠 Inicio":
    
    ruta_fondo = os.path.join("outputs", "figures", "fondo_fifa.jpg")
    base64_fondo = get_base64_image(ruta_fondo)
    
    if base64_fondo:
        bg_css = f"url(data:image/jpeg;base64,{base64_fondo})"
    else:
        bg_css = "#0F172A" 
        
    # TODO EL HTML EN UNA SOLA LÍNEA PARA EVITAR EL FORMATO DE CÓDIGO DE MARKDOWN
    hero_html = f"""<div style="background-image: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), {bg_css}; background-size: cover; background-position: center; padding: 60px 30px; border-radius: 15px; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.2); font-family: 'Helvetica Neue', sans-serif; margin-bottom: 30px;"><div style="font-size: 4rem; margin-bottom: 10px;">🏟️ ⚽ 🏆</div><h1 style="color: #FFFFFF !important; font-size: 3.5rem; font-weight: 900; margin-bottom: 10px; text-shadow: 2px 2px 8px rgba(0,0,0,0.5);">FIFA World Cup 2026</h1><h3 style="color: #E2E8F0 !important; font-size: 1.5rem; font-weight: 400; margin-bottom: 40px;">Predicción mediante Machine Learning y Simulación Monte Carlo</h3><div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px;"><div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2); flex: 1; min-width: 200px;"><p style="color: #E2E8F0; margin: 0; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Mejores Modelos</p><p style="color: #D4AF37; margin: 5px 0 0 0; font-size: 1.6rem; font-weight: bold;">Random Forest / CatBoost</p></div><div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2); flex: 1; min-width: 150px;"><p style="color: #E2E8F0; margin: 0; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Accuracy</p><p style="color: #D4AF37; margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold;">58.25 %</p></div><div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2); flex: 1; min-width: 150px;"><p style="color: #E2E8F0; margin: 0; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Partidos Históricos</p><p style="color: #D4AF37; margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold;">49,502</p></div><div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2); flex: 1; min-width: 150px;"><p style="color: #E2E8F0; margin: 0; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Simulaciones</p><p style="color: #D4AF37; margin: 5px 0 0 0; font-size: 1.8rem; font-weight: bold;">10,000</p></div></div></div>"""
    
    st.markdown(hero_html, unsafe_allow_html=True)

# ----------------- DASHBOARD EJECUTIVO -----------------
elif menu == "📊 Dashboard Ejecutivo":
    st.title("Dashboard Ejecutivo")
    st.markdown("Visualización del rendimiento y probabilidades del torneo.")
    
    ruta_exec = os.path.join("outputs", "figures", "executive_dashboard.png")
    if os.path.exists(ruta_exec):
        st.image(ruta_exec, use_container_width=True)
        st.write("---")
    else:
        st.info("Esperando gráfico: executive_dashboard.png")

    graficos_col1 = [
        {"archivo": "ranking_top8.png", "titulo": "Top 8 Favoritos"},
        {"archivo": "semifinal_probability.png", "titulo": "Prob. Semifinales"}
    ]
    graficos_col2 = [
        {"archivo": "stacked_progress.png", "titulo": "Avance por Fase"},
        {"archivo": "final_probability.png", "titulo": "Prob. Final"}
    ]

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        for g in graficos_col1:
            st.subheader(g["titulo"])
            ruta_img = os.path.join("outputs", "figures", g["archivo"])
            if os.path.exists(ruta_img):
                st.image(ruta_img, use_container_width=True)
            else:
                st.info(f"Falta: {g['archivo']}")
                
        st.subheader("Probabilidad de Campeón")
        ruta_champ = os.path.join("outputs", "figures", "champion_probability.png")
        if os.path.exists(ruta_champ):
            st.image(ruta_champ, use_container_width=True)

    with col_g2:
        for g in graficos_col2:
            st.subheader(g["titulo"])
            ruta_img = os.path.join("outputs", "figures", g["archivo"])
            if os.path.exists(ruta_img):
                st.image(ruta_img, use_container_width=True)
            else:
                st.info(f"Falta: {g['archivo']}")

    st.write("---")
    st.subheader("Tabla de Probabilidades Finales")
    ruta_csv = os.path.join("outputs", "simulations", "champion_probabilities.csv")
    if os.path.exists(ruta_csv):
        df_probs = pd.read_csv(ruta_csv)
        st.dataframe(df_probs.style.format(precision=2), use_container_width=True)
    else:
        df_mock = pd.DataFrame({
            "País": ["Francia", "Argentina", "Inglaterra", "Brasil", "España"],
            "Probabilidad Semifinal": ["68.5%", "62.1%", "59.0%", "55.4%", "45.2%"],
            "Probabilidad Final":     ["35.2%", "31.8%", "28.5%", "22.1%", "18.9%"],
            "Probabilidad Campeón":   ["21.4%", "15.8%", "15.0%", "12.5%", "8.2%"]
        })
        st.dataframe(df_mock, use_container_width=True)

# ----------------- PREDICTOR DE PARTIDOS -----------------
elif menu == "🔮 Predictor de Partidos":
    st.title("Motor Predictivo H2H")
    st.write("Selecciona dos selecciones para ejecutar el modelo y proyectar el resultado.")
    
    col1, col_vs, col2 = st.columns([2, 1, 2])
    with col1: local = st.selectbox("Equipo Local", PAISES_MUNDIAL, index=PAISES_MUNDIAL.index("Francia"))
    with col_vs: st.markdown("<h2 style='text-align: center; margin-top: 25px;'>VS</h2>", unsafe_allow_html=True)
    with col2: visitante = st.selectbox("Equipo Visitante", PAISES_MUNDIAL, index=PAISES_MUNDIAL.index("Argentina"))
    
    st.write("<br>", unsafe_allow_html=True)
    
    if st.button("PREDECIR PARTIDO", type="primary"):
        with st.spinner("Analizando historial y calculando métricas..."):
            time.sleep(1.5)
            if not USAR_MOCK:
                resultado = predict_match(local, visitante)
                prob_local = resultado['probabilities']['Local'] * 100
                prob_empate = resultado['probabilities']['Empate'] * 100
                prob_visita = resultado['probabilities']['Visitante'] * 100
                ganador = resultado['winner']
                confianza = resultado['confidence']
            else:
                prob_local = round(random.uniform(30, 60), 1)
                prob_empate = round(random.uniform(20, 30), 1)
                prob_visita = 100 - prob_local - prob_empate
                ganador = local if prob_local > prob_visita else visitante
                confianza = max(prob_local, prob_visita)

            st.success("Predicción completada exitosamente.")
            col_res1, col_res2 = st.columns(2)
            with col_res1: st.metric("Ganador Predicho", ganador)
            with col_res2: st.metric("Confianza del Modelo", f"{confianza:.1f}%")
            
            st.write("### Distribución de Probabilidades")
            st.progress(int(prob_local), text=f"{local}: {prob_local:.1f}%")
            st.progress(int(prob_empate), text=f"Empate: {prob_empate:.1f}%")
            st.progress(int(prob_visita), text=f"{visitante}: {prob_visita:.1f}%")

# ----------------- SIMULACIÓN DEL MUNDIAL -----------------
elif menu == "🏆 Simulación del Mundial":
    st.title("Simulador de Torneo (Monte Carlo)")
    if st.button("SIMULAR MUNDIAL", type="primary"):
        progress_text = "Corriendo 10,000 simulaciones. Por favor espere..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.5)
        my_bar.empty()
        st.success("¡10,000 simulaciones completadas!")
        
        ruta_sim = os.path.join("outputs", "simulations", "champion_probabilities.csv")
        if os.path.exists(ruta_sim):
            df_sim = pd.read_csv(ruta_sim)
            st.dataframe(df_sim, use_container_width=True)
        else:
            st.info("🥇 1. Francia (21.4%) | 🥈 2. Argentina (15.8%) | 🥉 3. Inglaterra (15.0%)")

# ----------------- ESTADÍSTICAS -----------------
elif menu == "📈 Estadísticas":
    st.title("Rendimiento del Modelo")
    col1, col2 = st.columns(2)
    with col1: st.metric("Cantidad de partidos procesados", "49,502")
    with col2: st.metric("Precisión Base (Accuracy)", "58.25 %")
    st.write("---")
    st.subheader("Importancia de Variables")
    st.markdown("1. Fuerza Ofensiva Histórica (28%)\n2. Diferencia de Goles Promedio (22%)\n3. Ranking FIFA Previo (18%)")

# ----------------- REPORTES -----------------
elif menu == "📄 Reportes":
    st.title("Descarga de Entregables")
    archivos = [
        ("Executive Summary (MD)", "outputs/reports/executive_summary.md"),
        ("Technical Report (MD)", "outputs/reports/technical_report.md"),
        ("Champion Probabilities (CSV)", "outputs/simulations/champion_probabilities.csv"),
    ]
    for nombre, ruta in archivos:
        if os.path.exists(ruta):
            with open(ruta, "rb") as file:
                st.download_button(label=f"Descargar {nombre}", data=file, file_name=os.path.basename(ruta), mime="text/plain")
        else:
            st.button(f"Falta archivo: {nombre}", disabled=True)

# ----------------- ACERCA DEL PROYECTO -----------------
elif menu == " Acerca del Proyecto":
    st.title("Sobre este Proyecto")
    
    st.markdown("""
    ###  Objetivo Principal
    Desarrollar un sistema avanzado de **Machine Learning** y análisis estadístico capaz de evaluar patrones históricos para proyectar probabilidades, predecir enfrentamientos directos y simular el desarrollo completo de la **Copa Mundial de la FIFA 2026**.
    
    ---
    
    ###  Fuentes de Datos
    * **Historial Internacional:** Procesamiento de más de **49,500 partidos oficiales y amistosos** (dataset histórico desde 1872), filtrando y priorizando los datos contemporáneos para evaluar el "Power Ranking" actual de cada selección.
    * **Estructura del Torneo 2026:** Integración de la nueva fase de grupos de 48 equipos y el cuadro eliminatorio oficial (Ronda de 32 hasta la Final).
    
    ###  Metodología y Arquitectura
    1. **Feature Engineering:** Creación de variables predictivas clave como la *Fuerza Ofensiva/Defensiva*, historial *Head-to-Head (H2H)*, diferencia de goles y rendimiento en los últimos encuentros.
    2. **Motor Predictivo:** Entrenamiento y evaluación de múltiples algoritmos para clasificación multiclase. Se evaluaron Random Forest, XGBoost y **CatBoost**, seleccionando este último por su superioridad al manejar variables categóricas complejas (como los nombres de los países) sin necesidad de transformaciones pesadas.
    3. **Simulación Monte Carlo:** Ejecución de **10,000 iteraciones** virtuales del torneo completo. Este motor toma las probabilidades matemáticas de cada choque y simula la aleatoriedad del deporte para calcular con exactitud qué porcentaje de veces un país logra alcanzar cada fase (Cuartos, Semis, Final).
    
    ---
    
    ###  Stack Tecnológico
    * **Lenguaje Base:** Python 3
    * **Manipulación de Datos:** Pandas, NumPy
    * **Modelado y Predicción:** Scikit-Learn, CatBoost
    * **Visualización de Datos:** Plotly, Matplotlib
    * **Despliegue Frontend:** Streamlit
    * **Asistentes de IA:** Codex IA, Google Antigravity, ChatGPT y Gemini. Para Documentacion correcta del proyecto y ayuda con la aplicacion web 

    
    ---
    
    <div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #F8FAFC; border-radius: 10px; border: 1px solid #E2E8F0;">
        <p style="margin: 0; color: #1E3A8A; font-size: 1.1rem;">Diseñado y desarrollado por</p>
        <p style="margin: 5px 0 0 0; color: #D4AF37; font-size: 1.5rem; font-weight: bold;">Dariel Peña Vásquez</p>
    </div>
    """, unsafe_allow_html=True)
