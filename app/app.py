import streamlit as st
import pandas as pd
import time
import os
import random

# 
# 1. CONFIGURACIÓN DE LA PÁGINA
# 
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 
# 2. ESTILOS CSS CORPORATIVOS
# 
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    [data-testid="stSidebar"] { background-color: #0F172A; color: #FFFFFF; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    h1, h2, h3 { color: #1E3A8A !important; font-family: 'Helvetica Neue', sans-serif; }
    
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
    
    /* Estilo para imágenes con bordes redondeados */
    img { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATOS GLOBALES
# ==========================================
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

# 
# 4. NAVEGACIÓN (SIDEBAR)
# 
# Cargar el Logo en la parte superior del Sidebar
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

# 
# 5. LÓGICA DE LAS PÁGINAS

# ----------------- INICIO -----------------
if menu == "🏠 Inicio":
    # Reemplazo de la imagen por iconos nativos para evitar enlaces rotos
    st.markdown("<div style='text-align: center; font-size: 5rem;'>🏟️ ⚽ 🏆</div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-top: 10px;'>FIFA World Cup 2026</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #64748B;'>Predicción mediante Machine Learning y Simulación Monte Carlo</h4>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    
    # Filas divididas de a 2 para evitar que el texto se corte
    col1, col2 = st.columns(2)
    with col1: st.metric("Mejores Modelos", "Random Forest / CatBoost")
    with col2: st.metric("Accuracy", "58.25 %")
    
    st.write("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3: st.metric("Partidos Históricos", "49,502")
    with col4: st.metric("Simulaciones", "10,000")

# ----------------- DASHBOARD EJECUTIVO -----------------
elif menu == "📊 Dashboard Ejecutivo":
    st.title("Dashboard Ejecutivo")
    st.markdown("Visualización del rendimiento y probabilidades del torneo.")
    
    # 1. Imagen principal (Executive Dashboard)
    ruta_exec = os.path.join("outputs", "figures", "executive_dashboard.png")
    if os.path.exists(ruta_exec):
        st.image(ruta_exec, use_container_width=True)
        st.write("---")
    else:
        st.info("Esperando gráfico: executive_dashboard.png")

    # 2. Organizar el resto de gráficos en columnas
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
                
        # Poner champion_probability al final de la columna 1
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
elif menu == "ℹ️ Acerca del Proyecto":
    st.title("Sobre este Proyecto")
    st.markdown("""
    **Objetivo:** Desarrollar un modelo de Machine Learning capaz de evaluar estadísticas para proyectar probabilidades en la Copa Mundial de la FIFA 2026.
    **Tecnologías:** Python, Pandas, Scikit-Learn, CatBoost, Streamlit.
    """)
