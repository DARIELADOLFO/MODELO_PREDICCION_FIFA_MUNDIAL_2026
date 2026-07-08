"""
Módulo de inferencia y predicción para el Mundial FIFA 2026.

Este módulo carga el modelo pre-entrenado y los datos actuales de las
selecciones para construir vectores de características al vuelo y predecir
el resultado de partidos futuros (ej. Fase de Grupos, Octavos, etc.) de
forma robusta, sin modificar el dataset original ni re-hacer Feature Engineering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import joblib
import unicodedata

def normalize_string(s: str) -> str:
    """Normaliza un string eliminando tildes, espacios y pasándolo a minúsculas."""
    if pd.isna(s): return s
    return ''.join(c for c in unicodedata.normalize('NFD', str(s).strip()) if unicodedata.category(c) != 'Mn').lower()

# ==========================================
# Configuración de Logging y Rutas Base
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Definición de rutas base de forma dinámica
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "predictions"


class PredictionEngine:
    """
    Motor central para la predicción de partidos.
    Se encarga de mantener en memoria el modelo, el encoder y los datos actuales
    para no recargar archivos repetidamente en predicciones por lotes.
    """
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.wc_data = None
        
        # Lista exacta de variables usadas en el entrenamiento (Random Forest)
        # generadas por el feature_builder.py
        self.feature_columns = [
            'neutral',
            'home_matches', 'home_wins', 'home_draws', 'home_losses',
            'home_goals_for', 'home_goals_against', 'home_goal_difference',
            'home_win_rate', 'home_avg_goals_for', 'home_avg_goals_against',
            'home_form_5', 'home_form_10', 'home_avg_goals_5', 'home_avg_goals_10',
            'home_consecutive_wins', 'home_unbeaten_streak', 'home_consecutive_losses',
            'away_matches', 'away_wins', 'away_draws', 'away_losses',
            'away_goals_for', 'away_goals_against', 'away_goal_difference',
            'away_win_rate', 'away_avg_goals_for', 'away_avg_goals_against',
            'away_form_5', 'away_form_10', 'away_avg_goals_5', 'away_avg_goals_10',
            'away_consecutive_wins', 'away_unbeaten_streak', 'away_consecutive_losses',
            'goal_difference_strength', 'win_rate_difference', 
            'goals_for_difference', 'goals_against_difference',
            'is_world_cup', 'is_qualification', 'is_friendly',
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_matches'
        ]

    def load_model(self) -> None:
        """Carga automáticamente models/best_model.pkl."""
        model_path = MODELS_DIR / "best_model.pkl"
        if not model_path.exists():
            logger.error(f"No se encontró el modelo en {model_path}")
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        self.model = joblib.load(model_path)
        logger.info("Modelo best_model.pkl cargado exitosamente.")

    def load_encoder(self) -> None:
        """Carga automáticamente models/label_encoder.pkl."""
        encoder_path = MODELS_DIR / "label_encoder.pkl"
        if not encoder_path.exists():
            logger.error(f"No se encontró el encoder en {encoder_path}")
            raise FileNotFoundError(f"Encoder no encontrado: {encoder_path}")
        self.encoder = joblib.load(encoder_path)
        logger.info("Encoder label_encoder.pkl cargado exitosamente.")

    def load_world_cup_data(self) -> pd.DataFrame:
        """Carga automáticamente Dataset_Mundial_2026_Actualizado.xlsx y retorna el DataFrame."""
        data_path = DATA_DIR / "Dataset_Mundial_2026_Actualizado.xlsx"
        if not data_path.exists():
            logger.error(f"No se encontró el dataset en {data_path}")
            raise FileNotFoundError(f"Dataset no encontrado: {data_path}")
            
        try:
            self.wc_data = pd.read_excel(data_path)
            
            if "País" not in self.wc_data.columns:
                raise ValueError(
                    f"No existe la columna 'País'. Columnas encontradas: {list(self.wc_data.columns)}"
                )
                
            self.wc_data.set_index("País", inplace=True)
            self.wc_data.index = self.wc_data.index.map(normalize_string)
            
            # Mapeo de columnas español -> inglés para que _safe_get las encuentre
            col_mapping = {
                'partidos jugados': 'matches',
                'victorias': 'wins',
                'empates': 'draws',
                'derrotas': 'losses',
                'goles a favor': 'goals_for',
                'goles en contra': 'goals_against',
                'diferencia de goles': 'goal_difference'
            }
            
            new_cols = {}
            for c in self.wc_data.columns:
                norm_c = normalize_string(c)
                if norm_c in col_mapping:
                    new_cols[c] = col_mapping[norm_c]
                else:
                    new_cols[c] = norm_c
            self.wc_data.rename(columns=new_cols, inplace=True)
                
            logger.info(f"Columnas: {self.wc_data.columns.tolist()}")
            logger.info(f"Primeros índices: {self.wc_data.index[:10].tolist()}")
            logger.info("Dataset Dataset_Mundial_2026_Actualizado.xlsx cargado exitosamente.")
            return self.wc_data
        except Exception as e:
            logger.error(f"Error al cargar el Excel del Mundial: {e}")
            raise

    def get_team_features(self, team: str) -> pd.Series:
        """Obtiene las estadísticas actuales de un país desde el Excel."""
        if self.wc_data is None:
            self.load_world_cup_data()
            
        team_norm = normalize_string(team)
            
        if team_norm not in self.wc_data.index:
            logger.warning(f"El equipo '{team}' no se encontró en el dataset. Verifique la escritura.")
            return pd.Series(dtype='float64')
            
        return self.wc_data.loc[team_norm]
        
    def _safe_get(self, series: pd.Series, var_name: str, default: float = 0.0) -> float:
        """Extrae un valor del series tolerando variaciones en el nombre de la columna."""
        keys = [var_name, var_name.lower(), var_name.replace('_', ' ')]
        for k in keys:
            if k in series.index:
                val = series[k]
                return float(val) if pd.notna(val) else default
        return default

    def build_match_vector(self, home_team: str, away_team: str, neutral: bool = True) -> pd.DataFrame:
        """Construye exactamente el mismo vector de variables utilizado durante el entrenamiento."""
        home_stats = self.get_team_features(home_team)
        if home_stats.empty:
            raise ValueError(f"No se encontraron estadísticas reales para el equipo local: {home_team}")
            
        away_stats = self.get_team_features(away_team)
        if away_stats.empty:
            raise ValueError(f"No se encontraron estadísticas reales para el equipo visitante: {away_team}")
        
        vector = {}
        
        # 1. Variables de torneo
        vector['neutral'] = 1 if neutral else 0
        vector['is_world_cup'] = 1
        vector['is_qualification'] = 0
        vector['is_friendly'] = 0
        
        # 2. Variables de equipo local (Home)
        vector['home_matches'] = self._safe_get(home_stats, 'matches')
        vector['home_wins'] = self._safe_get(home_stats, 'wins')
        vector['home_draws'] = self._safe_get(home_stats, 'draws')
        vector['home_losses'] = self._safe_get(home_stats, 'losses')
        vector['home_goals_for'] = self._safe_get(home_stats, 'goals_for')
        vector['home_goals_against'] = self._safe_get(home_stats, 'goals_against')
        
        # Cálculos de relleno por si el excel no los trae directamente
        hm = vector['home_matches']
        hw = vector['home_wins']
        hg_for = vector['home_goals_for']
        hg_ag = vector['home_goals_against']
        
        vector['home_goal_difference'] = self._safe_get(home_stats, 'goal_difference') or (hg_for - hg_ag)
        vector['home_win_rate'] = self._safe_get(home_stats, 'win_rate') or (hw / hm if hm > 0 else 0.0)
        vector['home_avg_goals_for'] = self._safe_get(home_stats, 'avg_goals_for') or (hg_for / hm if hm > 0 else 0.0)
        vector['home_avg_goals_against'] = self._safe_get(home_stats, 'avg_goals_against') or (hg_ag / hm if hm > 0 else 0.0)
        
        vector['home_form_5'] = self._safe_get(home_stats, 'form_5')
        vector['home_form_10'] = self._safe_get(home_stats, 'form_10')
        vector['home_avg_goals_5'] = self._safe_get(home_stats, 'avg_goals_5')
        vector['home_avg_goals_10'] = self._safe_get(home_stats, 'avg_goals_10')
        vector['home_consecutive_wins'] = self._safe_get(home_stats, 'consecutive_wins')
        vector['home_unbeaten_streak'] = self._safe_get(home_stats, 'unbeaten_streak')
        vector['home_consecutive_losses'] = self._safe_get(home_stats, 'consecutive_losses')

        # 3. Variables de equipo visitante (Away)
        vector['away_matches'] = self._safe_get(away_stats, 'matches')
        vector['away_wins'] = self._safe_get(away_stats, 'wins')
        vector['away_draws'] = self._safe_get(away_stats, 'draws')
        vector['away_losses'] = self._safe_get(away_stats, 'losses')
        vector['away_goals_for'] = self._safe_get(away_stats, 'goals_for')
        vector['away_goals_against'] = self._safe_get(away_stats, 'goals_against')
        
        am = vector['away_matches']
        aw = vector['away_wins']
        ag_for = vector['away_goals_for']
        ag_ag = vector['away_goals_against']
        
        vector['away_goal_difference'] = self._safe_get(away_stats, 'goal_difference') or (ag_for - ag_ag)
        vector['away_win_rate'] = self._safe_get(away_stats, 'win_rate') or (aw / am if am > 0 else 0.0)
        vector['away_avg_goals_for'] = self._safe_get(away_stats, 'avg_goals_for') or (ag_for / am if am > 0 else 0.0)
        vector['away_avg_goals_against'] = self._safe_get(away_stats, 'avg_goals_against') or (ag_ag / am if am > 0 else 0.0)
        
        vector['away_form_5'] = self._safe_get(away_stats, 'form_5')
        vector['away_form_10'] = self._safe_get(away_stats, 'form_10')
        vector['away_avg_goals_5'] = self._safe_get(away_stats, 'avg_goals_5')
        vector['away_avg_goals_10'] = self._safe_get(away_stats, 'avg_goals_10')
        vector['away_consecutive_wins'] = self._safe_get(away_stats, 'consecutive_wins')
        vector['away_unbeaten_streak'] = self._safe_get(away_stats, 'unbeaten_streak')
        vector['away_consecutive_losses'] = self._safe_get(away_stats, 'consecutive_losses')

        # 4. Variables de diferencias entre los equipos
        vector['goal_difference_strength'] = vector['home_goal_difference'] - vector['away_goal_difference']
        vector['win_rate_difference'] = vector['home_win_rate'] - vector['away_win_rate']
        vector['goals_for_difference'] = vector['home_avg_goals_for'] - vector['away_avg_goals_for']
        vector['goals_against_difference'] = vector['home_avg_goals_against'] - vector['away_avg_goals_against']
        
        # 5. Head To Head (Historial Directo)
        # Extraemos el H2H en caso de que esté en el excel (ej. cruzando nombres). 
        # Si no existe, se asumirá 0 partidos entre ellos (o lo que dicten los datos).
        vector['h2h_home_wins'] = self._safe_get(home_stats, f'h2h_wins_vs_{away_team}')
        vector['h2h_away_wins'] = self._safe_get(away_stats, f'h2h_wins_vs_{home_team}')
        vector['h2h_draws'] = self._safe_get(home_stats, f'h2h_draws_vs_{away_team}')
        vector['h2h_matches'] = vector['h2h_home_wins'] + vector['h2h_away_wins'] + vector['h2h_draws']
        
        # Consolidar manteniendo estrictamente el orden esperado por el modelo RandomForest
        ordered_vector = {col: vector.get(col, 0.0) for col in self.feature_columns}
        
        return pd.DataFrame([ordered_vector])

    def predict_match(self, home_team: str, away_team: str, neutral: bool = True) -> Dict[str, Any]:
        """Realiza la predicción integral de un partido, devolviendo diccionarios de resultados."""
        if self.model is None:
            self.load_model()
        if self.encoder is None:
            self.load_encoder()
            
        # 1. Construir vector
        vector_df = self.build_match_vector(home_team, away_team, neutral)
        
        # Orden de columnas de seguridad (por si el modelo fue entrenado en otra versión de sklearn)
        if hasattr(self.model, 'feature_names_in_'):
            vector_df = vector_df[self.model.feature_names_in_]
        
        # 2. Predecir probabilidades
        probs = self.model.predict_proba(vector_df)[0]
        
        # Obtener las clases del encoder
        classes = self.encoder.classes_ if hasattr(self.encoder, 'classes_') else self.model.classes_
        
        # 3. Crear mapeo de probabilidades a nombres legibles
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        
        # 4. Determinar ganador final y traducir etiqueta de Local/Visitante a nombres de países
        winner_class = str(classes[np.argmax(probs)])
        if winner_class == 'Local':
            predicted_winner = home_team
        elif winner_class == 'Visitante':
            predicted_winner = away_team
        else:
            predicted_winner = 'Empate'
            
        # 5. Calcular confianza (probabilidad de la clase ganadora en porcentaje)
        confidence = float(np.max(probs) * 100)
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "winner": predicted_winner,
            "probabilities": prob_dict,
            "confidence": round(confidence, 2)
        }

    def predict_matches(self, matches: List[Tuple[str, str]], neutral: bool = True) -> pd.DataFrame:
        """Realiza predicciones por lotes devolviendo un DataFrame formateado."""
        results = []
        for home, away in matches:
            try:
                res = self.predict_match(home, away, neutral)
                probs = res['probabilities']
                
                results.append({
                    "Partido": f"{home} vs {away}",
                    "Equipo Local": home,
                    "Equipo Visitante": away,
                    "Probabilidad Local": round(probs.get("Local", 0.0), 4),
                    "Probabilidad Empate": round(probs.get("Empate", 0.0), 4),
                    "Probabilidad Visitante": round(probs.get("Visitante", 0.0), 4),
                    "Ganador Predicho": res["winner"],
                    "Confianza": res["confidence"]
                })
            except Exception as e:
                logger.error(f"Error procesando {home} vs {away}: {str(e)}")
                
        return pd.DataFrame(results)


# ==========================================
# Instancia Singleton y Funciones Públicas
# ==========================================
_engine = None

def _get_engine() -> PredictionEngine:
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
    return _engine


def load_model() -> None:
    """Carga automáticamente models/best_model.pkl."""
    _get_engine().load_model()

def load_encoder() -> None:
    """Carga automáticamente models/label_encoder.pkl."""
    _get_engine().load_encoder()

def load_world_cup_data() -> pd.DataFrame:
    """Carga automáticamente el dataset Excel en memoria y lo retorna."""
    return _get_engine().load_world_cup_data()

def get_team_features(team: str) -> pd.Series:
    """Obtiene las estadísticas actuales de un país desde el Excel."""
    return _get_engine().get_team_features(team)

def build_match_vector(home_team: str, away_team: str, neutral: bool = True) -> pd.DataFrame:
    """Construye el vector idéntico al usado en entrenamiento para predicción."""
    return _get_engine().build_match_vector(home_team, away_team, neutral)

def predict_match(home_team: str, away_team: str, neutral: bool = True) -> Dict[str, Any]:
    """
    Predice el ganador de un enfrentamiento directo.
    
    Args:
        home_team (str): Nombre del equipo local.
        away_team (str): Nombre del equipo visitante.
        neutral (bool): Si el partido se juega en sede neutral.
        
    Returns:
        Dict[str, Any]: Diccionario con las probabilidades, el equipo ganador y la confianza.
    """
    return _get_engine().predict_match(home_team, away_team, neutral)

def predict_matches(matches: List[Tuple[str, str]], neutral: bool = True) -> pd.DataFrame:
    """
    Predice el resultado de una lista de partidos y lo devuelve en formato tabla.
    
    Args:
        matches (List[Tuple[str, str]]): Lista de tuplas conteniendo pares de equipos.
        neutral (bool): Si los partidos se juegan en sede neutral.
        
    Returns:
        pd.DataFrame: Tabla con columnas listas para ser exportadas o visualizadas.
    """
    return _get_engine().predict_matches(matches, neutral)

def save_predictions(predictions_df: pd.DataFrame, filename: str = "quarterfinal_predictions.csv") -> None:
    """
    Guarda el DataFrame de predicciones en disco respetando la estructura de carpetas.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame generado por predict_matches().
        filename (str): Nombre del archivo .csv
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUTS_DIR / filename
    
    try:
        predictions_df.to_csv(filepath, index=False, float_format="%.4f", encoding='utf-8')
        logger.info(f"Predicciones exportadas exitosamente en {filepath}")
    except Exception as e:
        logger.error(f"Fallo al guardar predicciones: {e}")
        raise
