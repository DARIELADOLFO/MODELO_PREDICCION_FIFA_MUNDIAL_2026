"""
Módulo para la construcción de características (Feature Engineering).

Este módulo procesa el historial de partidos internacionales y calcula
estadísticas acumuladas, rachas y rendimiento histórico (H2H) sin
introducir filtración de datos (data leakage).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class TeamStats:
    """Clase para mantener el estado histórico de una selección.
    
    Attributes:
        matches (int): Cantidad total de partidos jugados.
        wins (int): Cantidad de victorias.
        draws (int): Cantidad de empates.
        losses (int): Cantidad de derrotas.
        goals_for (int): Goles a favor acumulados.
        goals_against (int): Goles en contra acumulados.
        recent_results (List[str]): Historial de resultados recientes ('W', 'D', 'L').
        recent_goals_for (List[int]): Historial de goles anotados recientemente.
        consecutive_wins (int): Racha actual de victorias consecutivas.
        unbeaten_streak (int): Racha actual de partidos sin perder.
        consecutive_losses (int): Racha actual de derrotas consecutivas.
    """
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    
    recent_results: List[str] = field(default_factory=list)
    recent_goals_for: List[int] = field(default_factory=list)
    
    consecutive_wins: int = 0
    unbeaten_streak: int = 0
    consecutive_losses: int = 0


@dataclass
class H2HStats:
    """Clase para mantener el historial de enfrentamientos directos entre dos equipos.
    
    Attributes:
        matches (int): Cantidad de partidos entre ambos.
        team_a_wins (int): Victorias del equipo A (alfabéticamente primero).
        team_b_wins (int): Victorias del equipo B (alfabéticamente segundo).
        draws (int): Cantidad de empates.
    """
    matches: int = 0
    team_a_wins: int = 0
    team_b_wins: int = 0
    draws: int = 0


class MatchFeatureBuilder:
    """Constructor principal de características para los partidos.
    
    Recorre el historial de partidos de manera cronológica y asegura
    que cada partido solo utilice información disponible estrictamente
    antes del pitazo inicial.
    """

    def __init__(self):
        """Inicializa los diccionarios de seguimiento para equipos y H2H."""
        self.teams: Dict[str, TeamStats] = defaultdict(TeamStats)
        self.h2h: Dict[Tuple[str, str], H2HStats] = defaultdict(H2HStats)

    def _get_win_rate(self, stats: TeamStats) -> float:
        """Calcula el porcentaje de victorias histórico de un equipo."""
        return stats.wins / stats.matches if stats.matches > 0 else 0.0

    def _get_avg_goals_for(self, stats: TeamStats) -> float:
        """Calcula el promedio de goles anotados históricamente."""
        return stats.goals_for / stats.matches if stats.matches > 0 else 0.0

    def _get_avg_goals_against(self, stats: TeamStats) -> float:
        """Calcula el promedio de goles recibidos históricamente."""
        return stats.goals_against / stats.matches if stats.matches > 0 else 0.0

    def _calculate_form(self, stats: TeamStats, n: int) -> float:
        """Calcula los puntos obtenidos en los últimos n partidos.
        
        Victoria suma 3 puntos, empate suma 1, derrota 0.
        """
        if not stats.recent_results:
            return 0.0
        recent = stats.recent_results[-n:]
        return float(sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in recent))

    def _calculate_recent_avg_goals(self, stats: TeamStats, n: int) -> float:
        """Calcula el promedio de goles a favor en los últimos n partidos."""
        if not stats.recent_goals_for:
            return 0.0
        recent = stats.recent_goals_for[-n:]
        return sum(recent) / len(recent)

    def _get_h2h_stats(self, home_team: str, away_team: str) -> Dict[str, int]:
        """Obtiene las estadísticas de enfrentamientos directos orientadas al partido actual."""
        teams_sorted = tuple(sorted([home_team, away_team]))
        h2h_record = self.h2h[teams_sorted]
        
        is_home_team_a = (home_team == teams_sorted[0])
        home_wins = h2h_record.team_a_wins if is_home_team_a else h2h_record.team_b_wins
        away_wins = h2h_record.team_b_wins if is_home_team_a else h2h_record.team_a_wins
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_draws': h2h_record.draws,
            'h2h_matches': h2h_record.matches
        }

    def _build_team_features(self, team: str, prefix: str) -> Dict[str, Any]:
        """Extrae y formatea todas las variables de un equipo con un prefijo dado."""
        st = self.teams[team]
        goal_diff = st.goals_for - st.goals_against
        
        return {
            f'{prefix}matches': st.matches,
            f'{prefix}wins': st.wins,
            f'{prefix}draws': st.draws,
            f'{prefix}losses': st.losses,
            f'{prefix}goals_for': st.goals_for,
            f'{prefix}goals_against': st.goals_against,
            f'{prefix}goal_difference': goal_diff,
            f'{prefix}win_rate': self._get_win_rate(st),
            f'{prefix}avg_goals_for': self._get_avg_goals_for(st),
            f'{prefix}avg_goals_against': self._get_avg_goals_against(st),
            f'{prefix}form_5': self._calculate_form(st, 5),
            f'{prefix}form_10': self._calculate_form(st, 10),
            f'{prefix}avg_goals_5': self._calculate_recent_avg_goals(st, 5),
            f'{prefix}avg_goals_10': self._calculate_recent_avg_goals(st, 10),
            f'{prefix}consecutive_wins': st.consecutive_wins,
            f'{prefix}unbeaten_streak': st.unbeaten_streak,
            f'{prefix}consecutive_losses': st.consecutive_losses
        }

    def _determine_target(self, home_score: int, away_score: int) -> Tuple[str, str, str]:
        """Determina el resultado del partido (Local, Empate, Visitante)."""
        if home_score > away_score:
            return 'Local', 'W', 'L'
        elif home_score == away_score:
            return 'Empate', 'D', 'D'
        else:
            return 'Visitante', 'L', 'W'
            
    def _parse_tournament(self, tournament: str) -> Dict[str, int]:
        """Crea variables binarias basadas en el nombre del torneo."""
        tournament_lower = str(tournament).lower()
        is_world_cup = 1 if 'world cup' in tournament_lower and 'qualification' not in tournament_lower and 'qualifier' not in tournament_lower else 0
        is_qualification = 1 if 'qualification' in tournament_lower or 'qualifier' in tournament_lower else 0
        is_friendly = 1 if 'friendly' in tournament_lower else 0
        
        return {
            'is_world_cup': is_world_cup,
            'is_qualification': is_qualification,
            'is_friendly': is_friendly
        }

    def _update_streaks(self, st: TeamStats, result: str) -> None:
        """Actualiza las rachas de victorias, invictos y derrotas."""
        if result == 'W':
            st.wins += 1
            st.consecutive_wins += 1
            st.unbeaten_streak += 1
            st.consecutive_losses = 0
        elif result == 'D':
            st.draws += 1
            st.consecutive_wins = 0
            st.unbeaten_streak += 1
            st.consecutive_losses = 0
        else:
            st.losses += 1
            st.consecutive_wins = 0
            st.unbeaten_streak = 0
            st.consecutive_losses += 1

    def _update_team(self, team: str, goals_scored: int, goals_conceded: int, result: str) -> None:
        """Actualiza el registro histórico completo de un equipo luego de un partido."""
        st = self.teams[team]
        st.matches += 1
        st.goals_for += goals_scored
        st.goals_against += goals_conceded
        st.recent_goals_for.append(goals_scored)
        st.recent_results.append(result)
        
        self._update_streaks(st, result)

    def _update_h2h(self, home_team: str, away_team: str, target: str) -> None:
        """Actualiza el registro de enfrentamientos directos entre dos equipos."""
        teams_sorted = tuple(sorted([home_team, away_team]))
        h2h_record = self.h2h[teams_sorted]
        h2h_record.matches += 1
        
        is_home_team_a = (home_team == teams_sorted[0])
        
        if target == 'Local':
            if is_home_team_a:
                h2h_record.team_a_wins += 1
            else:
                h2h_record.team_b_wins += 1
        elif target == 'Visitante':
            if is_home_team_a:
                h2h_record.team_b_wins += 1
            else:
                h2h_record.team_a_wins += 1
        else:
            h2h_record.draws += 1

    def _build_match(self, row: pd.Series) -> Dict[str, Any]:
        """Procesa una fila individual (partido) integrando todas las lógicas."""
        home = row['home_team']
        away = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        tournament = row.get('tournament', '')
        neutral = int(row.get('neutral', 0))
        
        # 1. Extraer características ANTES de actualizar historiales (Evita data leakage)
        home_features = self._build_team_features(home, 'home_')
        away_features = self._build_team_features(away, 'away_')
        h2h_features = self._get_h2h_stats(home, away)
        tournament_features = self._parse_tournament(tournament)
        
        target, home_res, away_res = self._determine_target(home_score, away_score)
        
        # Variables relativas (Diferencias)
        diff_features = {
            'goal_difference_strength': home_features['home_goal_difference'] - away_features['away_goal_difference'],
            'win_rate_difference': home_features['home_win_rate'] - away_features['away_win_rate'],
            'goals_for_difference': home_features['home_avg_goals_for'] - away_features['away_avg_goals_for'],
            'goals_against_difference': home_features['home_avg_goals_against'] - away_features['away_avg_goals_against']
        }
        
        # Compilar la fila de características
        row_features = {
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'tournament': tournament,
            'neutral': neutral,
            'target': target
        }
        
        row_features.update(home_features)
        row_features.update(away_features)
        row_features.update(diff_features)
        row_features.update(tournament_features)
        row_features.update(h2h_features)
        
        # 2. Actualizar historiales POST-partido
        self._update_team(home, home_score, away_score, home_res)
        self._update_team(away, away_score, home_score, away_res)
        self._update_h2h(home, away, target)
        
        return row_features

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Punto de entrada para la generación del DataFrame final."""
        # Ordenamiento cronológico estricto
        df_sorted = df.copy()
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        df_sorted = df_sorted.sort_values('date').reset_index(drop=True)
        
        features_list = []
        for _, row in df_sorted.iterrows():
            features_list.append(self._build_match(row))
            
        return pd.DataFrame(features_list)

def build_match_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Función pública y núcleo principal para generar el DataFrame de características sin data leakage.
    
    Toma un DataFrame de resultados históricos filtrados (p. ej., desde el año 2000), los ordena
    cronológicamente y calcula el rendimiento histórico, la forma reciente, rachas y 
    enfrentamientos directos estrictamente antes de que se juegue cada partido.
    
    Args:
        results_df (pd.DataFrame): DataFrame original de partidos. Debe contener al menos:
            'date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'neutral'.
            
    Returns:
        pd.DataFrame: Nuevo DataFrame listo para algoritmos de Machine Learning (XGBoost, etc).
    """
    builder = MatchFeatureBuilder()
    return builder.build_features(results_df)
