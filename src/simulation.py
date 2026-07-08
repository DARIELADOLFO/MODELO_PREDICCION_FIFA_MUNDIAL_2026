"""
Módulo de Simulación Monte Carlo para el Mundial FIFA 2026.

Este módulo se apoya en el motor de predicciones (prediction.py) para
simular de manera probabilística el desarrollo de la fase final del torneo.
Incluye lógicas para el avance de rondas y definiciones por penales.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.prediction import predict_match

# 
# Configuración de Logging y Rutas Base
# 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
OUTPUTS_DIR = ROOT_DIR / "outputs" / "simulations"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"


@dataclass
class TeamStats:
    """Mantiene el conteo de los logros alcanzados por una selección en las simulaciones."""
    semifinalist: int = 0
    finalist: int = 0
    champion: int = 0


class WorldCupSimulator:
    """
    Motor encargado de simular completamente la fase final del Mundial.
    
    Utiliza un enfoque probabilístico (Monte Carlo) donde cada resultado
    es muestreado según las probabilidades pre-calculadas por el modelo,
    asegurando que los partidos se desarrollen con incertidumbre realista.
    """

    def __init__(self, simulations: int = 10000):
        """
        Inicializa el simulador y la estructura oficial del torneo.
        
        Args:
            simulations (int): Cantidad de mundiales virtuales a simular.
        """
        self.simulations = simulations
        self.stats: Dict[str, TeamStats] = defaultdict(TeamStats)
        
        # Caché de probabilidades para optimizar drásticamente la ejecución
        # ya que evaluar Random Forest decenas de miles de veces sería costoso.
        self.match_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        
        # Llaves Oficiales de Cuartos de Final
        self.quarterfinals = [
            ("Francia", "Marruecos"),
            ("España", "Bélgica"),
            ("Inglaterra", "Noruega"),
            ("Argentina", "Suiza")
        ]
        
        # Contadores específicos para generación de DataFrames
        self.qf_wins = defaultdict(int)
        self.sf_wins = defaultdict(int)
        self.final_wins = defaultdict(int)

    def run_penalty_shootout(self, home_team: str, away_team: str) -> str:
        """
        Resuelve un empate mediante una simulación de penales (probabilidad 50-50).
        
        Args:
            home_team (str): Equipo local.
            away_team (str): Equipo visitante.
            
        Returns:
            str: Nombre del equipo ganador.
        """
        return np.random.choice([home_team, away_team])

    def simulate_match(self, home_team: str, away_team: str) -> str:
        """
        Simula probabilísticamente el resultado de un partido.
        
        Args:
            home_team (str): Equipo local.
            away_team (str): Equipo visitante.
            
        Returns:
            str: El equipo ganador (resuelve penales automáticamente si hay empate).
        """
        pair = (home_team, away_team)
        
        # Para un rendimiento extremo (Monte Carlo), consultamos al modelo 
        # una única vez por cada combinación única de partido.
        if pair not in self.match_cache:
            res = predict_match(home_team, away_team, neutral=True)
            self.match_cache[pair] = res['probabilities']
            
        probs = self.match_cache[pair]
        
        p_home = probs.get("Local", 0.0)
        p_draw = probs.get("Empate", 0.0)
        p_away = probs.get("Visitante", 0.0)
        
        # Normalización matemática de seguridad
        total = p_home + p_draw + p_away
        if total == 0:
            p_home, p_draw, p_away = 1/3, 1/3, 1/3
        else:
            p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total
            
        # Selección probabilística del resultado
        result = np.random.choice(['Local', 'Empate', 'Visitante'], p=[p_home, p_draw, p_away])
        
        if result == 'Local':
            return home_team
        elif result == 'Visitante':
            return away_team
        else:
            return self.run_penalty_shootout(home_team, away_team)

    def simulate_round(self, matches: List[Tuple[str, str]]) -> List[str]:
        """
        Ejecuta todos los partidos de una ronda completa.
        
        Args:
            matches (List[Tuple[str, str]]): Lista de enfrentamientos.
            
        Returns:
            List[str]: Lista con los ganadores de cada partido.
        """
        winners = []
        for home, away in matches:
            winner = self.simulate_match(home, away)
            winners.append(winner)
        return winners

    def simulate_final(self, home_team: str, away_team: str) -> str:
        """
        Simula el partido de la gran final.
        
        Args:
            home_team (str): Finalista 1.
            away_team (str): Finalista 2.
            
        Returns:
            str: Nombre del equipo campeón.
        """
        return self.simulate_match(home_team, away_team)

    def update_statistics(self, sf_teams: List[str], final_teams: List[str], champion: str) -> None:
        """
        Suma los logros al historial global tras finalizar un torneo completo.
        
        Args:
            sf_teams (List[str]): Equipos que llegaron a la semifinal.
            final_teams (List[str]): Equipos que llegaron a la final.
            champion (str): Equipo ganador de la final.
        """
        for team in sf_teams:
            self.stats[team].semifinalist += 1
            self.qf_wins[team] += 1
            
        for team in final_teams:
            self.stats[team].finalist += 1
            self.sf_wins[team] += 1
            
        self.stats[champion].champion += 1
        self.final_wins[champion] += 1

    def generate_summary(self) -> pd.DataFrame:
        """
        Construye la tabla maestra con las probabilidades finales del mundial.
        
        Returns:
            pd.DataFrame: Tabla con porcentajes y conteos agregados.
        """
        data = []
        for team, st in self.stats.items():
            data.append({
                "Pais": team,
                "Campeon": st.champion,
                "Finalista": st.finalist,
                "Semifinalista": st.semifinalist,
                "Campeon %": round((st.champion / self.simulations) * 100, 2),
                "Final %": round((st.finalist / self.simulations) * 100, 2),
                "Semifinal %": round((st.semifinalist / self.simulations) * 100, 2)
            })
            
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(by="Campeon %", ascending=False).reset_index(drop=True)
        return df

    def save_outputs(self, prob_df: pd.DataFrame) -> None:
        """
        Exporta automáticamente todos los DataFrames y el reporte Markdown.
        
        Args:
            prob_df (pd.DataFrame): DataFrame general (champion_probabilities.csv).
        """
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Guardar tabla principal de probabilidades
        prob_df.to_csv(OUTPUTS_DIR / "champion_probabilities.csv", index=False)
        
        # 2. Archivos detallados por ronda
        df_qf = pd.DataFrame([{"Pais": k, "Victorias Cuartos": v} for k, v in self.qf_wins.items()])
        if not df_qf.empty:
            df_qf.sort_values(by="Victorias Cuartos", ascending=False).to_csv(OUTPUTS_DIR / "quarterfinal_results.csv", index=False)
            
        df_sf = pd.DataFrame([{"Pais": k, "Victorias Semifinal": v} for k, v in self.sf_wins.items()])
        if not df_sf.empty:
            df_sf.sort_values(by="Victorias Semifinal", ascending=False).to_csv(OUTPUTS_DIR / "semifinal_results.csv", index=False)
            
        df_f = pd.DataFrame([{"Pais": k, "Victorias Final": v} for k, v in self.final_wins.items()])
        if not df_f.empty:
            df_f.sort_values(by="Victorias Final", ascending=False).to_csv(OUTPUTS_DIR / "final_results.csv", index=False)
            
        # 3. Guardar sumario
        top_champ = prob_df.iloc[0]['Pais'] if not prob_df.empty else "N/A"
        top_prob = prob_df.iloc[0]['Campeon %'] if not prob_df.empty else 0
        summary_df = pd.DataFrame({
            "Total Simulaciones": [self.simulations],
            "Modelo Utilizado": ["Random Forest"],
            "Top Campeon": [top_champ],
            "Probabilidad Maxima %": [top_prob]
        })
        summary_df.to_csv(OUTPUTS_DIR / "simulation_summary.csv", index=False)
        
        # 4. Generar Reporte Markdown
        self._generate_markdown_report(prob_df, top_champ)

    def _generate_markdown_report(self, prob_df: pd.DataFrame, top_campeon: str) -> None:
        """Helper para construir el reporte en Markdown."""
        report_path = REPORTS_DIR / "tournament_report.md"
        
        df_final = prob_df.sort_values(by="Final %", ascending=False)
        top_finalista = df_final.iloc[0]['Pais'] if not df_final.empty else "N/A"
        
        report = f"""# Reporte Oficial - Simulación del Mundial 2026

## Resumen Ejecutivo
- **Modelo utilizado:** Random Forest
- **Cantidad de simulaciones:** {self.simulations:,}
- **Equipo con mayor probabilidad de campeonato:** **{top_campeon}**
- **Equipo con mayor probabilidad de llegar a la final:** **{top_finalista}**

## Ranking Completo (Métricas)

{prob_df.to_markdown(index=False)}

---
*Reporte generado automáticamente mediante simulación probabilística de Monte Carlo.*
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Reporte automatizado MD guardado en {report_path}")

    def simulate_world_cup(self) -> Dict[str, Any]:
        """
        Orquesta toda la lógica para ejecutar la simulación masiva del torneo.
        
        Returns:
            Dict[str, Any]: Diccionario con DataFrames y metadatos resultantes.
        """
        logger.info(f"Iniciando simulación Monte Carlo ({self.simulations} iteraciones)...")
        
        # Uso de TQDM para barra de progreso amigable en la terminal/consola
        for _ in tqdm(range(self.simulations), desc="Simulando Mundiales"):
            # 1. Jugar Cuartos de final
            qf_winners = self.simulate_round(self.quarterfinals)
            sf_teams = qf_winners # 4 Semifinalistas
            
            # 2. Emparejar y jugar Semifinales
            semifinals = [
                (qf_winners[0], qf_winners[1]),  # Ganador Partido 1 vs Ganador Partido 2
                (qf_winners[2], qf_winners[3])   # Ganador Partido 3 vs Ganador Partido 4
            ]
            sf_winners = self.simulate_round(semifinals)
            final_teams = sf_winners # 2 Finalistas
            
            # 3. Jugar la Gran Final
            champion = self.simulate_final(final_teams[0], final_teams[1])
            
            # 4. Acumular historial
            self.update_statistics(sf_teams, final_teams, champion)
            
        # Generar archivos
        prob_df = self.generate_summary()
        self.save_outputs(prob_df)
        
        top_champ = prob_df.iloc[0]['Pais'] if not prob_df.empty else "None"
        
        summary_result_df = pd.DataFrame({
            "Métrica": ["Simulaciones", "Modelo", "Top Campeón"],
            "Valor": [self.simulations, "Random Forest", top_champ]
        })
        
        logger.info("Simulación finalizada exitosamente.")
        
        return {
            "champion_probabilities": prob_df,
            "summary": summary_result_df,
            "champion": top_champ,
            "simulations": self.simulations
        }


def simulate_world_cup(simulations: int = 10000) -> Dict[str, Any]:
    """
    Función pública, expuesta para iniciar toda la simulación con un comando.
    
    Args:
        simulations (int): Cantidad de veces a simular el torneo. Por defecto 10000.
        
    Returns:
        Dict: Contiene el dataframe con los resultados, resúmenes y el equipo favorito.
    """
    simulator = WorldCupSimulator(simulations=simulations)
    result = simulator.simulate_world_cup()
    
    # Generar reportes y gráficos de forma automática
    try:
        from src.visualization import generate_all_post_simulation_outputs
        generate_all_post_simulation_outputs()
    except Exception as e:
        logger.error(f"Fallo al generar visualizaciones/reportes automáticos: {e}")
        
    return result
