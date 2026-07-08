import sys
sys.path.append('C:/Users/darie/Downloads/PROYECTO_FIFA2026')
from src.simulation import simulate_world_cup

print("--- RUNNING SIMULATION (10 iterations) ---")
result = simulate_world_cup(simulations=10)
print("Champion:", result['champion'])
print("--- DONE ---")
