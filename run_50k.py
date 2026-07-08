import sys
sys.path.append('C:/Users/darie/Downloads/PROYECTO_FIFA2026')
from src.simulation import simulate_world_cup

print("--- RUNNING SIMULATION (50,000 iterations) ---")
result = simulate_world_cup(simulations=50000)
print("\n--- FINAL RESULTS ---")
print("Champion:", result['champion'])
print("Top 5 DataFrame:")
print(result['champion_probabilities'].head(5))
print("--- DONE ---")
