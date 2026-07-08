import sys
sys.path.append('C:/Users/darie/Downloads/PROYECTO_FIFA2026')
from src.prediction import load_world_cup_data, get_team_features, build_match_vector, predict_match

print("--- TESTING LOAD ---")
world = load_world_cup_data()
print("Type of world:", type(world))
print("Cols:", world.columns.tolist())

print("\n--- TESTING GET TEAM ---")
arg = get_team_features("Argentina")
print("Argentina stats length:", len(arg))
if not arg.empty:
    print("Argentina matches:", arg.get('matches'))

fra = get_team_features("Francia")
print("Francia stats length:", len(fra))

print("\n--- TESTING VECTORS ---")
try:
    v1 = build_match_vector("Argentina", "Suiza")
    print("Arg vs Sui vector shape:", v1.shape)
    # Just sum to check not 0
    print("Arg vs Sui vector sum:", v1.sum().sum())
    
    v2 = build_match_vector("Francia", "Marruecos")
    print("Fra vs Mar vector sum:", v2.sum().sum())
    print("Vectors are different:", v1.sum().sum() != v2.sum().sum())
except Exception as e:
    print("Error in vector:", e)
    
print("\n--- TESTING PREDICT ---")
try:
    p1 = predict_match("Argentina", "Suiza")
    print("Arg vs Sui probs:", p1['probabilities'])
    p2 = predict_match("Francia", "Marruecos")
    print("Fra vs Mar probs:", p2['probabilities'])
except Exception as e:
    print("Error in predict:", e)
