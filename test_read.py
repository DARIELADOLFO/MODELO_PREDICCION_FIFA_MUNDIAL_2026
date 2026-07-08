import pandas as pd
import unicodedata

def remove_accents(s):
    if pd.isna(s): return s
    return ''.join(c for c in unicodedata.normalize('NFD', str(s).strip()) if unicodedata.category(c) != 'Mn').lower()

df = pd.read_excel('C:/Users/darie/Downloads/PROYECTO_FIFA2026/data/raw/Dataset_Mundial_2026_Actualizado.xlsx')
print("Original cols:", df.columns.tolist())
team_col = None
for c in df.columns:
    if remove_accents(c) in ['team', 'equipo', 'pais', 'country']:
        team_col = c
        break

print("Found team col:", team_col)
if team_col:
    df.set_index(team_col, inplace=True)
    df.index = df.index.map(remove_accents)
    print("New index:", df.index.tolist()[:5])
    print("Is francia in index?", 'francia' in df.index)
    print("Is argentina in index?", 'argentina' in df.index)
