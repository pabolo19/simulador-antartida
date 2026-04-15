import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Datos_Completos_Antartida.csv")
df = df.dropna(subset=['Latitud', 'Longitud', 'Presion', 'Temperatura', 'Salinidad', 'pH', 'Densidad'])
X = df[['Latitud', 'Longitud', 'Presion', 'Temperatura', 'Salinidad']]

for var in ['pH', 'Densidad']:
    print(f"Reconstruyendo cerebro de {var}...")
    modelo = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    modelo.fit(X, df[var])
    joblib.dump(modelo, f'modelo_rf_{var.lower()}.pkl')
    print(f"Guardado: modelo_rf_{var.lower()}.pkl")
