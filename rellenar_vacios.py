import pandas as pd
import numpy as np

# 1. Fórmula de Haversine para calcular distancia real en kilómetros
def calcular_distancia(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radio de la Tierra en km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# 2. Cargar los datos originales
archivo = 'datos_antartida.csv'
print(f"Cargando {archivo}...")

# AQUÍ ESTÁ LA MAGIA: Le indicamos el punto y coma, el idioma, y que los guiones son vacíos
df = pd.read_csv(archivo, sep=';', encoding='latin-1', na_values=['-', '--', '---', '----', '-----'])

# 3. Lista de las columnas a rellenar
columnas_a_rellenar = [
    'Conductivity [mS/cm]', 'Salinity [PSU]', 'TSM [mg/L]', 
    'NO3- [ug-at/L]', '(PO4)3- [ug-at/L]', 'NO3- [mg/L]', '(PO4)3- [mg/L]', 
    'NO3- [mmol/m3]', 'NO3- (umol/L)', 'PO43- (umol/L)'
]

print("Iniciando Interpolación Espacial (Promedio de los 3 puntos más cercanos)...")

# 4. El Algoritmo
for col in columnas_a_rellenar:
    if col not in df.columns: 
        continue
    
    # Nos aseguramos de que la columna sea tratada matemáticamente (como números decimales)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    filas_nulas = df[df[col].isnull()]
    filas_validas = df[df[col].notnull()]
    
    if len(filas_nulas) > 0 and len(filas_validas) >= 3:
        for idx_nula, fila_nula in filas_nulas.iterrows():
            lat_nula = fila_nula['Latitude']
            lon_nula = fila_nula['Longitude']
            
            # Calcular distancias
            distancias = filas_validas.apply(
                lambda row: calcular_distancia(lat_nula, lon_nula, row['Latitude'], row['Longitude']), 
                axis=1
            )
            
            # Obtener los 3 más cercanos
            indices_cercanos = distancias.nsmallest(3).index
            promedio_cercanos = filas_validas.loc[indices_cercanos, col].mean()
            
            # Rellenar
            df.at[idx_nula, col] = promedio_cercanos
            print(f"[{col}] Fila {idx_nula} rellenada con: {promedio_cercanos:.4f}")

# 5. Guardar el nuevo archivo maestro (lo guardaremos con comas estándar de IA)
nuevo_archivo = 'Datos_Biologia_Completos.csv'
df.to_csv(nuevo_archivo, index=False, encoding='utf-8')
print(f"\n✅ ¡Misión cumplida! Datos guardados en '{nuevo_archivo}'")