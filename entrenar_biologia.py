import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# 1. Cargar los datos limpios
archivo = 'Datos_Biologia_Completos.csv'
print(f"Cargando el archivo maestro: {archivo}...\n")
df = pd.read_csv(archivo)

# --- SOLUCIÓN AL SÍMBOLO CORRUPTO ---
# Buscamos dinámicamente cualquier columna que contenga "Surface Sea Temp"
# sin importar los caracteres raros que tenga al final.
col_temp_original = [c for c in df.columns if 'Surface Sea Temp' in c][0]

# La renombramos a algo totalmente seguro
df.rename(columns={col_temp_original: 'Temperatura_Superficial'}, inplace=True)
# ------------------------------------

# 2. Definir las 6 entradas (Físicas + Geográficas) usando el nombre limpio
variables_entrada = [
    'Latitude', 
    'Longitude', 
    'Depth: average (m)',
    'Temperatura_Superficial',  
    'Conductivity [mS/cm]',
    'Salinity [PSU]'
]

# 3. Definir las 5 variables objetivo
variables_objetivo = {
    'tsm': 'TSM [mg/L]',
    'po4': 'PO43- (umol/L)',
    'no3': 'NO3- (umol/L)',
    'sat_ch4': 'Saturation CH4 (%)',
    'sat_n2o': 'Saturation N2O (%)'
}

print("🌱 Iniciando la plantación de los Bosques Aleatorios (con variables Físicas)...\n" + "-"*50)

# 4. Entrenar los modelos
for nombre_corto, columna_objetivo in variables_objetivo.items():
    
    columnas = variables_entrada + [columna_objetivo]
    
    # Sistema de seguridad
    columnas_faltantes = [col for col in columnas if col not in df.columns]
    if columnas_faltantes:
        print(f"⚠️ Error en {nombre_corto}: Faltan estas columnas en el CSV: {columnas_faltantes}")
        continue

    # Extraer los datos y limpiar nulos
    df_modelo = df[columnas].copy()
    df_modelo = df_modelo.dropna()
    
    # Validar que tengamos datos suficientes
    if len(df_modelo) < 10:
        print(f"⚠️ Omitiendo {nombre_corto}: Muy pocos datos.")
        continue
        
    X = df_modelo[variables_entrada]
    y = df_modelo[columna_objetivo]
    
    # Dividir: 80% Entrenamiento, 20% Prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construir el Bosque Aleatorio
    modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    
    # ¡Entrenar!
    modelo_rf.fit(X_train, y_train)
    
    # Examen final
    predicciones = modelo_rf.predict(X_test)
    
    # Calificar
    precision_r2 = r2_score(y_test, predicciones)
    error_medio = mean_absolute_error(y_test, predicciones)
    
    print(f"📊 Modelo para: {columna_objetivo}")
    print(f"   -> Precisión (R2): {precision_r2:.4f}")
    print(f"   -> Error Promedio: ±{error_medio:.4f}")
    
    # Guardar
    nombre_archivo_pkl = f'modelo_rf_{nombre_corto}.pkl'
    joblib.dump(modelo_rf, nombre_archivo_pkl)
    print(f"   -> Guardado en disco como: {nombre_archivo_pkl}")
    print("-" * 50)

print("\n🎉 ¡Modelos actualizados y guardados!")