import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar el archivo maestro (¡Ahora con los lentes de codificación puestos!)
archivo = 'Datos_Biologia_Completos.csv'
print(f"Cargando {archivo}...\n")
df = pd.read_csv(archivo, encoding='latin-1')

# 2. DEFINIR LA INTELIGENCIA A LA MEDIDA
modelos_optimizados = {
    'no3': {
        'objetivo': 'NO3- (umol/L)',
        'entradas': ['Latitude', 'Longitude', 'Salinity [PSU]'],
        'descripcion': 'Geografía + Salinidad (Masas de agua)'
    },
    'sat_n2o': {
        'objetivo': 'Saturation N2O (%)',
        'entradas': ['Latitude', 'Longitude', 'Depth: average (m)'],
        'descripcion': 'Geografía + Profundidad (Presión y estratificación)'
    }
}

print("🏆 ENTRENANDO LOS MODELOS BIOLÓGICOS DEFINITIVOS 🏆\n" + "="*60)

for nombre_corto, conf in modelos_optimizados.items():
    
    columna_objetivo = conf['objetivo']
    variables_entrada = conf['entradas']
    
    # Preparar datos (Filtrar nulos)
    columnas = variables_entrada + [columna_objetivo]
    df_modelo = df[columnas].copy().dropna()
        
    X = df_modelo[variables_entrada]
    y = df_modelo[columna_objetivo]
    
    # 80% entrenamiento, 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # El modelo Campeón: Bosque Aleatorio (Profundidad 5)
    modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    modelo_rf.fit(X_train, y_train)
    
    # Examen final
    predicciones = modelo_rf.predict(X_test)
    precision_r2 = r2_score(y_test, predicciones)
    error_medio = mean_absolute_error(y_test, predicciones)
    
    # Imprimir el boletín oficial
    print(f"🔬 Variable: {columna_objetivo}")
    print(f"   -> Estrategia: {conf['descripcion']}")
    print(f"   -> Precisión (R2): {precision_r2:.4f}")
    print(f"   -> Error Promedio: ±{error_medio:.4f}")
    
    # Guardar el modelo físico en tu Mac
    nombre_archivo = f'modelo_final_{nombre_corto}.pkl'
    joblib.dump(modelo_rf, nombre_archivo)
    print(f"   💾 ¡Guardado con éxito como '{nombre_archivo}'!")
    print("-" * 60)

print("\n✅ Sesión finalizada. Tienes la mejor IA posible para tus datos de la Antártida.")