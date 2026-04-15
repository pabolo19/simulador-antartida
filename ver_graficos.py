import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar datos
df = pd.read_csv('Datos_Biologia_Completos.csv', encoding='latin-1')

modelos = {
    'no3': {
        'objetivo': 'NO3- (umol/L)',
        'entradas': ['Latitude', 'Longitude', 'Salinity [PSU]'],
        'titulo': 'Nitrato (NO3-): Real vs Predicción'
    },
    'sat_n2o': {
        'objetivo': 'Saturation N2O (%)',
        'entradas': ['Latitude', 'Longitude', 'Depth: average (m)'],
        'titulo': 'Saturación N2O (%): Real vs Predicción'
    }
}

# 2. Configurar el lienzo (1 fila, 2 columnas)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (nombre_corto, conf) in enumerate(modelos.items()):
    columna_objetivo = conf['objetivo']
    variables_entrada = conf['entradas']
    
    cols = variables_entrada + [columna_objetivo]
    df_modelo = df[cols].copy().dropna()
        
    X = df_modelo[variables_entrada]
    y = df_modelo[columna_objetivo]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    modelo_rf.fit(X_train, y_train)
    
    y_train_pred = modelo_rf.predict(X_train)
    y_test_pred = modelo_rf.predict(X_test)
    
    ax = axes[idx]
    
    # Dibujar puntos
    ax.scatter(y_train, y_train_pred, color='#3498db', alpha=0.7, s=50, label='Entrenamiento (Vista previa)')
    ax.scatter(y_test, y_test_pred, color='#e74c3c', s=100, edgecolor='black', label='Prueba (Datos ocultos)')
    
    # Dibujar la línea diagonal perfecta
    min_val = min(y.min(), min(y_train_pred.min(), y_test_pred.min()))
    max_val = max(y.max(), max(y_train_pred.max(), y_test_pred.max()))
    buffer = (max_val - min_val) * 0.05
    ax.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 
            'k--', lw=2, alpha=0.6, label='Línea de Perfección')
    
    # Estilos
    ax.set_title(conf['titulo'], fontsize=14, fontweight='bold')
    ax.set_xlabel('Valor Real Medido', fontsize=12)
    ax.set_ylabel('Valor Predicho', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Guardar la imagen en tu carpeta
nombre_imagen = 'Mis_Resultados_IA.png'
plt.savefig(nombre_imagen, dpi=300)
print(f"✅ ¡Gráfico generado y guardado en tu Mac como '{nombre_imagen}'!")