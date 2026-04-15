import joblib
import numpy as np
import warnings

# Ignorar advertencias de formato de scikit-learn para que la terminal se vea limpia
warnings.filterwarnings("ignore")

print("\n" + "="*60)
print("   INICIANDO SIMULADOR ANTÁRTICO V1.0")
print("   Cargando 9 cerebros de IA y traductores...")
print("="*60)

try:
    # 1. Cargar el traductor de entrada maestro
    scaler_X = joblib.load('traductor_entradas_XYZ.pkl')

    # 2. Cargar modelos de Redes Neuronales y sus traductores
    variables_nn = ['temperatura', 'salinidad', 'conductividad', 'oxigeno', 
                    'fluorescencia', 'saturacion_n2', 'vel_sonido']
    redes_neuronales = {}
    traductores_y = {}
    
    for var in variables_nn:
        redes_neuronales[var] = joblib.load(f'modelo_nn_{var}.pkl')
        traductores_y[var] = joblib.load(f'traductor_salida_{var}.pkl')

    # 3. Cargar modelos Random Forest (pH y Densidad)
    rf_ph = joblib.load('modelo_rf_ph.pkl')
    rf_densidad = joblib.load('modelo_rf_densidad.pkl')
    
    print("✅ Todos los sistemas en línea. Listo para inmersión.")

except FileNotFoundError as e:
    print(f"❌ Error al cargar archivos. Asegúrate de ejecutar los códigos de guardado antes. Detalle: {e}")
    exit()

# 4. EL BUCLE INTERACTIVO DE SIMULACIÓN
while True:
    print("\n" + "-"*60)
    print("Ingresa coordenadas para lanzar un CTD virtual (o escribe 'salir').")
    
    entrada = input("Presiona Enter para iniciar o escribe 'salir': ")
    if entrada.lower() == 'salir':
        print("\nApagando Simulador... ¡Hasta la próxima investigación!")
        break
        
    try:
        # Pedir datos al usuario
        lat = float(input("🌎 Latitud (ej. -62.5) : "))
        lon = float(input("🌎 Longitud (ej. -59.2): "))
        pres = float(input("⚓ Profundidad (m / db): "))
        
        print("\nCalculando termodinámica y biogeoquímica con Inteligencia Artificial...")
        
        # --- FASE A: Predicciones Base (Redes Neuronales) ---
        coordenada_cruda = np.array([[lat, lon, pres]])
        coordenada_traducida = scaler_X.transform(coordenada_cruda)
        
        # Predecimos Temp y Salinidad (las necesitamos para el siguiente paso)
        temp_scaled = redes_neuronales['temperatura'].predict(coordenada_traducida)
        sal_scaled = redes_neuronales['salinidad'].predict(coordenada_traducida)
        
        temp_real = traductores_y['temperatura'].inverse_transform(temp_scaled.reshape(-1, 1))[0][0]
        sal_real = traductores_y['salinidad'].inverse_transform(sal_scaled.reshape(-1, 1))[0][0]
        
        # --- FASE B: Predicciones Complejas (Random Forest) ---
        # El RF requiere: Lat, Lon, Presion, Temperatura, Salinidad
        entradas_rf = np.array([[lat, lon, pres, temp_real, sal_real]])
        ph_real = rf_ph.predict(entradas_rf)[0]
        densidad_real = rf_densidad.predict(entradas_rf)[0]
        
        # --- FASE C: Resto del ecosistema (Redes Neuronales) ---
        resultados = {
            'Temperatura (°C)': temp_real,
            'Salinidad (PSU)': sal_real,
            'pH': ph_real,
            'Densidad (kg/m³)': densidad_real
        }
        
        variables_restantes = ['conductividad', 'oxigeno', 'fluorescencia', 'saturacion_n2', 'vel_sonido']
        nombres_bonitos = ['Conductividad (mS/cm)', 'Oxígeno (ml/l)', 'Fluorescencia (mg/m³)', 'Sat. N2 (mg/l)', 'Vel. Sonido (m/s)']
        
        for var, nombre in zip(variables_restantes, nombres_bonitos):
            pred_scaled = redes_neuronales[var].predict(coordenada_traducida)
            pred_real = traductores_y[var].inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            resultados[nombre] = pred_real

        # --- IMPRIMIR EL PERFIL ---
        print("\n" + "="*45)
        print(" 📍 PERFIL OCEANOGRÁFICO ESTIMADO POR IA")
        print(f"    Lat: {lat} | Lon: {lon} | Prof: {pres}m")
        print("="*45)
        
        for nombre, valor in resultados.items():
            # Formatear a 4 decimales para que parezca salida de equipo Sea-Bird
            print(f" {nombre:<25}: {valor:>10.4f}")
        print("="*45)

    except ValueError:
        print("❌ Error: Por favor ingresa solo números válidos usando punto para decimales.")
    except KeyboardInterrupt: # Por si haces Control+C
        print("\n\nApagando Simulador forzosamente... ¡Adiós!")
        break