import joblib
import warnings
warnings.filterwarnings('ignore') # Para ocultar advertencias molestas

print("🧠 Despertando a los modelos de Inteligencia Artificial...\n")

try:
    # 1. Cargar los modelos guardados
    modelo_no3 = joblib.load('modelo_final_no3.pkl')
    modelo_n2o = joblib.load('modelo_final_sat_n2o.pkl')
except FileNotFoundError:
    print("❌ Error: No encuentro los archivos .pkl. Asegúrate de haberlos generado.")
    exit()

print("🤖 Modelos listos. Ingresa los datos de tu nueva expedición:")
print("-" * 50)

# 2. Pedir datos al usuario desde la terminal
try:
    lat = float(input("📍 Latitud (ej. -67.82): "))
    lon = float(input("📍 Longitud (ej. -67.35): "))
    prof = float(input("⬇️  Profundidad en metros (ej. 17.0): "))
    sal = float(input("🧂 Salinidad en PSU (ej. 34.6): "))
except ValueError:
    print("\n❌ Error: Debes ingresar números válidos.")
    exit()

print("\n" + "=" * 50)
print("🔮 RESULTADOS DE LA PREDICCIÓN 🔮")
print("=" * 50)

# 3. Ejecutar las predicciones
# OJO: Le pasamos exactamente la combinación que cada modelo necesita
prediccion_no3 = modelo_no3.predict([[lat, lon, sal]])[0]
prediccion_n2o = modelo_n2o.predict([[lat, lon, prof]])[0]

# 4. Mostrar resultados
print(f"🧬 Nitrato (NO3-) Estimado: {prediccion_no3:.4f} umol/L")
print(f"   (Basado en Latitud, Longitud y Salinidad)")
print("-" * 50)
print(f"🫧 Saturación N2O Estimada: {prediccion_n2o:.4f} %")
print(f"   (Basado en Latitud, Longitud y Profundidad)")
print("=" * 50)