from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app) 

print("🧠 Iniciando el Gran Cerebro Antártico...")

# 1. Diccionario con todos tus modelos
archivos_modelos = {
    'no3': 'modelo_final_no3.pkl',
    'sat_n2o': 'modelo_final_sat_n2o.pkl',
    'temperatura': 'modelo_rf_temperatura.pkl',
    'salinidad': 'modelo_rf_salinidad.pkl',
    'ph': 'modelo_rf_ph.pkl',
    'densidad': 'modelo_rf_densidad.pkl',
    'conductividad': 'modelo_rf_conductividad.pkl',
    'oxigeno': 'modelo_rf_oxigeno.pkl',
    'fluorescencia': 'modelo_rf_fluorescencia.pkl',
    'sat_n2': 'modelo_rf_sat_n2.pkl',
    'vel_sonido': 'modelo_rf_vel_sonido.pkl'
}

modelos_cargados = {}

# 2. Cargar todos los modelos que existan en la carpeta
for variable, archivo in archivos_modelos.items():
    try:
        modelos_cargados[variable] = joblib.load(archivo)
        print(f"✅ Cargado: {archivo}")
    except FileNotFoundError:
        print(f"⚠️ Omitido (No encontrado): {archivo}")

@app.route('/', methods=['GET'])
def home():
    return f"¡Radar Oceanográfico en línea! Modelos activos: {list(modelos_cargados.keys())}"

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = request.json
        lat = float(datos['latitud'])
        lon = float(datos['longitud'])
        prof = float(datos['profundidad'])
        
        # Como NO3 necesita salinidad, podemos usar un valor por defecto o pedirlo
        sal = float(datos.get('salinidad', 34.5)) 

        resultados = {}

        # 3. Hacer predicciones masivas
        # Todos estos usan Lat, Lon, Prof
        entradas_fisicas = [[lat, lon, prof]]
        
        # Iterar sobre los modelos cargados para predecir
        for variable, modelo in modelos_cargados.items():
            if variable == 'no3':
                resultados[variable] = round(modelo.predict([[lat, lon, sal]])[0], 4)
            else:
                resultados[variable] = round(modelo.predict(entradas_fisicas)[0], 4)

        return jsonify(resultados)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
