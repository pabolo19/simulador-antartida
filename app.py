from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import gc # Herramienta para limpiar la memoria RAM

app = Flask(__name__)
CORS(app) 

print("🧠 Servidor iniciado en Modo Carga Perezosa (Ahorro de Memoria)...")

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

@app.route('/', methods=['GET'])
def home():
    return "¡Radar Oceanográfico en línea! (Modo Ahorro de Memoria Activado 🔋)"

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = request.json
        lat = float(datos['latitud'])
        lon = float(datos['longitud'])
        prof = float(datos['profundidad'])
        sal = float(datos.get('salinidad', 34.5)) 

        entradas_fisicas = [[lat, lon, prof]]
        resultados = {}

        # 🚀 LA MAGIA: Cargar un modelo, usarlo y borrarlo.
        for variable, archivo in archivos_modelos.items():
            try:
                # 1. Cargar solo este modelo a la RAM
                modelo = joblib.load(archivo)
                
                # 2. Hacer predicción
                if variable == 'no3':
                    resultados[variable] = round(modelo.predict([[lat, lon, sal]])[0], 4)
                else:
                    resultados[variable] = round(modelo.predict(entradas_fisicas)[0], 4)
                
                # 3. Borrar el modelo de la memoria RAM inmediatamente
                del modelo
                
            except FileNotFoundError:
                resultados[variable] = "No disponible"
        
        # 4. Pasar la "escoba" para vaciar la basura de la RAM
        gc.collect()

        return jsonify(resultados)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
