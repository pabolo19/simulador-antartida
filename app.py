from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import gc

app = Flask(__name__)
CORS(app) 

print("🛡️ Servidor iniciado con el Diccionario de Modelos Corregido...")

# Aquí están los nombres EXACTOS de tu lista
archivos_modelos = {
    'no3': 'modelo_final_no3.pkl',
    'sat_n2o': 'modelo_final_sat_n2o.pkl',
    'po4': 'modelo_rf_po4.pkl',
    'sat_ch4': 'modelo_rf_sat_ch4.pkl',
    'tsm': 'modelo_rf_tsm.pkl',
    'temperatura': 'modelo_nn_temperatura.pkl',
    'salinidad': 'modelo_nn_salinidad.pkl',
    'ph': 'modelo_rf_ph.pkl',
    'densidad': 'modelo_rf_densidad.pkl',
    'conductividad': 'modelo_nn_conductividad.pkl',
    'oxigeno': 'modelo_nn_oxigeno.pkl',
    'fluorescencia': 'modelo_nn_fluorescencia.pkl',
    'sat_n2': 'modelo_nn_saturacion_n2.pkl',
    'vel_sonido': 'modelo_nn_vel_sonido.pkl'
}

@app.route('/', methods=['GET'])
def home():
    return "¡Radar Oceanográfico en línea! (Modelos Enlazados Correctamente 🔗)"

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.json
    lat = float(datos.get('latitud', -67.8))
    lon = float(datos.get('longitud', -67.3))
    prof = float(datos.get('profundidad', 20.0))
    sal = float(datos.get('salinidad', 34.5)) 

    resultados = {}

    for variable, archivo in archivos_modelos.items():
        modelo = None
        try:
            modelo = joblib.load(archivo)
            
            # Asumimos que los biológicos usan salinidad y los físicos profundidad
            if variable in ['no3', 'po4']:
                resultados[variable] = round(modelo.predict([[lat, lon, sal]])[0], 4)
            else:
                resultados[variable] = round(modelo.predict([[lat, lon, prof]])[0], 4)
                
        except ValueError as e:
            if hasattr(modelo, 'feature_names_in_'):
                esperadas = list(modelo.feature_names_in_)
                resultados[variable] = f"Error: Requiere {len(esperadas)} datos -> {esperadas}"
            else:
                resultados[variable] = "Error de formato (Probablemente necesite sus traductores.pkl)"
        except FileNotFoundError:
            resultados[variable] = "Modelo no encontrado."
        except Exception as e:
            resultados[variable] = f"Error: {str(e)}"
        finally:
            if modelo is not None:
                del modelo # Borramos de RAM
    
    gc.collect() # Limpiamos basura
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)