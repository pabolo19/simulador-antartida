from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import gc

app = Flask(__name__)
CORS(app) 

print("🛡️ Servidor iniciado en Modo Carga Perezosa y Resiliente...")

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
    return "¡Radar Oceanográfico en línea! (Modo Ahorro de Memoria y Escudo Activado 🛡️)"

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.json
    lat = float(datos.get('latitud', -67.8))
    lon = float(datos.get('longitud', -67.3))
    prof = float(datos.get('profundidad', 20.0))
    sal = float(datos.get('salinidad', 34.5)) 

    resultados = {}

    # El servidor intentará predecir uno por uno sin que un error detenga al resto
    for variable, archivo in archivos_modelos.items():
        modelo = None
        try:
            modelo = joblib.load(archivo)
            
            if variable == 'no3':
                resultados[variable] = round(modelo.predict([[lat, lon, sal]])[0], 4)
            else:
                resultados[variable] = round(modelo.predict([[lat, lon, prof]])[0], 4)
                
        except ValueError as e:
            # Si el modelo pide más de 3 variables, leemos su cerebro para saber cuáles son
            if hasattr(modelo, 'feature_names_in_'):
                esperadas = list(modelo.feature_names_in_)
                resultados[variable] = f"Error: El modelo requiere {len(esperadas)} datos exactos -> {esperadas}"
            else:
                resultados[variable] = "Error: El modelo requiere más o diferentes variables."
        except FileNotFoundError:
            resultados[variable] = "Modelo no encontrado o no subido."
        except Exception as e:
            resultados[variable] = f"Error desconocido: {str(e)}"
        finally:
            if modelo is not None:
                del modelo # Borrar de RAM
    
    gc.collect() # Limpiar basura
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)