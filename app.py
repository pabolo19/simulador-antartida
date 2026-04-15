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
    return "¡Radar Oceanográfico en línea! (Modo Escudo Activado 🛡️)"

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
            
            if variable == 'no3':
                resultados[variable] = round(modelo.predict([[lat, lon, sal]])[0], 4)
            else:
                resultados[variable] = round(modelo.predict([[lat, lon, prof]])[0], 4)
                
        except ValueError as e:
            resultados[variable] = "Error: Este modelo antiguo pide 5 variables en vez de 3."
        except FileNotFoundError:
            resultados[variable] = "Modelo no encontrado."
        except Exception as e:
            resultados[variable] = f"Error desconocido: {str(e)}"
        finally:
            if modelo is not None:
                del modelo # Borrar de RAM
    
    gc.collect() # Limpiar memoria
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)