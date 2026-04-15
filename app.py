from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import gc
import numpy as np

app = Flask(__name__)
CORS(app) 

print("🧠 Servidor iniciado con Redes Neuronales y Traductores Activados...")

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

archivos_traductores_salida = {
    'temperatura': 'traductor_salida_temperatura.pkl',
    'salinidad': 'traductor_salida_salinidad.pkl',
    'conductividad': 'traductor_salida_conductividad.pkl',
    'oxigeno': 'traductor_salida_oxigeno.pkl',
    'fluorescencia': 'traductor_salida_fluorescencia.pkl',
    'sat_n2': 'traductor_salida_saturacion_n2.pkl',
    'vel_sonido': 'traductor_salida_vel_sonido.pkl'
}

@app.route('/', methods=['GET'])
def home():
    return "¡Radar Oceanográfico en línea! (Traductores Activados 🌐)"

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.json
    lat = float(datos.get('latitud', -67.8))
    lon = float(datos.get('longitud', -67.3))
    prof = float(datos.get('profundidad', 20.0))
    sal = float(datos.get('salinidad', 34.5)) 

    resultados = {}
    
    # 1. Cargar el Traductor de Entradas (Para Lat, Lon, Prof)
    traductor_entradas = None
    try:
        traductor_entradas = joblib.load('traductor_entradas_XYZ.pkl')
        entradas_nn = traductor_entradas.transform([[lat, lon, prof]])
    except Exception as e:
        entradas_nn = [[lat, lon, prof]] # Si falla, usa los normales

    for variable, archivo in archivos_modelos.items():
        modelo = None
        try:
            modelo = joblib.load(archivo)
            
            # Si el modelo es una Red Neuronal
            if 'modelo_nn_' in archivo:
                # Predecimos con los datos en idioma máquina
                pred_maquina = modelo.predict(entradas_nn)
                
                # Traducimos de vuelta a idioma humano
                if variable in archivos_traductores_salida:
                    traductor_salida = joblib.load(archivos_traductores_salida[variable])
                    # Descomprimimos el número
                    valor_humano = traductor_salida.inverse_transform(np.array(pred_maquina).reshape(-1, 1))[0][0]
                    resultados[variable] = round(float(valor_humano), 4)
                    del traductor_salida
                else:
                    resultados[variable] = round(float(pred_maquina[0]), 4)
                    
            # Si es un Bosque Aleatorio (Random Forest)
            else:
                if variable in ['no3', 'po4']:
                    resultados[variable] = round(modelo.predict([[lat, lon, sal]])[0], 4)
                else:
                    resultados[variable] = round(modelo.predict([[lat, lon, prof]])[0], 4)
                
        except ValueError as e:
            if hasattr(modelo, 'feature_names_in_'):
                esperadas = list(modelo.feature_names_in_)
                resultados[variable] = f"Error: Requiere {len(esperadas)} datos -> {esperadas}"
            else:
                resultados[variable] = "Error de formato de datos."
        except FileNotFoundError:
            resultados[variable] = "Modelo o traductor no encontrado."
        except Exception as e:
            resultados[variable] = f"Error: {str(e)}"
        finally:
            if modelo is not None:
                del modelo # Borrar de RAM
    
    if traductor_entradas is not None:
        del traductor_entradas
        
    gc.collect() # Limpieza de memoria
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)