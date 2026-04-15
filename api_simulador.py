from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

print("Cargando el arsenal oceanográfico completo (9 Modelos)...")
try:
    # 1. Traductor maestro
    scaler_X = joblib.load('traductor_entradas_XYZ.pkl')
    
    # 2. Las 7 Redes Neuronales y sus traductores
    variables_nn = ['temperatura', 'salinidad', 'conductividad', 'oxigeno', 
                    'fluorescencia', 'saturacion_n2', 'vel_sonido']
    redes_neuronales = {}
    traductores_y = {}
    
    for var in variables_nn:
        redes_neuronales[var] = joblib.load(f'modelo_nn_{var}.pkl')
        traductores_y[var] = joblib.load(f'traductor_salida_{var}.pkl')

    # 3. Los 2 Random Forest
    rf_ph = joblib.load('modelo_rf_ph.pkl')
    rf_densidad = joblib.load('modelo_rf_densidad.pkl')
    
    print("✅ Todos los sistemas en línea. Listo para recibir peticiones web.")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")

@app.route('/api/simular', methods=['POST'])
def simular_perfil():
    try:
        # 1. Recibir coordenadas de la página web
        datos_web = request.json
        lat = float(datos_web['latitud'])
        lon = float(datos_web['longitud'])
        pres = float(datos_web['profundidad'])
        
        # 2. Traducir coordenadas para las redes neuronales
        coordenada_cruda = np.array([[lat, lon, pres]])
        coordenada_traducida = scaler_X.transform(coordenada_cruda)
        
        # 3. Calcular la Base Físico-Química (Redes Neuronales)
        resultados_finales = {}
        
        for var in variables_nn:
            pred_scaled = redes_neuronales[var].predict(coordenada_traducida)
            pred_real = traductores_y[var].inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            # Guardamos el resultado con la primera letra mayúscula para que se vea bonito
            resultados_finales[var.capitalize()] = round(float(pred_real), 4)
            
        # 4. Rescatar Temp y Salinidad para dárselas a los Bosques
        temp_real = resultados_finales['Temperatura']
        sal_real = resultados_finales['Salinidad']
        
        # 5. Calcular pH y Densidad (Random Forest)
        entradas_rf = np.array([[lat, lon, pres, temp_real, sal_real]])
        ph_real = rf_ph.predict(entradas_rf)[0]
        densidad_real = rf_densidad.predict(entradas_rf)[0]
        
        resultados_finales['Ph'] = round(float(ph_real), 4)
        resultados_finales['Densidad'] = round(float(densidad_real), 4)
        
        # 6. Añadir estatus y enviar a la web
        resultados_finales['Estatus'] = "Éxito"
        return jsonify(resultados_finales)
        
    except Exception as e:
        return jsonify({"Estatus": "Error", "Mensaje": str(e)}), 400

if __name__ == '__main__':
    # Render asigna un puerto automáticamente en la variable de entorno PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)