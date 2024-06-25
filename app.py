from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd 
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home(): 
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        max_power = float(request.form['max_power (in bph)'])
        year = float(request.form['year'])
        driven = float(request.form['km_driven'])
        gas = float(request.form['fuel'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[max_power, year, driven, gas]], columns=['max_power (in bph)', 'year','km_driven','fuel'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

