from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd  
import logging
import re

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el scaler
try:
    model = joblib.load('modeloNeuR2.pkl')
    scaler = joblib.load('dataSetScalado.pkl')
    app.logger.debug('Modelo y escalador cargados correctamente.')
except Exception as e:
    app.logger.error(f'Error al cargar el modelo: {str(e)}')

# Función para limpiar y convertir valores
def clean_and_convert(value):
    clean_value = re.sub(r'[^\d.-]', '', value)
    try:
        return float(clean_value)
    except ValueError:
        return None

@app.route('/')
def home(): 
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        max_power_raw = request.form['max_power (in bph)']
        year_raw = request.form['year']
        driven_raw = request.form['km_driven']
        fuel = request.form['fuel']

        # Limpiar y convertir los valores
        max_power = clean_and_convert(max_power_raw)
        year = clean_and_convert(year_raw)
        driven = clean_and_convert(driven_raw)

        # Mapeo de valores de combustible a numéricos si es necesario
        fuel_map = {
            'Diesel': 0,
            'Petrol': 1,
            'CNG': 2,
            'LPG': 3,
            'Electric': 4
        }
        gas = fuel_map.get(fuel, None)

        # Verificar que los datos hayan sido convertidos correctamente
        if None in [max_power, year, driven, gas]:
            raise ValueError("Datos inválidos recibidos")

        input_data = pd.DataFrame({
            'Unnamed: 0': [0],
            'name': [0],
            'year': [year],
            'km_driven': [driven],
            'fuel': [gas],
            'seller_type': [0],
            'owner': [0],
            'seats': [0],
            'max_power (in bph)': [max_power],
            'Mileage': [0], 
            'Engine (CC)': [0],
            'Mileage Unit_km/kg': [0],
            'Mileage Unit_kmpl': [0],
            'transmission_Automatic': [0],
            'transmission_Manual': [0]
        })

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        # Asegúrate de que estos índices sean correctos según cómo escalaste tus datos
        scaled_data_for_prediction = scaled_data[:, [2, 3, 4, 8]]

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)

        # Devolver la predicción como JSON
        prediction_value = round(float(prediccion[0]), 2)

        return jsonify({'prediction': prediction_value})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
