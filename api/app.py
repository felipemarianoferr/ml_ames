from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from funcoes import *

MODEL_FILE_PATH = 'linear_pipeline_final.joblib'

fifteen_selected_features = [
    'Overall.Qual', 'Gr.Liv.Area', 'Garage.Cars', 'Garage.Area', 'Total.Bsmt.SF', 'Year.Built',
    'X1st.Flr.SF', 'TotRms.AbvGrd', 'Lot.Frontage', 'X2nd.Flr.SF', 'Bedroom.AbvGr', 'Exter.Qual',
    'Foundation', 'Kitchen.Qual', 'Bsmt.Qual'
]

def linhas_validas(df):
    linhas = (
             (df['Overall.Qual'] > 0)
           & (df['Gr.Liv.Area'] < 3700)
           & (df['Garage.Cars'] < 5)
           & (df['Total.Bsmt.SF'] < 3000)
           & (df['Year.Built'] > 1820)
           & (df['X1st.Flr.SF'] < 2800)
           & (df['TotRms.AbvGrd'] < 13)
           & (df['Lot.Frontage'] < 175)
           & (df['X2nd.Flr.SF'] < 1800)
           & (df['Bedroom.AbvGr'] < 7)
           )
    if 'Bsmt.Qual' in df.columns:
        linhas &= (df['Bsmt.Qual'] != 'Po')
    if 'Foundation' in df.columns:
        linhas &= (df['Foundation'] != 'Wood')
    if 'Kitchen.Qual' in df.columns:
        linhas &= (df['Kitchen.Qual'] != 'Po')
    return linhas

app = Flask(__name__)

model_pipeline = joblib.load(MODEL_FILE_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])
    input_df_selected = input_df[fifteen_selected_features]
    mascara_valida = linhas_validas(input_df_selected)

    if not mascara_valida.iloc[0]:

        return jsonify({'message': 'Dados de entrada considerados inválidos/outlier.', 'valid_row': False}), 200

    data_to_predict = input_df_selected[mascara_valida]
    log10_prediction = model_pipeline.predict(data_to_predict)
    log10_pred_value = float(log10_prediction[0])
    predicted_price = float(10**log10_pred_value)

    result = {
        'predicted_price': round(predicted_price, 2),
        'log10_prediction': round(log10_pred_value, 5)
    }
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)