import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import joblib

# Load your models and scalers
model_simple_path = './artifacts/model_1.pkl'
model_engineered_path = './artifacts/model_2.pkl'
scaler_basic_path = './artifacts/scaler_basic.pkl'
scaler_engineered_path = './artifacts/scaler_engineered.pkl'

try:
    model_simple = joblib.load(model_simple_path)
    print("Simple model loaded successfully.")
except Exception as e:
    print(f"Failed to load simple model: {e}")

try:
    model_engineered = joblib.load(model_engineered_path)
    print("Engineered model loaded successfully.")
except Exception as e:
    print(f"Failed to load engineered model: {e}")

try:
    scaler_basic = joblib.load(scaler_basic_path)
    print("Basic scaler loaded successfully.")
except Exception as e:
    print(f"Failed to load basic scaler: {e}")

try:
    scaler_engineered = joblib.load(scaler_engineered_path)
    print("Engineered scaler loaded successfully.")
except Exception as e:
    print(f"Failed to load engineered scaler: {e}")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Stock Price Trend Prediction"),
    html.Div([
        html.Div([
            html.Label("Open Price:"),
            dcc.Input(id='open', type='number', placeholder="Enter opening price"),
        ], className="two columns"),

        html.Div([
            html.Label("High Price:"),
            dcc.Input(id='high', type='number', placeholder="Enter highest price"),
        ], className="two columns"),

        html.Div([
            html.Label("Low Price:"),
            dcc.Input(id='low', type='number', placeholder="Enter lowest price"),
        ], className="two columns"),

        html.Div([
            html.Label("Close Price:"),
            dcc.Input(id='close', type='number', placeholder="Enter closing price"),
        ], className="two columns"),

        html.Div([
            html.Label("Volume:"),
            dcc.Input(id='volume', type='number', placeholder="Enter volume"),
        ], className="two columns"),
    ], className="row"),

    html.Label("Select Model:"),
    dcc.RadioItems(
        id='model-type',
        options=[
            {'label': 'Simple Model', 'value': 'simple'},
            {'label': 'Engineered Model', 'value': 'engineered'}
        ],
        value='simple',
        style={'marginTop': 20, 'marginBottom': 20}
    ),

    html.Button('Predict Stock Trend', id='submit-val', n_clicks=0),
    html.Div(id='output-prediction', style={'marginTop': 20, 'fontSize': 20})
])

# Callback to update prediction
@app.callback(
    Output('output-prediction', 'children'),
    [Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('open', 'value'),
     dash.dependencies.State('high', 'value'),
     dash.dependencies.State('low', 'value'),
     dash.dependencies.State('close', 'value'),
     dash.dependencies.State('volume', 'value'),
     dash.dependencies.State('model-type', 'value')])
def update_output(n_clicks, open_price, high_price, low_price, close_price, volume, model_type):
    if None in [open_price, high_price, low_price, close_price, volume]:
        return "Please enter all fields to predict the stock trend."

    features = np.array([[open_price, high_price, low_price, close_price, volume]])
    
    if model_type == 'simple':
        features_scaled = scaler_basic.transform(features)
        prediction = model_simple.predict(features_scaled)
    else:
        features_scaled = scaler_engineered.transform(features)
        prediction = model_engineered.predict(features_scaled)

    result = "Increase" if prediction[0] > 0.5 else "Decrease"
    return f"The predicted stock trend is: {result}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
