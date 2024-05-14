import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import joblib

# Loading models and scalers
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

# Styling dictionaries
base_color = '#007BFF'
hover_color = '#0056b3'
error_color = '#ff0000'
background_color = '#f8f9fa'
text_color = '#343a40'
input_style = {
    'width': '100%', 'padding': '10px', 'margin': '10px 0', 
    'border': '2px solid #ccc', 'border-radius': '5px',
    'transition': 'border-color 0.3s'
}
column_style = {
    'margin': '10px', 'flex': '1'
}
button_style = {
    'fontSize': '16px', 'padding': '10px 20px', 'background-color': base_color,
    'color': 'white', 'border': 'none', 'border-radius': '5px', 'cursor': 'pointer',
    'transition': 'background-color 0.3s ease'
}

# Define the layout
app.layout = html.Div(style={'padding': '20px', 'fontFamily': 'Arial', 'backgroundColor': background_color}, children=[
    html.H1("Apple Stock Price Trend Prediction", style={'textAlign': 'center', 'color': text_color}),
    html.H3("Please enter stock prices as they appear (e.g., 19.8457) and large volumes scaled down (e.g., enter 100,000 as '100' for thousands)", style={'textAlign': 'center', 'color': text_color}),
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
        html.Div(style=column_style, children=[
            html.Label("Open Price:"),
            dcc.Input(id='open', type='number', placeholder="Enter opening price", style=input_style),
        ]),
        html.Div(style=column_style, children=[
            html.Label("High Price:"),
            dcc.Input(id='high', type='number', placeholder="Enter highest price", style=input_style),
        ]),
        html.Div(style=column_style, children=[
            html.Label("Low Price:"),
            dcc.Input(id='low', type='number', placeholder="Enter lowest price", style=input_style),
        ]),
        html.Div(style=column_style, children=[
            html.Label("Close Price:"),
            dcc.Input(id='close', type='number', placeholder="Enter closing price", style=input_style),
        ]),
        html.Div(style=column_style, children=[
            html.Label("Volume:"),
            dcc.Input(id='volume', type='number', placeholder="Enter volume", style=input_style),
        ]),
    ]),

    html.Label("Select Model:", style={'marginTop': '20px'}),
    dcc.RadioItems(
        id='model-type',
        options=[
            {'label': 'Simple Model', 'value': 'simple'},
            {'label': 'Engineered Model', 'value': 'engineered'}
        ],
        value='simple',
        style={'marginTop': '10px', 'marginBottom': '20px'}
    ),

    html.Button('Predict Stock Trend', id='submit-val', n_clicks=0, style=button_style),
    html.Div(id='output-prediction', style={'marginTop': '20px', 'fontSize': '20px', 'textAlign': 'center', 'color': text_color})
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
        return html.Div("Please enter all fields to predict the stock trend.", style={'color': error_color})

    features = np.array([[open_price, high_price, low_price, close_price, volume]])
    engineered_features = np.array([[open_price, high_price, low_price, close_price, volume, 0, 0, 0, 0]])

    if model_type == 'simple':
        features_scaled = scaler_basic.transform(features)
        prediction = model_simple.predict(features_scaled)
    else:
        features_scaled = scaler_engineered.transform(engineered_features)
        prediction = model_engineered.predict(features_scaled)

    result = "Increase" if prediction[0] > 0.5 else "Decrease"
    result_color = '#28a745' if result == "Increase" else '#dc3545'
    
    return html.Div(f"The predicted stock trend for the next day is: {result}",
                    style={'marginTop': '20px', 'fontSize': '20px', 'textAlign': 'center', 'color': result_color})


if __name__ == '__main__':
    app.run_server(debug=True)
