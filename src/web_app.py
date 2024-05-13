import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import pickle

# Load your models
with open('../artifacts/model_1.pkl', 'rb') as file:
    model_simple = pickle.load(file)
with open('../artifacts/model_2.pkl', 'rb') as file:
    model_engineered = pickle.load(file)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Stock Price Trend Prediction"),
    html.Div([
        html.Div([
            html.Label("Open Price:"),
            dcc.Input(id='open', type='number', placeholder="Enter opening price"),
        ], className="three columns"),

        html.Div([
            html.Label("High Price:"),
            dcc.Input(id='high', type='number', placeholder="Enter highest price"),
        ], className="three columns"),

        html.Div([
            html.Label("Low Price:"),
            dcc.Input(id='low', type='number', placeholder="Enter lowest price"),
        ], className="three columns"),

        html.Div([
            html.Label("Close Price:"),
            dcc.Input(id='close', type='number', placeholder="Enter closing price"),
        ], className="three columns"),
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
    dash.dependencies.State('model-type', 'value')])
def update_output(n_clicks, open_price, high_price, low_price, close_price, model_type):
    features = [[open_price, high_price, low_price, close_price]]
    if model_type == 'simple':
        prediction = model_simple.predict(features)
    else:
        prediction = model_engineered.predict(features)
    result = "Increase" if prediction[0] == 1 else "Decrease"
    return f"The predicted stock trend using {model_type} model is: {result}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
