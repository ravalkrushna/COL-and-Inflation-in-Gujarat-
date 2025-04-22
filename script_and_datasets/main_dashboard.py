
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import os

app = dash.Dash(__name__)
server = app.server

# Load available forecast CSVs
data_dir = "../data"
forecast_files = [f for f in os.listdir(data_dir) if f.startswith("forecast_") and f.endswith(".csv")]
cities = [f.replace("forecast_", "").replace(".csv", "") for f in forecast_files]

app.layout = html.Div([
    html.H1("Gujarat City Forecast Dashboard", style={'textAlign': 'center'}),
    html.Label("Select a City:"),
    dcc.Dropdown(
        id="city-dropdown",
        options=[{"label": city, "value": city} for city in cities],
        value=cities[0]
    ),
    dcc.Graph(id="forecast-graph")
])

@app.callback(
    Output("forecast-graph", "figure"),
    Input("city-dropdown", "value")
)
def update_graph(selected_city):
    file_path = os.path.join(data_dir, f"forecast_{selected_city}.csv")
    df = pd.read_csv(file_path)
    if 'Date' in df.columns and 'Forecast' in df.columns:
        fig = px.line(df, x='Date', y='Forecast', title=f"{selected_city} - Forecast Over Time")
    else:
        fig = px.line(title=f"No forecast data for {selected_city}")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
