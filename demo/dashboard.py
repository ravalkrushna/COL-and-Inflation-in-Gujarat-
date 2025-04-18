import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load dataset
df = pd.read_csv('gujarat_cost_of_living_full_dataset.csv')

# Forecast Accuracy results (you can merge your forecast accuracy results CSV file if needed)
forecast_accuracy_df = pd.read_csv('forecast_accuracy_results.csv')

# Calculate liveability score (simple average of the normalized values)
# You may want to tweak this calculation as per your desired model for liveability
columns_for_liveability = ['Rent', 'Utilities', 'Grocery', 'Transportation', 'Healthcare', 'Education']
df['Liveability_Score'] = df[columns_for_liveability].mean(axis=1)

# Create a Plotly Express line chart for expense trends
fig_expense_trends = px.line(df, x='Date', y='Total_Monthly_Expense', color='District', title='Expense Trends by District')

# Create a bar chart for Forecast Accuracy (MAPE)
fig_forecast_accuracy = px.bar(forecast_accuracy_df, x='District', y='MAPE (%)', title='Forecast Accuracy (MAPE) by District')

# Create a bar chart for Liveability Score Ranking
liveability_df = df.groupby('District')['Liveability_Score'].mean().reset_index()
fig_liveability_ranking = px.bar(liveability_df, x='District', y='Liveability_Score', title='Liveability Score Ranking')

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children="Gujarat City Expense and Liveability Dashboard"),

    html.Div(children="""
        This dashboard presents key metrics such as city expense trends, forecasting accuracy (MAPE), and liveability rankings based on various expense categories.
    """),

    dcc.Graph(
        id='expense-trends',
        figure=fig_expense_trends
    ),
    
    dcc.Graph(
        id='forecast-accuracy',
        figure=fig_forecast_accuracy
    ),

    dcc.Graph(
        id='liveability-ranking',
        figure=fig_liveability_ranking
    )
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
