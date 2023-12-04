import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import requests

import pandas as pd
import plotly.express as px

def get_data():
    weather_requests = requests.get(
        "http://api-service:3000/obtener_datos"
    )
    json_data = weather_requests.json()
    print(json_data)
    df = pd.DataFrame(json_data)
    return df

df = get_data()


app = dash.Dash(__name__)

app.layout = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050,debug=True)