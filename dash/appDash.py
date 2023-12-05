import pandas as pd
import numpy as np
import math
import datetime as dt

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly
import json
import dash
import dash_bootstrap_components as dbc
from dash import dcc,dash_table,html
import dash_daq as daq
from dash.dependencies import Input, Output, State, ClientsideFunction
import glob
import requests
import logging

import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.DEBUG)
logging.debug("Servicio iniciado")

principal_df = pd.DataFrame()
principal_product = ""

def get_product_names():
    data_request = requests.get(
        "http://api-service:3000/get_products_names"
    )
    json_data = data_request.json()
    values_names = [x['column_name'] for x in json_data]

    return values_names

def get_product_historical_data(selected_column):
    url = "http://api-service:3000/get_product_historical_data"
    payload = {'column': selected_column}
    json_data = requests.post(url, json=payload).json()
    
    df = pd.DataFrame(json_data)
    df['fechas'] = pd.to_datetime(df['fechas'], format='%Y-%m-%d')
    df[selected_column] = pd.to_numeric(df[selected_column])
    df.set_index('fechas', inplace=True)
    
    return df

def get_exog_historical_data(selected_column):
    url = "http://api-service:3000/get_historical_training"
    payload = {'column': selected_column}
    json_data = requests.post(url, json=payload).json()
    
    df = pd.DataFrame(json_data)
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
    col_numeric = [x for x in df.columns.tolist() if x != 'fecha']
    logging.debug(col_numeric)
    
    df[col_numeric] = df[col_numeric].apply(pd.to_numeric, errors='coerce')
    df.set_index('fecha', inplace=True)
    
    return df

def get_forecast( model_name):
    url = "http://api-service:3000/get_forecast"
    info = {'model_name': model_name}
    json_data = requests.post(url, json=info).json()
    logging.debug('REgreso del forecast')
    logging.debug(type(json_data))
    data = json_data[0]
    df = pd.DataFrame(data['variables'])
    forecast = data['forecast']
    forecast = [float(x) for x in forecast]
    
    #df = pd.DataFrame(json_data)
    df['fecha_pronos'] = pd.to_datetime(df['fecha_pronos'], format='%Y-%m-%d')

    col_numeric = [x for x in df.columns.tolist() if x != 'fecha_pronos']
    df[col_numeric] = df[col_numeric].apply(pd.to_numeric, errors='coerce')
    df[model_name] = forecast

    return df

def get_forecast_simulacion(data, model_name):
    url = "http://api-service:3000/get_forecast_simulacion"
    info = [{'variables':data, 'model_name': model_name}]
    json_data = requests.post(url, json=json.dumps(info)).json()
    data = json_data[0]
    logging.debug(data)
    df = pd.DataFrame(data['variables'])
    forecast = data['forecast']
    forecast = [float(x) for x in forecast]
    
    #df = pd.DataFrame(json_data)
    df['fecha_pronos'] = pd.to_datetime(df['fecha_pronos'])

    col_numeric = [x for x in df.columns.tolist() if x != 'fecha_pronos']
    df[col_numeric] = df[col_numeric].apply(pd.to_numeric, errors='coerce')
    df[model_name] = forecast
    
    
    return df



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


R=['Fechas', 'lower Empresas MN', 'Forecast Empresas MN',
       'upper Empresas MN']

ls=['Fechas', 'Empleo IMSS', 'Cartera Mayorista Mercado', 'CETES 3m (%)']

colors = {
    'background': "#023858",
    'text': '#7FDBFF'}

bbva_logo = "https://region6.bfp.gov.ph/wp-content/uploads/2020/10/Icon_47-512.png"

navbar = dbc.NavbarSimple(

    children=[
        dbc.Button("Ocultar", outline=True, color="#2DCCCD", className="mr-2", id="btn_sidebar",
                  style={'font-family': 'Verdana', 'color':'#f8f9fa'}),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Simulador", href="/page-1", id="page-1-link",),
                dbc.DropdownMenuItem("Análisis Productos", href="/page-2", id="page-2-link",),
            ],
            nav=True,
            in_navbar=True,
            label="Contenido",
        )

       ],
    brand="BBVA",
    brand_style={'font-family': 'Verdana', 'color':'white','font-size': '40px'},
    color="#004481",
    dark=True,
    fluid=True)


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 95,
    "left": 0,
    "bottom": 0,
    "width": "14rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": '#f8f9fa',#"#f8f9fa",#c9daf8ff
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 95,
    "left": "-14rem",
    "bottom": 0,
    "width": "14rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "15rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "1rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
    "background-color": "#f8f9fa",
}

pronos_checkbox = dcc.Checklist(
                                options=[
                                            {'label': 'Base', 'value': 'Base'},
                                            {'label': 'Adverso', 'value': 'Adverso'}
                                        ],
                                value=[],
                                id = 'cbx1'
                                )  

edit_vars = dcc.Input(
    id = 'edit_vars',
    type = 'number',
    placeholder='Ingresa un valor',
    value=0,
    step=0.001,
    style={'float': 'left','color':'#004481','OutlineColor': '#004481'},
    className="mr-2",
)

sidebar = html.Div(
    [
        html.H2("Simulador", style={'textAlign': 'center', 'width': '30px','font-family': 'Verdana','color':'#004481'}),
        html.Hr(),
        html.H4("Integrantes"),
        html.Span("Edgar López Gónzalez"),
        html.Br(),
        html.Span("Aline Perez López")
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div([html.Div([
                            html.H1('Comportamiento histórico de la actividad bancaria'),
                            html.Div([    
                                        html.P('Elige un producto:', style={'display': 'inline-block', 'marginRight': '10px'}),
                                        dcc.Dropdown(
                                        id='demo-dropdown',
                                        options=[{'label': x, 'value': x} for x in get_product_names()],
                                        placeholder = 'Elige un producto',
                                        value='hipotecaria',
                                        style={'textAlign': 'center','font-family': 'Verdana','color':'#004481'}
                                        #multi =True
                                        ),
                                     ]),
                             
                             html.Div([
                                        dcc.Graph(id='x-time-series')      
                                     ]),

                            ]),
                    html.H3('Datos utilizados para pronósticar'),
                    html.Div([
                                    dash_table.DataTable(
                                    id='table-sims',
                                    fixed_rows = {'headers':True, 'data':0},
                                    column_selectable="multi",
                                    merge_duplicate_headers=True,
                                    page_action='native',
                                    page_current=0,
                                    page_size=10,
                                    style_table={'height': '300px','overflowY': 'auto','overflowX': 'auto','font-family':
                                                 'Verdana','color':'#004481'},
                                    style_cell={'minWidth': 95, 'width': 150, 'maxWidth': 200, 'textAlign': 'left'},
                                    style_header={
                                                        'backgroundColor': 'rgb(230, 230, 230)',
                                                        'font-family': 'Verdana','color':'white', 'backgroundColor': '#004481'
                                                    },
                                    )
                             ]),
                    html.Hr(),
                    html.H1('Pronósticos'),
                    html.Div([
                            html.Button('Generar pronósticos', id='button-forecast', style= {
                                                                                    'background-color': '#004481',  # Color azul de BBVA
                                                                                    'border': 'none',
                                                                                    'color': 'white',
                                                                                    'text-align': 'center',
                                                                                    'text-decoration': 'none',
                                                                                    'display': 'inline-block',
                                                                                    'font-size': '16px',
                                                                                    'padding': '10px 20px',
                                                                                    'margin': '4px 2px',
                                                                                    'cursor': 'pointer',
                                                                                    'border-radius': '12px'
                                                                                })]),
                    html.Div([
                                    dash_table.DataTable(
                                    id='table-forecast',
                                    fixed_rows = {'headers':True, 'data':0},
                                    merge_duplicate_headers=True,
                                    page_action='native',
                                    page_current=0,
                                    page_size=10,
                                    style_cell={
                                        'height': 'auto',
                                        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'
                                    },
                                    style_table={'height': '300px','overflowY': 'auto','font-family':
                                                 'Verdana','color':'#004481'},
                                    style_header={
                                                        'backgroundColor': 'rgb(600, 600, 600)',
                                                        'font-family': 'Verdana','color':'white', 'backgroundColor': '#004481'
                                                    },
                                )]),
                    html.Div([
                             html.Div([
                                        dcc.Graph(id='ve-time-series'),
                                      ]),

                             ]),
                    html.H1('Simulación de pronósticos'),
                    html.Div([
                            html.Button('Generara simulación', id='button-simulacion', style= {
                                                                                    'background-color': '#004481',  # Color azul de BBVA
                                                                                    'border': 'none',
                                                                                    'color': 'white',
                                                                                    'text-align': 'center',
                                                                                    'text-decoration': 'none',
                                                                                    'display': 'inline-block',
                                                                                    'font-size': '16px',
                                                                                    'padding': '10px 20px',
                                                                                    'margin': '4px 2px',
                                                                                    'cursor': 'pointer',
                                                                                    'border-radius': '12px'
                                                                                })]),
                    html.Div([
                                    dash_table.DataTable(
                                    id='table-simulacion',
                                    editable = True,
                                    fixed_rows = {'headers':True, 'data':0},
                                    merge_duplicate_headers=True,
                                    style_cell={
                                        'height': 'auto',
                                        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'
                                    },
                                    style_table={'height': '300px','overflowY': 'auto','font-family':
                                                 'Verdana','color':'#004481'},
                                    style_header={
                                                        'backgroundColor': 'rgb(600, 600, 600)',
                                                        'font-family': 'Verdana','color':'white', 'backgroundColor': '#004481'
                                                    },
                                )]),
                    html.Div([
                             html.Div([
                                        dcc.Graph(id='simulacion'),
                                      ]),

                             ])
                    ],
        id="page-content",
        style=CONTENT_STYLE)

app.layout = html.Div([

                    html.Div(
                        [
                            dcc.Store(id='side_click'),
                            dcc.Location(id="url"),
                            navbar,
                            sidebar,
                            content,
                        ],
                    ),

                     ])




@app.callback(
        [
            Output("sidebar", "style"),
            Output("page-content", "style"),
            Output("side_click", "data"),
        ],

        [Input("btn_sidebar", "n_clicks")],
        [
            State("side_click", "data"),
        ]
    )
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

def create_time_series(dff):
        try:
            #logging.debug(dff.columns)
            #logging.debug(dff)
            print()
            fig = px.line(dff, x=dff.index, y=dff.columns)
            fig.update_layout(paper_bgcolor='#f8f9fa', font_color = '#004481')
            fig.update_xaxes(color='#004481')
            fig.update_yaxes(color='#004481',title='Montos en Miles de Millones')
        except Exception as e:
            logging.debug(f"Eror en la grafica {e}")
        return fig

def create_time_series_col(dff, colname):
        try:
            #logging.debug(dff.columns)
            #logging.debug(dff)
            print()
            fig = px.line(dff, x=dff.index, y=dff[colname])
            fig.update_layout(paper_bgcolor='#f8f9fa', font_color = '#004481')
            fig.update_xaxes(color='#004481')
            fig.update_yaxes(color='#004481',title='Montos en Miles de Millones')
        except Exception as e:
            logging.debug(f"Eror en la grafica {e}")
        return fig

# Define el callback para actualizar el gráfico
@app.callback(
    Output('x-time-series', 'figure'),
    Output('table-sims', 'data'),
    Output('table-sims', 'columns'),
    [Input('demo-dropdown', 'value')]
)
def update_graph(selected_column):
    if selected_column:
        # Cargar datos usando la función load_data
        try:
            data = get_product_historical_data(selected_column)
            # Generar la gráfica usando la función create_time_series
            fig = create_time_series(data)

            data_exog = get_exog_historical_data(selected_column).reset_index()
            merged_df = pd.merge(data_exog, data, left_on='fecha', right_on='fechas', how='inner')
            data_index = merged_df.sort_index()

            table_columns = [{'name': col, 'id': col} for col in data_index.columns]
        except Exception as e:
            logging.debug(e)
        return fig, data_index.to_dict('records'), table_columns
    else:
        # En caso de que no se haya seleccionado un producto
        return go.Figure(), [], []



@app.callback(
    Output('ve-time-series', 'figure'),
    Output('table-forecast', 'data'),
    Output('table-forecast', 'columns'),
    [Input('button-forecast', 'n_clicks')],
    [State('table-sims', 'data'), State('demo-dropdown', 'value')]
)
def update_forecast(n_clicks, table_data, column_selct):
    if n_clicks is not None and n_clicks > 0:
        try:
            df = get_forecast(column_selct)
            
            fig = create_time_series_col(df, column_selct)
            table_columns = [{'name': col, 'id': col} for col in df.columns]

            return fig, df.to_dict('records'), table_columns
        except Exception as e:
            logging.debug(e)
    else:    
        return go.Figure(),[],[]

@app.callback(
    Output('simulacion', 'figure'),
    Output('table-simulacion', 'data'),
    Output('table-simulacion', 'columns'),
    [Input('button-simulacion', 'n_clicks')],
    [State('table-sims', 'data'), State('table-simulacion', 'data'), State('demo-dropdown', 'value')]
)
def update_forecast(n_clicks, table_data, table_simulado,column_selct):
    if n_clicks is not None and n_clicks > 0:
        logging.debug('Hola')
        logging.debug(table_simulado)
        try:
            if table_simulado != []:
                logging.debug("Ejecuta la función")
                #logging.debug(table_simulado)
                table_simulado = [{k: v for k, v in d.items() if k!= column_selct} for d in table_simulado]
                logging.debug("Como se manda la información")
                logging.debug(table_simulado)
                df = get_forecast_simulacion(table_simulado, column_selct)
                logging.debug(df)
            else:
                df = get_forecast(column_selct)

            fig = create_time_series_col(df, column_selct)
            table_columns = [{'name': col, 'id': col} for col in df.columns]

            return fig, df.to_dict('records'), table_columns
        except Exception as e:
            logging.debug(e)
    else:    
        return go.Figure(),[],[]



if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050,debug=True)