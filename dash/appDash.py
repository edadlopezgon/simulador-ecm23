import pandas as pd
import numpy as np
import math
import datetime as dt

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly

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
                                        value='vista_minorista_mn',
                                        style={'textAlign': 'center','font-family': 'Verdana','color':'#004481'}
                                        #multi =True
                                        ),
                                     ]),
                             
                             html.Div([
                                        dcc.Graph(id='x-time-series')      
                                     ]),

                            ]),
                    html.Div([
                                    dash_table.DataTable(
                                    id='table-sims',
                                    fixed_rows = {'headers':True, 'data':0},
                                    column_selectable="multi",
                                    merge_duplicate_headers=True,
                                    page_action='native',
                                    page_current=0,
                                    page_size=10,
                                    fixed_columns={'headers': True, 'data': 1 },
                                    style_table={'height': '300px','width':'500px','overflowY': 'auto','font-family':
                                                 'Verdana','color':'#004481'},
                                    style_header={
                                                        'backgroundColor': 'rgb(230, 230, 230)',
                                                        'font-family': 'Verdana','color':'white', 'backgroundColor': '#004481'
                                                    },
                                    )
                             ]),
                    html.Hr(),
                    html.Div([
                                    dash_table.DataTable(
                                    id='table',
                                    columns=[{"name": i, "id": i} for i in ls],
                                    editable = True,
                                    fixed_rows = {'headers':True, 'data':0},
                                    export_format='xlsx',
                                    export_headers='display',
                                    merge_duplicate_headers=True,
                                    fixed_columns={'headers': True, 'data': 1 },
                                    style_cell={
                                        'height': 'auto',
                                        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'
                                    },
                                    style_table={'height': '300px','width':'3000px','overflowY': 'auto','font-family':
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
            data_index = data.sort_index(ascending=False).reset_index()
            table_columns = [{'name': col, 'id': col} for col in data_index.columns]
        except Exception as e:
            logging.debug(e)
        return fig, data_index.to_dict('records'), table_columns
    else:
        # En caso de que no se haya seleccionado un producto
        return go.Figure(), [], []



if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050,debug=True)