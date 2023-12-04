#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import math
import datetime as dt
import pickle
import sklearn
import statsmodels

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

import warnings
warnings.filterwarnings("ignore")

# In[2]:


np.__version__


# In[3]:


import glob
paths_dictionaries = glob.glob('C:/Users/lizet/OneDrive/Documents/Estadistica4/ProyectoFinal/simulador-ecm23/dash/Insumos/Insumos/Diccionarios_Covariables/*')
paths_models = glob.glob('C:/Users/lizet/OneDrive/Documents/Estadistica4/ProyectoFinal/simulador-ecm23/dash/Insumos/Insumos/Models/*')
paths_pronos_base = glob.glob('C:/Users/lizet/OneDrive/Documents/Estadistica4/ProyectoFinal/simulador-ecm23/dash/Insumos/Insumos/Pronos_Base/*.txt')


# In[4]:


# import glob
# paths_dictionaries = glob.glob('Insumos/Diccionarios_Covariables/*')
# paths_models = glob.glob('Insumos/Models/*')
# paths_pronos_base = glob.glob('Insumos/Pronos_Base/*.txt')


# In[5]:


objects = []
for path in paths_dictionaries:
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
D = objects
D2 = {}
for elem in D: 
    new_elem = list(elem[0].keys())[0]
    s = new_elem.split('_')
    elem[0][new_elem]['transf'] = s[0]
    D2['_'.join(s[1:-1])] = elem[0][new_elem]

D2


# In[6]:


productos = {}
for path in paths_pronos_base:
    name = '_'.join(path.split('\\')[1].split('_')[:-1])
    path_model = 'C:/Users/lizet/OneDrive/Documents/Estadistica4/ProyectoFinal/simulador-ecm23/dash/Insumos/Insumos/Models/{}.sav'.format(name)
    with (open(path_model, "rb")) as openfile:
        temp_model = pickle.load(openfile)
    productos[name] = {'pronos_base': pd.read_csv(path, delimiter = '\t'),
                       'feat_dict': D2[name],
                       'models': temp_model}


# In[7]:


data = pd.read_csv('C:/Users/lizet/OneDrive/Documents/Estadistica4/ProyectoFinal/simulador-ecm23/dash/Insumos/Insumos/variables_exog.txt', delimiter = "\t")
data['Fechas'] = pd.to_datetime(data['Fechas'],format='%d/%m/%Y')
data.set_index('Fechas', inplace=True)
# for c in data.columns:
#     data[c] = data[c].apply(lambda x: float("{:.2f}".format(x)))
data = data.drop(['Auto Mercado','Originaciones Nominales',
       'Originaciones Personales', 'Originaciones TDC', 'Originaciones PyME',
       'Originaciones Auto', 'Originaciones Hipotecario'],axis=1)
data['Nominales Mercado']=data['Nominales Mercado'].fillna(0)


# In[10]:


cartera = pd.read_excel("C:/Users/lizet/OneDrive/Documents/Estadistica4/ProyectoFinal/simulador-ecm23/dash/Insumos/Insumos/carteraP_2021.xlsx")
cartera['Fechas'] = pd.to_datetime(cartera['Fechas'],format='%d/%m/%Y')
cartera.set_index('Fechas', inplace=True)
cartera.columns = [' '.join(c.split('_')) for c in cartera.columns]

captacion = pd.read_excel("C:/Users/lizet/OneDrive/Documents/Estadistica4/ProyectoFinal/simulador-ecm23/dash/Insumos/Insumos/CaptacionP_2021.xlsx")
captacion['Fechas'] = pd.to_datetime(captacion['Fechas'],format='%d/%m/%Y')
captacion.set_index('Fechas', inplace=True)
captacion.columns = [' '.join(c.split('_')) for c in captacion.columns]


# In[11]:


otras_variables = pd.read_csv('C:/Users/lizet/OneDrive/Documents/Estadistica4/ProyectoFinal/simulador-ecm23/dash/Insumos/Insumos/otras_variables.txt', delimiter = "\t")
otras_variables['Fechas'] = pd.to_datetime(otras_variables['Fechas'],format='%d/%m/%Y')
otras_variables['Fechas'] = pd.to_datetime(otras_variables['Fechas']).dt.date
otras_variables.set_index('Fechas', inplace=True)
otras_variables


# In[12]:


seleccion = '_'.join('Empresas MN'.split())
seleccion = 'Empresas MN'


# In[13]:


historia = cartera[seleccion].to_frame()
temp_f = productos['_'.join(seleccion.split(' '))]['pronos_base'].copy()
temp_f.columns = ['Fechas', 'lower '+seleccion, 'Forecast '+seleccion, 'upper '+seleccion]
f = temp_f.set_index('Fechas')
R = historia.join(f, how = 'outer')
R = R.reset_index()
R['Fechas'] = pd.to_datetime(R['Fechas']).dt.date



# In[14]:


# a = catalogo_dict[seleccion]
ls = list(productos['_'.join(seleccion.split(' '))]['feat_dict'].keys())
ls = [key for key in ls if key != 'transf']
ls.insert(0, "Fechas")


# In[15]:


data = data.reset_index().dropna()


# In[16]:


feat_list = productos['_'.join(seleccion.split(' '))]['feat_dict']
[k for k in feat_list.keys() if k != 'transf']


# In[24]:


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

R['Fechas'] = pd.to_datetime(R['Fechas']).dt.date
data['Fechas'] = pd.to_datetime(data['Fechas']).dt.date
efecto_inflacion = data.set_index('Fechas').INPC / 63.02
colors = {
    'background': "#023858",
    'text': '#7FDBFF'}

bbva_logo = "https://region6.bfp.gov.ph/wp-content/uploads/2020/10/Icon_47-512.png"

navbar = dbc.NavbarSimple(
                   
    children=[
        dbc.Button("Esconder Menú", outline=True, color="#2DCCCD", className="mr-2", id="btn_sidebar",
                  style={'font-family': 'Verdana', 'color':'#f8f9fa'}),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Simulador", href="/page-1", id="page-1-link",),
                dbc.DropdownMenuItem("Análisis Productos", href="/page-2", id="page-2-link",),
            ],
            nav=True,
            in_navbar=True,
            label="Contenido",
        ),
            html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=bbva_logo, height="40px"))
                ],
                align="center",
            ))

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
        dbc.Nav(
            [
                html.Br(style={'color':'#004481'}),
                dcc.Dropdown(
                        id='select_vars',
                        placeholder = 'Elige una variable',
                        style={'textAlign': 'center','font-family': 'Verdana','color':'#004481'}
                      ),
                html.Hr(),
                edit_vars,
                html.Br(),
                html.Button('Editar Tabla',id='boton-editar_tabla',n_clicks=0,
                            style={'font-family': 'Verdana','color':'#f8f9fa','backgroundColor': '#004481'},
                           className="mr-2"
                          ),
                html.Br(),
                html.Button('Simular', id='boton-simular', n_clicks = 0,
                            style={'font-family': 'Verdana','color':'#f8f9fa','backgroundColor': '#004481'},
                           className="mr-2",
                        ),
                html.Br(),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div([html.Div([
                        html.Div([       
                                    dcc.Dropdown(
                                    id='demo-dropdown',
                                    options=[{'label': x, 'value': x} for x in [' '.join(p.split('_')) for p in productos.keys()]],
                                    placeholder = 'Elige un producto',
                                    value='Empresas MN',
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
                                columns=[{"name": i, 
                                          "id": i, 
                                          "selectable": True} for i in R.drop('Empresas MN', axis = 1).columns],
                                data=R.drop('Empresas MN', axis = 1).dropna().to_dict('records'),
                                fixed_rows = {'headers':True, 'data':0},
                                column_selectable="multi",
                                selected_columns=['Fechas'],
                                export_format='xlsx',
                                export_headers='display',
                                merge_duplicate_headers=True,
                                fixed_columns={'headers': True, 'data': 1 },
                                style_table={'height': '300px','width':'3000px','overflowY': 'auto','font-family': 'Verdana','color':'#004481'},
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
                                data=data[ls].to_dict('records'),
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
                                style_table={'height': '300px','width':'3000px','overflowY': 'auto','font-family': 'Verdana','color':'#004481'},
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

#####Funciones de Gráficos página 1

def create_time_series(dff):
    print('cts')
    fig = px.line(dff, x=dff.index, y=dff.columns)
    fig.update_layout(paper_bgcolor='#f8f9fa', font_color = '#004481')
    fig.update_xaxes(color='#004481')
    fig.update_yaxes(color='#004481',title='Montos en Miles de Millones')
    return fig

def update_vars(producto):
    print('update_vars')
    feat_list = productos['_'.join(producto.split(' '))]['feat_dict']
    feat_list = [k for k in feat_list.keys()]
    return feat_list

def filter_exog(V):
    print('filterexog')
    filtered_vars = V        
    if 'dummy' in filtered_vars:
        filtered_vars.remove('dummy')
    if 'transf' in filtered_vars:
        filtered_vars.remove('transf')
    if 'AR' in filtered_vars:
        filtered_vars.remove('AR')
    return filtered_vars

def plot_datamacro(dataset):
    n = len(dataset.columns)
    fig = make_subplots(rows=n, cols=1,subplot_titles=dataset.columns)
    
    for i in range(0,n):
        fig.append_trace(go.Scatter(x =dataset.index,y=dataset.iloc[:,i],name=dataset.iloc[:,i].name,showlegend=False),row = i+1,col=1)
    fig.update_layout(paper_bgcolor='#f8f9fa',title_text="Variables Macroeconómicas Seleccionadas",
                     title_font_color='#004481',font_color='#004481')#,plot_bgcolor = '#023858'
    fig.update_xaxes(color='#004481')
    fig.update_yaxes(color='#004481')
    return fig

def forecast_new_sim(df, temp_dict):
    df = df
    cols = df.columns
    dict_transf = {}
    for c in cols:
        transformations = temp_dict[c]  ### lista que incluye transfromacion y lag
        for elem in transformations:
            dict_transf['_'.join([c, elem[0], elem[1]])] = transform_col(df[c], elem)
    transformed_df = pd.DataFrame(dict_transf)
    return transformed_df

def transform_col(S, transformations):
    print('transf_col')
    S = S.astype(float)
    ddd = {'m': 1, 'a': 12}
    t1, t2 = transformations[0][:-1], transformations[0][-1]   ### tcm, t1 = 'tc', t2 = 'm'
    lag = int(transformations[1][-1])
    if t1 == 'tc':
        return S.pct_change(ddd[t2]).shift(lag)
    elif t1 == 'dif':
        return S.diff(ddd[t2]).shift(lag)
    elif t1 == 'log':
        return np.log(S).diff(ddd[t2]).shift(lag)
    return S.pct_change(ddd[t2]).shift(lag)


####################################### CALLBACKS

@app.callback(Output('x-time-series', 'figure'),
              [Input('table-sims', 'data'),
              Input('table-sims', 'columns'),
              Input('table-sims', 'selected_columns'),
              State('demo-dropdown', 'value')])

def update_simulations(rows, columns, selection, producto):
    print('us')
    if producto in cartera.columns:
        historia = cartera[producto]
    else:
        historia = captacion[producto]
    
    tc = data.set_index('Fechas')['Tipo de Cambio USD']
    if producto[-3:] == ' ME':
        historia = historia/tc
    
    historia.name = producto
    historia = historia.to_frame().dropna()
    
    cols = [c for c in selection if c == 'Fechas' or producto in c]

    f = pd.DataFrame(rows, columns=cols)
    f['Fechas'] = f['Fechas'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    f = f.set_index('Fechas')

    R2 = historia.join(f, how = 'outer')

    return create_time_series(R2)

@app.callback([Output('ve-time-series', 'figure'),
               Output('select_vars', 'options')],
              [Input('demo-dropdown', 'value'), 
               Input('table','data'),
               Input('table','columns')])

def update_tables(producto, ren, col):
    dd_vars = [{'label': i, 'value': i} for i in productos['_'.join(producto.split(' '))]['feat_dict'].keys() if i not in ['transf', 'dummy', 'AR']]
    exog_vars = pd.DataFrame(ren, columns=[c['name'] for c in col]).set_index('Fechas')
    return plot_datamacro(exog_vars), dd_vars

@app.callback([Output('table', 'data'),
               Output('table', 'columns')],
              [Input('boton-editar_tabla','n_clicks'),
               Input('demo-dropdown', 'value'),
               State('select_vars','value'),
               State('edit_vars','value'),
               State('table','data'),
               State('table','columns')])

def altere_exog_table(n_clicks, producto, variable, const, ren, col):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger == 'demo-dropdown':
        
        vars_tablero = filter_exog(update_vars(producto))
        temp_data = data.set_index('Fechas')[vars_tablero].loc[cartera.index[-12]:].reset_index()
        rows, columns = temp_data.to_dict('rows'), [{"name": i, "id": i} for i in temp_data.columns]
        return rows, columns
    elif trigger == 'boton-editar_tabla':
        temp_data = pd.DataFrame(ren, columns=[c['name'] for c in col])
        temp_data[variable]= temp_data[variable]*const
        rows, columns = temp_data.to_dict('rows'), [{"name": i, "id": i} for i in temp_data.columns] 
        return rows, columns
    else:
        raise dash.exceptions.PreventUpdate

@app.callback([Output('table-sims', 'data'),
               Output('table-sims', 'columns')],
              [Input('boton-simular', 'n_clicks'),
               Input('demo-dropdown', 'value'),
               State('table', 'data'), 
               State('table', 'columns'),
               State('table-sims', 'data'), 
               State('table-sims', 'columns')])

def altere_sims_table(n_clicks, producto, rows, columns, rows_sims, cols_sims):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if producto in cartera.columns:
        historia = cartera[producto]
    else:
        historia = captacion[producto]
         
    tc = data.set_index('Fechas')['Tipo de Cambio USD']
    if producto[-3:] == ' ME':
        historia = historia/tc
    
    historia.name = producto
    historia = historia.to_frame()
    
    print('historia')
    print(historia.tail())
    print('prev_f')
    f = productos['_'.join(producto.split(' '))]['pronos_base']
    f.columns = ['Fechas', 'lower '+producto, 'Forecast '+producto, 'upper '+producto]
    f = f.set_index('Fechas')
    print(f.head())
    
    R = historia.join(f, how = 'outer')
    R = R.reset_index()
    R['Fechas'] = pd.to_datetime(R['Fechas']).dt.date
    
    print(R.tail())
    sims_df = pd.DataFrame(rows_sims, columns=[c['name'] for c in cols_sims]).set_index('Fechas')
    filtered = [s for s in sims_df.columns if 'sim' in s]
    pronos_prod = producto
    temp_sims = R.set_index('Fechas')[[p for p in R.columns if producto in p]].loc[cartera.index[-1]:]
    temp_sims = temp_sims.reset_index()

    if trigger == 'demo-dropdown':
        rows_sims, cols_sims = temp_sims.drop(producto, axis = 1).dropna().to_dict('rows'), [{"name": i, 
                                                                                              "id": i,
                                                                                              "selectable": True} for i in temp_sims.drop(producto, axis = 1).columns]
        return rows_sims, cols_sims
    
    elif trigger == 'boton-simular':
        m = productos['_'.join(producto.split(' '))]['models']
        vars_list = productos['_'.join(producto.split(' '))]['feat_dict'].copy()
        
        temp_d = vars_list.copy()
        for k in vars_list.keys():
            if k in ['transf', 'dummy', 'AR']:
                del temp_d[k]
        
        vars_tablero = filter_exog(update_vars(producto))

        special_features = []
        AR = 0
        for c in vars_list.keys():
            if c == 'dummy':
                for feat in vars_list[c]:
                    if feat not in ['Crisis', 'renglon']:
                        special_features.append('Mes_'+feat)
                    else:
                        special_features.append(feat)
            if c == 'AR':
                AR = int(vars_list[c][0])
                print('AR', AR)
                
        temp_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        temp_df['Fechas'] = temp_df['Fechas'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
        temp_df = temp_df.set_index('Fechas')
        print('exog_retrieved')

        transformed_df = forecast_new_sim(temp_df, temp_d)  ### variables exog transformadas
                                   
        if len(special_features) > 0:
            transformed_df = transformed_df.join(otras_variables[special_features])
                    
        print('transf_df')
        print(transformed_df)
    
        if producto in cartera.columns:
            serie_defla = (historia[producto] / efecto_inflacion).dropna()
        else:
            serie_defla = (historia[producto] / efecto_inflacion).dropna()
            
        if producto in ['Auto', 'PPI', 'PyME']:#Ajustar temporalidades:
            sim = m.get_forecast(steps= 57,exog = transformed_df.loc[dt.date(2021,4,1):]).predicted_mean.values
        elif AR == 0:
            sim = m.predict(transformed_df.loc[cartera.index[-1]:sims_df.index[-1]].iloc[0:])
        else:
            last_values = serie_defla.iloc[-AR-1:]
            print('lv')
            print(last_values)
            t = vars_list['transf']
            if t == 'difm':
                sim = list(last_values.diff().values.flatten())#.flatten()
            elif t == 'tcm':
                sim = list(last_values.pct_change().values.flatten())#.flatten()
            
            print('sim', sim)
            print('last_value', sim[-AR])
            print(sim[-1])
            for idx in transformed_df.loc[cartera.index[-1]:sims_df.index[-1]].iloc[1:].index:
                X = transformed_df.loc[idx].copy()
                X['AR'] = sim[-AR]
                sim.append(m.predict(X.values.reshape(1,-1))[0])
        
        
        sim = sim[AR:]
        if producto in ['Auto', 'PPI', 'PyME']:
            print(pd.DataFrame({'FAR': sim}, index = transformed_df.loc[cartera.index[-1]:sims_df.index[-1]].iloc[1:].index))
        else:
            print(pd.DataFrame({'FAR': sim}, index = transformed_df.loc[cartera.index[-1]:sims_df.index[-1]].index))
        
        uv = serie_defla.tail(1).values[0]
        
        results = [uv]
        
        for i in range(0,len(sim)):
            t = vars_list['transf']
            if t == 'difm':
                results.append((results[-1]+sim[i]))
            elif t == 'tcm':
                results.append((results[-1]+results[-1]*sim[i]))
        
    
        if AR == 0 and producto not in ['Auto', 'PPI', 'PyME']:
            results = results[2:]
        elif producto in ['Auto', 'PPI', 'PyME']:
            results = results[1:]
        else:
            results = results[2:]
        print('what', len(sim), len(results), len(sims_df.index))
        print(sims_df.index)
        forecasts = pd.DataFrame({'sim_'+producto+'_'+str(n_clicks): np.array(results)*efecto_inflacion[-len(results):].values}, 
                                 index = sims_df.index)
        print('what1')
        joined = sims_df.join(forecasts, how = 'outer').reset_index()
        
        ren = joined.to_dict('rows')
        cols = [{"name": i, "id": i, 'selectable': True} for i in joined.columns] 

        return ren, cols
    else:
        raise dash.exceptions.PreventUpdate

# if __name__ == "__main__":
#     app.run_server(debug = True, use_reloader=False)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
