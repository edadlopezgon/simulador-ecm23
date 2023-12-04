from flask import Flask, request, jsonify
from sqlalchemy import create_engine
import logging
import json
import pickle
import pandas as pd

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logging.debug("Servicio iniciado")

def execute_queries(list_queries_string=[], querie_type=''):
    db_name = 'rainbow_database'
    db_user = 'unicorn_user'
    db_pass = 'magical_password'
    db_host = 'database-service' # este es el servicio database declarado en el docker-compose
    db_port = '5432'
    try:
        db_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_user, db_pass, db_host, db_port, db_name)
        db = create_engine(db_string)
        if querie_type == 'SELECT':
            for querie in list_queries_string:
                data_profiles = db.execute(querie).fetchall()
                logging.debug("Este es el resultado de la consulta")
            return json.dumps([dict(ix) for ix in data_profiles], default=str) #CREATE JSON
    except Exception as e:
	    logging.debug(f"error en conexion con la base de datos {e}")


     

@app.route('/simular_serie')
def sim_serie():
    try:
        return {"message": "hola"}
    except: 
        return{"message": "error"}

@app.route('/obtener_datos')
def get_data():
    querie_data = 'SELECT *  FROM db_historical_data ;'
    logging.debug("Iniciando obtener datos")
    try:
        logging.debug(execute_queries([querie_data]))
        data_profiles = json.loads(execute_queries([querie_data],"SELECT"))
        
        return json.dumps(data_profiles)
    except Exception as ex:
        logging.debug(ex)
        return {"message":"error en consulta de datos en servicio faker-service"}
    
@app.route('/get_products_names')
def get_products_names():
    querie_data = 'SELECT column_name FROM information_schema.columns WHERE table_name = \'db_historical_data\'  AND column_name in (\'hipotecaria\',\'vista_red_mn\',\'ahorro_red_mn\');'
    try:
        logging.debug(execute_queries([querie_data]))
        data_profiles = json.loads(execute_queries([querie_data],"SELECT"))
        
        return json.dumps(data_profiles)
    except Exception as ex:
        logging.debug(ex)
        return {"message":"error en consulta de datos en servicio faker-service"}
    
@app.route('/get_product_historical_data', methods=['POST'])
def receive_data():
    if request.method == 'POST':
        data = request.get_json()
        selected_column = data['column']
        querie_data = f'SELECT fechas, {selected_column}  FROM db_historical_data ;'
        try:
            data_profiles = json.loads(execute_queries([querie_data],"SELECT"))            
        except Exception as ex:
            logging.debug(ex)
            return {"message":"error en consulta de datos en servicio faker-service"}
        
        return json.dumps(data_profiles)
    
@app.route('/get_historical_training', methods=['POST'])
def receive_data_training():
    if request.method == 'POST':

        data = request.get_json()
        selected_column = data['column']
        querie_data = f'SELECT  variables_model  FROM db_models_features WHERE model_name= \'{selected_column}\' ;'
        try:
            data_profiles = json.loads(execute_queries([querie_data],"SELECT"))   
            columns_names = ','.join([x['variables_model'] for x in data_profiles ]) 
            logging.debug(columns_names)        
        except Exception as ex:
            logging.debug(ex)
            return {"message":"error en consulta de datos en servicio faker-service"}
        
        querie_data = f'SELECT  fecha, {columns_names} FROM db_historical_exogen_variables;'
        try:
            data_profiles = json.loads(execute_queries([querie_data],"SELECT"))   
            #logging.debug(data_profiles)        
        except Exception as ex:
            logging.debug(ex)
            return {"message":"error en consulta de datos en servicio faker-service"}
        
        return json.dumps(data_profiles)

@app.route('/get_forecast', methods=['POST'])
def generate_forecast_of():
    if request.method == 'POST':
        try:
            data = request.get_json()
            selected_column = data['model_name']
            querie_data = f'SELECT  variables_model  FROM db_models_features WHERE model_name= \'{selected_column}\' ;'
            try:
                data_profiles = json.loads(execute_queries([querie_data],"SELECT"))   
                columns_names = ','.join([x['variables_model'] for x in data_profiles ]) 
                logging.debug(columns_names)        
            except Exception as ex:
                logging.debug(ex)
                return {"message":"error en consulta de datos en servicio faker-service"}
            
            querie_data = f'SELECT  fecha_pronos, {columns_names} FROM db_forecast_exogen_variables;'
            try:
                data_profiles = json.loads(execute_queries([querie_data],"SELECT"))   
                logging.debug(data_profiles)        
            except Exception as ex:
                logging.debug(ex)
                return {"message":"error en consulta de datos en servicio faker-service"}

            try:
                with open(f'models/{selected_column}.pkl', 'rb') as f:
                    loaded_model = pickle.load(f)
                    logging.debug(type(loaded_model)) 
                    forecast = generate_forecast(data_profiles, loaded_model)
                    data = [{"variables":data_profiles, "forecast": forecast}]
            except Exception as ex:
                logging.debug(ex)
                return {"message":"error en consulta de datos en servicio faker-service"}
            


        except Exception as e:
            logging.debug(e)
    
    return json.dumps(data)

@app.route('/get_forecast_simulacion', methods=['POST'])
def generate_forecast_simulate():
    if request.method == 'POST':
        try:
            data = request.get_json()
            data = json.loads(data)[0]
            model_name = data['model_name']
            data = data['variables']
            
            try:
                with open(f'models/{model_name}.pkl', 'rb') as f:
                    loaded_model = pickle.load(f)
                    logging.debug(type(loaded_model)) 
                    generate_forecast(data, loaded_model)
            except Exception as ex:
                logging.debug(ex)
                return {"message":"error en consulta de datos en servicio faker-service"}


        except Exception as e:
            logging.debug(e)
    
    return {"message":"Recibido"}
         

def generate_forecast(data, model):
    data_pd = pd.DataFrame(data)
    if 'fecha_pronos' in data_pd.columns:
        data_pd.drop('fecha_pronos', axis=1, inplace=True) 
    forecast = model.predict(data_pd)
    logging.debug(forecast)
    forecast = [str(x) for x in forecast]
    return forecast


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)