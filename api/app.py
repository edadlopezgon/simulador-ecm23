from flask import Flask
from sqlalchemy import create_engine
import logging
import json

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
    querie_data = 'SELECT fechas,vista_red_mn  FROM db_historical_data ;'
    logging.debug("Iniciando obtener datos")
    try:
        logging.debug(execute_queries([querie_data]))
        data_profiles = json.loads(execute_queries([querie_data],"SELECT"))
        
        return json.dumps(data_profiles)
    except Exception as ex:
        logging.debug(ex)
        return {"message":"error en consulta de datos en servicio faker-service"}
    
@app.route('/ejemplo')
def get_ejemplo():
    try:
        aux = [{"fechas": "2007-01-01", "vista_red_mn": "68562.32798"}, 
               {"fechas": "2007-02-01", "vista_red_mn": "68639.77640"}]
        
        return json.dumps(aux)
    except Exception as ex:
        logging.debug(ex)
        return {"message":"error en consulta de datos en servicio faker-service"}




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)