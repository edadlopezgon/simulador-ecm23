CREATE TABLE db_simulations (
    id_sim int,
    another VARCHAR(40),
    producto VARCHAR(255),
    variables VARCHAR(255),
    values_variables NUMERIC
);

CREATE TABLE db_models_features (
    model_name VARCHAR(40),
    variables_model VARCHAR(255)
);

CREATE TABLE db_historical_exogen_variables (
    fecha DATE NOT NULL,
    pib DECIMAL,
    tasa_desempleo DECIMAL,
    inpc DECIMAL,
    tipo_cambio_usd DECIMAL,
    cetes_6m DECIMAL,
    cetes_12m DECIMAL,
    bonos_m_10a DECIMAL,
    empleo_imss DECIMAL,
    salario DECIMAL
);

CREATE TABLE db_forecast_exogen_variables (
    fecha_pronos DATE NOT NULL,
    pib DECIMAL,
    tasa_desempleo DECIMAL,
    inpc DECIMAL,
    tipo_cambio_usd DECIMAL,
    cetes_6m DECIMAL,
    cetes_12m DECIMAL,
    bonos_m_10a DECIMAL,
    empleo_imss DECIMAL,
    salario DECIMAL
);

CREATE TABLE db_historical_data (
    fechas DATE,
    vista_red_mn NUMERIC,
    vista_patrimonial_mn NUMERIC,
    vista_minorista_mn NUMERIC,
    vista_empresas_mn NUMERIC,
    vista_gobierno_mn NUMERIC,
    vista_empresas_gobierno_mn NUMERIC,
    vista_corporativa_mn NUMERIC,
    plazo_red_mn NUMERIC,
    plazo_patrimonial_mn NUMERIC,
    plazo_minorista_mn NUMERIC,
    plazo_empresas_mn NUMERIC,
    plazo_gobierno_mn NUMERIC,
    plazo_empresas_gobierno_mn NUMERIC,
    plazo_corporativa_mn NUMERIC,
    plazo_mayorista_mn NUMERIC,
    vista_red_me NUMERIC,
    vista_patrimonial_me NUMERIC,
    vista_minorista_me NUMERIC,
    vista_empresas_me NUMERIC,
    vista_gobierno_me NUMERIC,
    vista_empresas_gobierno_me NUMERIC,
    vista_corporativa_me NUMERIC,
    vista_mayorista_me NUMERIC,
    plazo_red_me NUMERIC,
    plazo_patrimonial_me NUMERIC,
    plazo_minorista_me NUMERIC,
    plazo_empresas_me NUMERIC,
    plazo_gobierno_me NUMERIC,
    plazo_empresas_gobierno_me NUMERIC,
    plazo_corporativa_me NUMERIC,
    plazo_mayorista_me NUMERIC,
    ahorro_red_mn NUMERIC,
    ahorro_patrimonial_mn NUMERIC,
    ahorro_minorista_mn NUMERIC,
    ahorro_empresas_mn NUMERIC,
    ahorro_gobierno_mn NUMERIC,
    ahorro_empresas_gobierno_mn NUMERIC,
    ahorro_corporativa_mn NUMERIC,
    empresas_mn NUMERIC,
    gobierno_mn NUMERIC,
    corporativa_mn NUMERIC,
    promotor NUMERIC,
    empresas_me NUMERIC,
    gobierno_me NUMERIC,
    corporativa_me NUMERIC,
    gobierno_total NUMERIC,
    pyme NUMERIC,
    tdc NUMERIC,
    nomina NUMERIC,
    ppi NUMERIC,
    auto NUMERIC,
    hipotecaria NUMERIC
);
COPY db_historical_data
  FROM '/data/historical_data.csv'
  DELIMITER ','
  CSV HEADER
  NULL as 'NA';

COPY db_historical_exogen_variables
  FROM '/data/exog_historical_data.csv'
  DELIMITER ','
  CSV HEADER
  NULL as 'NA';

COPY db_forecast_exogen_variables
  FROM '/data/exog_forecast_data.csv'
  DELIMITER ','
  CSV HEADER
  NULL as 'NA';

INSERT INTO db_simulations (id_sim, another, producto, variables, values_variables) VALUES (1,'HOLA','Producto A', 'Variable A', 10);
INSERT INTO db_models_features (model_name, variables_model) VALUES ('hipotecaria','empleo_imss');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('hipotecaria','tasa_desempleo');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('hipotecaria','bonos_m_10a');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('hipotecaria','inpc');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('empresas_mn','empleo_imss');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('empresas_mn','cetes_12m');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('empresas_mn','tasa_desempleo');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('empresas_mn','salario');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('empresas_mn','inpc');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('vista_red_mn','empleo_imss');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('vista_red_mn','tasa_desempleo');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('vista_red_mn','cetes_6m');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('vista_red_mn','inpc');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('ahorro_red_mn','empleo_imss');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('ahorro_red_mn','tasa_desempleo');
INSERT INTO db_models_features (model_name, variables_model) VALUES ('ahorro_red_mn','inpc');