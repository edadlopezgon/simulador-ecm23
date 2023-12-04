CREATE TABLE db_simulations (
    id_sim int,
    another VARCHAR(40),
    producto VARCHAR(255),
    variables VARCHAR(255),
    values_variables NUMERIC
);

INSERT INTO db_simulations (id_sim, another, producto, variables, values_variables) VALUES (1,'HOLA','Producto A', 'Variable A', 10);