CREATE DATABASE budget_data;

\connect budget_data

CREATE TABLE real_budget (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    category TEXT,
    amount REAL
);




