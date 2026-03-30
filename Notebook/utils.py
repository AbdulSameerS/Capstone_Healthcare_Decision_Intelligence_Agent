import duckdb
import os

def get_db_connection():
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    DB_PATH = os.path.join(DATA_PATH, "hf_project.duckdb")
    con = duckdb.connect(DB_PATH)
    print("Connected to DuckDB at:", DB_PATH)
    return con
