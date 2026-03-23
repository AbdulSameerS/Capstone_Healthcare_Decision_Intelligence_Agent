import duckdb
import os

def get_db_connection():
    DATA_PATH = "/Users/sameer/Documents/DataScience_Capstone_Project/Capstone_Healthcare_Decision_Intelligence_Agent/dataset/"
    DB_PATH = os.path.join(DATA_PATH, "hf_project.duckdb")
    con = duckdb.connect(DB_PATH)
    print("Connected to DuckDB at:", DB_PATH)
    return con
