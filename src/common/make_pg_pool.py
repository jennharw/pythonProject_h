import json
import psycopg2
from psycopg2 import pool

def make_pgbb_pool():
    config_path='../config/BB.json'
    with open(config_path, 'r') as f:
        db_config = json.load(f)

    pg_pool = psycopg2.pool.SimpleConnectionPool(1,5, user= db_config["user"],
                                                 password=db_config["password"],
                                                 host=db_config["addr"],
                                                 port=db_config["port"],
                                                 database=db_config["SID"])
    return pg_pool



