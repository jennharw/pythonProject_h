import cx_Oracle as co
import os
import json

def make_ora_pool():
    config_path = '../config/KUISDB.json'
    with open(config_path, 'r') as f:
        db_config = json.load(f)

    os.environ["NLS_LANG"] = ".AL32UTF8"

    dsn_tns = co.makedsn(db_config["addr"], db_config["port"], db_config["SID"])
    pool = co.SessionPool(user = db_config["user"], password = db_config["password"], dsn = dsn_tns,
                          min=1, max=5, increment=1, threaded=True, getmode = co.SPOOL_ATTRVAL_WAIT)

    return pool