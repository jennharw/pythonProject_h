import pandas as pd

def execute_query(conn, query, initial = 'Y'):
    query_result = pd.read_sql(query, conn)
    if initial == 'N':
        query_result.columns=query_result.columns.str.lower()
    return query_result
