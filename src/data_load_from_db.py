from common.make_ora_conn_pool import make_ora_pool
from common.make_pg_pool import make_pgbb_pool
from common.execute_query import execute_query

import pandas as pd
from tqdm import tqdm
import time
import pickle
import os
import csv
def load_data():


    if os.path.exists('data/graduate.pkl'):
        data = pd.read_csv('data/graduateeda.csv'
                    )
        file = open('data/graduate.pkl', 'rb')
        graduate = pickle.load(file)
        file.close()
        return graduate

    pool_kuis = make_ora_pool()
    conn_kuis = pool_kuis.acquire()


    # tuition = open('../sql/ora/tuition.sql', 'r', encoding='utf-8').read()
    # tui_df = execute_query(conn_kuis, tuition, 'N')
    # tui_df.to_pickle('data/tui_df.pkl')


    s_surv = open('../sql/ora/s_surv.sql', 'r', encoding = 'utf-8').read()
    ku_std1 = execute_query(conn_kuis, s_surv, 'N')
    s_drop = open('../sql/ora/s_drop.sql', 'r', encoding='utf-8').read()
    ku_std2= execute_query(conn_kuis, s_drop, 'N')
    s_break = open('../sql/ora/s_break.sql', 'r', encoding='utf-8').read()
    ku_std3 = execute_query(conn_kuis, s_break, 'N')
    s_study= open('../sql/ora/s_study.sql', 'r', encoding='utf-8').read()
    ku_std4 = execute_query(conn_kuis, s_study, 'N')
    s_tuition = open('../sql/ora/s_tuition.sql', 'r', encoding='utf-8').read()
    ku_std5 = execute_query(conn_kuis, s_tuition, 'N')
    s_from = open('../sql/ora/s_from.sql', 'r', encoding='utf-8').read()
    ku_std6 = execute_query(conn_kuis, s_from, 'N')
    s_gpa = open('../sql/ora/s_gpa.sql', 'r', encoding='utf-8').read()
    ku_std7 = execute_query(conn_kuis, s_gpa, 'N')

    df = ku_std1.merge(ku_std2, left_on='rec014_std_id', right_on = 'rec012_std_id', how = 'left') #outer, campus 2 제거
    df = df.merge(ku_std3,  left_on='rec014_std_id', right_on = 'std_id', how = 'left')
    df = df.merge(ku_std4,  left_on='rec014_std_id', right_on = '학번', how = 'left')
    df = df.merge(ku_std5,  left_on='rec014_std_id', right_on = 'sch214_std_id', how = 'left')
    df = df.merge(ku_std6,  left_on='rec014_std_id', right_on = 'rec511_std_id', how = 'left')
    df = df.merge(ku_std7,  left_on='rec014_std_id', right_on = 'rec012_std_id', how = 'left')

    for i in range(len(df)):
        if pd.notna(df['period'][i]):
            df['기간'][i] = df['period'][i]

    #이름 바꾸고, column 선택 하기
    surv_ku = df[['rec014_std_id', 'rec014_ent_year', 'rec014_ent_term','기간','학과', '과정', '학적', 'chg_nm', '휴학횟수', '휴학기간', '인건비횟수' ,'인건비합', '인건비과제수', '등록금장학', 'etc_장학', '성적', '입학성적', '학부출신', '석사출신']]
    surv_ku.to_pickle('data/surv_ku.pkl')

    file = open('data/surv_ku.pkl', 'rb')
    surv_ku = pickle.load(file)
    file.close()

    start_dt = [
                '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01',
                  '2019-09-01', '2019-10-01', '2019-11-01','2019-12-01',
                '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01',
                  '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01',
                '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01'  ]

    end_dt = ['2019-04-01', '2019-05-01', '2019-06-01','2019-06-27',
                '2019-10-01', '2019-11-01', '2019-12-01','2019-12-22',
                '2020-04-01', '2020-05-01', '2020-06-01','2020-06-27 00:00:00',
                '2020-10-01', '2020-11-01', '2020-12-01','2020-12-22',
                 '2021-04-01', '2021-05-01', '2021-06-01','2021-06-22']

    pool_bb = make_pgbb_pool()
    conn_bb = pool_bb.getconn()


    # start = time.time()
    # bb_sql = open('../sql/pg/login_attempt.sql', 'r', encoding='utf-8').read()
    log_info = pd.DataFrame()
    # for x in tqdm(range(len(start_dt))):
    #     log = bb_sql.format(
    #         start_dt[x], end_dt[x])
    #     log_attempt = execute_query(conn_bb, log, 'N')
    #     log_attempt.to_pickle('data/log{}.pkl'.format(start_dt[x]))
    #     #log_info = log_info.append(log_attempt)
    #     print(f'Time:{time.time() - start}')
    # print(f'Time:{time.time()-start}')

    for x in tqdm(range(len(start_dt))):
        file = open('data/log{}.pkl'.format(start_dt[x]), 'rb')
        log_attempt = pickle.load(file)
        file.close()
        log_info = log_info.append(log_attempt)
    log_info['date'] = pd.to_datetime(log_info['date'], format = '%Y-%m-%d')
    log_info['year_month']=log_info['date'].dt.strftime('%Y-%m')

    user = open('../sql/pg/users.sql', 'r', encoding='utf-8').read()
    user_log = execute_query(conn_bb, user, 'N')

    df2 = user_log.merge(log_info, left_on='pk1', right_on='user_pk1', how='inner')

    s_surv_ori = open('../sql/ora/test.sql', 'r', encoding = 'utf-8').read()
    ku_std6 = execute_query(conn_kuis, s_surv_ori, 'N')

    df3 = df2.merge(ku_std6, left_on='user_id', right_on='rec014_std_id', how='inner')

    #EDA
    df_eda = df3.groupby(["rec014_std_id","year_month"])["count"].sum().to_frame().reset_index()

    df4 = df3.groupby("rec014_std_id")["count"].sum().to_frame().reset_index()

    df_F = surv_ku.merge(df4,  on='rec014_std_id', how = 'outer')
    df_Feda = surv_ku.merge(df_eda,  on='rec014_std_id', how = 'outer')


    df_F.to_pickle('data/graduate.pkl')
    df_Feda.to_csv('data/graduateeda.csv')
    return df_F
