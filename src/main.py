"""
강의실 출입기록 , portal 자퇴 신청 로그

eda -> 요인,
Model :sksurv, pycox  ;train eval test
output -> 학과마다 위험학생정보 내리기, 그래프 + 장학금
"""

import pandas as pd
import numpy as np
import os
from data_load_from_db import load_data
from eda import eda
from randomSurv import ForestSurvival
from simplemodel import survivalSimple
from competingrisk import DeepHitCompetingRisk

from tuition import tuition_cluster

#time varying variable -> normalize 효과를 줄인다.

# 검증 데이터나 테스트 데이터가 아닌 학습데이터에서만 오버샘플링 사용할 것
#from imblearn.over_sampling import SMOTE
#smote = SMOTE(random_state=11)
#smote = SMOTE(random_state = 42, k_neighbors = 5)

def main(): #dept_cd = '경영학과'

    graduate = load_data()

    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 1000)

    #19, 20학번 TRAIN , 21학번 1학기  PREDICT
    graduate = graduate.drop(graduate[(graduate['rec014_ent_year'] == 2021) & (graduate['rec014_ent_term'] =='2R')].index)
    graduate = graduate[graduate['rec014_ent_year'] > '2018']

    graduate['event'] = graduate.apply(lambda x: 1 if (x['학적'] == '제적') else 0, axis=1)  # event 1 censor 0
   # graduate['event1'] = graduate.apply(lambda x: 1 if x['학적'] == '제적' else (0 if x['학적'] == '졸업' else 2), axis=1) #competing  risk

    graduate['자타'] = graduate.apply(lambda x: 1 if (x['학부출신'] == '고려대학교') else 0, axis=1)
    graduate = graduate.fillna(0)

    graduate = graduate[['기간', 'event', '자타', 'count', '인건비횟수', '인건비과제수', '인건비합', '등록금장학', 'etc_장학', '성적', '입학성적', '휴학횟수','휴학기간', '과정', '학과', 'rec014_std_id', 'rec014_ent_year', 'rec014_ent_term']]

    graduate['기간'] = graduate['기간'].astype('float64')
    graduate['event'] = graduate['event'].astype('int64')

    print(graduate['event'].max())
    #print(graduate.dtypes)

    # #OverSampling
    # graduate['학적2'] = graduate.apply(lambda x: 1 if x['학적'] == '제적' else (0 if x['학적'] == '졸업' else 2), axis=1)
    # graduate['과정2'] = graduate.apply(lambda x: 1 if x['과정'] == '박사과정' else (0 if x['과정'] == '석사과정' else 2), axis=1)
    # y = graduate['event']
    # X = graduate[['기간', '자타', 'count', '인건비합', '등록금장학','etc_장학', '성적','휴학기간', '학적2', '과정2']]
    # X_train_over, y_train_over = smote.fit_resample(X, y)
    # print(X_train_over.head())
    # print(y_train_over.head())
    #
    # graduate_sampled = pd.concat([y_train_over, X_train_over], axis = 1)
    # graduate_sampled = graduate_sampled.rename(columns={'과정2':'과정', "학적2":'학적'})
    # graduate_sampled['과정'] = graduate_sampled.apply(lambda x: '박사과정' if x['과정'] == 1 else ('석사과정' if x['과정'] == 0 else '석박사통합과정'), axis=1)
    # graduate_sampled['학적'] = graduate_sampled.apply(lambda x: '제적' if x['학적'] == 1 else ('졸업' if x['학적'] == 0 else '수료'), axis=1)

    #Explonatory Data Analysis
    #eda(graduate)
    #eda(graduate_sampled)

    #Random Survival Forest
    # print("--------------------------Random Survival Forest------------------------------")
    # survF = ForestSurvival(graduate)
    # #rsf = survF.randomforest()
    # RS_plt, RS_hazard_students = survF.predict_students_by_dept(rsf=True, dept_cd = '경영학과')
    # RS_plt.savefig('data/RSF_plt.png')
    #RS_hazard_students.to_csv('data/RSF_hazard_students.csv')

    #Deephit single Model
    print("--------------------------Deephit single Model------------------------------")
    simpleSurv = survivalSimple(graduate)
    #DHSmodel = simpleSurv.simplemodel()
    #DS_plt, DS_hazard_students = simpleSurv.predict_by_dept(dept_cd = '경영학과')


    #Competing risk
    print("--------------------------Deephit Competing risk------------------------------")
    graduate = load_data()
    graduate = graduate.drop(
        graduate[(graduate['rec014_ent_year'] == 2021) & (graduate['rec014_ent_term'] == '2R')].index)
    graduate = graduate[graduate['rec014_ent_year'] > '2018']
    graduate['event'] = graduate.apply(lambda x: 1 if x['학적'] == '제적' else (0 if x['학적'] == '졸업' else 2),
                                        axis=1)  # competing  risk
    graduate['자타'] = graduate.apply(lambda x: 1 if (x['학부출신'] == '고려대학교') else 0, axis=1)
    graduate = graduate.fillna(0)
    graduate = graduate[
        ['기간', 'event', '자타', 'count', '인건비횟수', '인건비과제수', '인건비합', '등록금장학', 'etc_장학', '성적', '입학성적', '휴학횟수', '휴학기간', '과정',
         '학과', 'rec014_std_id', 'rec014_ent_year', 'rec014_ent_term']]

    CRM = DeepHitCompetingRisk(graduate)
    #cmModel, labtrans_cuts = CRM.training()
    CR_plt, CR_hazard_students= CRM.predict_competingrisk(dept_cd="경영학과")

    #장학금
    print("--------------------------대학별 장학금 K prototypes Clustering------------------------------")
    tc = tuition_cluster()
    tc.data_preprocessing()
    fig = tc.umap()
    tc.kmeans_clustering()
    tc.kprototypes_clustering()
    rec_tuition = tc.kprototypes_group("경영학과")
    print(rec_tuition)

    #학과 -> plot 목록 학생 장학금
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()