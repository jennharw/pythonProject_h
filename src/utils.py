import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

def get_graduate():
    if os.path.exists('data/graduate.pkl'):
        data = pd.read_csv('data/graduateeda.csv'
                           )
        file = open('data/graduate.pkl', 'rb')
        graduate = pickle.load(file)
        file.close()

    # pd.set_option('display.max_columns', 40)
    # pd.set_option('display.width', 1000)

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
    return graduate

def predict_students_by_dept_rsf(graduate,rsf = True ,dept_cd = '경영학과'):
    if rsf:
        rsf = pickle.load(open('model/randomSurvivalForest.pkl', 'rb'))
    # x_test 21년 1학기 입학생
    test_set = graduate[(graduate['rec014_ent_year'] == '2021') & (graduate['rec014_ent_term'] == '1R') & (graduate['학과'] == dept_cd)]
    X_test_set = test_set[['자타', 'count', '인건비합', 'etc_장학','등록금장학', '성적','입학성적', '휴학기간']]
    test_set['rec014_std_id'] = test_set['rec014_std_id'].astype('string')

    test_set['hazard_predict'] = rsf.predict(X_test_set)
    hazard_students = test_set[test_set['hazard_predict'] > 6] #top 6 ? 10% 상위
    hazard_students = test_set.sort_values(by = 'hazard_predict', ascending=False).head(int(len(test_set)*0.1)) # 상위 ?

    surv = rsf.predict_cumulative_hazard_function(X_test_set, return_array=True)  # predict cumulative hazard function
    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    for i, s in enumerate(surv):
        if test_set.iloc[i]['rec014_std_id'] in hazard_students['rec014_std_id'].tolist():
            mylabel = "std %s" % (str(test_set.iloc[i]['rec014_std_id']))
        else:
            mylabel = None
        ax.step(rsf.event_times_, s, where="post", label=mylabel)
    #ax.ylabel("Hazard probability")
    #ax.xlabel("Time in Semester")
    #ax.legend()
    ax.grid(True)
    # plt.show()

    #print(hazard_students)
    return fig3, hazard_students
#
# def predict ()
#     simpleSurv = survivalSimple(graduate)
#     DHSmodel = simpleSurv.simplemodel()
#     DS_plt, DS_hazard_students = simpleSurv.predict_by_dept(DHSmodel, dept_cd='경영학과')

from simplemodel import survivalSimple
from competingrisk import DeepHitCompetingRisk
from tuition import tuition_cluster
from plotly.tools import mpl_to_plotly
def get_fig(dept_cd = '경영학과'):
    graduate = get_graduate()

    rsf_plt, rsf_stu = predict_students_by_dept_rsf(graduate, rsf = True ,dept_cd = dept_cd)

    simpleSurv = survivalSimple(graduate)
    DHS_plt, DHS_hazard_students = simpleSurv.predict_by_dept(dept_cd =dept_cd)

    CRM = DeepHitCompetingRisk(graduate)
    CR_plt, CR_hazard_students = CRM.predict_competingrisk(dept_cd=dept_cd)
    DHS_hazard_students = DHS_hazard_students[['기간', 'event', '자타', 'count', '인건비횟수', '인건비과제수', '인건비합',
                                               '등록금장학', 'etc_장학', '성적', '입학성적', '휴학횟수', '휴학기간', '과정',
                                               '학과', 'rec014_std_id']]
    CR_hazard_students = CR_hazard_students[['기간', 'event', '자타', 'count', '인건비횟수', '인건비과제수', '인건비합',
                                             '등록금장학', 'etc_장학', '성적', '입학성적', '휴학횟수', '휴학기간', '과정',
                                             '학과', 'rec014_std_id']]

    tc = tuition_cluster()
    tc.data_preprocessing()
    fig = tc.umap()
    tc.kmeans_clustering()
    tc.kprototypes_clustering()
    rec_tuition = tc.kprototypes_group(dept_cd)

    fig3= plt.figure()
    ax= fig3.add_subplot(111)
    ax.plot(range(10), [i**2 for i in range(10)])
    ax.grid(True)
    plotly_fig = mpl_to_plotly(rsf_plt)
    plotly_fig2 = mpl_to_plotly(DHS_plt)
    plotly_fig3 = mpl_to_plotly(CR_plt)
    return plotly_fig, plotly_fig2, plotly_fig3, rsf_stu, DHS_hazard_students,CR_hazard_students, rec_tuition
#
# plotly_fig, plotly_fig2, plotly_fig3, rec_tuition, rsf_stu = get_fig('경영학과')
# print('complete')