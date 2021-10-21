import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#한글 폰트 사용
from matplotlib import font_manager,rc
import matplotlib
import seaborn as sns
from heatmap import heatmap, corrplot
from data_load_from_db import load_data
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 1000)

from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from randomSurv import ForestSurvival

def factor_ana_haz(dept_cd = '경영학과'):

    graduate = load_data()


    #19, 20학번 TRAIN , 21학번 1학기  PREDICT
    #graduate = graduate.drop(graduate[(graduate['rec014_ent_year'] == 2021) & (graduate['rec014_ent_term'] =='2R')].index)
    graduate = graduate[graduate['rec014_ent_year'] > '2018']


    graduate['event'] = graduate.apply(lambda x: 1 if (x['학적'] == '제적') else 0, axis=1)  # event 1 censor 0

    graduate['자타'] = graduate.apply(lambda x: 1 if (x['학부출신'] == '고려대학교') else 0, axis=1)
    graduate = graduate.fillna(0)

    graduate = graduate[['기간', 'event', '자타', 'count', '인건비횟수', '인건비과제수', '인건비합', '등록금장학', 'etc_장학', '성적', '입학성적', '휴학횟수','휴학기간', '과정', '학과', 'rec014_std_id', 'rec014_ent_year', 'rec014_ent_term']]

    graduate['기간'] = graduate['기간'].astype('float64')
    graduate['event'] = graduate['event'].astype('int64')

    deep_graduate = graduate.copy(deep=True)

    # 위험학생만!!!
    survF = ForestSurvival(deep_graduate)
    RS_plt, RS_hazard_students = survF.predict_students_by_dept(rsf=True, dept_cd=dept_cd)
    # 경영학과 위험학생 , RS_hazard_students
    RS_hazard_students = RS_hazard_students.reset_index().drop('index', axis =1)
    print(RS_hazard_students)

    graduate2 = RS_hazard_students[['인건비횟수', '휴학횟수', '성적', '등록금장학']]
    std = RS_hazard_students['rec014_std_id']
    #ERROR
    """
    졸업생 휴학 기간 -> 1 
    """

    #(위험) 학생 유형
    ## Primary Factor 로 clustering = Factor analysis, K-means

    #dropping unnecessary columns
    RS_hazard_students.drop(['event', '과정', '학과','rec014_std_id', 'rec014_ent_year', 'rec014_ent_term'], axis =1, inplace = True) #?기간

    plt.figure(figsize=(8,8))
    corr = RS_hazard_students.corr()
    corrplot(corr, size_scale = 300)
    #ax = sns.heatmap(corr)
    plt.show()


    #PCA, visualization
    x= StandardScaler().fit_transform(graduate2)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])

    # Clustering
    kmeans = KMeans(n_clusters=4).fit(graduate2)
    kmeans_labels = kmeans.labels_
    graduate2['kmeans_labels'] = kmeans_labels
    print("------------------------------------")
    graduate2['rec014_std_id'] = 0
    graduate2['rec014_std_id'] = std

    finalDf = pd.concat([principalDf, graduate2['kmeans_labels']], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1, 2,3]
    colors = ['r', 'g', 'b', '#1f77b4']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['kmeans_labels'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)

    p_list = [0 for i in range(4)]
    for i, txt in enumerate(zip(graduate2['rec014_std_id'], graduate2['kmeans_labels'])):
        for p in range(4):
            if txt[1] == p:
                ax.annotate(txt[0], (finalDf['principal component 1'][i], finalDf['principal component 2'][i]+0.1*p_list[p]))
                p_list[p] += 1
                break
    #text adjust ???
    ax.legend(targets)
    ax.grid()
    plt.show()

    print(graduate2)
    show_df = graduate2.groupby('kmeans_labels').aggregate('mean').reset_index()
    print(show_df)
    return fig, show_df


if __name__ == '__main__':
    factor_ana_haz()