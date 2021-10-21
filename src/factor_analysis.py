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

def factor_ana(dept_cd = '전기전자공학과'):

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

    #ERROR
    """
    졸업생 휴학 기간 -> 1 
    """
    graduate1 = graduate[graduate['휴학횟수']>6]
    graduate1['휴학횟수'] = 1
    graduate1['휴학기간'] = 1
    graduate = graduate[graduate['휴학횟수']<6]
    graduate =  pd.concat([graduate, graduate], axis = 0)
    #boxplot
    plt.boxplot(graduate['휴학횟수'])
    plt.title("휴학 box plot")
    plt.show()



    #(위험) 학생 유형
    ## Primary Factor 로 clustering = Factor analysis, K-means

    #dropping unnecessary columns
    #------>
    if dept_cd != False:
        graduate = graduate[graduate['학과'] == dept_cd]
    graduate = graduate.reset_index()
    graduate_std = graduate['rec014_std_id']
    graduate_event = graduate['event']
    graduate.drop(['index', 'event', '과정', '학과','rec014_std_id', 'rec014_ent_year', 'rec014_ent_term'], axis =1, inplace = True) #?기간
    graduate.dropna(inplace = True)
    print(graduate.info())

    #corrlation
    # 폰트 경로
    font_path = os.path.join(os.getcwd(), "ChosunCentennial_ttf.ttf")
    # 폰트 이름 얻어오기
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    # font 설정
    matplotlib.rc('font', family=font_name)

    plt.figure(figsize=(8,8))
    corr = graduate.corr()
    corrplot(corr, size_scale = 300)
    #ax = sns.heatmap(corr)
    plt.show()

    #KMO
    kmo_model = calculate_kmo(graduate) #0.7 이상 진행
    print(kmo_model)

    #choosing number of factor
    fa = FactorAnalyzer(rotation = None, n_factors = graduate.shape[1])
    fa.fit(graduate)
    ev, v = fa.get_eigenvalues()
    print(ev) # 몇개나 1보다 큰지? 6,4
    plt.scatter(range(1, graduate.shape[1] + 1), ev)
    plt.plot(range(1, graduate.shape[1] + 1), ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show() #4

    fa = FactorAnalyzer(n_factors = 4, rotation = 'varimax')
    fa.fit(graduate)
    print(pd.DataFrame(fa.loadings_, index = graduate.columns))


    graduate2 = graduate[['인건비횟수', '휴학횟수', '성적', '등록금장학']]

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
    # aggregate mean columns  by clustering
    print(graduate2.groupby('kmeans_labels').aggregate(['min', 'mean', 'max', np.median]))
    graduate2['rec014_std_id'] = 0
    graduate2['rec014_std_id'] = graduate_std
    print(graduate2)

    # 위험학생 추가
    survF = ForestSurvival(deep_graduate)
    RS_plt, RS_hazard_students = survF.predict_students_by_dept(rsf=True, dept_cd=dept_cd)
    # 경영학과 위험학생 , RS_hazard_students
    h_list = list(RS_hazard_students['rec014_std_id'])
    graduate2['label'] = 0
    graduate2['hazard'] = 0
    graduate2['event'] = graduate_event
    for i in range(len(graduate)):
        if graduate2['rec014_std_id'][i] in h_list:

            graduate2['label'][i] = graduate2['rec014_std_id'][i]
            graduate2['hazard'][i] = 1
        else:
            graduate2['label'][i] = ""
            graduate2['hazard'][i] = 0

    finalDf = pd.concat([principalDf, graduate2[['kmeans_labels', 'label']]], axis=1)
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

    for i, txt in enumerate(graduate2['label']):
        ax.annotate(txt, (finalDf['principal component 1'][i], finalDf['principal component 2'][i]))

    for i, txt in enumerate(graduate2['event']):
        if txt == 1:
            ax.annotate(txt, (finalDf['principal component 1'][i], finalDf['principal component 2'][i]))
    ax.legend(targets)
    ax.grid()
    plt.show()

    show_df = pd.DataFrame()
    for i in range(len(graduate)):
        if graduate2['rec014_std_id'][i] in h_list:
            show_df = show_df.append(graduate2.iloc[i])
    #event0, 1, 또는 위험학생이 어디 속하는가?
    print(show_df)

    return fig




if __name__ == '__main__':
    factor_ana()