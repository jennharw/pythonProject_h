import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style = "whitegrid")
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

import statsmodels.api as sm
from statsmodels.formula.api import ols
import researchpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

def eda(graduate):
    print(researchpy.summary_cont(graduate.기간))
    print(researchpy.summary_cont(graduate.기간.groupby(graduate.event)))
    ax = sns.countplot(x = "event", data = graduate)

    plt.rc('font', family='NanumGothic')
    print(f"설정 폰트 글꼴: {plt.rcParams['font.family']}, 설정 폰트 사이즈: {plt.rcParams['font.size']}")

    # 1) 현황
    n_students = graduate.shape[0]
    students = np.arange(n_students)
    fig, ax = plt.subplots(figsize=(8, 6))
    blue, _, red = sns.color_palette()[:3]
    ax.hlines(
        students[graduate.event.values == 0], 0, graduate[graduate.event.values == 0].기간, color=blue,
        label="Censored"
    )
    ax.hlines(
        students[graduate.event.values == 1], 0, graduate[graduate.event.values == 1].기간, color=red,
        label="Uncensored"
    )
    ax.scatter(
        graduate.기간,
        students,
        color="k",
        zorder=10,
        label="Architecture",
    )
    ax.set_xlim(left=0)
    ax.set_xlabel("Semester")
    ax.set_yticks([])
    ax.set_ylabel("Students")
    ax.set_ylim(-0.25, n_students + 0.25)
    plt.show()

    # 2) Kaplan Meier
    T = graduate['기간']
    E = graduate['event']
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter().fit(T, E)
    kmf.plot_survival_function(ax=ax)
    plt.show()

    # 3) 그룹별 KAPLAN MERIER
    ax = plt.subplot(111)
    dem = (graduate['자타'] == 1)
    a = T[dem]
    b = E[dem]
    kmf = KaplanMeierFitter().fit(T[dem], E[dem], label="KU")
    kmf.plot_survival_function(ax=ax)

    kmf = KaplanMeierFitter().fit(T[~dem], E[~dem], label="Not KU")
    kmf.plot_survival_function(ax=ax)
    plt.title("Dropout KU")
    plt.show()

    # 3-2) log rank
    results = logrank_test(T[dem], T[~dem], E[dem], E[~dem], alpha=.99)
    results.print_summary()

    # 4) 장학금 boxplot, histogram
    pilt = sns.boxplot(x=graduate['등록금장학'], orient="h", palette="Set2")
    plt.title('Tuition Boxplot')
    plt.xlabel('')

    fig = pilt.get_figure()
    fig.show()

    graduate['등록금장학'].plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.title('Tuition Histogram')
    plt.xlabel('')
    plt.ylabel('')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # 5) Cox Regression
    graduate2 = graduate[['기간', 'event', '인건비합', '등록금장학', '성적']]
    cf = CoxPHFitter()
    cf.fit(graduate2, '기간', 'event')
    cf =cf.summary
    print(cf)

    ## Normalization
    scaler = MinMaxScaler()
    graduate['등록금장학'] = scaler.fit_transform(graduate.iloc[:, 5:6].to_numpy())
    # 4-2) 장학금 boxplot, histogram
    pilt = sns.boxplot(x=graduate['등록금장학'], orient="h", palette="Set2")
    plt.xlabel('')

    plt.title('Tuition Boxplot')
    fig = pilt.get_figure()
    fig.show()

    graduate['등록금장학'].plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.title('Normalized Tuition Histogram')
    plt.ylabel('')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    # 5-2) Cox Regression
    graduate2 = graduate[['기간', 'event', '인건비합', '등록금장학', '성적']]
    cf = CoxPHFitter()
    cf.fit(graduate2, '기간', 'event')
    cf.print_summary()
    cf2 = cf.summary
    print("start")

    #5-3) Cox Regression 예측
    fig, axis = plt.subplots(nrows=1, ncols=1)
    cf.baseline_survival_.plot(ax=axis, title="Baseline Survival")
    # Solution to plotting multiple regressors
    fig, axis = plt.subplots(nrows=1, ncols=1, sharex=True)
    regressor1 = np.array([[0, 0, 60]])
    regressor2 = np.array([[0, 1000000, 100]])
    survival_1 = cf.predict_survival_function(regressor1)
    survival_2 = cf.predict_survival_function(regressor2)
    plt.plot(survival_1, label="성적 60")
    plt.plot(survival_2, label="성적 100")
    plt.legend(loc="upper right")
    plt.show()

    odds = survival_1 / survival_2
    plt.plot(odds, c="red")
    plt.show()

    #7) ANOVA, Regression
    pilt = sns.boxplot(x="과정", y='기간', data=graduate)  # hue palette = "Set3", orient = "h"
    fig = pilt.get_figure()
    fig.show()
    anova_re = ols("기간 ~ 과정", data=graduate).fit()
    # sm.stats.anova_lm(anova_re, type = 2) parametric 이분산, 등분산
    kruskal = stats.kruskal(graduate.loc[graduate['과정'] == '석사과정', "기간"],
                               graduate.loc[graduate['과정'] == '박사과정', "기간"],
                               graduate.loc[graduate['과정'] == '석박사통합과정', "기간"],
                               )
    print("kruskal")
    print(kruskal)

    grad_scatter = graduate[graduate['과정'] == '박사과정']
    plt.rcParams['figure.figsize'] = [10, 8]
    sns.scatterplot(x='기간', y='등록금장학', hue='과정', style='과정', data=grad_scatter)
    plt.show()

    line_fitter = LinearRegression()
    line_fitter.fit(grad_scatter['기간'].values.reshape(-1, 1), grad_scatter['count'])
    # line_fitter.intercept_
    # line_fitter.coef_
    est = sm.OLS(grad_scatter['count'], sm.add_constant(grad_scatter['기간'])).fit().summary()
    print("Linear Regression")
    print(est)
    plt.scatter(grad_scatter['기간'], grad_scatter['count'])
    plt.plot(grad_scatter['기간'], line_fitter.predict(grad_scatter['기간'].values.reshape(-1, 1)))
    plt.show()

    line_fitter = LinearRegression()
    line_fitter.fit(grad_scatter['기간'].values.reshape(-1, 1), grad_scatter['휴학기간'])
    # line_fitter.intercept_
    # line_fitter.coef_
    est = sm.OLS(grad_scatter['휴학기간'], sm.add_constant(grad_scatter['기간'])).fit().summary()
    print("Linear Regression")
    print(est)
    plt.scatter(grad_scatter['기간'], grad_scatter['휴학기간'])
    plt.plot(grad_scatter['기간'], line_fitter.predict(grad_scatter['기간'].values.reshape(-1, 1)))
    plt.show()

    return "Explanatory Data Analysis 데이터 탐색"