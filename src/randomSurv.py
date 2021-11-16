import pandas as pd
import matplotlib.pyplot as plt
import sksurv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import numpy as np
import IPython
import joblib
import pickle
from IPython.display import display
from IPython.core.display import Image as image
from PIL import Image
import plotly.express as px

from eli5.formatters import format_as_dataframe
class ForestSurvival:
    def __init__(self, graduate):
        #self.path
        self.graduate = graduate

    def randomforest(self):

        self.graduate['event'] = self.graduate.apply(lambda x: True if (x['event'] == 1) else False, axis=1) #제적

        cat_curriculum = pd.read_excel('data/계열.xlsx')

        #휴학
        graduate1 = self.graduate[self.graduate['휴학횟수']>6]
        graduate1['휴학횟수'] = 1
        graduate1['휴학기간'] = 1
        graduate = self.graduate[self.graduate['휴학횟수']<6]
        self.graduate =  pd.concat([graduate1, graduate], axis = 0)
        plt.boxplot(self.graduate['휴학횟수'])
        plt.title("휴학Dropout box plot")
        plt.show()

        #계열
        self.graduate = pd.merge(self.graduate, cat_curriculum, how='inner', on='학과')
        self.graduate = self.graduate[self.graduate['계열'] == '인문·사회'].reset_index()
        #self.graduate = self.graduate[self.graduate['계열'] == '공학'].reset_index()
        #self.graduate = self.graduate[self.graduate['계열'] == '자연과학'].reset_index()

        data_y = self.graduate[['event', '기간']]#.to_numpy()
        data_x = self.graduate[['자타', 'count', '인건비합', 'etc_장학','등록금장학', '성적','입학성적', '휴학기간']]

        time, survival_prob = kaplan_meier_estimator(data_y["event"], data_y["기간"])
        plt.step(time, survival_prob, where="post")
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$")
        plt.show()

        print(data_x["자타"].value_counts())

        #km
        for value in data_x["자타"].unique():
            mask = data_x["자타"] == value
            time_cell, survival_prob_cell = kaplan_meier_estimator(data_y["event"][mask],
                                                                   data_y["기간"][mask])
            plt.step(time_cell, survival_prob_cell, where="post",
                     label="%s (n = %d)" % (value, mask.sum()))
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$")
        plt.legend(loc="best")
        plt.show()
        from sklearn.preprocessing import StandardScaler

        #cox
        print(data_x.head(5))

        data_x["자타"] = data_x["자타"].astype('category')

        data_x_numeric = OneHotEncoder().fit_transform(data_x)
        estimator = CoxPHSurvivalAnalysis()
        data_y = self.graduate[['event', '기간']].to_numpy()
        aux = [(e1, e2) for e1, e2 in data_y]
        new_data_y = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        arr = new_data_y
        data_Y = np.array([tuple(row) for row in arr], dtype = [('Status', '?'), ('Survival_in_days', '<f8')])
        print(data_x_numeric)
        print("afterrrr ")
        data_x_numeric_cr = StandardScaler().fit_transform(data_x_numeric)
        estimator.fit(data_x_numeric_cr, data_Y)
        print("--------------cos regression----------------")
        print(pd.Series(np.round(estimator.coef_,2), index= data_x_numeric.columns))

        #randomforest
        feature_names = data_x_numeric.columns.tolist()
        print("Feature names-------")
        print(feature_names)

        random_state = 20
        print("----------------Random Forest Survival Prediction Using Sksurv : Training-----------------------------")
        X_train, X_test, y_train, y_test = train_test_split(data_x_numeric, data_Y, test_size = 0.10, random_state=20)

        with open('X_test_NS.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open('y_test_NS.pkl', 'wb') as f:
            pickle.dump(y_test, f)

        #training
        rsf = RandomSurvivalForest(n_estimators=1000,
                                   min_samples_split=10,
                                   min_samples_leaf=15,
                                   max_features="sqrt",
                                   n_jobs=-1,
                                   random_state=20)
        rsf.fit(X_train, y_train)

        print("Test, C-index")
        print(rsf.score(X_test, y_test))
        X_test_sel = X_test.iloc[:10]
        surv = rsf.predict_survival_function(X_test_sel, return_array=True) #predict cumulative hazard function
        for i, s in enumerate(surv):
            plt.step(rsf.event_times_, s, where="post", label=str(i))
        plt.ylabel("Survival probability")
        plt.xlabel("Time in semester")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(X_test_sel[:10])

        saved_model = pickle.dump(rsf,open('model/randomSurvivalForest_NaturalScience.pkl', 'wb'))
        #rsf = pickle.load(open('model/randomSurvivalForest_engineering.pkl', 'rb'))
        import eli5
        from eli5.sklearn import PermutationImportance
        feature_names =  data_x_numeric.columns.tolist()
        perm = PermutationImportance(rsf, n_iter=15, random_state=20)
        perm.fit(X_test, y_test)
        eli5.show_weights(perm, feature_names=feature_names)
        return rsf

    def predict_students_by_dept(self, rsf = True ,dept_cd = '경영학과'):
        if rsf:
            rsf = pickle.load(open('model/randomSurvivalForest.pkl', 'rb'))
        # x_test 21년 1학기 입학생
        test_set = self.graduate[(self.graduate['rec014_ent_year'] == '2021') & (self.graduate['rec014_ent_term'] == '1R') & (self.graduate['학과'] == dept_cd)]
        X_test_set = test_set[['자타', 'count', '인건비합', 'etc_장학','등록금장학', '성적','입학성적', '휴학기간']]
        test_set['rec014_std_id'] = test_set['rec014_std_id'].astype('string')

        print("학과 평균 ------------------------------->")
        pd.options.display.float_format = '{:,.2f}'.format

        print(test_set.mean(axis = 0))
        plt.boxplot(x=test_set['인건비횟수'])
        plt.title("인건비과제 box plot")
        plt.show()

        test_set['hazard_predict'] = rsf.predict(X_test_set)

        hazard_students = test_set[test_set['hazard_predict'] > 6] #top 6 ? 10% 상위
        hazard_students = test_set.sort_values(by = 'hazard_predict', ascending=False).head(int(len(test_set)*0.1)) # 상위 ?

        surv = rsf.predict_cumulative_hazard_function(X_test_set, return_array=True)  # predict cumulative hazard function

        anonymous_label = ['A', 'B', 'C', 'D', 'E']
        for i, s in enumerate(surv):
            if test_set.iloc[i]['rec014_std_id'] in hazard_students['rec014_std_id'].tolist():
                #mylabel = "std %s"%(str(test_set.iloc[i]['rec014_std_id']))
                #mylabel = anonymous_label.pop(0)
                mylabel = None
            else:
                mylabel = None
            plt.step(rsf.event_times_, s, where="post", label=mylabel)
            px.line(x=rsf.event_times_,y= s)
        plt.ylabel("Hazard probability")
        plt.xlabel("Time in Semester")
        plt.legend()
        plt.grid(True)
        plt.show()

        #print(hazard_students)
        return plt, hazard_students
