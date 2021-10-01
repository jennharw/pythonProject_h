import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly
import plotly.graph_objects as go
from scipy import stats

from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import copy
import pickle

class tuition_cluster:
    def __init__(self):
        file = open('data/tui_df.pkl', 'rb')
        self.df = pickle.load(file)
        self.embedding = 0
        file.close()

    def data_preprocessing(self):
        pd.set_option('display.max_columns', 40)
        pd.set_option('display.width', 1000)

        self.df['등록금장학'] = self.df['sum(sch214_tuition_fee)'] + self.df['sum(sch214_tuition_fee)']
        self.df = self.df[~self.df['학생소속대학명'].isin(['스마트보안학부','정보보호학부'])]
        df1 = pd.pivot_table(self.df, index=['sch110_sch_cd', 'sch110_sch_nm', 'in_out', 'kedi장학금구분'], columns='학생소속대학명', values='등록금장학',
                                fill_value=0).reset_index()
        df2 = pd.pivot_table(self.df, index='sch110_sch_cd',
                                 columns='학생소속대학명', values='sum(sch214_etc_fee)',
                                 fill_value=0).reset_index()
        self.df = pd.merge(df1, df2, how ='inner',on = 'sch110_sch_cd')
        self.df = self.df.set_index('sch110_sch_nm')


    def umap(self): #tuition_df

        # umap
        # PREPROCESSING NUMERICAL
        numerical = self.df.select_dtypes(exclude = 'object')
        for c in numerical.columns:
          pt = PowerTransformer()
          numerical.loc[:,c] = pt.fit_transform(np.array(numerical[c]).reshape(-1,1))
        #PREPROCESSING CATEGORICAL
        categorical = self.df.select_dtypes(include = 'object')
        categorical = pd.get_dummies(categorical)

        #categorical_weight = len(self.df.select_dtypes(include='object').columns) / self.df.shape[1]


        #Embedding
        #fit1 = umap.UMAP(metric='cosine').fit_transform(numerical)
        #fit2 = umap.UMAP(metric='cosine').fit_transform(categorical)


        pio.templates.default = 'ggplot2'
        self.embedding = umap.UMAP(n_components = 2, n_neighbors = 4, init = 'random', random_state = 0, min_dist = 0.99, metric='cosine').fit_transform(numerical)

        fig_2d = px.scatter(self.embedding,
                        x = 0, y=1,
                        color = self.df['kedi장학금구분'],
                        #color = df["F_NCOM_CODE_NM('SCH001',SCH110_SCH_DIV)"],
                        #color_discrete_map = {"IN":'rgb(20,63,145)',
                        #                       "OUT":'rgb(240,138,1)'},
                        # symbol=self.data['on/off'],
                        text=self.df.index.tolist(),
                        labels={'color': 'Fields'},
                        width=2500, height=2500,
                        title="장학금 지도",
                    )

        fig_2d.update_traces(marker=dict(size=12, opacity=0.5),
                                          textfont_size=10)
        fig_2d.update_layout(
             plot_bgcolor='rgb(242, 242, 242)',
             legend=dict(font=dict(size=20)),
             hoverlabel=dict(font_size=30),
             title_font_size=50
        )
        plotly.offline.plot(fig_2d, filename = 'result/장학umap.html')
        return fig_2d


    def kmeans_clustering(self):
        #kmeans
        #ONE HOT ENCODING = > K-MEANS
        self.df = self.df.drop(columns ='sch110_sch_cd', axis = 1 )
        data = pd.get_dummies(self.df)
        for c in data.columns:
            pt = PowerTransformer()
            data.loc[:, c] = pt.fit_transform(np.array(data[c]).reshape(-1, 1))

        kmeans = KMeans(n_clusters=10).fit(data)
        kmeans_labels = kmeans.labels_

        self.df['kmeans_labels'] = kmeans_labels

        fig_2d = px.scatter(
                        self.embedding, x=0, y=1,
                        color = self.df['kmeans_labels'],
                        #color = df["F_NCOM_CODE_NM('SCH001',SCH110_SCH_DIV)"]
                        #,color_discrete_map = {"IN":'rgb(20,63,145)',
                        #                       "OUT":'rgb(240,138,1)'},
                        # symbol=self.data['on/off'],
                        color_discrete_map={
                            1: 'rgb(23,190,207)',
                            2: 'rgb(23,190,207)',
                            3: 'rgb(23,190,207)',
                            4: 'rgb(20,63,145)',
                            5: 'rgb(167,19,96)',
                            6: 'rgb(23,190,207)',
                            7: 'rgb(242,242,242)',
                            8:'rgb(23,190,207)',
                            9: 'rgb(240,138,1)',
                            10:'rgb(193,0,0)',
                            11:'rgb(0,151,78)',
                            12: 'rgb(94,3,15)',
                            13:'rgb(97,4,19)',
                            14:'rgb(25,69,12)',
                            15:'rgb(73,05,24)'},
                        text=self.df.index.tolist(),
                        labels={'color': 'Fields'},
                        width=1800, height=1200,
                        title="장학금 지도",
                    )

        fig_2d.update_traces(marker=dict(size=18, opacity=0.5),
                                         textfont_size=5)
        fig_2d.update_layout(
            plot_bgcolor='rgb(242, 242, 242)',
            legend=dict(font=dict(size=20)),
            hoverlabel=dict(font_size=30),
            title_font_size=50
        )
        plotly.offline.plot(fig_2d, filename = 'result/k_means_장학umap.html')




    def kprototypes_clustering(self):

        #kprototypes
        df_T = self.df.fillna(0)
        kprot_data = self.df.copy()
        #Pre-processing
        for c in self.df.select_dtypes(exclude='object').columns:
            pt = PowerTransformer()
            kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

        kproto = KPrototypes(n_clusters= 15, init='Cao', n_jobs = 4)
        clusters = kproto.fit_predict(kprot_data, categorical=[0,1])
        self.df['clusters'] = clusters
        fig_2d = px.scatter(
                        self.embedding, x=0, y=1,
                        color = self.df['clusters'],
                        #color = df["F_NCOM_CODE_NM('SCH001',SCH110_SCH_DIV)"]
                        #,color_discrete_map = {"IN":'rgb(20,63,145)',
                        #                       "OUT":'rgb(240,138,1)'},
                        # symbol=self.data['on/off'],
                        color_discrete_map={
                            0:'#1E2328',
                            1: '#787D82',
                            2: '#F0F5FA',
                            3: '#F0CDB4',
                            4: '#B49B5A',
                            5: '#F02328',
                            6: '#F07D28',
                            7: '#F0F528',
                            8:'#1EF528',
                            9: '#50C3FA',
                            10:'#1E23FA',
                            11:'#8C23FA',
                            12: '#F055C8',
                            13:'#782328',
                            14:'#B4B928',
                            15:'#1E2382'},
                        text=self.df.index.tolist(),
                        labels={'color': 'Fields'},
                        width=1500, height=1200,
                        title="장학금 지도",
                    )

        fig_2d.update_traces(marker=dict(size=13, opacity=0.55),
                                         textfont_size=7)
        fig_2d.update_layout(
            plot_bgcolor='rgb(242, 242, 242)',
            legend=dict(font=dict(size=20)),
            hoverlabel=dict(font_size=30),
            title_font_size=50
        )
        plotly.offline.plot(fig_2d, filename = 'result/k_proto_장학umap.html')


    def kprototypes_group(self, dept_cd):
        #경영학과 -> 경영대학
        dept_mrj = pd.read_excel('data/대학학과.xlsx')
        dept_cd = dept_mrj[dept_mrj['학과'] == dept_cd].iloc[0]['학생소속대학명']
        pd.set_option('display.max_columns', 40)
        pd.set_option('display.width', 1000)
        file = open('data/tui_df.pkl', 'rb')
        dataset = pickle.load(file)
        file.close()
        dataset['등록금장학'] = dataset['sum(sch214_tuition_fee)'] + dataset['sum(sch214_tuition_fee)']
        dataset = dataset[~dataset['학생소속대학명'].isin(['스마트보안학부', '정보보호학부'])]
        dept_cd_to_tui = dataset.groupby(['sch110_sch_nm', 'sch110_sch_cd'])['등록금장학'].sum()
        dataset = pd.merge(dataset, dept_cd_to_tui, how = 'inner', on = ['sch110_sch_nm', 'sch110_sch_cd'])
        dataset['등록금장학_y'] = dataset['등록금장학_y'] + 1
        dataset['ratio'] = dataset['등록금장학_x'] / dataset['등록금장학_y']

        #self.df = self.df.sort_values(by = ['{}_{}'.format(dept_cd, 'x')], axis = 0, ascending=False) 대학 에서 가장 많이 받은 장학금
        dataset_by_dept =dataset[dataset['학생소속대학명'] == dept_cd].sort_values(by=['ratio'], axis=0, ascending=False).reset_index()
        #경영대학에서 제일 많이 받은 그룹
        self.df= self.df.reset_index()
        clusterbydept = self.df[self.df['sch110_sch_nm'] == dataset_by_dept['sch110_sch_nm'][0]].iloc[0]['clusters']

        list_tui = self.df[self.df['clusters'] == clusterbydept]['sch110_sch_nm']
        #print(list_tui)
        return list_tui
        # dept cd 가 ~ 대학
        #  / 비율이 높은
        #그 장학금이 속한 그룹
        #