import dash
import dash_table
from flask import request
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
import base64
from plotly.tools import mpl_to_plotly

from utils import predict_students_by_dept_rsf, get_graduate
from simplemodel import survivalSimple
from competingrisk import DeepHitCompetingRisk
from tuition import tuition_cluster

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

plotly_fig, plotly_fig2, plotly_fig3, rsf_stu, DHS_hazard_students,CR_hazard_students, rec_tuition = get_fig()


print("start")
app = dash.Dash(__name__)

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

df2 = pd.DataFrame({
    "Fruit2": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount2": [4, 1, 2, 2, 4, 5],
    "City2": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
fig2 = px.bar(df2, x="Fruit2", y="Amount2", color="City2", barmode="group")



app.layout = html.Div([
                #html.Img(src=app.get_asset_url('data/myplot.png')),
                # html.Div(children=[
                #     html.H1(children='Hello Dash'),
                #
                #     html.Div(children='''
                #         Dash: A web application framework for your data.
                #     '''),
                #
                #     dcc.Graph(
                #         id='example-graph',
                #         figure=fig
                #     )
                # ]),
                #
                # html.Div(children=[
                #     html.H1(children='Hello Dash2'),
                #
                #     html.Div(children='''
                #                     Dash2: A web application framework for your data.
                #                 '''),
                #
                #     dcc.Graph(
                #         id='example-graph2',
                #         figure=fig2
                #     )
                # ]),
                html.Div([
                    html.H1('대학원 중도탈락 위험 학생 예측 DashBoard'),
                    #html.H3('작성자 : 고려대학교 데이터 허브팀'),
                    #html.H3('작성일 : 2020.09.30'),
                    html.H3('중도탈락 예측 모델 시각화'),
                    html.H3('경영학과'),
                    html.H4('1) Random Survival Forest'),
                    html.H4('2) DeepHit Survival Single'),
                    html.H4('3) DeepHit Competing Risks'),
                    html.H4('4) 대학별 장학금')
                    ], style={'margin-left': '20px'}),
                html.Div([
                    html.H2(children='1) Random Survival Forest'),
                    html.Div(children='''
                                     Dash: A web application framework for your data.
                                 '''),
                    dcc.Graph(id='matplotlib-graph1', figure=plotly_fig)

                ]),
                dash_table.DataTable(
                    id='dtTable1',
                    columns=[{"name": i, "id": i} for i in rsf_stu.columns],
                    data=rsf_stu.to_dict('records')
                ),
                html.Div([
                    html.H2(children='2) DeepHit Survival Single'),
                    dcc.Graph(id='matplotlib-graph2', figure=plotly_fig2)

                ]),
                dash_table.DataTable(
                    id='dtTable2',
                    columns=[{"name": i, "id": i} for i in DHS_hazard_students.columns],
                    data=DHS_hazard_students.to_dict('records')
                ),
                html.Div([
                    html.H2(children='3) DeepHit Competing Risks'),
                    dcc.Graph(id='matplotlib-graph3', figure=plotly_fig3)

                ]),
                dash_table.DataTable(
                    id='dtTable3',
                    columns=[{"name": i, "id": i} for i in CR_hazard_students.columns],
                    data=CR_hazard_students.to_dict('records')
                ),
                html.Div(
                    className = 'rec_tuition',
                    children = [ html.H1('대학원 학과(대학별) 장학금 DashBoard'),
                                 html.Ul(id='my-list', children=[html.Li(i) for i in rec_tuition])],
                    style={'textAlign': 'center'}
                ),
                html.Div([
                        html.H4('디지털정보처 데이터Hub팀')], style={'margin':'20px', 'textAlign':'right'})

            ])





# app.layout = html.Div([
#     # represents the URL bar, doesn't render anything
#     dcc.Location(id='url', refresh=False),
#
#     dcc.Link('Navigate to "/"', href='/'),
#     html.Br(),
#     dcc.Link('Navigate to "/page-2"', href='/page-2'),
#
#     # content will be rendered in this element
#     html.Div(id='page-content')
# ])
# def shutdown():
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()
# @app.callback(dash.dependencies.Output('page-content', 'children'),
#               [dash.dependencies.Input('url', 'pathname')])
# def display_page(pathname):
#     if pathname =='/shutdown':
#         shutdown()
#     return html.Div([
#         html.H3('You are on page {}'.format(pathname))
#     ])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':
    app.run_server(debug=True, host='163.152.6.195')
