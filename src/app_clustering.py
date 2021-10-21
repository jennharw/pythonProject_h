import dash
import dash_table
from flask import request
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go

from factor_analysis_hazard import factor_ana_haz
from plotly.tools import mpl_to_plotly


# fig , show_df = factor_ana_haz('경영학과')
# plotly_fig = mpl_to_plotly(fig)
# print("start")
# app = dash.Dash(__name__)
#
# app.layout = html.Div([
#                 html.Div([
#                     html.H1('대학원 중도탈락 위험 학생 예측 DashBoard'),
#                     #html.H3('작성자 : 고려대학교 데이터 허브팀'),
#                     #html.H3('작성일 : 2020.09.30'),
#                     html.H3('중도탈락 예측 모델 시각화'),
#                     html.H3('경영학과'),
#                     html.H4('1) Random Survival Forest'),
#                     html.H4('2) DeepHit Survival Single'),
#                     html.H4('3) DeepHit Competing Risks'),
#                     html.H4('4) 대학별 장학금')
#                     ], style={'margin-left': '20px'}),
#                 html.Div([
#                     html.H2(children='1) Random Survival Forest'),
#                     html.Div(children='''
#                                      Dash: A web application framework for your data.
#                                  '''),
#                     dcc.Graph(id='matplotlib-graph1', figure=plotly_fig)
#
#                 ]),
#                 dash_table.DataTable(
#                     id='dtTable1',
#                     columns=[{"name": i, "id": i} for i in show_df.columns],
#                     data=show_df.to_dict('records')
#                 ),
#                 html.Div([
#                         html.H4('디지털정보처 데이터Hub팀')], style={'margin':'20px', 'textAlign':'right'})
#
#             ])
#



#
#
print("start")
app = dash.Dash(__name__)

fig_names = ['경영학과', '기계공학과', '전기전자공학과']
fig_dropdown = html.Div([
    dcc.Dropdown(
        id='fig_dropdown',
        options=[{'label': x, 'value': x} for x in fig_names],
        value=None
    )])

fig_plot = html.Div(id='fig_plot')

table = dash_table.DataTable(
                id = 'data-table',
                columns=[{"name":i, "id":i} for i in ['kmeans_labels','인건비횟수', '휴학횟수',
                                                      '성적', '등록금장학']]
            )

title = html.Div([
                    html.H1('대학원 중도탈락 위험 학생 유형 Clustering DashBoard'),
                    #html.H3('작성자 : 고려대학교 데이터 허브팀'),
                    #html.H3('작성일 : 2020.09.30'),
                    ], style={'margin-left': '20px'})
description = html.Div([
                    html.H2(children='위험학생 유형')])

footer = html.Div([
                        html.H4('디지털정보처 데이터Hub팀')], style={'margin':'20px', 'textAlign':'right'})

app.layout = html.Div([title,fig_dropdown,description, fig_plot,table,footer])

@app.callback(
   [ dash.dependencies.Output('fig_plot', 'children'),
     dash.dependencies.Output('data-table', 'data')
     ],

    [dash.dependencies.Input('fig_dropdown', 'value')])

def update_output(fig_name):
     return name_to_figure(fig_name)

def name_to_figure(fig_name):
    figure = go.Figure()
    if fig_name == '경영학과':
        plotly_fig, show_df = factor_ana_haz('경영학과')
        figure =  mpl_to_plotly(plotly_fig)
    elif fig_name == '기계공학과':
        plotly_fig, show_df = factor_ana_haz('기계공학과')
        figure = mpl_to_plotly(plotly_fig)
    else:
        plotly_fig, show_df = factor_ana_haz('전기전자공학과')
        figure = mpl_to_plotly(plotly_fig)
    return dcc.Graph(figure=figure), show_df.to_dict('records')


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})



if __name__ == '__main__':
    app.run_server(debug=True, host='163.152.6.195')
