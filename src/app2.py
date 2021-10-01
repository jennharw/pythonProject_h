import dash
import dash_table
from flask import request
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go

from utils import get_fig

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
fig_plot2 = html.Div(id='fig_plot2')
fig_plot3 = html.Div(id='fig_plot3')
table = dash_table.DataTable(
                id = 'data-table',
                columns=[{"name":i, "id":i} for i in ['기간', 'event', '자타', 'count', '인건비횟수', '인건비과제수', '인건비합',
                                               '등록금장학', 'etc_장학', '성적', '입학성적', '휴학횟수', '휴학기간', '과정',
                                               '학과', 'rec014_std_id']]
            )
table2 = dash_table.DataTable(
                id = 'data-table2',
                columns=[{"name":i, "id":i} for i in ['기간', 'event', '자타', 'count', '인건비횟수', '인건비과제수', '인건비합',
                                               '등록금장학', 'etc_장학', '성적', '입학성적', '휴학횟수', '휴학기간', '과정',
                                               '학과', 'rec014_std_id']]
            )
table3 = dash_table.DataTable(
                id = 'data-table3',
                columns=[{"name":i, "id":i} for i in ['기간', 'event', '자타', 'count', '인건비횟수', '인건비과제수', '인건비합',
                                               '등록금장학', 'etc_장학', '성적', '입학성적', '휴학횟수', '휴학기간', '과정',
                                               '학과', 'rec014_std_id']]
            )
rec_tui = html.Div(
                    id = 'rec_tuition',
                    children = [
                                 ],
                    style={'textAlign': 'center'}
                )
title = html.Div([
                    html.H1('대학원 중도탈락 위험 학생 예측 DashBoard'),
                    #html.H3('작성자 : 고려대학교 데이터 허브팀'),
                    #html.H3('작성일 : 2020.09.30'),
                    html.H3('중도탈락 예측 모델 시각화'),
                    #html.H3('경영학과'),
                    html.H4('1) Random Survival Forest'),
                    html.H4('2) DeepHit Survival Single'),
                    html.H4('3) DeepHit Competing Risks'),
                    html.H4('4) 대학별 장학금')
                    ], style={'margin-left': '20px'})
description = html.Div([
                    html.H2(children='1) Random Survival Forest')])
description2= html.Div([
                    html.H2(children='2)  DeepHit Survival Single')])
description3= html.Div([
                    html.H2(children='3)  DeepHit Competing Risk')])
description_rec= html.Div([
                    html.H1('대학원 학과(대학별) 장학금 DashBoard')
])
footer = html.Div([
                        html.H4('디지털정보처 데이터Hub팀')], style={'margin':'20px', 'textAlign':'right'})

app.layout = html.Div([title,fig_dropdown,description, fig_plot,table,description2,fig_plot2,table2, description3 , fig_plot3, table3,description_rec,rec_tui,footer])

@app.callback(
   [ dash.dependencies.Output('fig_plot', 'children'),
    dash.dependencies.Output('fig_plot2', 'children'),
    dash.dependencies.Output('fig_plot3', 'children'),
     dash.dependencies.Output('data-table', 'data'),
     dash.dependencies.Output('data-table2', 'data'),
     dash.dependencies.Output('data-table3', 'data'),
     dash.dependencies.Output('rec_tuition', 'children')

     ],

    [dash.dependencies.Input('fig_dropdown', 'value')])

def update_output(fig_name):
     return name_to_figure(fig_name)

def name_to_figure(fig_name):
    figure = go.Figure()
    if fig_name == '경영학과':
        plotly_fig, plotly_fig2, plotly_fig3, rsf_stu, DHS_hazard_students, CR_hazard_students, rec_tuition = get_fig('경영학과')
        #figure.add_trace(go.Scatter(y=[4, 2, 1]))
        figure = plotly_fig
    elif fig_name == '기계공학과':
        plotly_fig, plotly_fig2, plotly_fig3, rsf_stu, DHS_hazard_students, CR_hazard_students, rec_tuition = get_fig('기계공학과')
        # figure.add_trace(go.Scatter(y=[4, 2, 1]))
        figure = plotly_fig
    else:
        plotly_fig, plotly_fig2, plotly_fig3, rsf_stu, DHS_hazard_students, CR_hazard_students, rec_tuition = get_fig(
            '전기전자공학과')
        # figure.add_trace(go.Scatter(y=[4, 2, 1]))
        figure = plotly_fig
    return dcc.Graph(figure=figure),  dcc.Graph(figure=plotly_fig2), dcc.Graph(figure=plotly_fig3), rsf_stu.to_dict('records'),DHS_hazard_students.to_dict('records'), CR_hazard_students.to_dict('records'),html.Ul(id='my-list', children=[html.Li(i) for i in rec_tuition])


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})



if __name__ == '__main__':
    app.run_server(debug=True, host='163.152.6.195')
