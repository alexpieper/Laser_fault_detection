import dash
import dash_html_components as html
from dash import dcc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


def make_evaluation_dashboard(X_test, y_test, *models):
    '''

    :param X_test:
    :param y_test:
    :param models: all the models, that should be included in the Evaluation
    :return: a dashboard, running on localhost with port 8050
    '''
    # the number of columns in the Plotly graphic.
    ncols = 3

    app = dash.Dash()

    df = pd.DataFrame(X_test)
    df['y_true'] = y_test.ravel()
    df = df.reset_index()

    # initialize the first evaluation DataFrame with all models
    evaluation_df =pd.DataFrame({
            'Model Name': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
        })
    for model in models:
        evaluation_df.loc[model.name, 'Model Name'] = model.name
        evaluation_df.loc[model.name, 'Precision'] = model.precision
        evaluation_df.loc[model.name, 'Recall'] = model.recall
        evaluation_df.loc[model.name, 'F1-Score'] = (2 * model.precision * model.recall) / (model.precision + model.recall)
        # this F1-Score corresponds to the F_{0.5} Measure from the slides. The Harmonic mean of Precision and Recall


    percentage = dash.dash_table.FormatTemplate.percentage(2)
    format = dash.dash_table.Format.Format()
    all_options = [{'label': 'All Models', 'value':
                         {'predictions': y_test.ravel(),
                          'recall': 1,
                          'precision': 1,
                          'label': 'All Models',
                          }}] + [
                        {'label': i.name, 'value': {'predictions':i.predictions,
                                                    'recall':i.recall,
                                                    'precision':i.precision,
                                                    'label':i.name,
                                                    }} for i in models
                    ]

    # This defines the layout of the Dashboard
    app.layout = html.Div(id='parent', children=[
        html.H1(id='H1', children='Laser Project Model Evaluation', style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),
        dcc.Dropdown(id='dropdown',
                    options = all_options,
                    placeholder = "Select a model",
                    value = {'predictions':y_test.ravel(),
                                'recall':1,
                                'precision':1,
                                'label':'All Models',
                                }
                     ),
        dash.dash_table.DataTable(data=evaluation_df.to_dict('records'),
                                  columns=[{"name": i, "id": i, "format": (percentage if i != 'Model Name' else format), 'type': ('numeric' if i != 'Model Name' else 'string')} for i in evaluation_df.columns],
                                  sort_action="native",
                                  filter_action="native",
                                  id='eval_table'),
        dcc.Graph(id='graph'),
    ])

    # Here we update the Table, when a value from the dropdown is chosen
    @app.callback(
        dash.Output('eval_table', 'data'),
        dash.Input('dropdown', 'value'))
    def update_table(selected_model):
        if selected_model['label'] == 'All Models':
            evaluation_df = pd.DataFrame({
                'Model Name': [],
                'Precision': [],
                'Recall': [],
                'F1-Score': [],
            })
            for model in models:
                evaluation_df.loc[model.name, 'Model Name'] = model.name
                evaluation_df.loc[model.name, 'Precision'] = model.precision
                evaluation_df.loc[model.name, 'Recall'] = model.recall
                evaluation_df.loc[model.name, 'F1-Score'] = (2 * model.precision * model.recall) / (
                            model.precision + model.recall)
        else:
            f1_score = (2 * selected_model['precision'] * selected_model['recall']) / (selected_model['precision'] + selected_model['recall'])
            evaluation_df = pd.DataFrame({
                'Model Name': [selected_model['label']],
                'Precision': [selected_model['precision']],
                'Recall': [selected_model['recall']],
                'F1-Score': [f1_score],
            })
        return evaluation_df.to_dict(orient='records')

    # Here we update the Graph, when a value from the dropdown is chosen
    @app.callback(
        dash.Output('graph', 'figure'),
        dash.Input('dropdown', 'value'))
    def update_figure(selected_model):
        if isinstance(selected_model['predictions'][0], list):
            selected_model['predictions'] = [i[0] for i in selected_model['predictions']]
        df['y_pred'] = selected_model['predictions']



        if len(y_test.ravel())%ncols != 0:
            n_rows = int(len(y_test.ravel())/ncols + 1)
        else:
            n_rows = int(len(y_test.ravel())/ncols)

        fig = make_subplots(cols=ncols, rows=n_rows)
        row_index=0
        column_index = 0
        for i in range(len(y_test.ravel())):
            if row_index >= n_rows:
                column_index += 1
                row_index = 0
            if (y_test.ravel()[i] == 1) & (selected_model['predictions'][i] == 1):
                fig.add_trace(
                    go.Scatter(x=list(range(len(df.iloc[i].drop(['index', 'y_true', 'y_pred'])))),
                               y=df.iloc[i].drop(['index', 'y_true', 'y_pred']), line=dict(color='rgba(177, 221, 140, 0.8)')), row=row_index + 1, col= column_index + 1)
            elif  (y_test.ravel()[i] == -1) & (selected_model['predictions'][i] == -1):
                fig.add_trace(
                    go.Scatter(x=list(range(len(df.iloc[i].drop(['index', 'y_true', 'y_pred'])))),
                               y=df.iloc[i].drop(['index', 'y_true', 'y_pred']), line=dict(color="rgba(255, 98, 81, 0.5)")), row=row_index + 1, col=column_index + 1)
            # In this and the next case we make the wrong classifications wider, to seperate them from the correct predctions.
            elif (y_test.ravel()[i] == 1) & (selected_model['predictions'][i] == -1):
                fig.add_trace(
                    go.Scatter(x=list(range(len(df.iloc[i].drop(['index', 'y_true', 'y_pred'])))),
                               y=df.iloc[i].drop(['index', 'y_true', 'y_pred']), line=dict(color="rgba(177, 221, 140, 1)", width=5)), row=row_index + 1, col=column_index + 1)
            elif (y_test.ravel()[i] == -1) & (selected_model['predictions'][i] == 1):
                fig.add_trace(
                    go.Scatter(x=list(range(len(df.iloc[i].drop(['index', 'y_true', 'y_pred'])))),
                               y=df.iloc[i].drop(['index', 'y_true', 'y_pred']), line=dict(color="rgba(255, 98, 81, 1)", width=5)), row=row_index + 1, col=column_index + 1)
            row_index += 1
        fig.update_layout(
            autosize=False,
            width=1200,
            height=60 * n_rows,
            showlegend=False,
            transition_duration=1000)
        return fig

    # this starts the app
    app.run_server(debug=False)

