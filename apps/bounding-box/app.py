import numpy as np
import json
import pandas as pd

import dash_canvas
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate

from dash_canvas.utils import parse_jsonstring_rectangle
import dash_table
from textwrap import dedent

filename = 'https://bestpostarchive.com/wp-content/uploads/2019/02/driving-in-the-streets-of-san-fr-800x445.jpg'

app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

list_columns = ['width', 'height', 'left', 'top', 'type']
columns = [{"name": i, "id": i} for i in list_columns]
columns[-1]['presentation'] = 'dropdown'

app.layout = html.Div([
    html.Div([
        html.H3('Annotate image with bounding boxes'),
        dcc.Markdown(dedent('''
        Annotations of vehicles can be used as training set for machine
        learning.
        ''')),
        dash_canvas.DashCanvas(
            id='canvas',
            width=500,
            tool="rectangle",
            lineWidth=2,
            filename=filename,
            hide_buttons=['pencil', 'line'],
            goButtonTitle='Get coordinates'
            ),
            ], className="six columns"),
    html.Div([
        html.Img(id='img-help',
                 src='assets/bbox.gif',
                 width='100%'),
        html.H4('Geometry of bounding boxes'),
        dash_table.DataTable(
              id='table',
              columns=columns,
              editable=True,
	      column_static_dropdown=[
            {
                'id': 'type',
                'dropdown': [
                    {'label': i, 'value': i}
                    for i in ['car', 'truck', 'bike', 'pedestrian']
                ]
            }],
              ),
        ], className="six columns")],# Div
    className="row")


@app.callback(Output('table', 'data'),
              [Input('canvas', 'json_data')])
def show_string(string):
    props = parse_jsonstring_rectangle(string)
    df = pd.DataFrame(props, columns=list_columns[:-1])
    df['type'] = 'car'
    return df.to_dict("records")


@app.callback(Output('img-help', 'width'),
              [Input('canvas', 'json_data')])
def reduce_help(json_data):
    if json_data:
        return '0%'
    else:
        raise PreventUpdate



if __name__ == '__main__':
    app.run_server(debug=True)

