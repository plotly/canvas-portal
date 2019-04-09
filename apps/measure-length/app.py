import numpy as np
import pandas as pd
from skimage import io, data

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.exceptions import PreventUpdate

import dash_canvas
from dash_canvas.utils import (image_string_to_PILImage, array_to_data_url,
                               parse_jsonstring_line)


filename = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/X-ray_of_normal_elbow_by_lateral_projection.jpg/756px-X-ray_of_normal_elbow_by_lateral_projection.jpg'
try:
    img = io.imread(filename)
except:
    img = data.coins() 
height, width = img.shape
canvas_width = 500
canvas_height = round(height * canvas_width / width)
scale = canvas_width / width

list_columns = ['length', 'width', 'height']
columns = [{"name": i, "id": i} for i in list_columns]

app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True


app.layout = html.Div([
    html.Div([
        dash_canvas.DashCanvas(
            id='canvas-line',
            width=canvas_width,
            height=canvas_height,
            scale=scale,
            lineWidth=2,
            lineColor='red',
            tool="line",
            hide_buttons=['pencil'],
            image_content=array_to_data_url(img),
            goButtonTitle='Measure',
            ),
    ], className="seven columns"),
    html.Div([
        html.H3('Draw lines and measure lengths'),
        html.H3(children='How to use this app', id='measure-subtitle'),
        html.Img(id='measure-help',
                 src='assets/measure.gif',
                 width='100%'),
        html.H4(children="Objects properties"),
        dash_table.DataTable(
            id='table-line',
            columns=columns,
            editable=True,
            ),
    ], className="four columns"),
    ])



@app.callback(Output('canvas-line', 'tool'),
                [Input('canvas-line', 'image_content')])
def modify_tool(string):
    return "line"


@app.callback(Output('measure-subtitle', 'children'),
                [Input('canvas-line', 'json_data')])
def reduce_help(json_data):
    if json_data:
        return ''
    else:
        raise PreventUpdate


@app.callback(Output('measure-help', 'width'),
                [Input('canvas-line', 'json_data')])
def reduce_help(json_data):
    if json_data:
        return '0%'
    else:
        raise PreventUpdate


@app.callback(Output('table-line', 'data'),
                [Input('canvas-line', 'json_data')])
def show_table(string):
    props = parse_jsonstring_line(string)
    df = pd.DataFrame(props[:, :3], columns=list_columns)
    return df.to_dict("records")


if __name__ == '__main__':
    app.run_server(debug=True)

