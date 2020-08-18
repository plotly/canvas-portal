import numpy as np
import json
from skimage import io, color, segmentation, img_as_ubyte, filters, measure, \
                    data
from PIL import Image


import dash_canvas
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

from dash_canvas.utils import (parse_jsonstring, image_string_to_PILImage,
                               array_to_data_url, modify_segmentation)


# Image to segment and shape parameters
filename = 'https://upload.wikimedia.org/wikipedia/commons/1/1b/HumanChromosomesChromomycinA3.jpg'
try:
    img = io.imread(filename, as_gray=True)
    mask = img > 1.2 * filters.threshold_otsu(img)
except:
    img = data.coins()
    mask = img > filters.threshold_otsu(img)

labels = measure.label(mask)
overlay = segmentation.mark_boundaries(img, labels)
overlay = img_as_ubyte(overlay)

height, width = img.shape[:2]
canvas_width = 500
canvas_height = round(height * canvas_width / width)
scale = canvas_width / width

app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.Div([
   	html.Img(src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png', width='30px'),
        html.A(
              id='gh-link',
              children=[
                  'View on GitHub'],
              href="http://github.com/plotly/canvas-portal/"
                          "blob/master/apps/correct-segmentation/app.py",
              style={'color': 'black',
                    'border':'solid 1px black',
                    'float':'left'}
                    ),
    html.H3(children='Manual correction of automatic segmentation'),
    dcc.Markdown('''
        Annotate the picture to delineate boundaries
        between objects (in split mode) or to join objects
        together (in merge mode), then press the 
        "Update segmentation" button to correct
        the segmentation.
    '''),
    html.H5(children='Annotations'),
    dcc.RadioItems(id='mode',
    options=[
        {'label': 'Merge objects', 'value': 'merge'},
        {'label': 'Split objects', 'value': 'split'},
    ],
    value='split',
    # labelStyle={'display': 'inline-block'}
    ),
    html.H5(children='Save segmentation'),
    dcc.RadioItems(id='save-mode',
    options=[
        {'label': 'png', 'value': 'png'},
        #{'label': 'raw', 'value': 'raw'},
    ],
    value='png',
    labelStyle={'display': 'inline-block'}
    ),
    html.A(
        'Download Data',
        id='download-link',
        download="correct_segmentation.png",
        href="",
        target="_blank"
    ),
    dcc.Store(id='cache', data=''),

    ], className="four columns"),
    html.Div([
    dash_canvas.DashCanvas(
        id='canvas_',
        width=canvas_width,
	height=canvas_height,
        scale=scale,
        lineWidth=2,
        lineColor='red',
        image_content=array_to_data_url(overlay),
        goButtonTitle='Update',
    ),
     ], className="six columns"),
       ])

# ----------------------- Callbacks -----------------------------


@app.callback(Output('cache', 'data'),
            [Input('canvas_', 'trigger'),],
            [State('canvas_', 'json_data'),
            State('canvas_', 'scale'),
            State('canvas_', 'height'),
            State('canvas_', 'width'),
            State('cache', 'data'),
            State('mode', 'value')])
def update_segmentation(toggle, string, s, h, w, children, mode):
    if len(children) == 0:
        labs = labels
    else:
        labs = np.asarray(children)
    mask = parse_jsonstring(string, shape=(height, width))
    new_labels = modify_segmentation(labs, mask, img=img, mode=mode)
    return new_labels


@app.callback(Output('canvas_', 'image_content'),
            [Input('cache', 'data')])
def update_figure(labs):
    if labs:
        new_labels = np.array(labs)
        overlay = segmentation.mark_boundaries(img, new_labels)
        overlay = img_as_ubyte(overlay)
        return array_to_data_url(overlay)
    else:
        raise PreventUpdate


@app.callback(Output('download-link', 'download'),
            [Input('save-mode', 'value')])
def download_name(save_mode):
    if save_mode == 'png':
        return 'correct_segmentation.png'
    else:
        return 'correct_segmentation.raw'


@app.callback(Output('download-link', 'href'),
            [Input('cache', 'data')],
            [State('save-mode', 'value')])
def save_segmentation(labs, save_mode):
    if labs:
        new_labels = np.array(labs)
        np.save('labels.npy', new_labels)
        if save_mode == 'png':
            color_labels = 255 * color.label2rgb(new_labels)
            uri = array_to_data_url(color_labels, dtype=np.uint8)
            return uri
    else:
        raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)

