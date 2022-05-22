import os 

from dash import Dash, dcc, html, Input, Output
from pandas.core import common
import plotly.express as px
import plotly.graph_objects as go 

import pandas as pd
import numpy as np

import glob
import itertools 
from matplotlib import cm
import matplotlib as mpl

import yaml

home_dir = './'
file_prefix = home_dir + 'data/exp1/'
plotly_layout_file = home_dir + 'assets/plotly/layout.yaml'
plotly_traces_file = home_dir + 'assets/plotly/traces.yaml'
description_file = home_dir + 'description.md'

def process_results(file_list, drop_const_var = False):
    cols_to_drop = ['_unq_id', '_trial_id', 'file_name'] 
    
    def get_key(file_name):
        return os.path.splitext(os.path.basename(file_name))[0]
    
    dfs = {get_key(x): pd.read_parquet(x).drop(cols_to_drop, axis=1)
           for x in file_list}
    
    unq_vars = dfs['variations']\
                .drop_duplicates(ignore_index=True)\
                .set_index('_var_id')
    var_cols = unq_vars.columns
    unq_vars = unq_vars[sorted(var_cols)]
    

    for k, v in dfs.items():
        if k == 'variations': 
            dfs[k] = unq_vars.reset_index()
            continue
        dfs[k] = v.drop(var_cols, axis=1)\
                .set_index('_var_id')\
                .groupby(['_var_id', 'time'], as_index=True)\
                .mean()\
                .join(unq_vars)\
                .reset_index()
        
    if drop_const_var: 
        const_vars = unq_vars.apply(lambda g: g.nunique() == 1)
        const_vars = [k for k, v in const_vars.items() if v]
        dfs = {k: v.drop(const_vars, axis=1) for k, v in dfs.items()}
        
    return dfs 

def create_cmap(cmap_name, num_colors, alpha=1.0, scale=1.0):
    cmap = cm.get_cmap(cmap_name, num_colors)(np.linspace(0,1,num_colors)) * 255 * scale
    cmap = ['rgba(%d,%d,%d,%f)' %(x[0], x[1], x[2], alpha) for x in cmap]
    return cmap

def load_results(file_prefix):
    result_files = glob.glob(file_prefix + '*.parq')
    dfs = process_results(result_files, drop_const_var=True)
    
    col_names = {k: set(v.columns) for k, v in dfs.items()}
    col_names = {k: list(v - col_names['variations'] - {'time'}) 
                 if k != 'variations' else list(v) 
                 for k, v in col_names.items()}

    label_names = {k: k.replace('_', ' ') 
                   for k in itertools.chain(*col_names.values())}

    return dfs, col_names, label_names

def load_layout_config(file_name, num_axis=50):
    with open(file_name, 'r') as f:
        cfg = yaml.safe_load(f)

    axis_configs = {}
    for i in range(num_axis):
        axis_id = '%d' %(i) if i > 0 else ''
        axis_configs['xaxis' + axis_id] = cfg['axis']
        axis_configs['yaxis' + axis_id] = cfg['axis']

    layout = go.Layout(
        **axis_configs,
        font=cfg['font'],
        **cfg['general'],
        **cfg['title']
    )
    
    return layout

def load_traces_config(file_name):
    with open(file_name, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_description(file_name):
    with open(file_name, 'r') as f: 
        txt = f.read()
    return txt

# Loading results and variables
dfs, col_names, label_names = load_results(file_prefix)

social_net_type_args = {
    "ER": 'social_net_args::p',
    "SBM": 'social_net_args::p_max'
}

# Labeling for plots

special_names = {
    'social_net_args::type': 'social network',
    'social_net_args::p': 'P(ER)',
    'social_net_args::p_max': 'P(SBM)',
    'utility_gamma': 'gamma'
}

special_names = {
    'social_net_args::type': 'social network',
    'social_content': r'$\textit{content}$',
    'social_net_args::p': r'$P_{\mathrm{neigh}}^{\mathrm{ER}}$',
    'social_net_args::p_max': r'$P_{\mathrm{block}}^{\mathrm{SBM}}$',
    'utility_gamma': r'$\gamma_{\mathrm{utility}}$'
}

label_names.update(special_names)

# Config for plotting
layout_cfg = load_layout_config(plotly_layout_file)
traces_cfg = load_traces_config(plotly_traces_file)

common_line_args = dict(
    facet_col = 'utility_gamma', 
    facet_row = 'social_content',
    facet_col_spacing=0.03, 
    facet_row_spacing=0.15, 
    labels=label_names
)

# Description 
desc_txt = load_description(description_file)

# Create app
app = Dash(__name__)

desc_div = dcc.Markdown(
    desc_txt,
    mathjax=True, 
    className='desc'
)

app.layout = html.Div([
    html.H1('Changed social influence integration with social network'),
    dcc.Markdown('*Note: wait and reload if there is an error*'),
    html.Div([
        html.Div([
            html.P("Network type", className="var-title"), 
            dcc.Dropdown(
                id="net_key",
                options=list(social_net_type_args.keys()),
                value="ER",
                clearable=False
            )
        ]), 
        html.Div([
            html.P("Reward measure", className="var-title"),
            dcc.Dropdown(
                id="reward_key",
                options=sorted(list(col_names['reward'])),
                value="mean_reward",
                clearable=False
            )
        ]),
        html.Div([
            html.P("Exploration measure", className="var-title"),
            dcc.Dropdown(
                id="explore_key",
                options=sorted(list(col_names['explore'])),
                value="explore_num",
                clearable=False
            )
        ])
        ],
        className = "var-section"
    ),
    dcc.Graph(id="reward_graph", mathjax=True),
    dcc.Graph(id="explore_graph", mathjax=True),
    desc_div, 
])


def filter_df(quant_key, net_key):
    df_plt = dfs[quant_key].fillna(0).query('`social_net_args::type` == "None" or `social_net_args::type` == @net_key')
    color_key = social_net_type_args[net_key]
    num_unq_colors = len(df_plt[color_key].unique())
    cmap = create_cmap('RdYlBu_r', num_unq_colors, scale=0.9, alpha=0.8)
    return df_plt, color_key, cmap 

def update_fig(fig):
    
    for a in fig.layout.annotations:
        a_key, a_val = a.text.split("=")
        if a_key.startswith('$') and a_key.endswith('$'):
            if type(a_val) is str: 
                a_val = '\\textbf{%s}' %(a_val)
            a.text = f'${a_key[1:-1]} = {a_val}$'


    fig.update_layout(layout_cfg)
    fig.update_traces(traces_cfg)

@app.callback(
    Output("reward_graph", "figure"), 
    [
        Input("net_key", "value"), 
        Input("reward_key", "value")
    ]
)
def update_reward(net_k, reward_k):
    df_plt, color_key, cmap = filter_df('reward', net_k)

    fig = px.line(
        df_plt,
        x = 'time', 
        y = reward_k, 
        title = "Reward graph: " + reward_k, 
        color = color_key,
        markers = True,
        color_discrete_sequence=cmap,
        **common_line_args
    )

    update_fig(fig)

    fig.for_each_xaxis(
        lambda axis:
            axis.update(
                tickmode = 'array',
                tickvals = [400, 800, 1200],
                ticktext = ['Child', 'Adol', 'Adult']
            )
    )

    return fig


@app.callback(
    Output("explore_graph", "figure"), 
    [
        Input("net_key", "value"), 
        Input("explore_key", "value")
    ]
)
def update_explore(net_k, explore_k):
    df_plt, color_key, cmap = filter_df('explore', net_k)

    fig = px.line(
        df_plt,
        x = 'time', 
        y = explore_k,
        title = "Exploration graph: " + explore_k, 
        color = color_key, 
        color_discrete_sequence=cmap,
        **common_line_args
    )
    
    for x in [400, 800]:
        fig.add_vline(
            x=x, 
            line_width=2, 
            line_dash="dot", 
            line_color="gray",
            opacity=0.5
        )

    update_fig(fig)
    
    return fig

app.run_server(debug=True)

