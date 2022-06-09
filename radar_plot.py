# %%
import pandas as pd
import numpy as np
import requests


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics.pairwise import cosine_similarity, paired_distances

import matplotlib.pyplot as plt
from plotly import tools
from plotly.subplots import make_subplots
import plotly as py
import plotly.express as px
import plotly.graph_objs as go

import ipywidgets as widgets
py.offline.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings ('ignore')

import dash
from dash import Dash, dcc, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from jupyter_dash import JupyterDash

# %%
data_Wy = pd.read_csv('roots_data.csv')

# %%
data_FB = data_Wy[data_Wy['FB'] == 'FB']
data_FB = data_FB[['Goals','Non-penalty goals','Assists','Key passes per 90','Successful defensive actions per 90',
                   'Duels per 90','Duels won, %','Shots blocked per 90','Interceptions per 90','Crosses per 90',
                   'Accurate crosses, %','Crosses to goalie box per 90','Received passes per 90','Passes per 90', 
                   'Accurate passes, %','Short / medium passes per 90','Accurate short / medium passes, %', 'Accurate long passes, %',
                   'Shot assists per 90','Passes to final third per 90','Accurate passes to final third, %','Rating']]
data_CB = data_Wy[data_Wy['CB'] == 'CB']
data_CB = data_CB[['Goals','Non-penalty goals','Assists','Key passes per 90','Successful defensive actions per 90',
                   'Duels per 90','Duels won, %','Shots blocked per 90','Interceptions per 90','Received passes per 90',
                   'Passes per 90','Accurate passes, %','Short / medium passes per 90','Accurate short / medium passes, %',
                   'Accurate long passes, %','Passes to final third per 90','Accurate passes to final third, %',
                   'Dribbles per 90','Successful dribbles, %','Through passes per 90','Accurate through passes, %','Rating']]
data_CM = data_Wy[data_Wy['CM'] == 'CM']
data_CM = data_CM[['Goals','Non-penalty goals','Assists','Duels per 90','Duels won, %','Aerial duels per 90',
                   'Touches in box per 90','Sliding tackles per 90','Interceptions per 90','Received passes per 90',
                   'Passes per 90','Accurate passes, %','Short / medium passes per 90','Accurate short / medium passes, %',
                   'Average pass length, m','Shot assists per 90','Passes to final third per 90','Accurate passes to final third, %',
                   'PAdj Interceptions','Smart passes per 90','Accurate smart passes, %','Rating']]
data_AM = data_Wy[data_Wy['AM'] == 'AM']
data_AM = data_AM[['Goals','Non-penalty goals','Assists','Duels per 90','Duels won, %','Key passes per 90',
                   'Touches in box per 90','Successful attacking actions per 90','Offensive duels won, %','Received passes per 90',
                   'Passes per 90','Accurate passes, %','Short / medium passes per 90','Accurate short / medium passes, %',
                   'Forward passes per 90','Accurate forward passes, %','Average pass length, m','Shot assists per 90',
                   'Second assists per 90','Third assists per 90','Passes to final third per 90',
                   'Accurate passes to final third, %','Passes to penalty area per 90',
                   'Accurate passes to penalty area, %','Rating']]
data_W = data_Wy[data_Wy['W'] == 'W']
data_W = data_W[['Goals','Non-penalty goals','Assists','Duels per 90','Duels won, %','Shots on target, %',
                 'Successful attacking actions per 90','Goal conversion, %','Crosses per 90',
                 'Accurate crosses, %','Crosses to goalie box per 90','Offensive duels won, %','Received passes per 90',
                 'Passes per 90','Accurate passes, %','Short / medium passes per 90','Accurate short / medium passes, %', 
                 'Shot assists per 90','Key passes per 90','Passes to penalty area per 90','Accurate passes to penalty area, %',
                 'Dribbles per 90','Successful dribbles, %','Rating']]
data_CF = data_Wy[data_Wy['CF'] == 'CF']
data_CF = data_CF[['Goals','Non-penalty goals','Assists','Aerial duels per 90',
                   'Aerial duels won, %','Duels per 90','Duels won, %','Successful attacking actions per 90',
                   'Shots on target, %','Goal conversion, %','Offensive duels won, %','Touches in box per 90',
                   'Received passes per 90','Passes per 90','Accurate passes, %','Short / medium passes per 90',
                   'Accurate short / medium passes, %','Shot assists per 90','Key passes per 90','Rating']]
data_GK = data_Wy[data_Wy['GK'] == 'GK']
data_GK = data_GK[['Short / medium passes per 90','Accurate short / medium passes, %','Long passes per 90',
                   'Accurate long passes, %','Conceded goals per 90','Shots against','Clean sheets','Save rate, %', 'Rating']]

# %%
def RandomForest(data):
    
    data_copy = data.copy()
    data_copy["Rating"] = data_copy["Rating"].astype("str")
    data_copy = data_copy.drop(data_copy[data_copy['Rating']=='-'].index)
    Y = data_copy.iloc[:, [-1]]
    Y.reset_index(drop=True,inplace=True)
    X = data_copy.iloc[:, 0:-1]
    X.reset_index(drop=True,inplace=True)

    # sort column index with number of nan value from smallest to largest
    sortindex = np.argsort(X.isnull().sum(axis=0)).values
    
    for i in sortindex:
        # set i column as target 
        feature_i = X.iloc[:, i]
    
        # set other columns as features, including 'y'
        tmp_df = pd.concat([X.iloc[:, X.columns != i], Y], axis=1)
    
        # Fill remaining column missing values with 0
        imp_mf = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        tmp_df_mf = imp_mf.fit_transform(tmp_df)

        # Use notnull samples in feature_i as training dataset
        y_notnull = feature_i[feature_i.notnull()]
        y_null = feature_i[feature_i.isnull()]   
        X_notnull = tmp_df_mf[y_notnull.index, :]
        X_null = tmp_df_mf[y_null.index, :] 
    
        # continue if this column has no nan value
        if y_null.shape[0] == 0:
            continue

        # RF Regression 
        rfc = RandomForestRegressor(n_estimators=100)
        rfc = rfc.fit(X_notnull, y_notnull)
    
        # predict nan value
        y_predict = rfc.predict(X_null)
    
        # fill
        X.loc[X.iloc[:, i].isnull(), X.columns[i]] = y_predict
        data_copy.loc[data_copy.iloc[:, i].isnull(), data_copy.columns[i]] = y_predict
    
    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    
    global selected_features, feat_lables
    
    feat_lables=data_copy.columns[1:]
    
    forest = RandomForestClassifier(oob_score=True, n_estimators=100, random_state=100, n_jobs=-1)
    forest.fit(x_train, y_train)
    
    # select features which threshold larger then 0.025
    selector = SelectFromModel(forest, threshold=0.025)
    features_important = selector.fit_transform(x_train, y_train)
    model = forest.fit(features_important, y_train)
    selected_features = model.feature_importances_
    
    # print features
    sorted_idx = np.argsort(selected_features)[::-1]
    
    for f in range(len(sorted_idx)):
        print("%2d) %-*s %f"%(f + 1,30,feat_lables[sorted_idx[f]],selected_features[sorted_idx [f]]))    
    

    data0 = pd.DataFrame()
    
    for i in range(len(sorted_idx)):
        data0.insert(i, feat_lables[sorted_idx[i]], data_copy[feat_lables[sorted_idx[i]]], True)
    
    if 'Rating' in data0.columns:
        data0.drop(['Rating'], axis=1, inplace=True)
    else:
        pass
    
    return data0

# %%
position_list = [data_FB, data_CB, data_CM, data_AM, data_W, data_CF, data_GK]
position_name = ['FB', 'CB', 'CM', 'AM', 'W', 'CF', 'GK']
model_name = [0, 1, 2, 3, 4, 5, 6]
num = 0
for i in position_list:
    print('For position', position_name[num])
    model_name[num] = RandomForest(i)
    num += 1
    
data_FB_Selected = model_name[0]
data_CB_Selected = model_name[1]
data_CM_Selected = model_name[2]
data_AM_Selected = model_name[3]
data_W_Selected = model_name[4]
data_CF_Selected = model_name[5]
data_GK_Selected = model_name[6]

# %%
position_selected_list = [data_FB_Selected, data_CB_Selected, data_CM_Selected, data_AM_Selected, data_W_Selected, data_CF_Selected, data_GK_Selected]
position_name = ['FB', 'CB', 'CM', 'AM', 'W', 'CF', 'GK']
playname_list = [1, 2, 3, 4, 5, 6, 7]


num = 0
for i in position_selected_list:
    player_name = []
    for j in i.index:
        player_name.append(data_Wy['Player'][j])
    playname_list[num] = player_name
    num += 1
    
Name_FB = playname_list[0]
Name_CB = playname_list[1]
Name_CM = playname_list[2]
Name_AM = playname_list[3]
Name_W = playname_list[4]
Name_CF = playname_list[5]
Name_GK = playname_list[6]

# %%
app = dash.Dash()

server = app.server
app.layout = html.Div([
    dcc.Markdown('''
        #### Introduction
        This engine applies to the player ability comparison of `Data_WyScout_Rating_2021.csv`.\n
        Before using it, you need to select the **type of position** you want to compare first.\n
        Next, **select the names of the two players separately**. This engine supports search function.\n
        Finally, click the **submit button** to generate the comparison chart.\n
        The engine generates interactive images, which can be interacted with by mouse clicks.\n
    '''),
    
    html.Div([
        "Position",
        dcc.Dropdown(id="position_select",
                     options = position_name,
                     placeholder = 'First step: select the position'
                    ),
    ]),
    html.Div([
        "First player",
        dcc.Dropdown(id="player1_select",
                    placeholder = 'Second step: select the first player'
                    ),
    ]),
    html.Div([
        "Second player",
        dcc.Dropdown(id="player2_select",
                    placeholder = 'Third step: select the second player'
                    ),
    ]),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',
             children='Complete the above three steps and press submit'),
    dcc.Graph(id='radar_plot')
])

@app.callback(
    [Output("player1_select", "options"),
     Output("player2_select", "options")],
    Input("position_select", "value")
)
def player_update(value):
    if not value:
        raise PreventUpdate
    elif value == 'FB':
        return Name_FB, Name_FB
    elif value == 'CB':
        return Name_CB, Name_CB
    elif value == 'CM':
        return Name_CM, Name_CM
    elif value == 'AM':
        return Name_AM, Name_AM
    elif value == 'W':
        return Name_W, Name_W
    elif value == 'CF':
        return Name_CF, Name_CF
    elif value == 'GK':
        return Name_GK, Name_GK
    
@app.callback(
    [Output("radar_plot", "figure"),
    Output("container-button-basic", "children"),
    Output("submit-val", "n_clicks")],
    [Input("player1_select", "value"),
     Input("player2_select", "value"),
     Input("position_select", "value"),
    Input("submit-val", "n_clicks")]
)
def Graph_update(Player1, Player2, Position, switch):
    if switch == 0:
        raise PreventUpdate
    elif not Player1:
        raise PreventUpdate
    elif not Player2:
        raise PreventUpdate
    elif not Position:
        raise PreventUpdate
    else:
        pass
    
    if Position == 'FB':
        Position_data = data_FB_Selected
    elif Position == 'CB':
        Position_data = data_CB_Selected
    elif Position == 'CM':
        Position_data = data_CM_Selected
    elif Position == 'AM':
        Position_data = data_AM_Selected
    elif Position == 'W':
        Position_data = data_W_Selected
    elif Position == 'CF':
        Position_data = data_CF_Selected
    elif Position == 'GK':
        Position_data = data_GK_Selected
    
    data_copy = Position_data.copy()
    
    #Fill back players' name after feature selection
    player_name = []
    for i in Position_data.index:
        player_name.append(data_Wy['Player'][i])
    data_copy.loc[:, 'Player'] = player_name
    
    #Positioning input player data and merge them into one dataframe in order to prepare for plot
    player1_data = data_copy[data_copy['Player'] == Player1].iloc[0:1,: -1]
    player2_data = data_copy[data_copy['Player'] == Player2].iloc[0:1,: -1]
    plot_data = pd.concat([player1_data, player2_data])
    
    #Radar plot
    fig = make_subplots(rows=1, cols=2,specs=[[{"type": "Polar"},{"type": "Polar"}]])

    R1=[]
    theta1=[]
    R2=[]
    theta2=[]
    R3=[]
    theta3=[]
    R4=[]
    theta4=[]
    
    for i in plot_data.columns:
        if max(plot_data[i]) < 20:
            R1.append(plot_data[i].iloc[0])
            theta1.append(i)
            R2.append(plot_data[i].iloc[1])
            theta2.append(i)
        else:   
            R3.append(plot_data[i].iloc[0])
            theta3.append(i)
            R4.append(plot_data[i].iloc[1])
            theta4.append(i)
            
    fig.add_trace(go.Scatterpolar(
          r = R1,
          theta = theta1,
          fill = 'toself',
          marker_color='rgb(47,138,196,30)',
          name = Player1),row=1, col=1)
    
    fig.add_trace(go.Scatterpolar(
          r = R2,
          theta = theta2,
          fill = 'toself',
          marker_color='rgb(229,210,245,0.2)',
          name = Player2),row=1, col=1)
    
    fig.add_trace(go.Scatterpolar(
          r = R3,
          theta = theta3,
          fill = 'toself',
          marker_color='rgb(47,138,196,30)',
          name = Player1),row=1, col=2)
    
    fig.add_trace(go.Scatterpolar(
          r = R4,
          theta = theta4,
          fill = 'toself',
          marker_color='rgb(229,210,245,0.2)',
          name = Player2),row=1, col=2)

    fig.layout.update(
        go.Layout(
        polar = dict(
            radialaxis = dict(
                visible = True,)),
        showlegend = True,
        title = "{} vs {}".format(Player1, Player2),
        height=400, width=1200,
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        ))
    
    children = 'Data confirm, graph generated.'
    switch = 0
    return fig, children, switch

if __name__ == '__main__':
    app.run_server(debug=True)

# %%



