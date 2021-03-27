import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np



# récupération des données, des informations et du modèle
data = pd.read_csv("../data/test_df.csv", index_col="SK_ID_CURR")
data.drop(columns=["index", "TARGET"], inplace=True)
data = data.replace(to_replace=np.nan, value=0)

model = pickle.load(open("../data/pickle_lgbm_classifier.pkl", "rb"))
cat_features = pickle.load(open("../data/pickle_cat_features.pkl", "rb"))
categorical_names = pickle.load(open("../data/pickle_categorical_names.pkl", "rb"))

shap_values = pickle.load(open("../data/pickle_shap_values.pkl", 'rb'))

# prédictions du modèle
prob = model.predict_proba(data, num_iteration=model.best_iteration_)[:,1]
pred = [0 if i<=0.19 else 1 for i in prob]

# graphiques
# camembert des prédictions
labels = ["Accepté","Refusé"]
values = pd.Series(pred).value_counts()

fig_pie_credit = go.Figure(data=[go.Pie(
    labels =labels, 
    values = values,
    pull=[0, 0.2]
)])
fig_pie_credit.update_traces(marker=dict(colors=["#0ecf10", "#f90531"]))

# histogramme de l'importance des features dans le modèle
df_features_importance = pd.DataFrame(model.feature_importances_, columns=["importance"])
df_features_importance["feature"] = model.feature_name_
df_features_importance = df_features_importance.sort_values(by="importance", ascending=False)
x_hist = list(df_features_importance.iloc[:,0])
x_displayed = x_hist[:5]
x_displayed.reverse()
y_hist = list(df_features_importance.iloc[:,1])
y_displayed = y_hist[:5]
y_displayed.reverse()
hist_colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]
fig_features_importance = go.Figure(data=[go.Bar(x=x_displayed,
                                                 y=y_displayed,
                                                 orientation='h',
                                                 marker_color=hist_colors
                                                )
                                         ])

# histogramme des shapley values
df_shap_importance = pd.DataFrame(data.columns, columns=["feature"])
df_shap_importance["importance"] = np.abs(shap_values[1]).sum(axis=0)/df_shap_importance.shape[0]
df_shap_importance = df_shap_importance.sort_values(by="importance", ascending=False)
x_shap = list(df_shap_importance.iloc[:, 1])
y_shap = list(df_shap_importance.iloc[:, 0])
x_displayed = x_shap[:5]
x_displayed.reverse()
y_displayed = y_shap[:5]
y_displayed.reverse()
fig_shap_importance = go.Figure(data=go.Bar(x=x_displayed,
                                            y=y_displayed,
                                            orientation='h',
                                            marker_color=hist_colors))

hist_colors.reverse()
hist_colors.append("#000000")


# dataframe avec les attributs catégoriels renommés 
df_filter_cat = pd.DataFrame()
for feat in cat_features:
    dict_map = {}
    names = categorical_names[feat]
    for i in range(len(names)):
        dict_map[i] = names[i]
    df_filter_cat[feat] = data[feat].replace(to_replace=dict_map)
df_filter_cat.index = data.index

# liste des attributs numériques
num_features = list(set(data.columns) - set(cat_features))

# premiers filtres
filter_1 = "CODE_GENDER"
filter_2 = "OCCUPATION_TYPE"
features_filter_1 = cat_features
values_filter_1 = ["Indifférent"] + sorted(list(df_filter_cat[filter_1].unique()))
features_filter_2 = cat_features
values_filter_2 = ["Indifférent"] + sorted(list(df_filter_cat[filter_2].unique()))
filter_3 = "DAYS_BIRTH"
filter_4 = "PAYMENT_RATE"
features_filter_3 = num_features
extreams_filter_3 = (min(data[filter_3].to_list()), max(data[filter_3].to_list()))
marks_filter_3 = {}
for i in np.linspace(extreams_filter_3[0], extreams_filter_3[1], 11):
    marks_filter_3[int(i)] = str(int(i))
features_filter_4 = num_features
extreams_filter_4 = (round(min(data[filter_4].to_list()),2), round(max(data[filter_4].to_list()),2))
marks_filter_4 = {}
for i in np.linspace(extreams_filter_4[0], extreams_filter_4[1], 11):
    marks_filter_4[round(i,2)] = str(round(i,2))


# feuille de style en ligne
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# application dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# premier onglet
tab_1 = html.Div(children=[
    html.Div(children=[
        html.H2(
            children="Pour chaque client un score est calculé.\
                Celui-ci correspond au risque de défaut de paiement. \
                Si ce score est supérieur à 19, le prêt est refusé",
            className="four columns"
        ),
        html.Div(children=[
            html.H3(
                children="Score moyen",
                className="centered_title"
            ),
            html.P(
                id = "mean_score",
                children = round(100*prob.mean(),1)
            )
        ], className="four columns"),
        html.Div(children=[
            html.H3(
                children="Proportion de prêts refusés",
                className="centered_title"
            ),
            html.Div(
                dcc.Graph(
                    id='pie_credits',
                    figure=fig_pie_credit,
                    responsive=True
                ),
                id = "pie_chart_container"
            )
        ], className="four columns")
    ], id="row"),
    html.Div(children=[       
        html.Div(children=[
            dcc.RadioItems(id="importance_choice",
                options=[
                    {"label": "Importances dans les décisions du modèle", "value": "lgbm"},
                    {'label': 'Importances dans les différences à une valeur de référence', 'value': "shap"}
                ],
                value="lgbm"
        ),
        html.Br(),
            html.H3(
                id="importance_title",
                children="Attributs intervenant le plus souvent dans les décisions du modèle"
            ),
            html.Div(
                children=dcc.Graph(
                    id="features_importance",
                    figure=fig_features_importance
                ),
                id="importance_container"
            )
        ], className="five columns", id="bottom_row1"),
        html.Div(children=[
            html.H3(children="Répartition des prêts refusés par attribut"),
            html.Div(children=[
                html.Label(children="Choisissez un attribut :", className = "four columns"),
                dcc.Dropdown(
                id="feature_choice",
                options=[{"label":y_hist[i], "value":i} for i in range(len(y_hist))],
                value=0,
                className="three columns")
            ], className="row"),
            html.Div(
                children=dcc.Graph(id="histogram"),
                id="histogram_container"
            ),
        ], className="seven columns", id="bottom_row2")
    ], className="row")
])



# affichage du diagramme en fonction du bouton radio
@app.callback(
    Output("features_importance", "figure"),
    Output("feature_choice", "options"),
    Output("importance_title", "children"),
    Input("importance_choice", "value")
)
def display_importance(value):
    if value=="lgbm":
        fig = fig_features_importance
        opt = [{"label":y_hist[i], "value":i} for i in range(len(y_hist))]
        title = "Attributs intervenant le plus souvent dans les décisions du modèle"
    else:
        fig = fig_shap_importance
        opt = [{"label":y_shap[i], "value":i} for i in range(len(y_shap))]
        title = "Attributs expliquant le plus les écarts à une valeur de référence"
    return fig, opt, title

# affichage de l'histogramme en fonction de la feature choisie
@app.callback(
    Output("histogram", "figure"),
    Input("feature_choice", "value"),
    Input("importance_choice", "value")
)
def display_histogram(feature_index, importance_choice_value):
    idx = int(feature_index)
    if importance_choice_value == "lgbm":
        feat = y_hist[idx]
        importance = df_features_importance.loc[df_features_importance["feature"]==feat, "importance"].values[0]
    else:
        feat = y_shap[idx]
        importance = df_shap_importance.loc[df_shap_importance["feature"]==feat, "importance"].values[0]
    feature_data = pd.DataFrame(data[feat])
    feature_data["TARGET"] = pred
    if feat in cat_features: # si la varaible est catégorielle
        grouped_data = feature_data.groupby(feat).mean()
        x = categorical_names[feat]
    else :
        min_feature = min(feature_data[feat])
        max_feature = max(feature_data[feat])
        feature_data["BINNED"] = pd.cut(feature_data[feat], bins=np.linspace(min_feature,max_feature,num=11))
        grouped_data = feature_data.groupby("BINNED").mean()
        x = grouped_data.index.astype(str)


    fig = go.Figure(data=go.Bar(x=x, 
                                y=100*grouped_data["TARGET"], 
                                marker_color=hist_colors[min(idx,5)]
                            ),
                    layout={"xaxis": {"title": f"{feat}\nimportance : {importance}"}, "yaxis": {"title": "% défauts de paiement"}}
                )
    
    return fig





# deuxième onglet
tab_2 = html.Div(children=[
    html.Div(children=[
        html.Div(children=[
            html.Div([
                html.H3(children=f"Client :", className="six columns"),
                dcc.Dropdown(id="customers", className="sic columns")
            ], className="row"),
            html.H4(id="default_proba"),
            html.H1(id="credit_agrement")
        ], className="four columns"),
        html.Div(children=[
            html.H3(children="Filtres"),
            html.Div(children=[
                html.Label(children="Accord de prêt :", className="three columns"),
                dcc.Dropdown(id="filter_agrement",
                    options=[
                        {"label": "Indifférent", "value": -1},
                        {'label': 'Accepté', 'value': 0},
                        {'label': 'Refusé', 'value': 1},
                    ],
                    value=-1,
                    className="nine columns"
                )                 
            ], className="row"),
            html.Div(children=[
                dcc.Dropdown(
                    id="features_filter_1",
                    options=[{"label": i, "value": i} for i in features_filter_1], 
                    value=filter_1,
                    className="three columns"
                ),
                dcc.Dropdown(id="values_filter_1",
                    options=[{"label": i, "value": i} for i in values_filter_1],
                    value="Indifférent",
                    className="nine columns"
                )                 
            ], className="row"),
            html.Div(children=[
                dcc.Dropdown(
                    id="features_filter_2",
                    options=[{"label": i, "value": i} for i in features_filter_2], 
                    value=filter_2,
                    className="three columns"
                ),
                dcc.Dropdown(id="values_filter_2",
                    options=[{"label": i, "value": i} for i in values_filter_2],
                    value="Indifférent",
                    className="nine columns"
                )                 
            ], className="row"),
            html.Div(children=[
                dcc.Dropdown(
                    id="features_filter_3",
                    options=[{"label": i, "value": i} for i in features_filter_3], 
                    value=filter_3,
                    className="three columns"
                ),
                dcc.RangeSlider(id="values_filter_3",
                    min=extreams_filter_3[0],
                    max=extreams_filter_3[1],
                    step=None,
                    marks=marks_filter_3,
                    value=list(extreams_filter_3),
                    className="nine columns"
                )                 
            ], className="row"),
            html.Div(children=[
                dcc.Dropdown(
                    id="features_filter_4",
                    options=[{"label": i, "value": i} for i in features_filter_4], 
                    value=filter_4,
                    className="three columns"
                ),
                dcc.RangeSlider(id="values_filter_4",
                    min=extreams_filter_4[0],
                    max=extreams_filter_4[1],
                    step=None,
                    marks=marks_filter_4,
                    value=list(extreams_filter_4),
                    className="nine columns"
                )                 
            ], className="row")
        ], className="eight columns")
    ], className="row"),
    html.Div(children=[
        html.Div([
            html.H3(children="Participation des attributs dans le risque de défaut de paiement"),
            html.Div(id="shap_figure_container", children=[            
                dcc.Graph(id="shap_waterfall")
            ])
        ], className="nine columns"),
        html.Div(children=[
            html.H3(children="Valeurs et importance des attributs"),
            html.P(id="display_customer"),
            dcc.Dropdown(id="value_importance_feature"),
            html.P(id="value_display"),
            html.P(id="importance_display")
        ], className="three columns")
    ], className="row")
])

# valeurs du premier filtre (hormis accord de prêt)
@app.callback(
    Output("values_filter_1", "options"),
    Output("values_filter_1", "value"),
    Input("features_filter_1", "value")
)
def change_feature_1(feat):
    values = ["Indifférent"] + sorted(list(df_filter_cat[feat].unique()))
    opt = [{"label": i, "value": i} for i in values]
    return opt, "Indifférent"

# valeurs du deuxième filtre (hormis accord de prêt)
@app.callback(
    Output("values_filter_2", "options"),
    Output("values_filter_2", "value"),
    Input("features_filter_2", "value")
)
def change_feature_2(feat):
    values = ["Indifférent"] + sorted(list(df_filter_cat[feat].unique()))
    opt = [{"label": i, "value": i} for i in values]
    return opt, "Indifférent"

# valeurs du troisième filtre (hormis accord de prêt)
@app.callback(
    Output("values_filter_3", "min"),
    Output("values_filter_3", "max"),
    Output("values_filter_3", "marks"),
    Output("values_filter_3", "value"),
    Input("features_filter_3", "value")
)
def change_feature_3(feat):
    if data[feat].dtype=="int64":
        extreams = (min(data[feat].to_list()), max(data[feat].to_list()))
        marks = {}
        for i in np.linspace(extreams[0], extreams[1], 11):
            marks[int(i)] = str(int(i))
    else:
        extreams = (round(min(data[feat].to_list()),2), round(max(data[feat].to_list()),2))
        marks = {}
        for i in np.linspace(extreams[0], extreams[1], 11):
            marks[round(i,2)] = str(round(i,2))
    return extreams[0], extreams[1], marks, list(extreams)

# valeurs du quatrième filtre (hormis accord de prêt)
@app.callback(
    Output("values_filter_4", "min"),
    Output("values_filter_4", "max"),
    Output("values_filter_4", "marks"),
    Output("values_filter_4", "value"),
    Input("features_filter_4", "value")
)
def change_feature_4(feat):
    if data[feat].dtype=="int64":
        extreams = (min(data[feat].to_list()), max(data[feat].to_list()))
        marks = {}
        for i in np.linspace(extreams[0], extreams[1], 11):
            marks[int(i)] = str(int(i))
    else:
        extreams = (round(min(data[feat].to_list()),2), round(max(data[feat].to_list()),2))
        marks = {}
        for i in np.linspace(extreams[0], extreams[1], 11):
            marks[round(i,2)] = str(round(i,2))
    return extreams[0], extreams[1], marks, list(extreams)


# sélection et tri des shapley value du client examiné
@app.callback(
    Output("intermediate_value", "children"),
    Input("customers", "value")
)
def calculate_shap(id_customer):
    idx_customer = list(data.index).index(id_customer)
    df_shap = pd.DataFrame(shap_values[1][idx_customer], columns=["shap_value"])
    df_shap["feature"] = data.columns
    df_shap = df_shap.sort_values(by="shap_value", key=abs, ascending=False)
    return df_shap.to_json(date_format="iso", orient="split")

# affichage du diagramme, du score et des importances
@app.callback(
    Output("default_proba", "children"),
    Output("credit_agrement", "children"), 
    Output("credit_agrement", "style"),
    Output("shap_waterfall", "figure"),
    Output("value_importance_feature", "options"),
    Output("value_importance_feature", "value"),
    Input("customers", "value"),
    Input("intermediate_value", "children")
)
def display_score(id_customer, df_intermediate):
    proba = model.predict_proba(np.array(data.loc[id_customer]).reshape(1, -1), num_iteration=model.best_iteration_)
    score = round(proba[0][1]*100,1)
    if score < 19 :
        agrement = "Prêt accepté"
        style = {"color":"green"}
    else :
        agrement = "Prêt refusé"
        style = {"color":"red"}
    # diagramme waterfall des shapley values
    df_shap = pd.read_json(df_intermediate, orient="split")
    y_shap = list(df_shap.iloc[:5,0])
    y_shap.append(df_shap.iloc[5:,0].sum())
    y_shap.append(0)
    y_shap.insert(0, 0)
    x_shap= [" "]
    for i in range(5):
        feat = df_shap.iloc[i,1]
        if feat in cat_features:
            val = categorical_names[feat][int(data.loc[id_customer,feat])]
        else :
            val = round(data.loc[id_customer, feat],3)
        x_shap.append(f"{feat} : {val}")
    x_shap.append("SUM OF OTHER FEATURES")
    x_shap.append("Écart total")
    fig_waterfall = go.Figure(data=go.Waterfall(
        name=f"Client {id_customer}",
        orientation='v',
        x=x_shap,
        y=y_shap,
        measure=["relative"]*7 + ["total"],
        text=["Risque de référence", " ", " ", " ", " ", " ", " ", f"Risque du client {id_customer}"],
        textposition="outside",
        decreasing = {"marker":{"color":"Green"}},
        increasing = {"marker":{"color":"Red"}}
    ),
    layout=go.Layout(yaxis=dict(
        title="Risque de défaut de paiement",
        showticklabels=False)
    ))
    fig_waterfall.add_trace(go.Scatter(
        name="Limite d'accord",
        x=[" ", "Écart total"],
        y=[-3.46, -3.46]
    ))

    opt = [{"label": i, "value": i} for i in df_shap["feature"]]

    return (f"Risque de défaut de paiement :  {score}", 
            agrement,
            style,
            fig_waterfall,
            opt,
            df_shap["feature"].iloc[5])

# affichage de l'importance lors d'un changement de feature
@app.callback(
    Output("display_customer", "children"),
    Output("value_display", "children"),
    Output("importance_display", "children"),
    Input("customers", "value"),
    Input("value_importance_feature", "value"),
    Input("intermediate_value", "children")
)
def display_value_importance(id_cust, feat, df_intermediate):
    df_shap = pd.read_json(df_intermediate, orient="split")
    val = data.loc[id_cust, feat]
    if feat in cat_features:
        val = categorical_names[feat][int(val)]
    imp = df_shap.loc[df_shap["feature"]==feat, "shap_value"].values[0]
    return (
        f"Pour le client {id_cust}, l'attribut",
        f"a pour valeur {val}", 
        f"et pour importance {imp}"
        )


# remplissage de la liste des clients disponibles en fonction des filtres
@app.callback(
    Output("customers", "options"),
    Output("customers", "value"),
    Input("filter_agrement", "value"),
    Input("features_filter_1", "value"),
    Input("values_filter_1", "value"),
    Input("features_filter_2", "value"),
    Input("values_filter_2", "value"),
    Input("features_filter_3", "value"),
    Input("values_filter_3", "value"),
    Input("features_filter_4", "value"),
    Input("values_filter_4", "value")
)
def filter_customers(
    value_agrement, 
    feature_filter_1, value_filter_1, 
    feature_filter_2, value_filter_2, 
    feature_filter_3, value_filter_3, 
    feature_filter_4, value_filter_4
    ):
    
    if value_agrement == -1 :
        df = data.copy()
    elif value_agrement == 0 :
        mask = (np.where(np.array(pred)==0))
        df = data.iloc[mask]
    else : 
        mask = (np.where(np.array(pred)==1))
        df = data.iloc[mask]
    
    if value_filter_1 != "Indifférent":
        df_temp = df_filter_cat.loc[df.index]
        df = df.loc[df_temp[feature_filter_1]==value_filter_1]
    
    if value_filter_2 != "Indifférent":
        df_temp = df_filter_cat.loc[df.index]
        df = df.loc[df_temp[feature_filter_2]==value_filter_2]

    df = df.loc[(df[feature_filter_3]>=float(value_filter_3[0])) & (df[feature_filter_3]<=float(value_filter_3[1]))]

    df = df.loc[(df[feature_filter_4]>=float(value_filter_4[0])) & (df[feature_filter_4]<=float(value_filter_4[1]))]

    list_cust = [{"label":i, "value":i} for i in list(df.index)]
    first_cust = df.index[0]
    return list_cust, first_cust


# application principale
app.layout = html.Div(children=[
    html.Div(id="intermediate_value", style={'display': 'none'}),
    html.Div(children= [
        html.H1(children='Home Credit')
    ], className="title"),
    dcc.Tabs([
        dcc.Tab(
            label="Généralités du modèle",
            children=tab_1
        ),
        dcc.Tab(
            label="Prédiction individuelle",
            children=tab_2
        )
        
    ])
])


# lancement de l'application
if __name__ == '__main__':
    app.run_server(debug=False)