from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
import numpy as np
import pickle



# récupération des données
X = pd.read_csv("../data/app_data.csv")

X = X.set_index("SK_ID_CURR")


# récupération du modèle
clf = pickle.load(open("../data/pickle_lgbm_classifier.pkl", "rb"))

# application Flask
app = Flask(__name__)

#affichage de l'interface
@app.route("/", methods=["GET"])
def interface():
    return render_template("interface.html")

# envoi de la probabilité de défaut de paiement à partir de l'URL
@app.route("/score/<id>", methods=["GET"])
def score(id):
    try :
        resp =  clf.predict_proba(np.array(X.loc[int(id)]).reshape(1, -1), num_iteration=clf.best_iteration_)[0,1]
    except KeyError:
        resp = f"Il n'y a pas de client dont l'identifiant est {id}"
    return jsonify(resp)

# envoi de la probabilité de défaut de paiement à partir de la méthode get
@app.route("/return_score/", methods=["GET"])
def return_score():
    id = request.args["id"]
    try :
            resp =  clf.predict_proba(np.array(X.loc[int(id)]).reshape(1, -1), num_iteration=clf.best_iteration_)[0,1]
    except KeyError:
        resp = f"Il n'y a pas de client dont l'identifiant est {id}"
    return jsonify(resp)

# affichage de la probabilité de défaut de paiement
@app.route("/display/", methods=["GET"])
def display():
    id = request.args["id"]
    try :
        prob =  clf.predict_proba(np.array(X.loc[int(id)]).reshape(1, -1), num_iteration=clf.best_iteration_)[0,1]
        resp = f"La probabilité de défaut de paiment du client {id} est {prob}"
    except KeyError:
        resp = f"Il n'y a pas de client dont l'identifiant est {id}"
    return render_template("display.html", response = resp )
    
# lancement de l'application
if __name__ == "__main__":
    app.run(debug=False)
