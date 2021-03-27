# Scoring du défaut de paiement

Ce projet s'appuie sur les données Home Credit disponibles sur la plateforme Kaggle à l'adresse https://www.kaggle.com/c/home-credit-default-risk/data.  
Il consiste à prédire les défauts de paiement d'un organisme de crédit. Puis de rendre les décision du modèle compréhensibles.  
L'exploration et le traitement des données sont inspirés des notebooks Kaggle suivants :
* https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
* https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features


Le modèle est un classifieur LightGBM dont les paramètres sont optimisés par optimisation bayésienne.   
Il est disponible sous forme d'API à l'adresse https://home-credit-app-sp.herokuapp.com/.


Le modèle est rendu interprétable via le dashboard dont une version en ligne est accessible à l'adresse https://home-credit-dashboard-sp.herokuapp.com/.  


Les premier notebook contient une exploration des données.  
Le deuxième et le troisième contiennent la préparation des données, y compris leur imputation.  
Le quatrième permet l'équilibrage de la cible pour un entraînement plus efficace. 
Le cinquième contient l'optimisation bayésienne des paramètres d'un classifieur LightGBM.  
Le sixième compare quelques modèles ou méthodes d'entraînement.  
Le septième contient l'entraînement et l'enregistrement du modèle final.  
Dans le huitième, les Shapley values sont calculées pour pouvoir interpréter les décisions du modèles.  

Les codes du dashboard et de l'API sont également disponibles dans le dossier nommé web.

