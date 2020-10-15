# Immothep
Immoteph - Groupe 1 - Orkaëlle & Thomas

![Screenshot](0eface24-8936-437e-89df-1eacf1e8fdbe.png)

## Contexte du projet : 

La société Immothep est une agence immobilière spécialisée dans le vente de biens de particuliers.
Possédant déjà un site internet, elle souhaite pouvoir intégrer à celui-ci, un module d'estimation. Elle possède les ressources nécessaires pour réaliser le code dit "front", ainsi que les ressources graphiques.
Elle ne possède cependant pas les compétences nécessaires pour la réalisation de l'API qui va permettre d'exposer ce nouveau service.
La société vous sollicite donc pour réaliser la partie API en utilisant les données Open Data des Demandes de Valeurs Foncières (DVF) sur l'année 2019.

## Sources utilisées : 

* https://ledatascientist.com/creer-un-modele-de-regression-lineaire-avec-python/
* https://fastapi.tiangolo.com/#installation
* https://blog.paperspace.com/anomaly-detection-isolation-forest/
* https://towardsdatascience.com/the-beginners-guide-to-selecting-machine-learning-predictive-models-in-python-f2eb594e4ddc
* https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/

## Librairies importées : 

* import requests
* import os
* from sklearn.model_selection import train_test_split
* from sklearn.linear_model import LinearRegression
* from sklearn.metrics import mean_squared_error
* from sklearn.metrics import r2_score
* import pandas as pd
* import numpy as np
* import seaborn as sns
* import seaborn as sns
* import matplotlib.pyplot as plt
* from sklearn.ensemble import IsolationForest
* from typing import Optional
* from fastapi import FastAPI

## Notre vision du projet : 

Nous nous sommes largement inspiré de la source : 
* https://ledatascientist.com/creer-un-modele-de-regression-lineaire-avec-python/

En effet, après quelques recherches sur notre moteur de recherche favori, nous avons choisis d'appliquer un Modèle de Régression linéaire avec Python.
Afin d'appliquer ce modèle, et donc d'avoir une estimation réaliste, nous avons du en amont:

* Une préparation rigoureuse de notre jeu de données (nettoyage des divers features, split des fichiers "Code departement")
Un exemple concret d'un des divers nettoayge de notre Dataframe : 

    ```PYTHON
    # Suppression ou mises à zéro des lignes vides
        df.dropna(subset = ["Valeur fonciere"], inplace = True)
        df.dropna(subset = ["Code postal"], inplace = True)
        df.dropna(subset = ["Surface reelle bati"], inplace = True)
        df.dropna(subset = ["Code departement"], inplace = True)
        df['Surface terrain'] = df['Surface terrain'].fillna(0)
    ```

* La création de l'isolation Forest

    ```PYTHON
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
    model.fit(df[['Valeur fonciere','Surface','Nombre pieces principales']])
    
    df['scores']=model.decision_function(df[['Valeur fonciere','Surface','Nombre pieces principales']])
    df['anomaly']=model.predict(df[['Valeur fonciere','Surface','Nombre pieces principales']])
    ```

* Creation matrice correlation
    
    ```PYTHON
    df_matrice = df[["Valeur fonciere",'Prix moyen m2 CP',"Nombre pieces principales","Surface","Surface terrain"]]
    matrice_corr = df_matrice.corr().round(3)
    sns.heatmap(data=matrice_corr, annot=True)
    ```
    
* L'étude de la correlation

![Screenshot](Capture.PNG)

Un ".plot" permet de mieux visualiser nos divers correaltions :

![Screenshot](Capture2.PNG)


* Entrainement de notre modèle via Régression Linéaire

    ```PYTHON
    df_train = df.sample(n=400, random_state=5)
    X = pd.DataFrame(np.c_[df_train["Surface"],df_train["Nombre pieces principales"]], columns= ["Surface","Nombre pieces principales"])
    Y = df_train["Valeur fonciere"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
    lmodellineaire = LinearRegression()
    lmodellineaire.fit(X_train, Y_train)
    ```
    
* Visualisation de nos résultats : 
    
    ```PYTHON
    y_train_predict = lmodellineaire.predict(X_train)
    rmse = round(np.sqrt(mean_squared_error(Y_train, y_train_predict)),2)
    r2 = round(r2_score(Y_train, y_train_predict),4)
    ```

Puis, nous sommes arrivés à un résutlat qui englobe ces divers élémeents : 
--------------------------------------
L'erreur quadratique moyenne est ()
le score R2 est ()

## Visualisation via FASTAPI :

Enfin, grâce à la librairie "FASTAPI", nous sommes parvenu à offrir une véritable experience ludique à l'utilisateur! :-D
En saisissant l'adresse http://127.0.0.1:8000/ l'utilisateur peux directement se conncecter à notre API.

    ```PYTHON
    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"welcome_message": "Bienvenue sur Immothep !"}

    @app.get("/estimation/{cp}/{surface}/{terrain}/{nbpieces}/")
    def api_estimate(cp: int, surface: int, terrain: int, nbpieces: int):
        estimation = oh.estimation(cp, surface, terrain, nbpieces)
        return {'estimation': estimation}
    ```

Exemple : 
http://127.0.0.1:8000/estimation/75017/115/50/5/ :
                                                    ---> estimation : "Notre chemin d'accès
                                                    ---> 75017 : code département (ici arondissement de Paris)
                                                    ---> 115 : Nombre de m²
                                                    ---> 50 : Surface terrain
                                                    ---> 5 : Nombre de pièce

## A vous de jouer ! :-)

Merci d'avoir pris le temps de lire ce README! 
N'hésitez pas à essayer vous même de choisir des valeurs, afin de determiner l'estimation du bien de vos rêves! 


