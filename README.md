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

## Notre vision du projet : 

Nous nous sommes largement inspiré de la source : 
* https://ledatascientist.com/creer-un-modele-de-regression-lineaire-avec-python/

En effet, après quelques recherches sur notre moteur de recherche favori, nous avons choisis d'appliquer un Modèle de Régression linéaire avec Python.
Afin d'appliquer ce modèle, et donc d'avoir une estimation réaliste, nous avons du en amont:

* Une préparation rigoureuse de notre jeu de données
* La création du modèle
* L'évaluation du modèle de régression linéaire

Puis, nous sommes arrivés à un résutlat qui englobe ces divers élémeents : 
--------------------------------------
L'erreur quadratique moyenne est ()
le score R2 est ()
