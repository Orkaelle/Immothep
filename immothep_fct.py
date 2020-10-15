import requests
import os
import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest


DATAS_LOCAL_PATH = './DATAS/'
RAW_LOCAL_PATH = DATAS_LOCAL_PATH + 'RAW/'
TXT_LOCAL_PATH = RAW_LOCAL_PATH + 'valeursfoncieres-2019.txt'
CURATED_LOCAL_PATH = DATAS_LOCAL_PATH + 'CURATED/'
URL = 'https://www.data.gouv.fr/fr/datasets/r/3004168d-bec4-44d9-a781-ef16f41856a2'


def ensure_data_loaded(dpt):
    '''
    Ensure if data are already loaded. Download if missing
    '''
    if os.path.exists(TXT_LOCAL_PATH) == False:
        dl_data()
    else :
        print('Datas already douwnloaded.')
    
    if os.path.exists(f'{CURATED_LOCAL_PATH}{str(dpt).zfill(2)}.csv') == False:
        split_data(dpt)
    else :
        print('Datas already splited.')


def dl_data ():
    print ('Downloading...')
    with open(TXT_LOCAL_PATH, "wb") as f:
        r = requests.get(URL)
        f.write(r.content)
    print ('Dataset dowloaded successfully.')


def check_folder ():
    PATH = [DATAS_LOCAL_PATH, RAW_LOCAL_PATH, CURATED_LOCAL_PATH]
    for p in PATH:
        if not os.path.exists(p):
            os.mkdir(p)


def split_data(dpt):
    '''
    Break raw data into many files
    '''
    COLUMN = 'Code departement'
    VALUE = str(dpt).zfill(2)

    filter = {}
    filter['Code departement'] = (VALUE)

    print (f'Spliting...')
    with open(TXT_LOCAL_PATH, encoding='utf-8') as file_stream:  
        csv.field_size_limit(10000000)  
        file_stream_reader = csv.DictReader(file_stream, delimiter='|')

        open_files_references = {}

        for row in file_stream_reader:
            column = row[COLUMN]

            # if column not in filter[COLUMN]:
            #     continue
        
            # Open a new file and write the header
            if column not in open_files_references:
                output_file = open(CURATED_LOCAL_PATH + f'{column}.csv', 'w', encoding='utf-8', newline='')
                dictionary_writer = csv.DictWriter(output_file, fieldnames=file_stream_reader.fieldnames)
                dictionary_writer.writeheader()
                open_files_references[column] = output_file, dictionary_writer
            # Always write the row
            open_files_references[column][1].writerow(row)
        # Close all the files
        for output_file, _ in open_files_references.values():
            output_file.close()

    print ('Done.')


def nettoyage (dpt):
    '''
    Nettoyage des données
    '''
    fichier = f'{CURATED_LOCAL_PATH}{str(dpt).zfill(2)}.csv'

    # lecture du fichier raw
    raw_df = pd.read_csv(fichier, decimal=',', encoding="UTF-8")

    #création dataframe pandas avec les colonnes souhaitées
    df = raw_df[["Nature mutation","Valeur fonciere","Nombre de lots","Code postal","Code departement","Code type local","Type local","Surface reelle bati","Surface Carrez du 1er lot","Nombre pieces principales","Surface terrain"]]

    # Suppression ou mises à zéro des lignes vides
    df.dropna(subset = ["Valeur fonciere"], inplace = True)
    df.dropna(subset = ["Code postal"], inplace = True)
    df.dropna(subset = ["Surface reelle bati"], inplace = True)
    df.dropna(subset = ["Code departement"], inplace = True)
    df['Surface terrain'] = df['Surface terrain'].fillna(0)

    # Suppression du département Corse pour éviter les str
    df = df.drop(df.loc[(df["Code departement"] == "2A") | (df["Code departement"] == "2B")].index)

    # selection du departement
    df = df[df["Code departement"] == dpt]

    # Conversion en int
    df["Code type local"] = df["Code type local"].astype('int8')
    df["Code departement"] = df["Code departement"].astype('int8')

    # Filrage sur les natures 'Ventes', et sur les 'Maisons' et 'Appartements'
    df = df.loc[(df['Nature mutation'] == 'Vente') & ((df['Type local'] == 'Maison') | (df['Type local'] == 'Appartement'))]

    # Suppression des ventes de lots multiples
    df = df.drop(df.loc[df["Nombre de lots"]> 1].index)

    # Nettoyage des surfaces (suppression incohérence, choix de la plus appropriée, suppression des NA)
    df = df.drop(df.loc[df["Surface Carrez du 1er lot"] > df["Surface reelle bati"]].index)
    df['Surface'] = np.where(df["Code type local"] == 1, df["Surface reelle bati"],df["Surface Carrez du 1er lot"])
    df.dropna(subset = ["Surface"], inplace = True)

    # Creation colonne prix m²
    df_prixmoyen = df.groupby('Code postal')[['Surface','Valeur fonciere']].sum().reset_index()
    df_prixmoyen['Prix moyen m2 CP'] = round(df_prixmoyen['Valeur fonciere'] / df_prixmoyen['Surface'],0)
    df_prixmoyen = df_prixmoyen[['Code postal','Prix moyen m2 CP']]
    df = pd.merge(df, df_prixmoyen, on='Code postal')
    df_prixmoyen.set_index('Code postal', inplace=True)

    return df, df_prixmoyen


def isolation (dpt) :
    '''
    Isolation forest
    '''
    df, df_prixmoyen = nettoyage(dpt)

    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
    model.fit(df[['Valeur fonciere','Surface','Nombre pieces principales','Prix moyen m2 CP']])

    # Ajouter une colonne de scores et d'anomalies
    df['scores']=model.decision_function(df[['Valeur fonciere','Surface','Nombre pieces principales','Prix moyen m2 CP']])
    df['anomaly']=model.predict(df[['Valeur fonciere','Surface','Nombre pieces principales','Prix moyen m2 CP']])

    # Évaluer le modèle
    outliers_counter = len(df[df['Valeur fonciere'] > 699999]) + len(df[df['Valeur fonciere'] < 10000]) + len(df[df['Surface'] > 270]) + len(df[df['Surface'] < 10]) + len(df[df['Nombre pieces principales'] > 6]) + len(df[df['Nombre pieces principales'] < 1])
    outliers_counter

    # Pourcentage de précision
    print("Accuracy percentage:", 100*list(df['anomaly']).count(-1)/(outliers_counter))

    # Suppression des anomalies
    df = df.loc[df['anomaly'] != -1]

    return df


def matrice (dpt):
    '''
    Creation matrice correlation
    '''
    initialisation(dpt)
    df = isolation(dpt)

    df_matrice = df[["Valeur fonciere",'Prix moyen m2 CP',"Nombre pieces principales","Surface","Surface terrain"]]
    matrice_corr = df_matrice.corr().round(3)
    sns.heatmap(data=matrice_corr, annot=True)

    
def train (dpt):
    '''
    Entrainement algo
    '''
    df = isolation(dpt)
    df_train = df.sample(n=3000, random_state=7)

    X = pd.DataFrame(np.c_[df_train["Surface"],df_train["Nombre pieces principales"],df_train['Prix moyen m2 CP'],df_train['Surface terrain']], columns= ["Surface","Nombre pieces principales",'Prix moyen m2 CP',"Surface terrain"])
    Y = df_train["Valeur fonciere"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

    lmodellineaire = LinearRegression()
    lmodellineaire.fit(X_train, Y_train)

    y_train_predict = lmodellineaire.predict(X_train)
    rmse = round(np.sqrt(mean_squared_error(Y_train, y_train_predict)),2)
    r2 = round(r2_score(Y_train, y_train_predict),4)
    
    print("La performance du modèle sur la base dapprentissage")
    print('--------------------------------------')
    print(f"L'erreur quadratique moyenne est {rmse}€")
    print(f"le score R2 est {r2}")
    print('\n')

    return lmodellineaire


def initialisation (dpt):
    check_folder()
    ensure_data_loaded(dpt)


def estimation (cp, surface, terrain, nbpieces):
    '''
    Renvoi estimation API
    '''
    dpt = int(cp/1000)
    initialisation(dpt)
    df, df_prixmoyen = nettoyage(dpt)
    lmodellineaire = train(dpt)

    prix_m2 = df_prixmoyen['Prix moyen m2 CP'][cp]
    df_estim = [[surface,nbpieces,prix_m2,0]]

    estimation = round(lmodellineaire.predict(df_estim)[0],2)
    print (f'Estimation du bien : {estimation} euros.')

    return estimation

