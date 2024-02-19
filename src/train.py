############################################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
import xgboost as xgb
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df.drop(['Class'],axis=1)
    y_train = df[['Class']]
    print(filename, ' cargado correctamente')
    
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    
    nm = NearMiss()
    
    X_res, y_res = nm.fit_resample(X_train, y_train)
    
    # Entrenamos el modelo con toda la muestra
    xgb_mod=xgb.XGBClassifier(colsample_bytree=0.7374195520571349,
                      n_estimators=1000, 
                      min_child_weight=0.0,
                      reg_alpha = 158.0,
                      reg_lambda=0.6983089924752687,
                      max_depth=11, 
                      gamma=6.917044807116284)
    xgb_mod.fit(X_res, y_res)
    
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(xgb_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('creditcard_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()