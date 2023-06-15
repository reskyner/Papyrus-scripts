from rdkit.Chem import Descriptors
import pandas as pd
import random 
import tqdm

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# halving grid search - https://scikit-learn.org/stable/modules/grid_search.html
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.metrics import make_scorer, f1_score

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as imbpipeline

import numpy as np
import tqdm

def getMolDescriptors(smi, lig_id, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    mol = Chem.MolFromSmiles(smi)
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            print('mol_id')
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    res['mol_id'] = lig_id
    return res


def get_balanced_data(df):
    actives = df['active'].sum()
    inactives = len(df['active']) - df['active'].sum()

    
    if actives > inactives:
        sel = 1
    elif inactives > actives:
        sel = 0
        
    select_from = df[df['active']==sel]
    select_from.reset_index(inplace=True, drop=True)

    idxs = random.sample(range(0, len(select_from)), len(df[df['active']==abs(sel-1)]))
    sample = select_from.iloc[idxs]
    
    full_set = pd.concat([sample, df[df['active']==abs(sel-1)]])

    return full_set

def get_xy(full_set):
    desc_names = [nm for nm,_ in Descriptors._descList]
    
    X = []
    # Y = []
    
    for i, row in full_set.iterrows():
        x = []
        y = []
        for nm in desc_names:
            x.append(row[nm])
        X.append(x)
    
    Y = list(full_set['active'])

    return X,Y


def find_best_models(X,Y, balance=None):

    names = [
        'Decision Tree Classifier',
        'Extra Tree Classifier',
        'Extra Trees Classifier',
        'Random Forest Classifier',
        'K-Neighbors Classifier',
        'MLP Classifier',
        'Ridge Classifier',
        'SVC',
        'AdaBoost',
        # 'Naive Bayes',
        # 'QDA',
        
    ]
    
    param_grids = [
        {
            'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__splitter': ['best', 'random'],
            'classifier__max_depth': [None, 10, 20, 40, 80, 160, 320],
            'classifier__max_features': ['sqrt', 'log2', None]
        }, # decision tree 
        {
            'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__splitter': ['best', 'random'],
            'classifier__max_depth': [None, 10, 20, 40, 80, 160, 320],
            'classifier__max_features': ['sqrt', 'log2', None]
        }, # extra tree
        {
            'classifier__n_estimators': [10, 20, 40, 80, 160, 320],
            'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__max_depth': [None, 10, 20, 40, 80, 160, 320],
            'classifier__max_features': ['sqrt', 'log2', None]
        }, # extra trees - takes a while...
        {
            'classifier__n_estimators': [10, 20, 40, 80, 160, 320],
            'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__max_depth': [None, 10, 20, 40, 80, 160, 320],
            'classifier__max_features': ['sqrt', 'log2', None]
        }, # random forest
        {
            'classifier__n_neighbors': [1, 2, 4, 8, 16, 32],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree'],
        }, # K-neighbors
        {
            'classifier__hidden_layer_sizes': [10, 20, 40, 80, 160, 320],
            'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'classifier__solver': ['lbfgs', 'sgd', 'adam'],
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
            'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
            
        }, # MLP
        {
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8],
            'classifier__fit_intercept': [True, False],
            'classifier__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
            'calssifier__max_iter': [500, 1000]
        }, # Ridge
        {
            'classifier__C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8],
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__degree': [2, 3, 4, 5, 6, 7],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__shrinking': [True, False]
        }, # SVC
        {
            'classifier__n_estimators': [10, 50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1, 10, 100, 1000],
            'classifier__algorithm': ['SAMME', 'SAMME.R'],
        }, # AdaBoost
        # {}, # Naive Bayes
        # {}, # QDA
    ]
    
    classifiers = [
        DecisionTreeClassifier(),
        ExtraTreeClassifier(),
        ExtraTreesClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier(),
        MLPClassifier(),
        RidgeClassifier(),
        SVC(),
        AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()
    ]
    
    best_models = {}
    
    for name, params, clf in tqdm.tqdm(list(zip(names, param_grids, classifiers))):
        # for key in Y_dict.keys():
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=0.3, random_state=42
                )

            if balance=='SMOTE':
                pipeline = imbpipeline(steps=[('smote', SMOTE(random_state=11)), ('scaler', StandardScaler()), ('classifier', clf)])

            if balance=='ADSYN':
                pipeline = imbpipeline(steps=[('adasyn', ADASYN(random_state=11)), ('scaler', StandardScaler()), ('classifier', clf)])

            if not balance:            
                pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', clf)])
            
            search = HalvingGridSearchCV(pipeline, param_grid=params, n_jobs=15, scoring=make_scorer(f1_score, zero_division=0))
            search.fit(np.array(X_train), np.array(Y_train))

    
            best_models[name] = {
                'model': search.best_estimator_,
                'CV_score': search.best_score_,
                'test_score': search.best_estimator_.score(np.array(X_test), np.array(Y_test)),
                'X_train': X_train,
                'X_test': X_test, 
                'Y_train': Y_train, 
                'Y_test': Y_test
            }
    
            print(search.best_estimator_.score(np.array(X_test), np.array(Y_test)))
            
        except:
            continue

    return best_models
