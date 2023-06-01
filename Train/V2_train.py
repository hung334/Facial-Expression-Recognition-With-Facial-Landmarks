# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:07:50 2021

@author: chen_hung
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,VotingRegressor,BaggingRegressor,StackingRegressor,ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from glob import glob
from sklearn import datasets, linear_model
import time
import datetime
import joblib



def show_his(values,name):
    plt.figure()
    n, bins, patches=plt.hist(values)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("{}-Histogram".format(name))
    plt.show()


def fit_model(X,Y,model,save=False):
    
    kf = KFold(n_splits=5,shuffle=True,random_state=0)
    Datasets_index = np.array([i for i in range(len(X))])
    
    score_a,r2_b,RMSE_c = [],[],[]
    
    print('\n{} 5-fold:'.format(model))
    for i,[train_index,test_index] in enumerate( kf.split(Datasets_index)):
        
        start = time.time()
        model.fit(X[train_index], Y[train_index])
        end = time.time()
        
        score = model.score(X[test_index], Y[test_index])
        pred = model.predict(X[test_index])
        
        train_pred = model.predict(X[train_index])
        
        train_r2 = model.score(X[train_index], Y[train_index])
        r2 = r2_score(Y[test_index], pred)
        rmse = (mean_squared_error(Y[test_index], pred))**0.5
        mae = mean_absolute_error(Y[test_index], pred)
        
        score_a.append(train_r2)
        r2_b.append(r2)
        RMSE_c.append(rmse)
        
        save_pth = "./save_model/{}_{}_fold_{}".format(str(model).split("(")[0],i+1,r2)
        if(save):joblib.dump(model,save_pth)
        
        print("train time:{}".format(str(datetime.timedelta(seconds=(end - start)))))
        print('{}-score:{}'.format(i+1,train_r2))
        print('r2:{} '.format(r2))
        print('RMSE:{} '.format( rmse))
        print("\n")
        
    print("score-mean:{}".format(np.mean(score_a)))
    print("score-std:{}".format(np.std(score_a)))
    print("r2-mean:{}".format(np.mean(r2_b)))
    print("r2-std:{}".format(np.std(r2_b)))
    print("RMSE-mean:{}".format(np.mean(RMSE_c)))
    print("RMSE-std:{}".format(np.std(RMSE_c)))

if __name__ == "__main__":
    
    #%%
    try:
        input_data = pd.read_csv("Datasets_v2.csv")
        print("Datasets.csv read ok.")
    except:
        
        files = glob('Data_v2/2021*.csv')
        input_data = pd.concat( pd.read_csv(file) for file in files )
        input_data.to_csv("Datasets_v2.csv")
        print("save Datasets.csv.")
    #%%
    #input_data = pd.read_csv('20211021_bad1_正.csv')
    ExtraTreesRegressor_f = np.load("ExtraTreesRegressor_Scores_combine_Feature_Importance.npy")
    select_ExtraTrees = [ i[0] for i in ExtraTreesRegressor_f[0:600]]
    '''
    Xgboost_f = np.load("XGBRegressor_Feature_Importance.npy")
    ExtraTreesRegressor_f = np.load("ExtraTreesRegressor_Feature_Importance.npy")
    RandomForest_f = np.load("RandomForestRegressor_Feature_Importance.npy")
    
    select_f = ['L1.z', 'L123.z', 'L134.z', 'L138.z', 'L147.z', 'L152.x', 'L171.x',
 'L177.z', 'L187.z', 'L188.x', 'L192.z', 'L208.x', 'L213.z',
 'L215.z', 'L220.z', 'L231.z', 'L264.y', 'L267.z', 'L280.y',
 'L291.y', 'L323.y', 'L347.y', 'L348.x', 'L35.z', 'L355.z',
 'L356.y', 'L361.y', 'L371.z', 'L389.y', 'L4.z', 'L448.x', 'L45.z','L454.y']
    
    select_XGB = [i[0] for i in Xgboost_f[0:500]]
    select_ExtraTrees = [ i[0] for i in ExtraTreesRegressor_f[0:600]]
    select_RandomForest = [ i[0] for i in RandomForest_f]
    select_Kbest = ['L123.z', 'L138.z', 'L147.z', 'L177.z', 'L187.z', 'L192.z', 'L213.z','L215.z', 'L355.z', 'L371.z']
    '''
    
    out_label = 'Scores_combine'
    
    drop_nan_data = input_data.dropna(axis=0,how='any') #input_data[input_data['Scores'].notna()]
    #X = drop_nan_data.values[:,3:-1]
    #print('X.shape',X.shape)
    print('In theory has 1434 features')
    
    #print("select_num:{}".format(len(select_ExtraTrees)))
    #X = drop_nan_data[select_ExtraTrees].values[:,2:-1]
    #X = drop_nan_data.values[:,2:-1]
    #Y = drop_nan_data.values[:,-1]
    F_col = input_data.columns[3:-3]
    
    
    #X = drop_nan_data[F_col].values
    X = drop_nan_data[select_ExtraTrees].values
    Y = drop_nan_data[out_label].values
    
    print('In fact has {} features'.format(len(X[0])))
    print('data read ok ({})'.format(out_label))
    
    #%%
    from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,VotingRegressor,BaggingRegressor,StackingRegressor,ExtraTreesRegressor
    from xgboost import XGBRegressor
    #model = XGBRegressor(random_state=0)#learning_rate=0.01, n_estimators=1000)
    model = ExtraTreesRegressor(random_state=0)
    #model = RandomForestRegressor(random_state=0)
    fit_model(X,Y,model,True)
    
    
    
    #%%
    #X = X#[0:100]
    #Y = Y#[0:100]
    
    #%%
    #from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,VotingRegressor,BaggingRegressor,StackingRegressor,ExtraTreesRegressor
    #model = RandomForestRegressor(n_estimators=100,random_state=0)
    #model = GradientBoostingRegressor(n_estimators=100,random_state=0)
    #model = AdaBoostRegressor(n_estimators=1000,random_state=0)
    #model = ExtraTreesRegressor(n_estimators=100,random_state=0)
    #print(model)
    #model.fit(X, Y)
    #print(model.score(X, Y))
    #val_fea = []
    #for i , col in enumerate(F_col):
    #    val_fea.append([col,model.feature_importances_[i]])
    #val_fea_sorted = sorted(val_fea, key=lambda x: x[1],reverse=True)
    #    print("{}:{}".format(col,model.feature_importances_[i]))
    #print(val_fea_sorted[0:10])