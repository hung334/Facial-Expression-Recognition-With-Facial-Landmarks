# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:45:19 2022

@author: chen_hung
"""

import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    #%%
    files = glob('Data/2021*.csv')
    input_data = pd.concat( pd.read_csv(file) for file in files )
    
    #%%
    #input_data = pd.read_csv('20211021_bad1_正.csv')
    
    
    drop_nan_data = input_data[input_data['Scores'].notna()]
    X = drop_nan_data.values[:,2:-1]
    Y = drop_nan_data.values[:,-1]
    F_col = input_data.columns[2:-1]
    print('data read ok')
    #%%
    X = X#[0:100]
    Y = Y#[0:100]
    
    #%%
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,VotingRegressor,BaggingRegressor,StackingRegressor,ExtraTreesRegressor
    #model = XGBRegressor(random_state=0)
    model = RandomForestRegressor(random_state=0)
    #model = GradientBoostingRegressor(n_estimators=100,random_state=0)
    #model = AdaBoostRegressor(n_estimators=1000,random_state=0)
    #model = ExtraTreesRegressor(random_state=0)
    print(model)
    model.fit(X, Y)
    print(model.score(X, Y))
    val_fea = []
    for i , col in enumerate(F_col):
        val_fea.append([col,model.feature_importances_[i]])
    val_fea_sorted = sorted(val_fea, key=lambda x: x[1],reverse=True)
    #    print("{}:{}".format(col,model.feature_importances_[i]))
    print(val_fea_sorted[0:10])
    
    val_fea_sorted = np.array(val_fea_sorted)
    np.save('{}_Feature_Importance'.format(str(model).split("(")[0]), val_fea_sorted)
    
    #model = RandomForestRegressor(n_estimators=100,random_state=0)
    #fit_model(X,Y,model)
    
    
    #from sklearn.feature_selection import RFE,RFECV
    # from xgboost import XGBRegressor
    
    # # Create the RFE object and rank each pixel
    # # RFR = LGBMRegressor()#RandomForestRegressor(n_estimators=100,random_state=0)#XGBRFRegressor()      
    # # rfe = RFE(estimator=RFR, n_features_to_select=18, step=1)
    # # rfe = rfe.fit(X, Y)
    # # print(F_col[rfe.support_])
    '''
    #%%
    #---------------------------------------------------xgboots
    from xgboost import XGBRegressor
    model = XGBRegressor()#learning_rate=0.01, n_estimators=1000)
    model.fit(X, Y)
    print(model.score(X, Y))
    # feature importance
    print(model.feature_importances_)
    val_fea = []
    for i , col in enumerate(F_col):
        val_fea.append([col,model.feature_importances_[i]])
    val_fea_sorted = sorted(val_fea, key=lambda x: x[1],reverse=True)
    #    print("{}:{}".format(col,model.feature_importances_[i]))
    print(val_fea_sorted[0:10])
    '''
    '''
    #%%
    from sklearn.feature_selection import SelectKBest, f_regression,SelectPercentile
    
    K_select = SelectKBest(f_regression)

    X_new = K_select.fit_transform(X, Y)
    
    print(F_col[K_select.get_support()])
    
    
    
    
    #%%
    '''