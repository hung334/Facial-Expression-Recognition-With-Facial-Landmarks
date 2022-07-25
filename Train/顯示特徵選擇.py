# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:14:40 2022

@author: chen_hung
"""


import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    XGBRegressor = "XGBRegressor_Feature_Importance.npy"
    ExtraTrees = "ExtraTreesRegressor_Feature_Importance.npy"
    RandomForest = "RandomForestRegressor_Feature_Importance.npy"
    
    input_data = np.load(XGBRegressor)
    
    
    val_fea_sorted = input_data[0:150].copy()
    
    plt.title("Feature Importance")
    plt.bar(range(len(val_fea_sorted)),
            val_fea_sorted[:,1].astype(float),
            color = "b",
            align="center")
    plt.xticks(range(len(val_fea_sorted)),val_fea_sorted[:,0],fontsize=3)
    plt.xlim([-1,len(val_fea_sorted)])
    plt.show()
    
    n, bins, patches=plt.hist(val_fea_sorted[:,1].astype(float))
    #plt.xlabel("Values")
    #plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()
    