# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:16:42 2022

@author: chen_hung
"""

import numpy as np

SelectKBest = ['L123.z', 'L138.z', 'L147.z', 'L177.z', 'L187.z', 'L192.z', 'L213.z','L215.z', 'L355.z', 'L371.z']

Xgboost = [['L208.x', 0.09071971], ['L280.y', 0.0828043], ['L347.y', 0.039791815], ['L448.x', 0.030080225], ['L348.x', 0.02667094], 
['L231.z', 0.02022399], ['L188.x', 0.016054781], ['L152.x', 0.014863459], ['L220.z', 0.014772245], ['L123.z', 0.014372892]]

RandomForest = [['L123.z', 0.05232155539998189], ['L134.z', 0.044573727170145246], ['L291.y', 0.04078995592719769], 
['L361.y', 0.03936660786778667], ['L171.x', 0.037479298094303226], ['L208.x', 0.03718431823322719], 
['L280.y',0.03271310174917876], ['L220.z', 0.022884563357444196], ['L267.z', 0.02117350092242145], 
['L323.y', 0.021103832274111663]]

ExtraTreesRegressor = [['L323.y', 0.01086341918383624], ['L356.y', 0.009985037099243893], ['L264.y', 0.007638156468928444],['L389.y',0.007162022109211577], ['L4.z', 0.006591941742807421], ['L45.z', 0.006232642073896701], ['L454.y', 0.005985714769240705], ['L134.z', 0.00580921813845053], 
['L1.z', 0.0056477305043641365], ['L35.z', 0.005605768069378909]]

features = []

for i in SelectKBest: features.append(i)
for i in Xgboost: features.append(i[0])
for i in RandomForest: features.append(i[0])
for i in ExtraTreesRegressor: features.append(i[0])

print("原始數量:{}".format(len(features)))

new_features = np.unique(np.array(np.unique(features)))

print("集合後數量:{}".format(len(new_features)))
