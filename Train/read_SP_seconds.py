# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:15:36 2022

@author: chen_hung
"""

import pandas as pd 
import numpy as np


SP_file = "SP秒數標記.xlsx"


    
df = pd.read_excel(SP_file, header = None)


#seconds_data = df[0].tolist()
#score_data = df[2].tolist()
#seconds_correspond_score={seconds_data[i]:score_data[i] for i in range(len(score_data))}