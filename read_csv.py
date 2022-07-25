# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:45:23 2022

@author: chen_hung
"""
import pandas as pd 
import numpy as np

def get_video_seconds(path):
    
    df = pd.read_excel(path, header = None)
    seconds_data = df[0].tolist()
    score_data = df[2].tolist()
    seconds_correspond_score={seconds_data[i]:score_data[i] for i in range(len(score_data))}
    
    reg ,record_1,record_2 = 0, None, None
    for i,data in enumerate(score_data):
        if(i>0):
            if (not(np.isnan(data)) and reg==0):
                record_1 = int(seconds_data[i])
                reg = 1
            if(np.isnan(data) and reg==1):
                record_2 = int(seconds_data[i-1])
                reg = 0
                
    print("秒數:{}-{}".format(record_1,record_2))
    return [record_1,record_2]
    



if __name__ == "__main__":



    path = r'H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\模板\110  11-12月\(7)20211209 江妮芝 吳曉語St\正面\20211209_bad2_正(7).xlsx'
    df = pd.read_excel(path, header = None)
    seconds_data = df[0].tolist()
    score_data = df[2].tolist()
    seconds_correspond_score={seconds_data[i]:score_data[i] for i in range(len(score_data))}
    
    reg ,record_1,record_2 = 0, None, None
    for i,data in enumerate(score_data):
        if(i>0):
            if (not(np.isnan(data)) and reg==0):
                record_1 = int(seconds_data[i])
                reg = 1
            if(np.isnan(data) and reg==1):
                record_2 = int(seconds_data[i-1])
                reg = 0
    
    print(record_1,record_2)

