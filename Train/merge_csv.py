# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:41:32 2022

@author: chen_hung
"""

import pandas as pd
import datetime
import os
import re

if __name__ == "__main__":
    
    
    file_path = "./outputs"
    
    max_time_str = '0:00:00'
    max_time_date = datetime.datetime.strptime(max_time_str, '%H:%M:%S')
    for person in os.listdir(file_path):
        input_data =  pd.read_csv(os.path.join(file_path,person))
        final_time_str = input_data.loc[input_data.shape[0]-1][0]
        final_time_date = datetime.datetime.strptime(final_time_str, '%H:%M:%S')
        if(final_time_date>max_time_date):
                    time_col = None
                    max_time_str = final_time_str
                    max_time_date = final_time_date
    total_s = max_time_date.minute*60+max_time_date.second+1
    time_col = [str(datetime.timedelta(seconds=(i))) for i in range(total_s)]
    
    
    Total_datas=[]
    for No_n in range(1,89):
        for person in os.listdir(file_path):
            re_s = re.compile(r'SP(\d+)正')
            number = int(re_s.findall(person)[0])
            name = "{}_{}".format(person.split(" ")[0],person.split(" ")[1])
            data_dict={}
            if(No_n == number):
                input_data =  pd.read_csv(os.path.join(file_path,person))
                data_dict['name'] = name
                #data_dict = {t:0.00 for t in time_col} 
                for t in time_col:
                    data_dict[t] = 0.00
                for i in range(input_data.shape[0]):
                    data_dict[input_data.loc[i][0]] = input_data.loc[i][1]
                #data_dict ={input_data.loc[i][0]:input_data.loc[i][1] for i in range(input_data.shape[0])}
                
                Total_datas.append(data_dict)
                print(No_n ,"ok")
    
    index = list(range(1,89))
    outputs = pd.DataFrame(Total_datas,index = index)
    outputs.to_csv("outputs.csv",index=False,encoding='utf_8_sig')
    #pd.DataFrame(columns=time_col,index=index)
    