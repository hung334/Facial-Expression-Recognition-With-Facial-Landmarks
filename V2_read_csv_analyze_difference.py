# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:45:23 2022

@author: chen_hung
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler


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
    return seconds_correspond_score,[record_1,record_2]
    

    #%%


if __name__ == "__main__":

    


    #path = r'H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\模板\110  11-12月\(7)20211209 江妮芝 吳曉語St\正面\20211209_bad2_正(7).xlsx'
    
    template_path = r'H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\模板'
    
    sun = []  #孫
    chang = []  #張
    file_names = []
    index = 0
    
    for month in os.listdir(template_path):#取 9-10月，10-11月
        if('月' in month):
            #print(month)
            template_month_path= os.path.join(template_path,month)
            for people in os.listdir(template_month_path):#取 人
                if(people != 'desktop.ini'):
                    people_path= os.path.join(template_month_path,people)
                    for angle in os.listdir(people_path):#取 正面
                        if('正'  in angle):
                            #print(angle)
                            angle_path= os.path.join(people_path,angle)
                            for file in os.listdir(angle_path):
                                if('.xlsx' in file ):
                                    #print(file)
                                    file_name = file#.replace('.xlsx','')
                                    file_names.append(file.replace('.xlsx',''))
                                    #print(file_name)
                                    input_path = os.path.join(angle_path,file_name)
                                    print(input_path)
                                    
                                    df = pd.read_excel(input_path, header = None,sheet_name=None)

                                    
                                    for sheet in list(df.keys()):
                                        #print(df.keys())
                                        seconds_data = df[sheet][0].tolist()
                                        score_data = df[sheet][2].tolist()
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
                                        
                                        score = [seconds_correspond_score[str(i).zfill(2)] for i in range(record_1,record_2+1)]
                                        max_n,min_n = max(score),min(score)
                                        difference = max_n-min_n
                                        Norm_score = score#[(i-min_n)/difference for i in score]
                                        
                                        
                                        if(sheet.replace(" ","")=='孫'):
                                            #sun = np.append(sun,score,axis=0)
                                            sun.append(Norm_score)
                                        elif(sheet.replace(" ","")=='張'):
                                            #chang = np.append(chang,score,axis=0)
                                            chang.append(Norm_score)
                                        else:
                                            print(sheet)
                                            
                                            
                                    '''
                                    #index = 0
                                    X = range(len(sun[index]))
                                    plt.title('{}'.format(index))
                                    plt.plot(X, sun[index], 'r',label='sun')
                                    plt.plot(X, chang[index], 'b',label='chang')
                                    plt.plot(X, (np.array(sun[index])+np.array(chang[index]))/2, 'g',label='mean')
                                    #plt.plot(X, score, 'b')
                                    plt.legend()
                                    plt.show()
                                    
                                    index += 1
                                    '''
    #%%
    
    max_sun = max(max(sun))
    min_sun = min(min(sun))
    max_chang = max(max(chang))
    min_chang = min(min(chang))
    
    
    new_sun,new_chang = [],[]
    for data in sun:
        score = [(i-min_sun)/(max_sun-min_sun) for i in data]
        new_sun.append(score)
    for data in chang:
        score = [(i-min_chang)/(max_chang-min_chang) for i in data]
        new_chang.append(score)
    
    
    #%%
    
    for index in range(len(new_sun)):
        file_name_in = file_names[index]
        print(file_name_in)
        data_path = r'C:\Users\chen_hung\Desktop\長庚科大_數據集\Data\{}.csv'.format(file_name_in)
        df_data = pd.read_csv(data_path)
        df_data['Scores2'] = np.NaN#pd.Series() 
        df_data['Scores_combine'] = np.NaN#pd.Series() 
        
        set_data_seconds = list(set(df_data['Seconds']))
        
        
        new_mean = (np.array(new_sun[index])+np.array(new_chang[index]))/2
        
        for i in range(len(df_data)):
            search_index = set_data_seconds.index(df_data.loc[i,'Seconds'])
            if(search_index<len(sun[index])):
                df_data.loc[i,'Scores'] = new_chang[index][search_index]
                df_data.loc[i,'Scores2'] = new_sun[index][search_index]
                df_data.loc[i,'Scores_combine'] = new_mean[search_index]
        df_data.to_csv(r'C:\Users\chen_hung\Desktop\長庚科大_數據集\Data_v2\{}.csv'.format(file_name_in),na_rep="nan", index=False)
    
    
    '''
    for index in range(len(new_sun)):
        X = range(len(sun[index]))
        plt.title('{}'.format(index))
        plt.plot(X, new_sun[index], 'r',label='sun')
        plt.plot(X, new_chang[index], 'b',label='chang')
        plt.plot(X, (np.array(new_sun[index])+np.array(new_chang[index]))/2, 'g',label='mean')
        #plt.plot(X, score, 'b')
        plt.legend()
        plt.show()
    '''


        
        
        
        

