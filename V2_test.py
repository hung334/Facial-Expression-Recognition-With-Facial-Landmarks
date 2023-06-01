# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:36:39 2022

@author: chen_hung
"""

import cv2
from tqdm import tqdm
import mediapipe as mp
import numpy as np
import csv
import os
from sklearn.ensemble import ExtraTreesRegressor
import joblib
import time
import datetime
import re
import pandas as pd 


save_local_path = r'C:\Users\chen_hung\Desktop\長庚科大_數據集\outputs_v2'  
#輸出答案
#model_file = r'C:\Users\chen_hung\Desktop\長庚科大_數據集\save_model\all_data\ExtraTreesRegressor_2_fold_0.9922081433698542'
#ExtraTreesRegressor_model = joblib.load(model_file)
#print(ExtraTreesRegressor_model)

ER_model_1 = joblib.load(r'C:\Users\chen_hung\Desktop\長庚科大_數據集\save_model\ExtraTreesRegressor_1_fold_0.9912041893050966')
ER_model_2 = joblib.load(r'C:\Users\chen_hung\Desktop\長庚科大_數據集\save_model\ExtraTreesRegressor_2_fold_0.9914882046020937')
ER_model_3 = joblib.load(r'C:\Users\chen_hung\Desktop\長庚科大_數據集\save_model\ExtraTreesRegressor_3_fold_0.9911791154121529')
ER_model_4 = joblib.load(r'C:\Users\chen_hung\Desktop\長庚科大_數據集\save_model\ExtraTreesRegressor_4_fold_0.9902297902124735')
ER_model_5 = joblib.load(r'C:\Users\chen_hung\Desktop\長庚科大_數據集\save_model\ExtraTreesRegressor_5_fold_0.9906612535908065')

print(ER_model_1 )

ExtraTreesRegressor_f = np.load(r"C:\Users\chen_hung\Desktop\長庚科大_數據集\ExtraTreesRegressor_Scores_combine_Feature_Importance.npy")
select_ExtraTrees = [ i[0] for i in ExtraTreesRegressor_f[0:600]]

Row_1=[]
for i in range(0,478):
    Row_1.append('L{}.x'.format(i))
    Row_1.append('L{}.y'.format(i))
    Row_1.append('L{}.z'.format(i))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_video_seconds(path):
    import pandas as pd 
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
                
    #print("秒數:{}-{}".format(record_1,record_2))
    return seconds_correspond_score,[record_1,record_2]
def get_mesh_box(image,landmark):
    h, w, c = image.shape
    cx_min=  w
    cy_min = h
    cx_max= cy_max= 0
    for id, lm in enumerate(landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy
    return [cx_min, cy_min,cx_max,cy_max]

def test_CV(path,start=0,end=None,wait=1):
    
    total_data = []
    
    '''
    strat-開始時間
    end-結束時間
    wait-每幀等待時間
    '''
    display_start = start
    
    vc = cv2.VideoCapture(path)
    opened = vc.isOpened()
    if not opened:
        print("open video error")
    
    vc.set(cv2.CAP_PROP_FPS,60)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    
    video_fps = vc.get(cv2.CAP_PROP_FPS)
    video_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"影片寬度:{video_width},影片高度:{video_height},影片總幀數:{video_count},幀率:{video_fps}")
    time.sleep(0.01)
    
    if(start):
        start_frame = min(int(start*video_fps),video_count) #初始幀
        vc.set(cv2.CAP_PROP_POS_FRAMES,start_frame)#設置開始幀位置
        video_count -=  start_frame
    if(end):
        video_count -= video_count - min(int((end-start)*video_fps),video_count)
    print(f"開始秒數:{start},結束秒數:{end},實際幀數:{video_count},\n")
    
    pbar = tqdm(total=video_count,desc="播放進度")#進度條
    f_num,r_num = 0,0 #讀取幀數
    
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
    min_detection_confidence=0.5,min_tracking_confidence=0.5)
    
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    drawing_spec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)
    
    p_time = 0

    while True:
            input_data = []
            reg_data = []
            
            f_num += 1
            r_num += 1
            
            if(f_num%int(video_fps)==0):
                display_start+=1
                r_num = 1
                
            #print(int(f_num/int(video_fps))+start,r_num)
            
            reg_data.append(int(f_num/int(video_fps))+start)#秒數
            reg_data.append(r_num)#秒數之張數
            #input_data.append(r_num)
            
            opened,frame = vc.read()
            block_frame = np.zeros(frame.shape,np.uint8)
            #frame.flags.writeable = False
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            aa = time.time()
            
            results = face_mesh.process(image)
            #return results
            #print(dir(results))
            #print(results.index[0])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            
            if results.multi_face_landmarks:
              for face_landmarks in results.multi_face_landmarks:
                mpDraw.draw_landmarks(image, face_landmarks)
                cy_bbox = get_mesh_box(image,face_landmarks.landmark)
                
                #print(len(face_landmarks.landmark))
                for landmarks in face_landmarks.landmark:
                    input_data.append(landmarks.x)
                    input_data.append(landmarks.y)
                    input_data.append(landmarks.z)
            

            c_time = time.time()
            fps = 1/(c_time-p_time)
            p_time = c_time
            
            
            if(len(input_data)>1):
                df = pd.DataFrame(np.array([input_data]),columns=Row_1,dtype=float)
                #print(df[select_ExtraTrees].values)
                #輸入score
                #ans_score =  ExtraTreesRegressor_model.predict(np.array([input_data]))
                ans_score = ER_model_5.predict(df[select_ExtraTrees].values)
                ans_score+=ER_model_4.predict(df[select_ExtraTrees].values)
                ans_score+=ER_model_3.predict(df[select_ExtraTrees].values)
                ans_score+=ER_model_2.predict(df[select_ExtraTrees].values)
                ans_score+=ER_model_1.predict(df[select_ExtraTrees].values)
                reg_score = ans_score/5
                ans_score = ((reg_score*75+20)+(reg_score*46+45))/2
                bb = time.time()
                
            else:
                ans_score = -1
            #print(bb-aa)
            #print('score',ans_score)
            reg_data.append(float(ans_score))
            
            
            total_data.append(reg_data)

            
            if(opened):
                #cv2.imshow('image',image)
                #cv2.imshow('image_2',crop_block_frame)
                pbar.update(1)
                if(cv2.waitKey(wait)==27 or f_num ==video_count):
                    break
            else:
                break
            
    pbar.close()
    vc.release()
    cv2.destroyAllWindows()
    
    return total_data

def make_datasets(input_path,save_path):
    
    
    seconds_correspond_score,seconds= get_video_seconds(input_path+'.xlsx')
    
    try:
        results = CV(input_path+'.MP4',seconds[0],seconds[1],seconds_correspond_score = seconds_correspond_score)
    except:
        results = CV(input_path+'.mov',seconds[0],seconds[1],seconds_correspond_score = seconds_correspond_score)
    
    with open(save_path, 'w', newline='') as csvfile:
          # 建立 CSV 檔寫入器
          writer = csv.writer(csvfile)
          
          Row_1 = ['Seconds','Frames']
          for i in range(0,478):
              Row_1.append('L{}.x'.format(i))
              Row_1.append('L{}.y'.format(i))
              Row_1.append('L{}.z'.format(i))
          Row_1.append('Scores')
          # 寫入一列資料
          writer.writerow(Row_1)
        
          # 寫入另外幾列資料
          for data in results:
              writer.writerow(data)

def debug_test(video_file,save_path):
        try:
            total_data = test_CV(video_file+'.MP4')
        except :
            total_data = test_CV(video_file+'.mp4')
        total_data = np.array(total_data)
        
        ans_csv = []
        
        for seconds in np.unique(total_data[:,0]):
            reg = []
            min_index = np.where(total_data[:,0]==seconds)[0][0]
            max_index = np.where(total_data[:,0]==seconds)[0][-1]+1
            
            score = np.mean(total_data[min_index:max_index,2])
            
            reg.append(str(datetime.timedelta(seconds=(seconds))))
            reg.append('{:1.2f}'.format(score))
            ans_csv.append(reg)
        
        with open(save_path, 'w', newline='') as csvfile:
              # 建立 CSV 檔寫入器
              writer = csv.writer(csvfile)
              
              Row_1 = ['Seconds','Scores']
              # 寫入一列資料
              writer.writerow(Row_1)
            
              # 寫入另外幾列資料
              for data in ans_csv:
                  writer.writerow(data)
                  
    
def run_test(S,E,video_file,save_path):
    try:
        try:
            total_data = test_CV(video_file+'.MP4',start=S,end=E)
        except :
            total_data = test_CV(video_file+'.mp4',start=S,end=E)
        total_data = np.array(total_data)
        
        ans_csv = []
        
        for seconds in np.unique(total_data[:,0]):
            reg = []
            min_index = np.where(total_data[:,0]==seconds)[0][0]
            max_index = np.where(total_data[:,0]==seconds)[0][-1]+1
            
            score = np.mean(total_data[min_index:max_index,2])
            
            reg.append(str(datetime.timedelta(seconds=(seconds))))
            reg.append('{:1.2f}'.format(score))
            ans_csv.append(reg)
        
        with open(save_path, 'w', newline='') as csvfile:
              # 建立 CSV 檔寫入器
              writer = csv.writer(csvfile)
              
              Row_1 = ['Seconds','Scores']
              # 寫入一列資料
              writer.writerow(Row_1)
            
              # 寫入另外幾列資料
              for data in ans_csv:
                  writer.writerow(data)
    except :
        print("{} 有問題".format(video_file))
    
if __name__ == "__main__":
    
    
    #pathA = r'H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\模板\110  9-10月\(1) 20211012 簡乃卉 張靜宜Tr\正面\20211012_bad1_正'
    #40~226
    #path = r'H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\模板\110  11-12月\(7)20211209 江妮芝 吳曉語St\正面\20211209_bad2_正(7)'
    #3~84
    
    #pathB = r'H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\模板\110  9-10月\(1) 20211012 簡乃卉 張靜宜Tr\正面\20211012_good1_正'
    '''
    template_path = r'H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\模板'
    
    get_now = os.listdir(save_local_path)
    
    
    video_path = r"H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\標準化病人"
    video_file = r"H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\標準化病人\SP1 游孟涵 20220111\SP1正 游孟涵 20220111"
    file_name = "SP1正 游孟涵 20220111"
    total_data = test_CV(video_file+'.MP4')
    total_data = np.array(total_data)
    
    ans_csv = []
    
    for seconds in np.unique(total_data[:,0]):
        reg = []
        min_index = np.where(total_data[:,0]==seconds)[0][0]
        max_index = np.where(total_data[:,0]==seconds)[0][-1]+1
        
        score = np.mean(total_data[min_index:max_index,2])
        
        reg.append(str(datetime.timedelta(seconds=(seconds))))
        reg.append('{:1.2f}'.format(score))
        ans_csv.append(reg)
    '''
    video_path = r"H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\標準化病人"
    
    #video_file = os.path.join(video_path,'SP12 潘子琦 20220113','SP12正 潘子琦 20220113')
    #save_path = os.path.join(save_local_path,'SP12正 潘子琦 20220113'+'.csv')
    #debug_test(video_file,save_path)
    
    SP_file = r"C:\Users\chen_hung\Desktop\長庚科大_數據集\SP秒數標記(1111028更新).xlsx"
    #SP_file = r"H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\標準化病人\SP秒數標記.xlsx"
    SP_df = pd.read_excel(SP_file, header = None)
    
    
    for people_name in os.listdir(video_path):
        if('SP' in people_name ):
            for video_name in os.listdir(os.path.join(video_path,people_name)):
                if all(i in video_name for i in ['SP','正'] ):
                    file_name = video_name.split(".")[0]
                    re_s = re.compile(r'SP(\d+)正')
                    number = int(re_s.findall(file_name)[0])
                    #print(number)
                    star_t = SP_df.loc[number][1]
                    end_t = SP_df.loc[number][3]
                    
                    star_s =  star_t.hour*60 + star_t.minute
                    end_s =  end_t.hour*60 + end_t.minute
                    
                    #print(file_name)
                    #print(number,star_s,end_s)
                    #print(video_name)
                    path = os.path.join(video_path,people_name,file_name)
                    save_path = os.path.join(save_local_path,file_name+'.csv')

                    if not(file_name+'.csv' in os.listdir(save_local_path)):
                        print(file_name)
                        run_test(star_s,end_s,path,save_path)

                    
    '''
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
                                    file_name = file.replace('.xlsx','')
                                    #print(file_name)
                                    input_path = os.path.join(angle_path,file_name)
                                    save_path = os.path.join(save_local_path,"{}.csv".format(file_name))
                                    if(file_name+'.csv' not in get_now):
                                        try:
                                            make_datasets(input_path,save_path)
                                        except:
                                            print("error:{}".format(file_name))
                            
    '''
                    
    
    '''
    input_path = path
    
    seconds_correspond_score,seconds= get_video_seconds(input_path+'.xlsx')
    
    results = CV(input_path+'.MP4',seconds[0],seconds[1])
    
    with open('output.csv', 'w', newline='') as csvfile:
          # 建立 CSV 檔寫入器
          writer = csv.writer(csvfile)
          
          Row_1 = ['Seconds','Frames']
          for i in range(0,478):
              Row_1.append('L{}.x'.format(i))
              Row_1.append('L{}.y'.format(i))
              Row_1.append('L{}.z'.format(i))
          Row_1.append('Scores')
          # 寫入一列資料
          writer.writerow(Row_1)
        
          # 寫入另外幾列資料
          for data in results:
              writer.writerow(data)
    '''