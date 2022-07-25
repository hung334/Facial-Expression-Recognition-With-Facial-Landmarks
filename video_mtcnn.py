# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:36:39 2022

@author: chen_hung
"""

import cv2
from PIL import Image
from tqdm import tqdm
import time
import numpy as np
from torch_mtcnn import detect_faces


cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.moveWindow("image", 0, 0)
cv2.resizeWindow("image", 1080, 720)


def draw_rectangle(img,boxs):
    
    for box in boxs:
        left_up = (int(box[0]),int(box[1]))
        right_down =  (int(box[2]),int(box[3]))
        color = (0, 0, 255) # red
        thickness = 2 # 寬度 (-1 表示填滿)
        cv2.rectangle(img, left_up, right_down, color, thickness) 
    
        #cv2.rectangle(img, (125, 20), (175, 40), (255, 0, 255), -1) 

    return img


def CV(path,start=0,end=None,wait=1):
    '''
    strat-開始時間
    end-結束時間
    wait-每幀等待時間
    '''
    
    vc = cv2.VideoCapture(path)
    opened = vc.isOpened()
    if not opened:
        print("open video error")
    
    vc.set(cv2.CAP_PROP_FPS,30)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    
    fps = vc.get(cv2.CAP_PROP_FPS)
    count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"影片寬度:{width},影片高度:{height},影片總幀數:{count},幀率:{fps}")
    time.sleep(0.01)
    
    if(start):
        start_frame = min(int(start*fps),count) #初始幀
        vc.set(cv2.CAP_PROP_POS_FRAMES,start_frame)#設置開始幀位置
        count -=  start_frame
    if(end):
        count -= count - min(int((end-start)*fps),count)
        
    pbar = tqdm(total=count,desc="播放進度")#進度條
    f_num = 0 #讀取幀數
    
    while opened:
            f_num += 1
            opened,frame = vc.read()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            bounding_boxes, landmarks = detect_faces(image)
            #print(bounding_boxes)
            #print(landmarks)
            frame = draw_rectangle(frame ,bounding_boxes)
            
            if(opened):
                cv2.imshow('image',frame)

                pbar.update(1)
                if(cv2.waitKey(wait)==27 or f_num ==count):
                    break
            else:
                break
    pbar.close()
    vc.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
        
    path = r'H:\.shortcut-targets-by-id\1vItTjppV9Gfw1svIRJVJtjZ9JFG4OgDB\IRB拍攝影片\模板\110  9-10月\(1) 20211012 簡乃卉 張靜宜Tr\正面\20211012_bad1_正.MP4'
    results = CV(path,40,226)