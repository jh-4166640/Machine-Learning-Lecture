import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

# 0 청사과
# 1 사과
# 2 바나나
# 3 블랙베리
# 4 오이
# 5 오렌지
# 6 복숭아
# 7 배
# 8 토마토
# 9 수박

# RGB
# 검정 [0,0,0]
# 흰 [255,255,255]

def MOST_COLOR(img):
    Rs = np.array([img[:,:,0], img[:,:,1], img[:,:,2]])

def select_features(directory):
    # 폴더 내 파일명 read
    file_list = os.listdir(directory)

    feature_1_list = []
    feature_2_list = []
    feature_3_list = []
    feature_4_list = []

    labels = []

    for name in file_list:
        path = os.path.join(directory, name)
        labels.append(int(name.split('_',1)[0]))
        # 이미지 Read & RGB 변환
        img_GRB = cv2.imread(path)
        img_RGB = cv2.cvtColor(img_GRB, cv2.COLOR_BGR2RGB)
        feature_2 = img_RGB[:,:,0].mean() # Red 평균 밝기
        feature_3 = img_RGB[:,:,1].mean() # Blue 평균 밝기
        feature_4 = img_RGB[:,:,2].mean() # Green 평균 밝기
        feature_5 = MOST_COLOR(img_RGB)
        #img_RGB=img_RGB.T    
        
        
        feature_2_list.append(feature_2)
        feature_3_list.append(feature_3)
        feature_4_list.append(feature_4)
    
    feature_2_list = np.array(feature_2_list)
    features = np.column_stack([feature_2_list,feature_3_list,feature_4_list])
    return features

    

directory = "C:\\Users\\USER\\Downloads\\train"
fets=select_features(directory)
