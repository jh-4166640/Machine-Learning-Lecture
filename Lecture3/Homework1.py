"""
* 2025-03-20
* 전자공학부 임베디드시스템 전공
* 2021146036
* 최지헌
* HomeWork #1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "//home//ian_ros2//pythonStudy//problem_1_data//"
img_mat = np.zeros([3,100,100])     
my_row = 0  # for image matrix row index


for file_name in range(0,100,1): # file_name.csv, file_name: 0~99
    open_file_buffer=pd.read_csv(path + str(file_name) +".csv", header=None) # path + file_name + .csv
    data_frame = pd.DataFrame(open_file_buffer) 
    temp=data_frame.values 
    
    if file_name % 10 == 0 and file_name != 0: # when file_name % 10 == 0, Current row stored complete
        my_row += 10                           # so index to next row
        
    img_mat[0][:,file_name] = temp[:,25]       # hw#1-1
    img_mat[1][file_name,:] = temp[10,:]       # hw#1-2
    # my_row:my_row + 10 is row_n~row_n+10 index
    # 10*(file_name%10):10*(file_name%10) + 10
    # 10 * (file_name % 10) is index to cur_start_column 0, 10, 20, 30, ... 80, 90
    # 10 * (file_name % 10) + 10 is index to cur_end_column 10, 20, 30, ... 90, 100. 
    #   actually index to 9, 19, 29, ... 99
    img_mat[2][my_row:my_row+10 , 10*(file_name%10):10*(file_name%10) + 10] = temp[70:80,80:90]
    
"""
-------------------------------------------------
Replace all data with one color
-------------------------------------------------
replace_val = 0.1
for num in range(0,3,1):
    for r in range(0,100,1):
        for c in range(0,100,1):
            if img_mat[num][r][c] != 0:
                img_mat[num][r][c] = replace_val
"""

subplot_val = 131   
# subplot figure index number, n/100 is row, (n%100)/10 is column, (n%100)%10 is index number
for idx in range(0,3,1):
    """
    One by one plot
    -------------------------
    plt.imshow(img_mat[idx], cmap='viridis')
    plt.axis('off')
    plt.show()
    -------------------------
    """
    """
    Once plot
    """
    plt.subplot(subplot_val+idx)
    plt.imshow(img_mat[idx], cmap='viridis')
    plt.axis('off')
plt.title("replace 0.1")
plt.show()
