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

path = "C:\\problem_1_data\\problem_1_data\\"
img_vec = np.zeros([3,100,100])
my_row = 0

for file_name in range(0,100,1):
    open_file_buffer=pd.read_csv(path + str(file_name) +".csv", header=None)
    data_frame = pd.DataFrame(open_file_buffer)
    temp=data_frame.values
    if file_name % 10 == 0:
        if file_name != 0:
            my_row += 10
    img_vec[0][:,file_name] = temp[:,25]    
    img_vec[1][file_name,:] = temp[10,:]    
    img_vec[2][my_row:my_row+10,10*(file_name%10):10*(file_name%10) + 10] = temp[70:80,80:90]


plt.subplot(131)
plt.imshow(img_vec[0], cmap='viridis')
plt.axis('off')
plt.subplot(132)
plt.imshow(img_vec[1], cmap='viridis')
plt.axis('off')
plt.subplot(133)
plt.imshow(img_vec[2], cmap='viridis')
plt.axis('off')
plt.show()
