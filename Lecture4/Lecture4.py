"""
* 2025-03-27
* 전자공학부 임베디드시스템 전공
* 2021146036 최지헌
* week 4
* HW#1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "C:\\Users\\USER\\lin_regression_data_01.csv"
open_file=pd.read_csv(path,header=None)
df = pd.DataFrame(open_file)

data_order = 0 # 0 : ascending, 1 : descending

numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 


# data sort ascending order
data = df.sort_values(by=[0], axis=data_order) # by=[standard]
data = data.values

input_mat = np.zeros([numberOfData,widthOfData]) # size initialize
input_mat[:,0] = data[:,0]
input_mat[:,1] = np.ones(numberOfData) # add to bias term
output_mat = data[:,1] # initialize

# 1)
plt.scatter(input_mat[:,0],output_mat)

plt.rc('font',size=20)
plt.xlabel('weight[g]',fontsize=20)
plt.ylabel('length[cm]', fontsize=20)
plt.title('Spring Length Graph',fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
x_ticks = np.arange(round(data[0,0]) -1 , round(data[numberOfData-1,0]) + 1,1) # x axis ticks interval
y_ticks = np.arange(round(data[0,1]) -1 , round(data[numberOfData-1,1]) + 1,1) # y axis ticks interval
plt.xticks(x_ticks,fontsize=15)
plt.yticks(y_ticks,fontsize=15)
plt.show()






"""
plt.rc('font',size=20)
plt.xlabel('Time (s)',fontsize=20)
plt.ylabel('Value', fontsize=20)
plt.title('Signal Graphs',fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
"""
