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

# 1)--------------------------------------------------------------------------
path = "C:\\Users\\USER\\lin_regression_data_01.csv"
open_file=pd.read_csv(path,header=None)
df = pd.DataFrame(open_file)

data_order = 0 # 0 : ascending, 1 : descending

numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 

# data sort ascending order
data = df.sort_values(by=[0], axis=data_order) # by=[standard]
data = data.values

#input_mat = np.zeros([numberOfData,widthOfData]) # size initialize
input_mat = data[:,0]
#input_mat[:,1] = np.ones(numberOfData) # add to bias term
output_mat = data[:,1] # initialize

plt.scatter(input_mat,output_mat,label = "Measured Data")

# 2)--------------------------------------------------------------------------
x_avg = np.mean(input_mat)
w0_child = np.mean(output_mat*(input_mat-x_avg))
w0_parent = np.mean((input_mat**2) - (x_avg**2))

w0_op = w0_child/w0_parent
w1_op = np.mean((output_mat - (w0_op*input_mat)))

new_x_start = 0 #Regression start point
new_x_end = 20  #Regression end point
new_x_step = 0.1 #New regression step

new_x = np.arange(new_x_start, new_x_end, new_x_step)
new_regression_y = w0_op*new_x + w1_op
plt.plot(new_x,new_regression_y,'r-',label="Regression")


plt.rc('font',size=20)
plt.xlabel('weight[g]',fontsize=20)
plt.ylabel('length[cm]', fontsize=20)
plt.title('Spring Length Graph',fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
x_ticks = np.arange(new_x_start , new_x_end+2, 2) # x axis ticks interval
y_ticks = np.arange(round(new_regression_y[0]) , round(new_regression_y[len(new_regression_y)-1]) ,1) # y axis ticks interval
plt.xlim(0,20) # x label range 0~20
plt.xticks(x_ticks,fontsize=15)
plt.yticks(y_ticks,fontsize=15)
plt.show()


# 3)--------------------------------------------------------------------------
y_regression=w0_op*input_mat + w1_op          # y^ = w0*x + w1
mse = np.mean((y_regression - output_mat)**2) 
