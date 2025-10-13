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
path = "lin_regression_data_01.csv"
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

plt.scatter(input_mat,output_mat,label = "Measured Data",s=50)

# 2)--------------------------------------------------------------------------
x_avg = np.mean(input_mat)
w0_child = np.mean(output_mat*(input_mat-x_avg))
w0_parent = np.mean((input_mat**2) - (x_avg**2))

w0_op = w0_child/w0_parent
w1_op = np.mean((output_mat - (w0_op*input_mat)))

new_x_start = 0 #New prediction start point
new_x_end = 20  #New prediction end point
new_x_step = 0.1 #New prediction step

new_x = np.arange(new_x_start, new_x_end, new_x_step)
new_regression_y = w0_op*new_x + w1_op

plt.plot(new_x,new_regression_y,'r-',label="Prediction")


plt.rc('font',size=20)
plt.xlabel('weight[g]',fontsize=20)
plt.ylabel('length[cm]', fontsize=20)
plt.title('Spring Length Graph',fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
x_ticks = np.arange(new_x_start , new_x_end+2, 2) # x axis ticks interval
y_ticks = np.arange(round(new_regression_y[0])-1 , round(new_regression_y[len(new_regression_y)-1]) ,1) # y axis ticks interval
plt.xlim(0,20) # x label range 0~20
plt.xticks(x_ticks,fontsize=15)
plt.yticks(y_ticks,fontsize=15)
plt.show()


# 3)--------------------------------------------------------------------------
y_prediction=w0_op*input_mat + w1_op          # y^ = w0*x + w1
mse = np.mean((y_prediction - output_mat)**2) 


w0_range = np.linspace(-2, 4,numberOfData)  # w_0 range
w1_range = np.linspace(-4.6, 10.6, numberOfData)  # w_1 range
w0_plot, w1_plot = np.meshgrid(w0_range, w1_range) # 1D -> 2D for broadcasting
mse_plot = np.zeros_like(w0_plot)
# Find the MSE for each weight
for i in range(numberOfData):
    for j in range(numberOfData):
        _w0 = w0_plot[i,j]
        _w1 = w1_plot[i,j]
        y_pred = _w0 * input_mat + _w1
        mse_plot[i,j] = np.mean((y_pred - output_mat)**2)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w0_plot,w1_plot,mse_plot,cmap='plasma')
plt.rc('font',size=12)
ax.set_xlabel('W0',fontsize=15)
ax.set_ylabel('W1', fontsize=15)
ax.set_zlabel('MSE', fontsize=16)
plt.title('Loss function',fontsize=24)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
ax.tick_params('z', labelsize=11)
ax.grid(True)
plt.show()
