
"""
* 2025-04-03
* 전자공학부 임베디드시스템 전공
* 2021146036 최지헌
* week 5
"""


def GradientDecent(x, y, a, epoch, init_start, init_space):
    """
    Gradient Decent Method Function
    
    x : (Matrix) input data
    y : (Matrix) output real data
    a : (float) learning rate
    epoch : (integer) training epoch
    """
    
    w_his = np.empty([0,2])
    mse_his = np.empty([0,1])
    for epc in range(0,epoch):
        if epc == 0:
            w0_init=np.random.rand(1,1)*init_space[0]+init_start[0]
            w1_init=np.random.rand(1,1)*init_space[1]+init_start[1]
            w_his = np.append(w_his, [w0_init, w1_init], axis=0)
            mse_his = np.append(mse_his,np.array(np.mean((w0_init*x[:,0]+w1_init)**2)),axis=0)
            
        cur_w = w_his[epc,:]
        
        new_w0=cur_w[0,0] - a * np.mean(np.dot(np.transpose(np.transpose(np.dot(cur_w,np.transpose(x))) - y),x))
        new_w1=cur_w[0,1] - a * np.mean(np.transpose(np.transpose(np.dot(cur_w,np.transpose(x))) - y))
        
        w_his = np.append(w_his, [[new_w0,new_w1]], axis=0)
        mse = np.mean((np.dot(cur_w, np.transpose(x)) - np.transpose(y))**2)
        mse_his = np.append(mse_his, mse, axis=0)
    
    return w_his, mse_his
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#1) 
file_path = "C:\\Users\\USER\\lin_regression_data_01.csv"
open_file = pd.read_csv(file_path,header=None)
df = pd.DataFrame(open_file)

data_order = 0 # 0 : ascending, 1 : descending

## -- User enter Variables --
learning_rate = 0.001 # learning rate
random_init_start = [1,1] # random start value # input size+1 by 1
random_init_space = [5,1] # random space       # input size+1 by 1
epoch = 1000 # training epoch
## --------------------------



numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 

data = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
data = data.values # sorted data

input_mat = np.ones([numberOfData,2]) # size initialize
input_mat[:,0] = data[:,0]
# 50by2
output_mat = data[:,1] # initialize
output_mat = output_mat.reshape([numberOfData,1]) # initialize
# 50by1


W_his, MSE_his = GradientDecent(input_mat, output_mat, learning_rate,epoch,random_init_start,random_init_space)
step = np.arange(0,epoch,1 )
plt.plot(step, W_his[:,0], 'r*')
plt.plot(step, W_his[:,1], 'b-')


plt.rc('font',size=20)
plt.xlabel('step',fontsize=20)
plt.ylabel('w0,w1', fontsize=20)
plt.title('weight',fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
#x_ticks = np.arange(0 , epoch, 500) # x axis ticks interval
#y_ticks = np.arange(round(new_regression_y[0])-1 , round(new_regression_y[len(new_regression_y)-1]) ,1) # y axis ticks interval
plt.xlim(0,epoch) # x label range 0~20
plt.xticks(x_ticks,fontsize=15)
plt.yticks(y_ticks,fontsize=15)
plt.show()











