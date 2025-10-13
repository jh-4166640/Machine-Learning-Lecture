
"""
* 2025-04-03
* 전자공학부 임베디드시스템 전공
* 2021146036 최지헌
* week 5
* #2
"""

"""    
* 2025-04-06
* #3 추가
"""

def GradientDescent(x, y, a, epoch, init_start, init_space, data_len, w0=0, w1=0):
    """
    Gradient Descent Method Function
    
    x : (Matrix) input data shape(Number of Data by Number of Feature + 1)
    y : (Matrix) output real data shape(Number of Data by Q) 
    a : (float) learning rate
    epoch : (integer) training epoch
    data_len : (interger) Number of Data
    """
    # initalize
    # weight, first MSE
    w_his = np.empty([0,2])
    mse_his = np.empty(0)
    w0_init=(np.random.rand()*init_space[0])+init_start[0]
    w1_init=(np.random.rand()*init_space[1])+init_start[1]
    if w0:
        w0_init=w0
    if w1:
        w1_init=w1
    w_his = np.append(w_his, [[w0_init, w1_init]], axis=0)
    print('random init w0, w1_ ', w_his)

    mse_his = np.append(mse_his,np.array(np.mean((w0_init*x[:,0]+w1_init)**2)))
    for epc in range(0,epoch-1):        
        cur_w = w_his[epc]              # load to current weight
        cur_w = cur_w.reshape([1,2])
        # weight update
        new_w=cur_w - a*((np.dot(np.transpose(np.transpose(np.dot(cur_w,np.transpose(x))) - y),x))/data_len)
        # new_w shape is (1,2)
        #new_w0=cur_w[0] - (a*np.mean(np.dot(np.transpose(np.transpose(np.dot(cur_w,np.transpose(x))) - y),x)))
        #new_w1=cur_w[1] - (a*np.mean(np.transpose(np.transpose(np.dot(cur_w,np.transpose(x))) - y)))
        #print(new_w)
        w_his = np.append(w_his, new_w, axis=0) # new weight store
        mse = np.mean((np.dot(new_w, np.transpose(x)) - np.transpose(y))**2) # Calculate MSE using new weights 
        mse_his = np.append(mse_his, mse) # MSE store
    
    return w_his, mse_his
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2) 
file_path = "lin_regression_data_01.csv"
open_file = pd.read_csv(file_path,header=None)
df = pd.DataFrame(open_file)

data_order = 0 # 0 : ascending, 1 : descending

## -- User enter Variables --
learning_rate = 0.003 # learning rate
random_init_start = [-10,-20] # random start value # input size+1 by 1
random_init_space = [100,100] # random space       # input size+1 by 1
epoch = 5000 # training epoch
## --------------------------


numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 

data = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
data = data.values # sorted data

input_mat = np.ones([numberOfData,2]) # size initialize
input_mat[:,0] = data[:,0]
# 50by2
output_mat = data[:,1] # initialize

inputs = input_mat[:,0]
x_avg = np.mean(inputs)
w0_child = np.mean(output_mat*(inputs-x_avg))
w0_parent = np.mean((inputs**2) - (x_avg**2))
w0_op = w0_child/w0_parent
w1_op = np.mean((output_mat - (w0_op*inputs)))

output_mat = output_mat.reshape([numberOfData,1]) # initialize
# 50by1

W_his, MSE_his = GradientDescent(input_mat, output_mat, learning_rate,epoch,random_init_start,random_init_space, numberOfData)
#print(W_his)
# MSE가 가장 작은 W로 그래프 그리기
mse_min = np.inf
mse_min_idx = 0
for ep in range(0,epoch):
    if MSE_his[ep] < mse_min:
        mse_min_idx = ep
        mse_min = MSE_his[ep]
        
print('result w0, w1',W_his[mse_min_idx])
print('mse min', mse_min)        
# 2-2) ---------------------------------
plot_step = 50
ticks = 500
step_start = 0
step_max = epoch
step = np.arange(step_start,step_max,plot_step)
plt.figure(figsize=(12,6))
plt.plot(step, W_his[step_start:step_max:plot_step,0], 'rs--',label='W0')
plt.plot(step, W_his[step_start:step_max:plot_step,1], 'bo--',label='W1')

plt.rc('font',size=20)
plt.xlabel('step',fontsize=20)
plt.ylabel('w0,w1', fontsize=20)
plt.title('Gradient Decent Weight variation. alpha='+str(learning_rate),fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
x_ticks = np.arange(step_start, step_max+plot_step,ticks) # x axis ticks interval
plt.xlim(step_start-plot_step,step_max) # x label range 0~20
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()


plot_mse_step = 50
mse_ticks = 250
mse_step_start = 1000
mse_step_max = epoch
mse_step = np.arange(mse_step_start,mse_step_max,plot_mse_step)

plt.figure(figsize=(12,6))
plt.plot(mse_step,MSE_his[mse_step_start:mse_step_max:plot_mse_step],'gD--',label='MSE')
plt.rc('font',size=20)
plt.xlabel('step',fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.title('Gradient Decent MSE variation. alpha='+str(learning_rate),fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
x_ticks = np.arange(mse_step_start, mse_step_max+plot_mse_step,mse_ticks) # x axis ticks interval
plt.xlim(mse_step_start-plot_mse_step,mse_step_max) # x label range 0~20
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()
# --------------------------------------


# 2-3) ---------------------------------
plt.figure(figsize=(12,6))
plt.scatter(input_mat[:,0],output_mat,c='#60BD7C',label = "Measured Data",s=70,alpha=0.7)
new_x_start = 0 #New prediction start point
new_x_end = 20  #New prediction end point # 왜 20까지 했는지 적자
new_x_step = 0.1 #New prediction step

new_x = np.arange(new_x_start, new_x_end, new_x_step)
optimal_solution = W_his[mse_min_idx][0]*new_x + W_his[mse_min_idx][1]
plt.plot(new_x,optimal_solution,'r--',label="Optimal solution learning rate="+str(learning_rate))
print(w0_op, w1_op)
analytic_solution = w0_op * new_x + w1_op
plt.plot(new_x,analytic_solution,'b--',label="Analytic solution")

W_his2, MSE_his2 = GradientDescent(input_mat, output_mat, 0.01,epoch,random_init_start,random_init_space, numberOfData)
mse_min2 = np.inf
mse_min_idx2 = 0
for ep in range(0,epoch):
    if MSE_his2[ep] < mse_min2:
        mse_min_idx2 = ep
        mse_min2 = MSE_his2[ep]

optimal_solution2 = W_his2[mse_min_idx2][0]*new_x + W_his2[mse_min_idx2][1]
plt.plot(new_x,optimal_solution2,color="orange", linestyle='--',label="Other Optimal solution2 learning rate="+str(0.01))

W_his3, MSE_his3 = GradientDescent(input_mat, output_mat, 0.0001,epoch,random_init_start,random_init_space, numberOfData)
mse_min3 = np.inf
mse_min_idx3 = 0
for ep in range(0,epoch):
    if MSE_his3[ep] < mse_min3:
        mse_min_idx3 = ep
        mse_min3 = MSE_his3[ep]


optimal_solution3 = W_his3[mse_min_idx3][0]*new_x + W_his3[mse_min_idx3][1]
plt.plot(new_x,optimal_solution3,color="green", linestyle='--',label="Other Optimal solution3 learning rate="+str(0.0001))


#not_sol = W_his[round(epoch/3)][0]*new_x + W_his[round(epoch/3)][1]
#plt.plot(new_x,not_sol,color='orange', linestyle='--',label="1/3 epoch weight")
#not_sol2 = W_his[2*round(epoch/3)][0]*new_x + W_his[2*round(epoch/3)][1]
#plt.plot(new_x,not_sol2,'b--',label="2/3 epoch weight")

plt.rc('font',size=20)
plt.xlabel('weight[g]',fontsize=20)
plt.ylabel('length[cm]', fontsize=20)
plt.title('Optimal solution compare Mesured Data(epoch='+str(epoch)+")",fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
x_ticks = np.arange(new_x_start , new_x_end+2, 2) # x axis ticks interval
#y_ticks = np.arange(round(optimal_solution[0])-1 , round(optimal_solution[len(optimal_solution)-1])+3 ,1) # y axis ticks interval
plt.xlim(0,20) # x label range 0~20
plt.xticks(x_ticks,fontsize=15)
plt.yticks(fontsize=14)
plt.show()
# --------------------------------------
