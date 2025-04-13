"""
* 2025-04-10
* 임베디드 시스템 전공
* 2021146036
* 최지헌
* week 6
* polynomial Basis Function
* 기본 과제 종료
"""

"""
u_k = input_mat[].min() + gauss_k*(input_mat[].max() - input_mat[].min())/count_gauss - 1
sigma = (input_mat[].max() - input_mat[].min()) / (count_gauss - 1)

gaussian = np.exp(0.5*((input_mat[] - u_k) / sigma)**2)
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def GeneratorBasis(k,x):
    """
    Parameters
    ----------
    k : interger
        Number of basis function.
    x : (float)matrix
        input value

    Returns
    -------
    basis : np.array[numData][width * k + 1]
        basis.

    """
    numData = x.shape[0]
    width = x.shape[1]
    basis=np.ones([numData, width * k + 1]) # array size [numData][width * k + 1] # M is Data feature, N is number of data
    for numx in range(0,width,1):
        for exp in range(0,k,1):
            basis[:,numx*k + exp + 1]=x[:,numx]**(exp+1)
    return basis


def PolynomialAnalyticSolution(k,x,y):
    """
    Parameters
    ----------
    k : interger
        Number of basis function.
    x : (float) matrix
        input.
    y : (float) matrix
        output.

    Returns
    -------
    weight, basis

    """
    basis= GeneratorBasis(k,x)
    # find the weight
    weight = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(basis),basis)),np.transpose(basis)),y)
    return weight, basis


def PolynomialGradientDescent(x, y, k, alpha, epoch, init_start, init_space):
    """
    Gradient Descent Method Function
    
    x : (Matrix) input data shape(Number of Data by Number of Feature + 1)
    y : (Matrix) output real data shape(Number of Data by Q) 
    k : (interger) Number of basis function.
    a : (float) learning rate
    epoch : (integer) training epoch
    
    Returns
    -------
    weight history, mse history
    """
    # initalize
    # weight, first MSE
    NumberOfData=x.shape[0]
    xwidth = x.shape[1]
    w_size = xwidth*k+1
    w_his = np.empty([0,w_size])
    mse_his = np.empty(0)
    w_init = []
    for idx in range(0, w_size):
        w_init.append((np.random.rand()*init_space)+init_start)
        
    basis = GeneratorBasis(k,x)
    
    w_his = np.append(w_his, [w_init], axis=0)
    #print('random init w0, w1, w2 ', w_his)
    w_init = np.reshape(w_init,[w_size,1])
    y_hat = np.dot(basis,w_init)
    mse = np.mean((y_hat - y)**2) # Calculate MSE using new weights 
    mse_his = np.append(mse_his, mse)
    
    for epc in range(0,epoch-1):        
        cur_w = w_his[epc]              # load to current weight
        cur_w = cur_w.reshape([w_size,1])
        # weight update
        # 얘 지금 4by4로 나옴

        new_w=np.reshape(cur_w,[1,w_size]) - alpha*(np.dot(np.transpose(np.dot(basis,cur_w) - y),basis)/NumberOfData)
        #new_w = new_w.re
        w_his = np.append(w_his, new_w.reshape([1,w_size]), axis=0) # new weight store
        mse = np.mean((np.dot(basis,new_w.reshape([w_size,1])) - y)**2) # Calculate MSE using new weights 
        mse_his = np.append(mse_his, mse) # MSE store
    return w_his, mse_his
    

#------------------- Data Organize -------------------
#file_path = "C:\\Users\\USER\\lin_regression_data_01.csv"
file_path = "lin_regression_data_01.csv"
open_file = pd.read_csv(file_path,header=None)
df = pd.DataFrame(open_file)

data_order = 0 # 0 : ascending, 1 : descending

numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 

data = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
data = data.values # sorted data

input_mat = np.ones([numberOfData,1]) # size initialize
input_mat[:,0] = data[:,0]

output_mat = data[:,1] # initialize
output_mat=np.reshape(output_mat, [50,1])



# 4-2)
"""
inputs = input_mat[:,0]
outputs = np.reshape(output_mat, (50,))
x_avg = np.mean(inputs)
w0_child = np.mean(outputs*(inputs-x_avg))
w0_parent = np.mean((inputs**2) - (x_avg**2))
w0_op = w0_child/w0_parent
w1_op = np.mean((outputs - (w0_op*inputs)))
print(w0_op,w1_op)


plt.figure(figsize=(12,6))
#565656
plt.scatter(data[:,0], data[:,1], label='Measured Data', c='#565656', s=30)
margin_rate = 0.1
x_plot_step = 0.1

x_axis_start = np.min(input_mat) - (np.max(input_mat)-np.min(input_mat)) * margin_rate
if x_axis_start < 0:
    x_axis_start = 0
x_axis_end = np.max(input_mat) + (np.max(input_mat)-np.min(input_mat)) * margin_rate
print(np.min(input_mat), np.max(input_mat))
print((np.max(input_mat)-np.min(input_mat)) * 0.5)
x_plot = np.arange(x_axis_start, x_axis_end, x_plot_step)
x_plot = np.reshape(x_plot,[x_plot.shape[0],1])

analytic_solution = w0_op * x_plot[:,0] + w1_op
#081890
plt.plot(x_plot,analytic_solution,'--',label="linear regression Analytic solution",color="#1880BA")

k_arr = [3, 4, 5, 6, 7, 8,9,10]
y_min = np.inf
y_max = 0
line_style=['-','-','-']
#colors = ['#EF8511', '#33B339','#F21684']
colors = ['darkorange', 'seagreen', 'deeppink']
#colors = ['cyan', 'limegreen', 'gold']

mse_his = np.empty(0)
l_idx = 0
for k in k_arr:
    weight,basis = PolynomialAnalyticSolution(k, input_mat, output_mat)
    print('k=',k)
    print('weight ',weight)
    y_real = np.dot(basis,weight)
    mse = np.mean((y_real - output_mat)**2) # Calculate MSE using new weights
    mse_his=np.append(mse_his, mse)
    print('mse',mse)
    #plt.plot(input_mat, y_real, label=' k='+str(k))

    new_basis = GeneratorBasis(k,x_plot)
    y_hat = np.dot(new_basis,weight)
    if np.min(y_hat) < y_min:
        y_min = np.min(y_hat)
    if np.max(y_hat) > y_max:
        y_max = np.max(y_hat)
    #plt.plot(x_plot, y_hat, label='Basis Function Analytic solution. k='+str(k),linestyle=line_style[l_idx], color=colors[l_idx])
    plt.plot(x_plot, y_hat, label='Basis Function Analytic solution. k='+str(k))
    l_idx = l_idx + 1

plt.rc('font',size=20)
plt.xlabel('weight[g]',fontsize=20)
plt.ylabel('length[cm]', fontsize=20)
plt.title('Analytical Solution with a Basis Function',fontsize=22)
#plt.title('Raw data',fontsize=22)
plt.grid(True)
plt.legend(fontsize=15)
x_ticks = np.arange(round(np.min(input_mat)-x_plot_step), round(np.max(input_mat)+x_plot_step)+1,1) # x axis ticks interval
#x_ticks = np.arange(x_axis_start, x_axis_end,1) # x axis ticks interval overfitting
#y_ticks = np.arange(round(y_min)+3,round(y_max)-10, 2)
plt.xlim(np.min(input_mat)-1,np.max(input_mat)+0.5)
#plt.xlim(x_axis_start-x_plot_step, x_axis_end+x_plot_step) overfitting
plt.ylim(0,40)
plt.xticks(x_ticks,fontsize=14)
#plt.yticks(y_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""
# 4-3) MSE 출력
"""
plt.figure(figsize=(12,6))
plt.plot(k_arr,mse_his)

plt.rc('font',size=20)
plt.xlabel('k',fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.title('MSE change according to k',fontsize=22)
plt.grid(True)
x_ticks = np.arange(3, 11,1) # x axis ticks interval
#y_ticks = np.arange(round(y_min)+3,round(y_max)-10, 2)
plt.xlim(2.5, 10.5)
#plt.ylim(y_min+3,y_max-10)
plt.xticks(x_ticks,fontsize=16)
#plt.yticks(y_ticks,fontsize=14)
plt.yticks(fontsize=16)
plt.show()
"""


#  Gradient Descent 
# -------- Hyper Parameter ---------
MIN_MAX = 0
Z_SCORE = 1
k_range = [3,4,5,6,7,8,9,10]
learning_rate = 0.05
epoch = 5000
random_start = 0
random_space = 2
Normalization_METHOD = MIN_MAX
# -----------------------------------
# Data Nomalization 
# ------ min-max normalization ------
if Normalization_METHOD == MIN_MAX: 
    x_min_val = np.min(input_mat, axis = 0)
    x_max_val = np.max(input_mat, axis = 0)
    Normalization_input_mat = (input_mat - x_min_val) / (x_max_val-x_min_val)
    y_min_val = np.min(output_mat, axis = 0)
    y_max_val = np.max(output_mat, axis = 0)
    Normalization_output_mat = (output_mat - y_min_val) / (y_max_val-y_min_val)
# ----------------------------------
# ------ z-score normalization ------
elif Normalization_METHOD == Z_SCORE:
    input_mean = np.mean(input_mat, axis = 0)
    input_std = np.std(input_mat, axis = 0)
    Normalization_input_mat = (input_mat-input_mean)/input_std
# ----------------------------------
plt.scatter(input_mat, output_mat, label="raw data", c='#565656', s=30)
inputs = input_mat[:,0]
outputs = np.reshape(output_mat, (50,))
x_avg = np.mean(inputs)
w0_child = np.mean(outputs*(inputs-x_avg))
w0_parent = np.mean((inputs**2) - (x_avg**2))
w0_op = w0_child/w0_parent
w1_op = np.mean((outputs - (w0_op*inputs)))
print(w0_op,w1_op)


margin_rate = 0.50 # minmax margin rate 0.3, z_score margin rate 0.1
x_plot_step = 0.1
x_matrix = input_mat[:,0]

x_axis_start = np.min(x_matrix) - ((np.max(x_matrix)-np.min(x_matrix)) * margin_rate)
if x_axis_start < 0:
    x_axis_start = 0
x_axis_end = np.max(x_matrix) + ((np.max(x_matrix)-np.min(x_matrix)) * margin_rate)
x_plot = np.arange(x_axis_start, x_axis_end, x_plot_step)
x_plot = np.reshape(x_plot,[x_plot.shape[0],1])

x_plot_normal = (x_plot - x_min_val) / (x_max_val-x_min_val)


analytic_solution = w0_op * x_plot + w1_op
x_plot_real = x_plot_normal * (x_max_val - x_min_val) + x_min_val
plt.plot(x_plot,analytic_solution,'--',label="linear regression Analytic solution",color="#1880BA")

mse_save = np.empty((epoch,0))
w_save = np.empty((epoch,0))
mse_min_save = np.empty((0))
for k in k_range:
    plot_basis = GeneratorBasis(k,x_plot_normal)
    w_GDmethod, mse_GDmethod = PolynomialGradientDescent(Normalization_input_mat,Normalization_output_mat, k,learning_rate,epoch,random_start,random_space)
    mse_save = np.hstack((mse_save, mse_GDmethod.reshape([epoch,1]))) # 열로 쌓기
    w_save = np.hstack((w_save, w_GDmethod)) # 열로 쌓기
    
    mse_min = np.inf
    mse_min_idx = 0
    for ep in range(0,epoch):
        if mse_GDmethod[ep] < mse_min:
            mse_min_idx = ep
            mse_min = mse_GDmethod[ep]
    #print('k=',k,' weight=',w_GDmethod[mse_min_idx])
    
    # MSE 계산만 하기 위한 텀
    mse_basis = GeneratorBasis(k,Normalization_input_mat)
    y_hat_only_mse_calculate = np.dot(mse_basis,  w_GDmethod[mse_min_idx,:])
    y_hat_real_only_mse_calculate = y_hat_only_mse_calculate * (y_max_val - y_min_val) + y_min_val
    y_hat_real_only_mse_calculate = np.reshape(y_hat_real_only_mse_calculate,(numberOfData,1)) # size 맞춰줘야 제대로 나옴
    
    
    mse_real = np.mean((y_hat_real_only_mse_calculate - output_mat)**2)
    mse_min_save = np.append(mse_min_save,mse_real)
    print('mse_min=',mse_min_save)
    # -----------|-----------|-----------|
    y_hat_normalized = np.dot(plot_basis,  w_GDmethod[mse_min_idx,:])
    y_hat_real = y_hat_normalized * (y_max_val - y_min_val) + y_min_val
    plt.plot(x_plot_real, y_hat_real, label='Gradient Descent y prediction k='+str(k))
    

plt.rc('font',size=20)
plt.xlabel('input',fontsize=20)
plt.ylabel('output', fontsize=20)
plt.title('Gradient Decent Method Nomalization. alpha='+str(learning_rate)+', epoch='+str(epoch),fontsize=24)
plt.grid(True)
plt.legend(fontsize=15)
x_plot_step = 1
print(np.max(input_mat[:,0]))
x_ticks = np.arange(0, np.max(input_mat[:,0]) + ((np.max(input_mat[:,0]) - np.min(input_mat[:,0]))*0.5),x_plot_step) # x axis ticks interval
lim_rate = 5
plt.xlim(0,round(np.max(input_mat[:,0])+(np.max(input_mat[:,0]) - np.min(input_mat[:,0]))*0.5))
plt.ylim(-20,60)
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()

"""
plot_step = 50
ticks = 500
step_start = 0
step_max = epoch
step = np.arange(step_start,step_max,plot_step)

my_k = 3
start_idx = 0
plt.figure(figsize=(12,6))
print(my_k*input_mat.shape[1]+1)
for k in k_range:
    
    for ws in range(start_idx, my_k*input_mat.shape[1]+1, 1):
        plt.plot(step, w_save[step_start:step_max:plot_step, ws], '--',label='W'+str(ws))

    plt.rc('font',size=20)
    plt.xlabel('step',fontsize=20)
    plt.ylabel('w_value', fontsize=20)
    plt.title('Gradient Decent Weight variation. k='+str(k)+ ' alpha='+str(learning_rate)+'. epoch='+str(epoch),fontsize=24)
    plt.grid(True)
    plt.legend(fontsize=20)
    x_ticks = np.arange(step_start, step_max+plot_step,ticks) # x axis ticks interval
    plt.xlim(step_start-plot_step,step_max) # x label range 0~20
    plt.xticks(x_ticks,fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
"""
plot_step = 1
ticks = 500
step_start = 0
step_max = epoch
step = np.arange(step_start,step_max,plot_step)
step_start = 1000
step = np.arange(step_start,step_max,plot_step)
plt.figure(figsize=(12,6))
print(mse_save.shape)

plt.bar(k_range, mse_min_save)
plt.rc('font',size=20)
plt.xlabel('k',fontsize=20)
plt.ylabel('mse', fontsize=20)
plt.title('Gradient Decent MSE. alpha='+str(learning_rate)+'. epoch='+str(epoch),fontsize=24)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
