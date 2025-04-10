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

def GeneratorBasis(k,numData,width, x):
    """
    Parameters
    ----------
    k : interger
        Number of basis function.
    numData : interger
        Number of data
    width : interger
        number of input feature
    x : (float)matrix
        input value

    Returns
    -------
    basis : np.array[numData][width * k + 1]
        basis.

    """
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
    x_width = x.shape[1]
    NumberOfData = x.shape[0]
    
    basis= GeneratorBasis(k,numberOfData,x_width,x)
    # weight[1,x_width*k+1]
    # find the weight
    for rm in range(0, x_width, 1):
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
        
    basis = GeneratorBasis(k,NumberOfData,xwidth, x)
    
    w_his = np.append(w_his, [w_init], axis=0)
    print('random init w0, w1, w2 ', w_his)
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
file_path = "C:\\Users\\USER\\lin_regression_data_01.csv"
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
#-----------------------------------------------------
"""
# k = 3)
weight,basis = PolynomialAnalyticSolution(8,input_mat,output_mat)

plt.scatter(data[:,0], data[:,1])
y_real = np.dot(basis,weight)
mse = np.mean((y_real - output_mat)**2) # Calculate MSE using new weights 
plt.plot(input_mat, y_real)

x_axis_start = np.min(input_mat) - (np.max(input_mat)-np.min(input_mat)) * 0.45
x_axis_end = np.max(input_mat) + (np.max(input_mat)-np.min(input_mat)) * 0.45
x_step = 0.1
x_plot = np.arange(x_axis_start, x_axis_end, x_step)

x_plot = np.reshape(x_plot,[x_plot.shape[0],1])
new_basis = GeneratorBasis(8,x_plot.shape[0], x_plot.shape[1],x_plot)
y_hat = np.dot(new_basis,weight)
plt.plot(x_plot, y_hat, label='margin')
"""
#----------------------------------------------------
plt.scatter(data[:,0], data[:,1])
margin_rate = 0
x_plot_step = 0.1

x_axis_start = np.min(input_mat) - (np.max(input_mat)-np.min(input_mat)) * margin_rate
x_axis_end = np.max(input_mat) + (np.max(input_mat)-np.min(input_mat)) * margin_rate
x_plot = np.arange(x_axis_start, x_axis_end, x_plot_step)
x_plot = np.reshape(x_plot,[x_plot.shape[0],1])
"""
k_arr = [3,6,8]
for k in k_arr:
    print(k)
    weight,basis = PolynomialAnalyticSolution(k,input_mat,output_mat)

    y_real = np.dot(basis,weight)
    mse = np.mean((y_real - output_mat)**2) # Calculate MSE using new weights 
    plt.plot(input_mat, y_real, label='real y_hat k='+str(k))

    new_basis = GeneratorBasis(k,x_plot.shape[0], x_plot.shape[1],x_plot)
    y_hat = np.dot(new_basis,weight)
    #plt.plot(x_plot, y_hat, label='margin y_hat k='+str(k))

plt.rc('font',size=20)
plt.xlabel('input',fontsize=20)
plt.ylabel('output', fontsize=20)
#plt.title('Gradient Decent Weight variation. alpha='+str(learning_rate),fontsize=24)
plt.grid(True)
plt.legend(fontsize=15)
x_ticks = np.arange(x_axis_start, x_axis_end+x_plot_step,1) # x axis ticks interval
plt.xlim(x_axis_start-x_plot_step,x_axis_end+x_plot_step) # x label range 0~20
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""

#  Gradient Descent 
wss, mse = PolynomialGradientDescent(input_mat,output_mat,3,0.01,5000,-10,100)
basis = GeneratorBasis(3,x_plot.shape[0], x_plot.shape[1],x_plot)
www = wss[4999,:]

y_descent = np.dot(new_basis,www)
plt.plot(x_plot, y_descent, label='grd k='+str(k))

plt.rc('font',size=20)
plt.xlabel('input',fontsize=20)
plt.ylabel('output', fontsize=20)
#plt.title('Gradient Decent Weight variation. alpha='+str(learning_rate),fontsize=24)
plt.grid(True)
plt.legend(fontsize=15)
x_ticks = np.arange(x_axis_start, x_axis_end+x_plot_step,1) # x axis ticks interval
plt.xlim(x_axis_start-x_plot_step,x_axis_end+x_plot_step) # x label range 0~20
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()










