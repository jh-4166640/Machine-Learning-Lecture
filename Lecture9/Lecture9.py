import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def GenerateGausianBasis(x,mu,sigma):
    
    mu = np.insert(np.reshape(mu,[1,2]),2,0,axis=1)
    return np.exp(-(x-mu[0,:])**2 / (2*sigma**2))

def Sigmoid(z):
    p = 1 / (1+np.exp(-z))
    return p

def GradientDescent(x, y, alpha, epoch, init_start, init_space):
    """
    Gradient Descent Method Function
    
    x : (Matrix) input data shape(Number of Data by Number of Feature + 1)
    y : (Matrix) output real data shape(Number of Data by Q) 
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
    w_his = np.empty([0,3])
    cee_his = np.empty(0)
    w_init = []
    for idx in range(0, 3):
        w_init.append((np.random.rand()*init_space)+init_start)
        
    w_his = np.append(w_his, [w_init], axis=0)
    #print('random init w0, w1, w2 ', w_his)
    w_init = np.reshape(w_init,[3,1])
    z = x@w_init
    p = Sigmoid(z)
    cee = -np.mean(y.T@np.log(p) + (1-y).T@np.log(1-p))
    cee_his = np.append(cee_his, cee) # MSE store
    
    for epc in range(0,epoch-1):        
        cur_w = w_his[epc]              # load to current weight
        cur_w = cur_w.reshape([3,1])
        # weight update
        # 얘 지금 4by4로 나옴
        z = x@cur_w
        p = Sigmoid(z)

        new_w = np.reshape(cur_w,[1,3]) - alpha*((p-y).T@x)/NumberOfData
        w_his = np.append(w_his, new_w.reshape([1,3]), axis=0) # new weight store
        cee = -np.mean(y.T@np.log(p) + (1-y).T@np.log(1-p))
        cee_his = np.append(cee_his, cee) # MSE store
    return w_his, cee_his


def MSEGradientDescent(x, y, alpha, epoch, init_start, init_space):
    """
    Gradient Descent Method Function
    
    x : (Matrix) input data shape(Number of Data by Number of Feature + 1)
    y : (Matrix) output real data shape(Number of Data by Q) 
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
    w_his = np.empty([0,3])
    mse_his = np.empty(0)
    w_init = []
    for idx in range(0, 3):
        w_init.append((np.random.rand()*init_space)+init_start)
        
    w_his = np.append(w_his, [w_init], axis=0)
    #print('random init w0, w1, w2 ', w_his)
    w_init = np.reshape(w_init,[3,1])
    z = x@w_init
    p = Sigmoid(z)
    mse = np.mean((p - y)**2) # Calculate MSE using new weights 
    mse_his = np.append(mse_his, mse)
    grad = 2 * (p - y) * p * (1 - p)
    for epc in range(0,epoch-1):        
        cur_w = w_his[epc]              # load to current weight
        cur_w = cur_w.reshape([3,1])
        # weight update
        # 얘 지금 4by4로 나옴
        z = x@cur_w
        p = Sigmoid(z)
        
        new_w = np.reshape(cur_w,[1,3]) - alpha*(((p - y) * p * (1 - p)).T@x)/NumberOfData
        w_his = np.append(w_his, new_w.reshape([1,3]), axis=0) # new weight store
        mse = np.mean((p - y)**2) # Calculate MSE using new weights 
        mse_his = np.append(mse_his, mse)
    return w_his, cee_his



def BasisGradientDescent(x, y, alpha, epoch, init_start, init_space, mu, sigma):
    """
    Gradient Descent Method Function
    
    x : (Matrix) input data shape(Number of Data by Number of Feature + 1)
    y : (Matrix) output real data shape(Number of Data by Q) 
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
    w_his = np.empty([0,3])
    cee_his = np.empty(0)
    w_init = []
    for idx in range(0, 3):
        w_init.append((np.random.rand()*init_space)+init_start)
        
    w_his = np.append(w_his, [w_init], axis=0)
    #print('random init w0, w1, w2 ', w_his)
    w_init = np.reshape(w_init,[3,1])
    z = x@w_init
    p = Sigmoid(z)
    cee = -np.mean(y.T@np.log(p) + (1-y).T@np.log(1-p))
    cee_his = np.append(cee_his, cee) # MSE store
    
    for epc in range(0,epoch-1):        
        cur_w = w_his[epc]              # load to current weight
        cur_w = cur_w.reshape([3,1])
        # weight update
        # 얘 지금 4by4로 나옴
        z = x@cur_w
        p = Sigmoid(z)

        new_w = np.reshape(cur_w,[1,3]) - alpha*((p-y).T@x)/NumberOfData
        w_his = np.append(w_his, new_w.reshape([1,3]), axis=0) # new weight store
        cee = -np.mean(y.T@np.log(p) + (1-y).T@np.log(1-p))
        cee_his = np.append(cee_his, cee) # MSE store
    return w_his, cee_his



epoch = 10000
alpha = 0.1
init_start = -0.1
init_space = 1
mu = np.mean(input_mat[:,0:2], axis=0)
sigma = 1

file_path = "C:\\Users\\USER\\Downloads\\logistic_regression_data.csv"
open_file = pd.read_csv(file_path,index_col = 0)
df = pd.DataFrame(open_file)


numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 

data = df.values
input_mat = np.ones([numberOfData,widthOfData]) # size initialize

for idx in range(widthOfData):
    if widthOfData-1 == idx:
        output_mat = data[:,idx] # initialize
    else :
        input_mat[:,idx] = data[:,idx]
    
output_mat = output_mat.reshape([500,1])

mu = np.mean(input_mat[:,0:2], axis=0)

basis = GenerateGausianBasis(input_mat,mu,sigma)

w_his, cee_his = GradientDescent(input_mat, output_mat, alpha, epoch, init_start, init_space)
w_his_mse, mse_his = MSEGradientDescent(input_mat, output_mat, alpha, epoch, init_start, init_space)

x0 = np.arange(-2, 8, 0.1)
x1 = -(w_his[epoch-1,0]/w_his[epoch-1,1])*x0-(w_his[epoch-1,2]/w_his[epoch-1,1])

xmse = -(w_his_mse[epoch-1,0]/w_his_mse[epoch-1,1])*x0-(w_his_mse[epoch-1,2]/w_his_mse[epoch-1,1])

plt.figure(figsize=(12,6))
plt.scatter(input_mat[0:250,0], input_mat[0:250,1], marker='o',label="input", s=10)
plt.scatter(input_mat[250:,0], input_mat[250:,1], marker='x',label="input", s=10)
plt.plot(x0,x1,label='cee')
plt.plot(x0,xmse,label='mse')
plt.rc('font',size=18)

plt.grid(True)
plt.legend(fontsize=16)
xtick = np.arange(np.min(input_mat[:,0])-0.5,np.max(input_mat[:,0])+0.5,0.5)
ytick = np.arange(np.min(input_mat[:,1])-0.5,np.max(input_mat[:,1])+0.5,0.5)
#plt.xticks(xtick,fontsize=14)
#plt.yticks(ytick,fontsize=14)
#plt.xlim(np.min(input_mat[:,0])-0.5,np.max(input_mat[:,0])+0.5)
#plt.ylim(np.min(input_mat[:,1])-0.5,np.max(input_mat[:,1])+0.5)
plt.show()

