
"""
* 2025-04-10
* 임베디드 시스템 전공
* 2021146036
* 최지헌
* week 6
"""

"""
u_k = input_mat[].min() + gauss_k*(input_mat[].max() - input_mat[].min())/count_gauss - 1
sigma = (input_mat[].max() - input_mat[].min()) / (count_gauss - 1)

gaussian = np.exp(0.5*((input_mat[] - u_k) / sigma)**2)
"""

#(max-min) * 0.25 # input max - min 범위의 몇%의 마진을 더 취하겠다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
    asdasd

    """
    x_width = x.shape[1]
    NumberOfData = x.shape[0]
    basis=np.ones([numberOfData,x_width * k + 1]) # array size [M][N][k] # M is Data feature, N is number of data
      
    for numx in range(0,x_width,1):
        for exp in range(0,k,1):
            basis[:,numx*k + exp + 1]=x[:,numx]**(exp+1)
    
    # find the weight
    for rm in range(0, x_width, 1):
        weight = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(basis),basis)),np.transpose(basis)),y)
    

    return weight, basis

    



file_path = "C:\\Users\\USER\\lin_regression_data_01.csv"
open_file = pd.read_csv(file_path,header=None)
df = pd.DataFrame(open_file)

data_order = 0 # 0 : ascending, 1 : descending

## -- User enter Variables --
learning_rate = 0.003 # learning rate
random_init_start = [-10,-20] # random start value # input size+1 by 1
random_init_space = [100,100] # random space       # input size+1 by 1
## --------------------------


numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 

data = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
data = data.values # sorted data

input_mat = np.ones([numberOfData,1]) # size initialize
input_mat[:,0] = data[:,0]
# 50by2
output_mat = data[:,1] # initialize
output_mat=np.reshape(output_mat, [50,1])

weight,basis = PolynomialAnalyticSolution(3,input_mat,output_mat)

plt.scatter(data[:,0], data[:,1])
y_hat = np.dot(basis,weight)
plt.plot(input_mat, y_hat)

