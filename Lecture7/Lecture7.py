import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Chap.2-1)
def RandomNoise_Augmentation(data, mult, rand_range, random_method = 0):
    """
    data : (Matrix) input data
    mult : (interger) Augmentat value
    rand_range : [xstart,ystart] 1by2 matrix random distribution
    random_method : 0 : uniform, 1 : standard normal distribution
    
    Returns
    -------
    Augemnted data : [numberOfData*mult,widthOfData]
    """
    UNIFORM = 0
    STDNORMAL = 1
    numberOfData = data.shape[0]
    widthOfData = data.shape[1]
    new_data = np.zeros([numberOfData*mult, widthOfData])
    x_random_range = rand_range[0] - (-1*rand_range[0])
    y_random_range = rand_range[1] - (-1*rand_range[1])
    
    
    for n in range(0,numberOfData,1):
        new_data[mult*n,:] = data[n,:]
        for aug in range(0,mult,1):
            
            if random_method == UNIFORM:
                randx = np.random.rand()*x_random_range+(-1*rand_range[0])
                randy = np.random.rand()*y_random_range+(-1*rand_range[1])
                
            elif random_method == STDNORMAL:
                 # 범위 계산 방식 수정 필요
                 randx = np.random.randn()
                 randy = np.random.randn()
                 
            new_data[(mult*n)+(1*aug),0] = data[n,0] + randx
            new_data[(mult*n)+(1*aug),1] = data[n,1] + randy
    return new_data
    
def DataDivide(data, train, validation, test):
    """
    data : (Matrix) input data
    train : (numerical) training set raito
    validation : (numerical) validation set raito
    test : (numerical) test set raito
    
    Returns
    -------
    divide data : 
    """
    
    n=data.shape[0]
    all_ratio = train+validation+test
    train = int(n*train/all_ratio)
    validation = int(n*validation/all_ratio)
    test = int(n*test/all_ratio)
    print(train, validation,test)
    print(train+validation+test)
    
    shuffle_data = np.random.permutation(data)
    
    training_set = shuffle_data[0:train,:]
    
    validation_set = shuffle_data[train:train+validation,:]
    test_set = shuffle_data[train+validation:,:]
    
    return training_set, validation_set, test_set
    


file_path = "C:\\Users\\USER\\Downloads\\lin_regression_data_01.csv"
open_file = pd.read_csv(file_path,header=None)
df = pd.DataFrame(open_file)

data_order = 0 # 0 : ascending, 1 : descending

numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 

data = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
data = data.values # sorted data


#1)

# 열에 대해서 최소간격 구하기
diff = np.diff(data, axis=0)
diff[diff==0] = np.inf
x_diff_min = np.min(abs(diff[:,0]))
y_diff_min = np.min(abs(diff[:,1]))
x_rate = 10
y_rate = 10
# 최소간격의 배율에 따른 분포 차이 작성
# 왜 최소 간격의 10배로 했는지 작성
rand_range = [x_diff_min*x_rate, y_diff_min*y_rate]
mult = 20 # 증가 비율 20배


newData = RandomNoise_Augmentation(data,mult,rand_range)
newData = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
newData = newData.values # sorted data
"""
plt.scatter(data[:,0], data[:,1], label="original data", c='blue', s=40)
plt.scatter(newData[:,0], newData[:,1], label="Augmented data", c='orange', s=15)
plt.rc('font',size=18)
plt.xlabel('weight[g]',fontsize=18)
plt.ylabel('length[g]', fontsize=18)
plt.title('Data Augmentation',fontsize=18)
plt.grid(True)
plt.legend(fontsize=15)
x_ticks = np.arange(0, round(np.max(newData[:,0])) + 1,2) # x axis ticks interval
plt.xlim(0,round(np.max(newData[:,0]))+1)
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""


#2)
training_set,validation_set,test_set = DataDivide(newData,5,3,2) # 소수점도 가능
"""
plt.scatter(training_set[:,0], training_set[:,1], label="training_set", c='blue', s=30, alpha=0.4)
plt.scatter(validation_set[:,0], validation_set[:,1], label="validation_set", c='red', s=30, alpha=0.4)
plt.scatter(test_set[:,0], test_set[:,1], label="test_set", c='green', s=30, alpha=0.4)
plt.rc('font',size=18)
plt.xlabel('weight[g]',fontsize=18)
plt.ylabel('length[g]', fontsize=18)
plt.title('Data Augmentation',fontsize=18)
plt.grid(True)
plt.legend(fontsize=15)
x_ticks = np.arange(0, round(np.max(newData[:,0])) + 1,2) # x axis ticks interval
plt.xlim(0,round(np.max(newData[:,0]))+1)
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""

#3)
training_set,validation_set,test_set = DataDivide(newData,8,0,2) # 소수점도 가능
training_set = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
training_set = training_set.values # sorted data

validation_set = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
validation_set = validation_set.values # sorted data

test_set = df.sort_values(by=[0], axis=data_order) # by=[standard] data sorting
test_set = test_set.values # sorted data

mse_train_his = np.empty(0)
mse_test_his = np.empty(0)

#k_arr = [2,3,4,5,6,7,8,9,10]
k_arr = np.arange(2,10,1)
train_x = training_set[:,0]
train_x = train_x.reshape([train_x.shape[0],1])
train_y = training_set[:,1]
train_y = train_y.reshape([train_y.shape[0],1])

test_x = test_set[:,0]
test_x = test_x.reshape([test_x.shape[0],1])
test_y = test_set[:,1]
test_y = test_y.reshape([test_y.shape[0],1])

for k in k_arr:
    weight,train_basis = PolynomialAnalyticSolution(k, train_x, train_y)
    
    y_train_hat = np.dot(train_basis,weight)
    mse_train = np.mean((y_train_hat - train_y)**2) # Calculate MSE using new weights
    mse_train_his=np.append(mse_train_his, mse_train)
    print('train','k=',k, 'mse=',mse_train)     
    
    if k == 7:
        plt.figure(figsize=(12,6))
        plt.scatter(train_x,train_y)
        plt.plot(train_x,y_train_hat)
        
        
    test_basis = GeneratorBasis(k,test_x)
    y_test_hat = np.dot(test_basis,weight)
    mse_test = np.mean((y_test_hat - test_y)**2) # Calculate MSE using new weights
    mse_test_his=np.append(mse_test_his, mse_test)       

    print('test','k=',k, 'mse=',mse_test)         
    
plt.figure(figsize=(12,6))
plt.plot(k_arr,mse_train_his,label='training MSE')
plt.plot(k_arr,mse_test_his,label='test MSE')
plt.rc('font',size=18)
plt.xlabel('Model Complexity[k]',fontsize=18)
plt.ylabel('MSE', fontsize=18)
plt.title('k MS',fontsize=18)
plt.grid(True)
plt.legend(fontsize=15)
x_ticks = np.arange(np.min(k_arr)-1, np.max(k_arr)+1,1) # x axis ticks interva
plt.xlim(np.min(k_arr)-0.5, np.max(k_arr)+0.5)
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()









