import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def GeneratorPolynomialBasis(k,x):
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

def AnalyticSolution(k,x,y):
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

    basis= GeneratorPolynomialBasis(k,x)
    # find the 
    
    weight = (np.linalg.pinv(basis.T @ basis) @ basis.T) @ y
    # 역행렬 존재하지 않음 오류 떠서 pinv사용
    return weight

# Chap.2-1)
def RandomNoise_Augmentation(data, mult, noise, random_method = 0):
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
    r = np.sqrt(data[0,0]**2 + data[0,1]**2) * noise
    for n in range(0,numberOfData,1):
        new_data[mult*n,:] = data[n,:]
        
        
        for aug in range(1,mult,1):
        
            if random_method == UNIFORM:
                randx = (np.random.rand()*2 -1) * r
                randy = (np.random.rand()*2 -1) * r
                
                
            elif random_method == STDNORMAL:
                 # 범위 계산 방식 수정 필요
                 randx = np.random.randn()* (r/2)
                 randy = np.random.randn()* (r/2)
            
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

    
    shuffle_data = np.random.permutation(data)
    
    training_set = shuffle_data[0:train,:]
    
    validation_set = shuffle_data[train:train+validation,:]
    test_set = shuffle_data[train+validation:,:]
    
    return training_set, validation_set, test_set
    

#file_path = "C:\\Users\\USER\\Downloads\\lin_regression_data_01.csv"
file_path = "lin_regression_data_01.csv"
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
diff_avg_x = np.mean(abs(diff[:,0]))
noise=0.085

mult = 20 # 증가 비율 20배
newData = RandomNoise_Augmentation(data,mult,noise,random_method = 0)
newData=newData[np.argsort(newData[:,0])]

newData2 = RandomNoise_Augmentation(data,mult,noise,random_method = 1)
newData2=newData2[np.argsort(newData2[:,0])]
"""
plt.scatter(data[:,0], data[:,1], label="original data", c='red', s=40)
#plt.scatter(newData[:,0], newData[:,1], label="Augmented data(uniform)", c='#ff7f00', s=10, alpha=0.4)
plt.scatter(newData2[:,0], newData2[:,1], label="Augmented data(normal)", c='green', s=10, alpha=0.4)
plt.rc('font',size=18)
plt.xlabel('weight[g]',fontsize=18)
plt.ylabel('length[cm]', fontsize=18)
plt.title('Data Augmentation noise='+str(noise*100)+'[%]',fontsize=22)
plt.grid(True)
plt.legend(fontsize=18)
x_ticks = np.arange(0, round(np.max(newData[:,0])) + 1,2) # x axis ticks interval
plt.xlim(0,round(np.max(newData[:,0]))+1)
plt.xticks(x_ticks,fontsize=16)
plt.yticks(fontsize=16)
plt.show()
"""

#2)
train=5
validation = 3
test = 2
"""
training_set,validation_set,test_set = DataDivide(newData,train,validation,test) # 소수점도 가능
#training_set,validation_set,test_set = DataDivide(data,train,validation,test) # 소수점도 가능
plt.figure(figsize=(12,6))
plt.scatter(training_set[:,0], training_set[:,1], label="training_set", c='blue', s=38, alpha=0.5)
plt.scatter(validation_set[:,0], validation_set[:,1], label="validation_set", c='red', s=38, alpha=0.5)
plt.scatter(test_set[:,0], test_set[:,1], label="test_set", c='green', s=38, alpha=0.5)
plt.rc('font',size=18)
plt.xlabel('weight[g]',fontsize=18)
plt.ylabel('length[cm]', fontsize=18)
ratio_text = '('+str(train)+':'+str(validation)+':'+str(test)+')'
#plt.title('Original Data divide(training:validation:test)='+ratio_text,fontsize=20)
plt.title('Augmentation Data divide(training:validation:test)='+ratio_text,fontsize=20)
plt.grid(True)
plt.legend(fontsize=16)
x_ticks = np.arange(0, round(np.max(newData[:,0])) + 1,2) # x axis ticks interval
plt.xlim(0,round(np.max(newData[:,0]))+1)
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""

#3)

training_set,validation_set,test_set = DataDivide(data,8,0,2) # 소수점도 가능

#데이터가 분할되고 증감을 해야 Training, Test가 큰 차이를 보일 수 있음(?)
#test 데이터는 원본데이터를 유지해야 하므로 하면 증강 하면 안돼



# 최소간격의 배율에 따른 분포 차이 작성
# 왜 최소 간격의 10배로 했는지 작성
mult = 20 # 증가 비율 20배
noise=0.085
aug_training_set = RandomNoise_Augmentation(training_set,mult,noise)
aug_training_set = aug_training_set[np.argsort(aug_training_set[:,0])]

aug_test_set = RandomNoise_Augmentation(test_set,mult,noise)
aug_test_set = aug_test_set[np.argsort(aug_test_set[:,0])]

# 증강된 training set

train_x = aug_training_set[:,0]
train_x = train_x.reshape([train_x.shape[0],1])
train_y = aug_training_set[:,1]
train_y = train_y.reshape([train_y.shape[0],1])
"""
train_x = training_set[:,0]
train_x = train_x.reshape([train_x.shape[0],1])
train_y = training_set[:,1]
train_y = train_y.reshape([train_y.shape[0],1])
"""

test_x = test_set[:,0]
test_x = test_x.reshape([test_x.shape[0],1])
test_y = test_set[:,1]
test_y = test_y.reshape([test_y.shape[0],1])

aug_test_x = aug_test_set[:,0]
aug_test_x = aug_test_x.reshape([aug_test_x.shape[0],1])
aug_test_y = aug_test_set[:,1]
aug_test_y = aug_test_y.reshape([aug_test_y.shape[0],1])
# 증강된 데이터들 출력
"""
plt.figure(figsize=(12,6))
plt.scatter(aug_training_set[:,0],aug_training_set[:,1],label='training set', alpha=0.6, s=15)
plt.scatter(test_set[:,0],test_set[:,1],label='test set', alpha=0.6, s=45, color='red')
plt.xlabel('weight[k]',fontsize=18)
plt.ylabel('length[cm]', fontsize=18)
plt.title('Divide and augmented data',fontsize=20)
plt.grid(True)
plt.legend(fontsize=17)
plt.show()
"""


# min - max Nomalization
x_min_val = np.min(train_x, axis = 0)

x_max_val = np.max(train_x, axis = 0)
train_x_norm = (train_x - x_min_val) / (x_max_val-x_min_val)
y_min_val = np.min(train_y, axis = 0)
y_max_val = np.max(train_y, axis = 0)
train_y_norm = (train_y - y_min_val) / (y_max_val-y_min_val)
test_x_norm = (test_x - x_min_val) / (x_max_val-x_min_val)

aug_test_x_norm = (aug_test_x - x_min_val) / (x_max_val-x_min_val)
# MSE 계산
mse_train_his = np.empty(0)
mse_test_his = np.empty(0)
mse_aug_test_his = np.empty(0)

#k_arr = [2,3,4,5,6,7,8,9,10]
k_arr = np.arange(2,81,1)
margin_rate = 0.2
x_plot_step = 0.1
x_matrix = train_x

x_axis_start = np.min(x_matrix) - ((np.max(x_matrix)-np.min(x_matrix)) * margin_rate)
if x_axis_start < 0:
    x_axis_start = 0
x_axis_end = np.max(x_matrix) + ((np.max(x_matrix)-np.min(x_matrix)) * margin_rate)
x_plot = np.arange(x_axis_start, x_axis_end, x_plot_step)
x_plot = np.reshape(x_plot,[x_plot.shape[0],1])

x_plot_norm = (x_plot - x_min_val) / (x_max_val-x_min_val)

plt.figure(figsize=(12,6))
for k in k_arr:
    weight = AnalyticSolution(k, train_x_norm, train_y_norm)
    train_basis= GeneratorPolynomialBasis(k,train_x_norm)
    plot_basis = GeneratorPolynomialBasis(k,x_plot_norm)
    # MSE Train
    y_train_hat_norm = train_basis @ weight    
    y_train_hat_real = y_train_hat_norm * (y_max_val - y_min_val) + y_min_val
    y_train_hat_real = np.reshape(y_train_hat_real,(train_y.shape[0],1)) # size 맞춰줘야 제대로 나옴
    mse_train = np.mean((y_train_hat_real - train_y)**2) # Calculate MSE using new weights
    mse_train_his=np.append(mse_train_his, mse_train)

    # MSE Test
    test_basis= GeneratorPolynomialBasis(k,test_x_norm)        
    y_test_hat_norm = test_basis @ weight
    y_test_hat_real = y_test_hat_norm * (y_max_val - y_min_val) + y_min_val
    y_test_hat_real = np.reshape(y_test_hat_real,(test_y.shape[0],1)) # size 맞춰줘야 제대로 나옴
    mse_test = np.mean((y_test_hat_real - test_y)**2) # Calculate MSE using new weights
    mse_test_his=np.append(mse_test_his, mse_test)       
    
    # MSE Test _ Augmentation
    aug_test_basis= GeneratorPolynomialBasis(k,aug_test_x_norm)        
    aug_y_test_hat_norm = aug_test_basis @ weight
    aug_y_test_hat_real = aug_y_test_hat_norm * (y_max_val - y_min_val) + y_min_val
    aug_y_test_hat_real = np.reshape(aug_y_test_hat_real,(aug_test_y.shape[0],1)) # size 맞춰줘야 제대로 나옴
    mse_aug_test = np.mean((aug_y_test_hat_real - aug_test_y)**2) # Calculate MSE using new weights
    mse_aug_test_his=np.append(mse_aug_test_his, mse_aug_test)    

    # plot 
    y_hat_plot_norm = plot_basis @ weight
    y_hat_plot_real = y_hat_plot_norm * (y_max_val - y_min_val) + y_min_val
    y_hat_plot_real = np.reshape(y_hat_plot_real,(x_plot.shape[0],1)) # size 맞춰줘야 제대로 나옴
    if k < 6:
        y_clipped = np.clip(y_hat_plot_real, 0, 35)
        plt.plot(x_plot, y_clipped, label='y_hat k='+str(k))


#plt.scatter(train_x,train_y,label='augmented train_set', s=10, alpha=0.4, color='grey')
plt.scatter(aug_test_x,aug_test_y,label='augmented test_set', s=10, alpha=0.4)
plt.scatter(test_x,test_y,label='test_set', s=30, color='red')
plt.rc('font',size=18)
plt.xlabel('weight[k]',fontsize=18)
plt.ylabel('length[cm]', fontsize=18)
plt.title('Prediction Model',fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
x_ticks = np.arange(int(x_axis_start), int(x_axis_end)-0.5,2) # x axis ticks interval
plt.xlim(int(x_axis_start)-0.5,int(x_axis_end)+0.5)
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(k_arr,mse_train_his,label='Augmented training MSE')
#plt.plot(k_arr,mse_train_his,label='Original training MSE')

plt.plot(k_arr,mse_test_his,label='test MSE')
plt.plot(k_arr,mse_aug_test_his,label='Augmented test MSE')
plt.rc('font',size=18)
plt.xlabel('Model Complexity[k]',fontsize=18)
plt.ylabel('MSE', fontsize=18)
plt.title('MSE graph according to k',fontsize=18)
plt.grid(True)
plt.legend(fontsize=15)
x_ticks = np.arange(np.min(k_arr)-1, np.max(k_arr)+1,2) # x axis ticks interva
plt.xlim(np.min(k_arr)-0.5, np.max(k_arr)+0.5)
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()



# 고차항일 수록 정규화를 하지 않아서 OVERFITTING되어 MSE가 매우 큼
# 정규화를 해서 깔끔하게 했다.
