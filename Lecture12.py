import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#01090711885 조교선배님 전화번호

def feature_1(input_data):
    col_sum = np.sum(input_data, axis=0)
    all_sum = np.sum(col_sum)
    pdf = col_sum / all_sum
    xx = np.arange(0,pdf.shape[0],1)
    exp = sum(xx*pdf)
    var = sum((xx-exp)**2*pdf)
    return var

def feature_2(input_data):
    # 세로에서 0으로 나뉜 갯수
    mask = np.zeros([input_data.shape[0],input_data.shape[1]])
    mask[:,9:19] = 1
    dt = input_data * mask
    row_sum = np.sum(dt, axis=1)
    row_sum[row_sum < 1.5] = 0
    first_idx = 0
    for i in range(0,input_data.shape[0]):
        if row_sum[i] != 0: 
            first_idx = i
            break
    
    row_sum = row_sum[first_idx:]
    row_sum = np.reshape(row_sum,[row_sum.shape[0],1])
    change = 0
    old_val = False

    if row_sum[0] == 0 : old_val = False
    else               : old_val = True
    for idx in range(1,row_sum.shape[0]):
        if row_sum[idx] == 0 : new_val = False
        else                 : new_val = True
        
        if old_val != new_val: change+=1
        old_val = new_val
    
    return change

def feature_3(input_data):
    #가로에서 0으로 나뉜 갯수
    mask = np.zeros([input_data.shape[0],input_data.shape[1]])
    mask[9:19,:] = 1
    dt = input_data * mask
    col_sum = np.sum(dt, axis=1)
    col_sum[col_sum < 1.5] = 0
    first_idx = 0
    for i in range(0,input_data.shape[0]):
        if col_sum[i] != 0: 
            first_idx = i
            break
    
    col_sum = col_sum[first_idx:]
    col_sum = np.reshape(col_sum,[col_sum.shape[0],1])
    change = 0
    old_val = False
    if col_sum[0] == 0 : old_val = False
    else               : old_val = True
    for idx in range(1,col_sum.shape[0]):
        if col_sum[idx] == 0 : new_val = False
        else                 : new_val = True
        
        if old_val != new_val: change+=1
        old_val = new_val
    
    return change


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


    
    
    
path = "C:\\Users\\USER\\Downloads\\MNIST Data"
my_dir = np.array(os.listdir(path))
DATA_CNT = np.size(my_dir)
my_dir=my_dir.reshape([DATA_CNT,1])
#train_set, val_set, test_set = DataDivide(my_dir, 7,0,3)
x_mat = np.empty((0,28**2))
y_mat = np.empty((0,1))
for i in range(0,DATA_CNT):
    sel_data = pd.read_csv(path+"\\"+my_dir[i,0],header=None)
    x=np.reshape(sel_data, [1,-1])
    classes = my_dir[i,0].split('_')
    y=np.reshape(classes[0],(1,1))
    x_mat = np.vstack([x_mat, x])
    y_mat = np.vstack([y_mat, y])

"""
input_mat = np.empty([0,0])
ftr1 = np.empty(0)
ftr2 = np.empty(0)
ftr3 = np.empty(0)
for i in range(0, DATA_CNT):
    sel_data = pd.read_csv(path+"\\"+my_dir[i,0],header=None)
    ftr1 = np.append(ftr1, feature_1(sel_data))
    # ftr2, ftr3에서 외곽선 0을 모두 제거 한 뒤 계산하는게 더 나을 듯
    ftr2 = np.append(ftr2, feature_2(sel_data))
    ftr3 = np.append(ftr3, feature_3(sel_data))
"""    
    
