import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Sigmoid(z):
    p = 1 / (1+np.exp(-z))
    return p

def OneHotEncoding(y):
    """
    parameters
    * y : output data
    return
    * new_y : one_hot processed y
    """
    numOfData = y.shape[0]
    onehot_set=np.unique(y)
    numOfQ = len(onehot_set)
    new_y = np.zeros([numOfData, numOfQ])
    
    
    for i in range(0,numOfData):
        for idx in range(0,numOfQ):
            if y[i] == onehot_set[idx]:
                new_y[i,idx] = 1

    return new_y


def Two_Layer_NN(x, y, L, init_start,init_space):
    x = x.T
    N = x.shape[1]
    M = x.shape[0]
    Q = len(np.unique(y))
    v = (np.random.rand(L,M+1)*init_space)+init_start # v weight 초기화
    w = (np.random.rand(Q,L+1)*init_space)+init_start # w weight 초기화
    
    bias = np.ones((1,N)) # bias 추가용
    
    x = np.concatenate((x, bias), axis=0)
    # hidn_1_i : first hidden layer input (before activation function)
    hidn_1_i = v@x
    # hidn_1_o : first hidden layer output (after activation function)
    hidn_1_o = Sigmoid(hidn_1_i)
    hidn_1_o = np.concatenate((hidn_1_o, bias), axis=0)
    # outp_i : output layer input (before activation function)
    outp_i = w@hidn_1_o
    # outp_o : output layer output (after activation function)
    outp_o = Sigmoid(outp_i)
    
    return outp_o # y hat Q by 1
    
file_path = "NN_data.csv"
open_file = pd.read_csv(file_path)
df = pd.DataFrame(open_file)

# -------- find input and output -------- #

col = list(df.columns)
temp = [col for col in col if 'y' in col]
split_idx=col.index(temp[0]) # output 데이터의 시작 열 번호
all_widht = df.shape[1] # 전체 데이터의 열의 갯수
data = df.values
N = data.shape[0] 

input_mat = data[:,:split_idx] # 데이터 자동 분할
output_mat = data[:,split_idx:] # 데이터 자동 분할

output_mat = np.reshape(output_mat, [N,all_widht-split_idx]) # 전체데이터 by 전체 열개수 - y 시작열번호

"""
# input 속성 수 : input_mat의 열의 갯수
# output class 수
"""
# -------- hyper parameter --------
init_space = 2
init_start = -1
hidden_layer_node = np.arange(2,100,1)
# ---------------------------------
one_hot_y = OneHotEncoding(output_mat)
acc_std_array = np.empty(0)
acc_max_array = np.empty(0)
# y_hat : Q by 1
for hln in hidden_layer_node:
    y_hat_mat=Two_Layer_NN(input_mat, output_mat, hln, init_start,init_space) # Neural network
    y_hat_mat = y_hat_mat.T
    

    #%% 0.5 기준으로 hot Encoding
    numOfQ = len(np.unique(output_mat))
    std_y_hat = np.zeros((N,numOfQ))
    data_idx = 0
    for y_hat in y_hat_mat:
        y_hat = np.where(y_hat >= 0.5, 1, 0)
        std_y_hat[data_idx,:] = y_hat
        data_idx = data_idx + 1
    # -------------------

    #%% max만 hot Encoding
    numOfQ = len(np.unique(output_mat))
    one_hot_y_hat = np.zeros((N,numOfQ))
    data_idx = 0
    for y_hat in y_hat_mat:
        idx = np.where(y_hat[:] == np.max(y_hat))
        one_hot_y_hat[data_idx,idx] = 1
        data_idx = data_idx + 1
    # -------------------    

    # sigmoid 0.5 기준으로 나눈거의 정확도
    compare_std = np.all(one_hot_y == std_y_hat, axis=1)
    acc_std_array = np.append(acc_std_array,(np.sum(compare_std == True) / N) * 100)

    # max에 정확도
    compare_max = np.all(one_hot_y == one_hot_y_hat, axis=1)
    acc_max_array = np.append(acc_max_array,(np.sum(compare_max == True) / N) * 100)

    compare_max= np.reshape(compare_max, [compare_max.shape[0],1])
    #print(acc_std_array.shape)
    #print(acc_max_array.shape)

    true_target = []
    realClass=np.unique(output_mat)

    for q in realClass:
        sel_class=(output_mat == q)
        sel_class_cnt  = np.sum(sel_class)
    
        correct = np.sum(compare_max[sel_class])
        true_target.append(correct / sel_class_cnt * 100)

    #print(true_target)

print('average accuracy std: ',np.average(acc_std_array))    
print('average accuracy max: ',np.average(acc_max_array))
plt.figure(figsize=(12,6))
plt.plot(hidden_layer_node, acc_std_array, label="0.5 standard 1 set accuracy")
plt.plot(hidden_layer_node, acc_max_array, label="max value 1 set accuracy")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=15)
xtick = np.arange(1,np.max(hidden_layer_node)+4,4)
ytick = np.arange(0,101,4)
plt.title('Accuracy for change in number of hidden layer nodes',fontsize=22)
plt.xlabel('Hidden Layer Node',fontsize=18)
plt.ylabel('accuracy[%]',fontsize=18)
plt.xticks(xtick,fontsize=18)
plt.yticks(ytick,fontsize=18)
plt.show()
