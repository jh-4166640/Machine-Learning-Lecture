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



def W_BackPropagation(hidn_o , y_hat, y_real, batch, Q, L):
    delta = (2*y_hat) @ (y_hat-y_real).T @ (1-y_hat)  # Q by N
    ww = []
    for n in range(0,batch):
        d = np.reshape(delta[:,n], [Q,1])
        hid = np.reshape(hidn_o[:,n], [L+1,1])
        ws = d @ hid.T
        ww.append(ws)
    ww = np.stack(ww)
    mean_w = np.mean(ww,axis=0)
    return mean_w

def V_BackPropagation(x_in, w_mat ,hidn_o , y_hat, y_real, batch, L, M, Q):
    delta = (2*y_hat) @ (y_hat-y_real).T @ (1-y_hat)  # Q by N
    vv=[]
    for n in range(0,batch):
        sums=0
        for q in range(0,Q):
            d = delta[q,n]
            ww = w_mat[q,:]
            sums = d*ww + sums
        kkk = hidn_o[:,n] * (1-hidn_o[:,n]) # 3by batch
        kkk = np.reshape(kkk, [L,1])
        x_inin = np.reshape(x_in[:,n], [M+1,1])
        vvs= sums * kkk @ x_inin.T
        vv.append( vvs)
    vv = np.stack(vv)
    mean_v = np.mean(vv,axis=0)
    return mean_v
   
    
def Two_Layer_NN(x, y, L, alpha, batch, epoch,  init_start,init_space):
    
    x = x.T
    N = x.shape[1]
    M = x.shape[0]
    Q = len(np.unique(y))
    _one_hot_y = OneHotEncoding(y)
    v = (np.random.rand(L,M+1)*init_space)+init_start # v weight 초기화 size v : (L by M+1)
    w = (np.random.rand(Q,L+1)*init_space)+init_start # w weight 초기화 size w : (Q by L+1)
    bias = np.ones((1,N)) # bias 추가용
    x = np.concatenate((x, bias), axis=0)
    
    step_max = round(N/batch)
    cnts = batch_size*step_max
    
    if cnts < N:
        if N-cnts <= batch :
            step_max = step_max + 1
        elif N-cnts > batch:
            step_max = step_max + 2
    
    for epc in range(0,epoch):
        x_shf = x[:,:]
        x_shf = np.concatenate((x_shf,_one_hot_y.T),axis=0)
        x_shf = x_shf.T
        np.random.shuffle(x_shf)
        x_shf = x_shf.T # shuffled data
        y_shf = x_shf[M+1:,:] # shuffled y data
        x_shf = x_shf[0:M+1,:] # shuffled x data
        
        start_idx = 0
        
        for step in range(0, step_max):
            cur_x = x_shf[:,start_idx : (step+1)*batch] # M+1 by batch
            cur_y = y_shf[:,start_idx : (step+1)*batch] # 1 by batch
            
            start_idx = (step+1)*batch
            # hidn_1_i : first hidden layer input (before activation function)
            hidn_1_i = v@cur_x # L by batch
            # hidn_1_o : first hidden layer output (after activation function)
            hidn_1_o = Sigmoid(hidn_1_i)
            bias_batch = np.ones((1,batch)) # bias 추가용
            hidn_1_o = np.concatenate((hidn_1_o, bias_batch), axis=0) # L+1 by batch
            
            # outp_i : output layer input (before activation function)
            outp_i = w@hidn_1_o # Q by batch
            # outp_o : output layer output (after activation function)
            outp_o = Sigmoid(outp_i) # Q by batch
            
            new_w = W_BackPropagation(hidn_1_o, outp_o, cur_y, batch, Q, L)
            new_v = V_BackPropagation(cur_x,w, hidn_1_o[0:L,:], outp_o, cur_y, batch, L, M, Q)
    
    
    
    
    #A = BackPropagation(hidn_1_o, outp_o.T, _one_hot_y) # Q by L+1
    #print(A.size)
    
    return outp_o # y hat: Q by N
    
file_path = "C:\\Users\\USER\\Downloads\\NN_data.csv"
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
hidden_layer_node = 3
hln=hidden_layer_node
batch_size = 16# 2^n
epoch = 5000
learning_rate = 0.01
# ---------------------------------
one_hot_y = OneHotEncoding(output_mat)
acc_std_array = np.empty(0)
acc_max_array = np.empty(0)
# y_hat : Q by 1


y_hat_mat=Two_Layer_NN(input_mat, output_mat, hln, learning_rate ,batch_size, epoch, init_start,init_space) # Neural network
y_hat_mat = y_hat_mat.T
        

