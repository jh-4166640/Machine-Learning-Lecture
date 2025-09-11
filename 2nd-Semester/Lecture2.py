"""
NN인수로 <- [hidden Layer 갯수, layer에서의 node 갯수]

activation function과
partial activation function을 만들어서 집어 넣는다.
Legacy = Legacy * output(1-output) # sigmoid니까
w = w_old-alpha * Legacy * input
Legacy = Legacy * w_old

backpropagation을 layer 갯수 만큼 반복문 반복
forwardpropagation을 layer 갯수 만큼 반복문 반복
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def ActivationFunc(z):
    #sigmoid
    p = 1 / (1+np.exp(-z))
    return p
def DifferentialByActiveFunc(arg):
    #sigmoid
    return arg * (1-arg)

def LossDifferentialFunc(y_hat,y_real):
    #MSE
    return 2 * (y_hat-y_real)

def BiasAdd(data, bch):
    bias = np.ones((1, bch))
    newData = np.concatenate((data, bias), axis=0)
    return newData


"""
class Neural_Network():
    
    param_mat=[]
    param_his=np.empty((0,0,0))
    loss_his =np.empty((0,0,0))
    
    def __init__(self, input_sz, layer_sz=1, hidden_node=[2], output_class=3):
        self.input_size = input_sz
        self.layer_size = layer_sz
        self.hidden_node = hidden_node
        self.output_class = output_class
        
        if hidden_node.size() == layer_sz
            
        for layer_cnt in range(0,self.layer_size+1,1):
            print(layer_cnt)
        
        
"""
### 여기부터
file_path = "C:\\Users\\user\\Downloads\\NN_data.csv"
open_file = pd.read_csv(file_path)
df = pd.DataFrame(open_file)

# -------- find input and output -------- #

col = list(df.columns)
temp = [col for col in col if 'y' in col]
split_idx=col.index(temp[0]) # output 데이터의 시작 열 번호
all_widht = df.shape[1] # 전체 데이터의 열의 갯수
data = df.values
N = data.shape[0] 
N = int(N*0.7) # train ratio
train_set, validation_set, test_set = DataDivide(data, 7, 0, 3)


input_mat = train_set[:,:split_idx] # 데이터 자동 분할
output_mat = train_set[:,split_idx:] # 데이터 자동 분할
bias = np.ones((N, 1)) # bias 추가용 # 0.7은 train set 70% 분할 해서
input_mat = np.concatenate((input_mat, bias), axis=1)

output_mat = np.reshape(output_mat, [train_set.shape[0],all_widht-split_idx]) # 전체데이터 by 전체 열개수 - y 시작열번호
x = input_mat.T
one_hot_y = OneHotEncoding(output_mat)
### -------------------------------------------------------------------------------------------------------

init_interval = 10
init_start    = -5
batch = 64
epoch = 10
alpha = 0.06

step_max = round(N/batch)
cnts = batch*step_max
if cnts < N:
    if N-cnts <= batch :
        step_max = step_max + 1
    elif N-cnts > batch:
        step_max = step_max + 2

input_size = np.shape(input_mat)[1]-1
layer_size = 4
hdn_node = [2,2,2]
output_class = len(np.unique(output_mat))


if layer_size != len(hdn_node)+1:
    print("size not match!")

hdn_node.insert(0,input_size)
hdn_node.append(output_class)

MAX_NODE_SIZE = max(hdn_node)+1


param_mat=np.zeros((layer_size,MAX_NODE_SIZE,MAX_NODE_SIZE))
param_his=np.zeros((layer_size,MAX_NODE_SIZE,MAX_NODE_SIZE))
loss_his =np.zeros((layer_size,MAX_NODE_SIZE,MAX_NODE_SIZE))
acc_his  =np.zeros((layer_size,MAX_NODE_SIZE,MAX_NODE_SIZE))
var_mat  =np.zeros((layer_size,MAX_NODE_SIZE-1, batch))
#layer_num = 0

# --- Initialize Parameters --- 
for layer_num in range(0,layer_size,1):
    rand_weight = (np.random.rand(hdn_node[layer_num+1],hdn_node[layer_num]+1)*init_interval)+init_start
    param_mat[layer_num,0:hdn_node[layer_num+1],0:hdn_node[layer_num]+1] = rand_weight
# -----------------------------     
for epc in range(0,epoch):
    x_shf = x[:,:]
    x_shf = np.concatenate((x_shf,one_hot_y.T),axis=0)
    x_shf = x_shf.T
    np.random.shuffle(x_shf)
    x_shf = x_shf.T # shuffled data
    y_shf = x_shf[input_size+1:,:] # shuffled y data
    x_shf = x_shf[0:input_size+1,:] # shuffled x data
    
    start_idx = 0
    for step in range(0, step_max):
        if (step+1)*batch >= N :
            end_idx = N-1
        else:
            end_idx = (step+1)*batch
            
        cur_x = x_shf[:,start_idx : end_idx] # M+1 by batch
        cur_y = y_shf[:,start_idx : end_idx] # 1 by batch
        batchs = end_idx - start_idx
        start_idx = end_idx
        
        
        forward_layer= 0
        # --- Forward Propagation ---
        cx = cur_x # input_node+1 by batch size
        for frd_lyr in range(0,layer_size,1):
            param = param_mat[frd_lyr,0:hdn_node[frd_lyr+1],0:hdn_node[frd_lyr]+1]
            alp = param@cx
            after_activeFn = ActivationFunc(alp)
            var_mat[frd_lyr,0:after_activeFn.shape[0],0:batch]=after_activeFn
            if frd_lyr == layer_size-1:
                y_hat=after_activeFn
            else:
                after_activeFn = BiasAdd(after_activeFn, batchs)
                cx = after_activeFn
        # ---------------------------
        
        # --- Back Propagation ---
        legacy = LossDifferentialFunc(y_hat, cur_y)
        #bck_lyr = 3
        for bck_lyr in range(layer_size, 0,-1):
            legacy = DifferentialByActiveFunc(y_hat) * legacy
            param_old = param_mat[bck_lyr-1, 0:hdn_node[bck_lyr],0:hdn_node[bck_lyr-1]+1]
            var = var_mat[bck_lyr-1,0:after_activeFn.shape[0],0:batch]
            param_mat[bck_lyr-1, 0:hdn_node[bck_lyr],0:hdn_node[bck_lyr-1]+1] = legacy * var
        # ------------------------
            
        
        
        #outp_o, hidn_1_o=ForwardPropagation(cur_x, batchs, w, v)
        #w_n, v_n = BackPropagation(cur_x, w, hidn_1_o, outp_o, cur_y)
        



