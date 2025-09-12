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

init_interval = 2
init_start    = -1
batch = 64
epoch = 1000
alpha = 0.005

step_max = round(N/batch)
cnts = batch*step_max
if cnts < N:
    if N-cnts <= batch :
        step_max = step_max + 1
    elif N-cnts > batch:
        step_max = step_max + 2

input_size = np.shape(input_mat)[1]-1
hdn_node = [10,10,10,10]
layer_size = len(hdn_node) + 1
output_class = len(np.unique(output_mat))


if layer_size != len(hdn_node)+1:
    print("size not match!")

hdn_node.insert(0,input_size)
hdn_node.append(output_class)

MAX_NODE_SIZE = max(hdn_node)+1


param_mat=np.zeros((layer_size,MAX_NODE_SIZE,MAX_NODE_SIZE))
param_his=np.zeros((epoch, layer_size,MAX_NODE_SIZE,MAX_NODE_SIZE))
loss_his =np.zeros(epoch)
acc_his  =np.zeros(epoch)
var_mat  =np.zeros((layer_size-1,MAX_NODE_SIZE, batch))
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
            
        # --- Forward Propagation ---
        cx = cur_x # input_node+1 by batch size
        for frd_lyr in range(0,layer_size,1):
            param = param_mat[frd_lyr,0:hdn_node[frd_lyr+1],0:hdn_node[frd_lyr]+1]
            alp = param@cx
            after_activeFn = ActivationFunc(alp)
            if frd_lyr == layer_size-1:
                y_hat=after_activeFn
            else:
                after_activeFn = BiasAdd(after_activeFn, batchs)
                var_mat[frd_lyr,0:after_activeFn.shape[0],0:batchs]=after_activeFn
                cx = after_activeFn
        # ---------------------------
        
        # --- Back Propagation ---
        legacy = LossDifferentialFunc(y_hat, cur_y)
        legacy = DifferentialByActiveFunc(y_hat) * legacy
        for bck_lyr in range(layer_size, 0,-1):
            param_old = param_mat[bck_lyr-1, 0:hdn_node[bck_lyr],0:hdn_node[bck_lyr-1]+1]
            var = var_mat[bck_lyr-2,0:hdn_node[bck_lyr-1]+1,0:batchs] # L+1 by batch
            param_mat[bck_lyr-1, 0:hdn_node[bck_lyr],0:hdn_node[bck_lyr-1]+1] -= alpha*(legacy @ var.T)
            param_old = param_old[:,:-1]
            legacy = param_old.T @ legacy
            legacy = DifferentialByActiveFunc(var[:-1,:]) * legacy
            #param_old 6by3 var 3by64 legacy 6 by 64
        # ------------------------
            
        #1 step uptate 완료 된거임
        
    # 1 epoch 완료
    #MSE 계산하기
    mse_cx = input_mat.T
    mse_y_real = one_hot_y.T
    
    for mse_fr in range(0,layer_size,1):
        mse_param = param_mat[mse_fr,0:hdn_node[mse_fr+1],0:hdn_node[mse_fr]+1]
        alp = mse_param@mse_cx
        mse_after_activeFn = ActivationFunc(alp)
        if mse_fr == layer_size-1:
            mse_y_hat=mse_after_activeFn
        else:
            mse_after_activeFn = BiasAdd(mse_after_activeFn, mse_cx.shape[1])
            mse_cx = mse_after_activeFn
    mse = np.mean((mse_y_real - mse_y_hat)**2)
    one_hot_y_hat = np.zeros_like(mse_y_hat.T)
    max_idx = np.argmax(mse_y_hat.T,axis=1)
    one_hot_y_hat[np.arange(mse_y_hat.shape[1]), max_idx] = 1

    compare_max = np.all(one_hot_y == one_hot_y_hat, axis=1)
    acc = (np.sum(compare_max == True) / one_hot_y.shape[0]) * 100
    loss_his[epc] = mse
    acc_his[epc] = acc
    
    param_his[epc,:,:,:] = param_mat[:,:,:]
    
    
plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,acc_his,label="accuracy")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("Accuracy graphs according to epoch\n"
          +"batch size="+str(batch)+", learning rate="+str(alpha),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('accuracy[%]',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
plt.yticks(fontsize=18)
plt.show()


plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,loss_his,label="loss")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("Loss graphs according to epoch\n"
          +"batch size="+str(batch)+", learning rate="+str(alpha),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('accuracy[%]',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
plt.yticks(fontsize=18)
plt.show()
