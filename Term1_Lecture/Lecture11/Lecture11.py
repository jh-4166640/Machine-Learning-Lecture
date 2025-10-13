"""
* 2025-05-18
* 임베디드 시스템 전공
* 2021146036
* 최지헌
* Two layer Neural Network
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


def BackPropagation(x_in, w_mat, hidn_o, y_hat, y_real):
    """
    parameters
    * x_in : input data
    * w_mat : weights output layer between Hidden layer
    * hidn_o : sigmoid passed on hidden layer
    * y_hat : output
    * y_real : real y (One Hot)
    return
    * mean_w : A matrix that divides the partially differentiated w weights into batches
    * mean_v : A matrix that divides the partially differentiated v weights into batches
    """
L = hidn_o.shape[0]-1
batch=y_hat.shape[1]
# --- 편미분 w matrix --- #
delta = 2 * (y_hat - y_real ) * y_hat * (1-y_hat) # Q byN
mean_w = (delta @ hidn_o.T) / batch # Q by L+1
# ---------------------- #

# --- 편미분 v matrix --- #
w_Nbias = w_mat[:,:L] # Q by L
hidn_o_Nbias =hidn_o[:L,:] # L by N
# delta*w
delta_hidden = w_Nbias.T @ delta # L by N
# b(1-b)
hidn_der = hidn_o_Nbias*(1-hidn_o_Nbias) # L by N
# delta*w*b(1-b)
delta_hidn = delta_hidden * hidn_der
# delta*w*b(1-b)*x
mean_v = (delta_hidn @ x_in.T) / batch # L by M+1
# ---------------------- #
return mean_w, mean_v

def ForwardPropagation(cx, bch, w, v):
    """
    parameters
    * cx : input data
    * bch : batch size
    * w : weights output layer between Hidden layer
    * v : weights input layer between Hidden layer
    
    return
    * outp_o : y hat result
    * hidn_1_o : sigmoid passed on hidden layer
    """
# hidn_1_i : first hidden layer input (before activation function)
hidn_1_i = v@cx # L by batch
# hidn_1_o : first hidden layer output (after activation function)
hidn_1_o = Sigmoid(hidn_1_i)
bias_batch = np.ones((1,bch)) # bias 추가용
hidn_1_o = np.concatenate((hidn_1_o, bias_batch), axis=0) # L+1 by batch

# outp_i : output layer input (before activation function)
outp_i = w@hidn_1_o # Q by batch
# outp_o : output layer output (after activation function)
outp_o = Sigmoid(outp_i) # Q by batch
return outp_o, hidn_1_o   

def MseAccuracy(x, y_real, w, v):
y_hat,temp =ForwardPropagation(x,x.shape[1],w,v)
y_hat = y_hat.T

mse = np.mean((y_real - y_hat)**2)

# gpt가 도와준 for문 없이 one hot 하기 gpt 최고
one_hot_y_hat = np.zeros_like(y_hat)
max_idx = np.argmax(y_hat, axis=1)
one_hot_y_hat[np.arange(y_hat.shape[0]), max_idx] = 1

compare_max = np.all(y_real == one_hot_y_hat, axis=1)
acc = (np.sum(compare_max == True) / y_real.shape[0]) * 100
return mse, acc


def Two_Layer_NN(x, y, L, alpha, batch, epoch,  init_start,init_space):

x = x.T
N = x.shape[1]
M = x.shape[0]
Q = len(np.unique(y))

w_his = []
v_his = []
mse_his = np.empty(0)
acc_his = np.empty(0)

_one_hot_y = OneHotEncoding(y)
v = (np.random.rand(L,M+1)*init_space)+init_start # v weight 초기화 size v : (L by M+1)
w = (np.random.rand(Q,L+1)*init_space)+init_start # w weight 초기화 size w : (Q by L+1)
bias = np.ones((1,N)) # bias 추가용
x = np.concatenate((x, bias), axis=0)

step_max = round(N/batch)
cnts = batch*step_max

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
if (step+1)*batch >= N :
end_idx = N-1
else:
end_idx = (step+1)*batch

cur_x = x_shf[:,start_idx : end_idx] # M+1 by batch
cur_y = y_shf[:,start_idx : end_idx] # 1 by batch
batchs = end_idx - start_idx
start_idx = end_idx
outp_o, hidn_1_o=ForwardPropagation(cur_x, batchs, w, v)
w_n, v_n = BackPropagation(cur_x, w, hidn_1_o, outp_o, cur_y)
w_new = w-alpha*w_n
v_new = v-alpha*v_n

w = w_new
v = v_new

w_his.append(w)
v_his.append(v)

mse, acc=MseAccuracy(x,_one_hot_y,w,v)
mse_his = np.append(mse_his,mse)
acc_his = np.append(acc_his,acc)
w_his = np.stack(w_his)    
v_his = np.stack(v_his)    
return w_his, v_his, mse_his, acc_his



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

train_set, validation_set, test_set = DataDivide(data, 7, 0, 3)


input_mat = train_set[:,:split_idx] # 데이터 자동 분할
output_mat = train_set[:,split_idx:] # 데이터 자동 분할

output_mat = np.reshape(output_mat, [train_set.shape[0],all_widht-split_idx]) # 전체데이터 by 전체 열개수 - y 시작열번호

"""
# input 속성 수 : input_mat의 열의 갯수
# output class 수
"""
# -------- hyper parameter --------
init_space = 2
init_start = -1
hidden_layer_node = 8
hidden_layer_node = 44
hln=hidden_layer_node
batch_size = 64# 2^n
epoch = 3000
learning_rate = 0.08
learning_rate = 0.06
# ---------------------------------
one_hot_y = OneHotEncoding(output_mat)

# y_hat : Q by 1


w_his, v_his, mse_his, acc_his=Two_Layer_NN(input_mat, output_mat, hln, learning_rate ,batch_size, epoch, init_start,init_space) # Neural network

plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,acc_his,label="acc")
plt.plot(plot_x,acc_his,label="accuracy")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("Accuracy graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('mse',fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('accuracy[%]',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
#ytick = np.arange(0, 101, 10)
#plt.yticks(ytick,fontsize=18)
plt.yticks(fontsize=18)
plt.show()

plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,mse_his,label="mse")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("MSE graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('mse',fontsize=18)
plt.xticks(fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
plt.yticks(fontsize=18)
plt.show()


#%% Test set
last_w = w_his[-1, :,:]
last_v = v_his[-1, :,:]
input_test = test_set[:,:split_idx] # 데이터 자동 분할
output_test = test_set[:,split_idx:] # 데이터 자동 분할=
output_test = np.reshape(output_test, [test_set.shape[0],all_widht-split_idx]) # 전체데이터 by 전체 열개수 - y 시작열번호
data_num = input_test.shape[0]
one_hot_y_test = OneHotEncoding(output_test)
bias_te = np.ones((1,data_num)) # bias 추가용
input_test = input_test.T
input_test = np.concatenate((input_test, bias_te), axis=0)

y_hat_test, temp_var =ForwardPropagation(input_test, data_num, last_w, last_v)
y_hat_test= y_hat_test.T
one_hot_y_hat_test = np.zeros_like(y_hat_test)
max_idx = np.argmax(y_hat_test, axis=1)
one_hot_y_hat_test[np.arange(y_hat_test.shape[0]), max_idx] = 1

# 고마운 GPT
# 벡터 연산으로 confusion matrix
classes = one_hot_y_hat_test.shape[1]
conf_mat = np.zeros((classes+1, classes+1),dtype=object)
true_labels = np.argmax(one_hot_y_test, axis=1)
pred_labels = np.argmax(one_hot_y_hat_test, axis=1)
np.add.at(conf_mat, (true_labels, pred_labels), 1)

conf_mat[-1, :-1] = np.sum(conf_mat[:-1, :-1], axis=0)  
conf_mat[:-1, -1] = np.sum(conf_mat[:-1, :-1], axis=1)  
conf_mat[-1,-1]=np.sum(conf_mat * np.eye(7)) / data_num * 100
headers = np.empty((1, classes+1),dtype=object)    
fcolumns = np.empty((classes+2,1),dtype=object)
for i in range(0,classes,1):
conf_mat[-1,i] = conf_mat[i,i] / conf_mat[-1,i] * 100
conf_mat[i,-1] = conf_mat[i,i] / conf_mat[i,-1] * 100
headers[0,i] = "pred Class" + str(i+1)
fcolumns[i+1,0] = "real Class" + str(i+1)

headers[0,-1] = "Recall"
fcolumns[-1,0] = "Precision"
fcolumns[0,0] = "\\"


conf_mat=np.concatenate((headers,conf_mat),axis=0)
conf_mat=np.concatenate((fcolumns,conf_mat),axis=1)
