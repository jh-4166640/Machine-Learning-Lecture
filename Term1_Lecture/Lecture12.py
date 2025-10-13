import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



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

def LearningRate_cosine_annealing(epoch, max_epoch, initial_lr, min_lr):
    # gpt
    cos_inner = np.pi * epoch / max_epoch
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(cos_inner))

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
        lr=LearningRate_cosine_annealing(epc,epoch,alpha, alpha*0.002)
        
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
            w_new = w-lr*w_n
            v_new = v-lr*v_n
            
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
    


def feature_1(input_data):
    # 
    col_sum = np.sum(input_data, axis=0)
    all_sum = np.sum(col_sum)
    pdf = col_sum / all_sum
    xx = np.arange(0,pdf.shape[0],1)
    exp = sum(xx*pdf)
    var = sum((xx-exp)**2*pdf)
    return var

def feature_2(input_data):
    # 
    row_sum = np.sum(input_data, axis=1)
    all_sum = np.sum(row_sum)
    pdf = row_sum / all_sum
    xx = np.arange(0,pdf.shape[0],1)
    exp = sum(xx*pdf)
    var = sum((xx-exp)**2*pdf)
    return var


def feature_3(input_data):
    # X 모양에 놓여져 있는 밝기 기준 값
    mask = np.eye(input_data.shape[0],input_data.shape[1])
    mask2 = np.fliplr(np.eye(input_data.shape[0],input_data.shape[1]))
    dt = input_data * mask + input_data * mask2
    all_sum= np.sum(dt)
    all_sum = np.sum(all_sum)
    return all_sum

def edges(input_data):
    dx = np.abs(np.diff(input_data, axis=1))
    dy = np.abs(np.diff(input_data, axis=0))
    return dx,dy
def feature_4(input_data):
    # 가로 세로 폭의 비율
    data = input_data.copy()
    data[data == 0]=5
    dx, dy=edges(data)
    dx[dx<1] = 0
    dy[dy<1] = 0
    columns = np.sum(dx, axis=0)
    rows    = np.sum(dy, axis=1)
    for i in range(0, rows.shape[0]):
        if columns[i] != 0:
            x_end = i
        if rows[i] != 0:
            y_end = i
        if columns[rows.shape[0]-i-1] != 0:
            x_start = rows.shape[0]-i-1
        if rows[rows.shape[0]-i-1] != 0:
            y_start = rows.shape[0]-i-1
            
    x_width = x_end - x_start        
    y_width = y_end - y_start
    return y_width/x_width
     

def feature_5(input_data):
    # 가운데 몰린 값
    mask = np.zeros((input_data.shape[0],input_data.shape[1]))
    mask[round(input_data.shape[0]/2)-3:round(input_data.shape[0]/2)+3+1, round(input_data.shape[1]/2)-3:round(input_data.shape[1]/2)+3+1]=1
    dt = input_data * mask
    col_sum = np.sum(input_data, axis=0)
    all_sum = np.sum(col_sum)
    return all_sum
    
def feature_6(input_data):
    y_idx, x_idx = np.indices(input_data.shape)
    total = input_data.sum()
    x_center = (x_idx * input_data).sum() / total
    y_center = (y_idx * input_data).sum() / total
    return x_center, y_center

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

# -------- hyper parameter --------
init_space = 2
init_start = -1
hidden_layer_node = 48
hln=hidden_layer_node
batch_size = 64# 2^n
epoch = 1000
learning_rate = 0.005
# ---------------------------------
    
path = "C:\\Users\\USER\\Downloads\\MNIST Data"
my_dir = np.array(os.listdir(path))
DATA_CNT = np.size(my_dir)
my_dir=my_dir.reshape([DATA_CNT,1])

#%% No Feature Extraction
"""
x_mat = np.empty((0,28**2))
y_mat = np.empty((0,1))
for i in range(0,DATA_CNT):
    sel_data = pd.read_csv(path+"\\"+my_dir[i,0],header=None)
    x=np.reshape(sel_data, [1,-1])
    classes = my_dir[i,0].split('_')
    y=np.reshape(classes[0],(1,1))
    y = y.astype(np.float64)  # float으로 변환
    x_mat = np.vstack([x_mat, x])
    y_mat = np.vstack([y_mat, y])
    
x_mat.shape[1]
one_hot_y = OneHotEncoding(y_mat)
datas = np.hstack([x_mat,y_mat])  
train_set, val_set, test_set = DataDivide(datas, 7,0,3)
train_x = train_set[:,:x_mat.shape[1]]
train_y = train_set[:,x_mat.shape[1]:]
test_x = test_set[:,:x_mat.shape[1]]
test_y = test_set[:,x_mat.shape[1]:]

w_his, v_his, mse_his, acc_his=Two_Layer_NN(train_x, train_y, hln, learning_rate ,batch_size, epoch, init_start,init_space) # Neural network


                               
plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,acc_his,label="accuracy")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("Accuracy graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
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
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
plt.yticks(fontsize=18)
plt.show()


last_w = w_his[500, :,:]
last_v = v_his[500, :,:]
data_num = test_x.shape[0]
one_hot_y_test = OneHotEncoding(test_y)
bias_te = np.ones((1,data_num)) # bias 추가용
test_x = test_x.T
test_x = np.concatenate((test_x, bias_te), axis=0)

y_hat_test, temp_var =ForwardPropagation(test_x, data_num, last_w, last_v)
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
conf_mat[-1,-1]=np.sum(conf_mat * np.eye(classes+1)) / data_num * 100
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

mse_his_test = np.empty(0)
acc_his_test = np.empty(0)
for i in range(0,epoch):    
    mse_test, acc_test=MseAccuracy(test_x,one_hot_y_test,w_his[i,:,:],v_his[i,:,:])
    mse_his_test = np.append(mse_his_test,mse_test)
    acc_his_test = np.append(acc_his_test,acc_test)
    

       
plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,acc_his_test,label="accuracy")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("Accuracy graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('accuracy[%]',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
#ytick = np.arange(0, 101, 10)
#plt.yticks(ytick,fontsize=18)
plt.yticks(fontsize=18)
plt.show()
        
plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,mse_his_test,label="mse")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("MSE graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('mse',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
plt.yticks(fontsize=18)
plt.show()
"""

#%% Feature Extraction
input_mat = np.empty([0,0])
ftr1 =ftr2 = ftr3 = ftr4 = ftr5 = np.empty(0)
ftr6 = np.empty((0,2))
x_mat = np.empty((1500,0))
y_mat = np.empty((0,1))
for i in range(0, DATA_CNT):
    sel_data = pd.read_csv(path+"\\"+my_dir[i,0],header=None)
    classes = my_dir[i,0].split('_')
    y=np.reshape(classes[0],(1,1))
    y = y.astype(np.float64)  # float으로 변환
    y_mat = np.vstack([y_mat, y])
    sel_data = sel_data.values
    ftr1 = np.append(ftr1, feature_1(sel_data))
    ftr2 = np.append(ftr2, feature_2(sel_data))
    ftr3 = np.append(ftr3, feature_3(sel_data))
    ftr4 = np.append(ftr4, feature_4(sel_data))    
    ftr5 = np.append(ftr5, feature_5(sel_data))    
    ftr6_x,ftr6_y=feature_6(sel_data)
    ftr6s=np.reshape(np.array([ftr6_x,ftr6_y]),[1,2])
    ftr6 = np.vstack([ftr6, ftr6s])

x_mat = np.hstack([x_mat, np.reshape(ftr1,[DATA_CNT,1])])    
x_mat = np.hstack([x_mat, np.reshape(ftr2,[DATA_CNT,1])])    
x_mat = np.hstack([x_mat, np.reshape(ftr3,[DATA_CNT,1])])    
x_mat = np.hstack([x_mat, np.reshape(ftr4,[DATA_CNT,1])])    
x_mat = np.hstack([x_mat, np.reshape(ftr5,[DATA_CNT,1])])     
x_mat = np.hstack([x_mat, np.reshape(ftr6,[DATA_CNT,2])])     
one_hot_y = OneHotEncoding(y_mat)
datas = np.hstack([x_mat,y_mat])  
train_set, val_set, test_set = DataDivide(datas, 7,0,3)
train_x = train_set[:,:x_mat.shape[1]]
train_y = train_set[:,x_mat.shape[1]:]
test_x = test_set[:,:x_mat.shape[1]]
test_y = test_set[:,x_mat.shape[1]:]

w_his, v_his, mse_his, acc_his=Two_Layer_NN(train_x, train_y, hln, learning_rate ,batch_size, epoch, init_start,init_space) # Neural network
                        
plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,acc_his,label="accuracy")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("Training Accuracy graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
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
plt.title("Training MSE graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('mse',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
plt.yticks(fontsize=18)
plt.show()


last_w = w_his[-1, :,:]
last_v = v_his[-1, :,:]
data_num = test_x.shape[0]
one_hot_y_test = OneHotEncoding(test_y)
bias_te = np.ones((1,data_num)) # bias 추가용
test_x = test_x.T
test_x = np.concatenate((test_x, bias_te), axis=0)

y_hat_test, temp_var =ForwardPropagation(test_x, data_num, last_w, last_v)
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
conf_mat[-1,-1]=np.sum(conf_mat * np.eye(classes+1)) / data_num * 100
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

mse_his_test = np.empty(0)
acc_his_test = np.empty(0)
for i in range(0,epoch):    
    mse_test, acc_test=MseAccuracy(test_x,one_hot_y_test,w_his[i,:,:],v_his[i,:,:])
    mse_his_test = np.append(mse_his_test,mse_test)
    acc_his_test = np.append(acc_his_test,acc_test)
    

       
plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,acc_his_test,label="accuracy")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("Test Accuracy graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('accuracy[%]',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
#ytick = np.arange(0, 101, 10)
#plt.yticks(ytick,fontsize=18)
plt.yticks(fontsize=18)
plt.show()
        
plt.figure(figsize=(12,6))
plot_x = np.arange(0,epoch,1)
plt.plot(plot_x,mse_his_test,label="mse")
plt.grid(True)
plt.rc('font',size=18)
plt.legend(fontsize=16)
plt.title("Test MSE graphs according to epoch\n"
          +"batch size="+str(batch_size)+", learning rate="+str(learning_rate)+", hidden layer nodes="+str(hln),fontsize=20)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('mse',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
plt.yticks(fontsize=18)
plt.show()
