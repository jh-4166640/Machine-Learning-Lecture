import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# 0 청사과
# 1 사과
# 2 바나나
# 3 블랙베리
# 4 오이
# 5 오렌지
# 6 복숭아
# 7 배
# 8 토마토
# 9 수박

# RGB
# 검정 [0,0,0]
# 흰 [255,255,255]

def MostColor(img):
    threshold = 240
    mask = np.all(img>threshold, axis=2)
    only_obj=img[~mask]
    obj = only_obj.reshape(-1,3)
    unique, counts = np.unique(obj, axis=0, return_counts=True)
    dominant = unique[np.argmax(counts)]
    return dominant
    
def AspectRatio(img):
    # 원형인지 길쭉인지 판단
    threshold = 240
    mask = np.all(img>threshold, axis=2)
    coordinate = np.argwhere(~mask)
    ymin, xmin = coordinate.min(axis=0)
    ymax, xmax = coordinate.max(axis=0)
    height = ymax - ymin + 1
    width = xmax - xmin + 1
    aspect_ratio = height / width
    return aspect_ratio
    

def MostUseColors(img, K=4):
    # k-mean clustring을 이용한 색상 값 구분
    # 반복 횟수로 멈추는 이유
    # * 안멈출 수 도 있어서
    pixels = np.reshape(img,[-1,3]) # 255x255,3으로 변경
    np.random.seed(77) # lucky 77
    idx=np.random.choice(pixels.shape[0],K,replace=False)
    centers = pixels[idx]
    old_centers= centers.copy()
    MAX_ITER = 20
    
    for _ in range(MAX_ITER):
        min_dist=np.full((pixels.shape[0], 1), np.inf)
        group = np.zeros((pixels.shape[0],1))
        distances = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)  # (N, K)
        group = np.argmin(distances, axis=1)  # (N,)
        for k in range(K):
            members = pixels[group == k]
            if len(members) > 0:
                centers[k] = np.mean(members, axis=0)

        new_centers=centers.copy()
        
        old_set = {tuple(np.round(c, 5)) for c in old_centers}
        new_set = {tuple(np.round(c, 5)) for c in centers}
        if old_set == new_set:
            break
        old_centers = new_centers.copy()
            
    return centers
    
        
def Projection_ROW(img):
    threshold = 240
    mask = np.all(img>threshold, axis=2)
    
    col_sum = np.sum(~mask, axis=0)
    all_sum = np.sum(col_sum)
    pdf = col_sum / all_sum
    xx = np.arange(0,pdf.shape[0],1)
    exp = sum(xx*pdf)
    var = sum((xx-exp)**2*pdf)
    return var

def Projection_COLUMN(img):
    threshold = 240
    mask = np.all(img>threshold, axis=2)
    
    row_sum = np.sum(~mask, axis=1)
    all_sum = np.sum(row_sum)
    pdf = row_sum / all_sum
    xx = np.arange(0,pdf.shape[0],1)
    exp = sum(xx*pdf)
    var = sum((xx-exp)**2*pdf)
    return var

def select_features(directory):
    K=3
    # 폴더 내 파일명 read
    file_list = os.listdir(directory)
    feature_1_list = []
    feature_2_list = []
    feature_3_list = []
    feature_4_list = []
    feature_5_list = []
    feature_6_list = []

    labels = []

    for name in file_list:
        path = os.path.join(directory, name)
        labels.append(int(name.split('_',1)[0]))
        # 이미지 Read & RGB 변환
        img_GRB = cv2.imread(path)
        img_RGB = cv2.cvtColor(img_GRB, cv2.COLOR_BGR2RGB)
        feature_1 = Projection_ROW(img_RGB)
        feature_2 = img_RGB[:,:,0].mean() # Red 평균 밝기
        feature_3 = img_RGB[:,:,1].mean() # Blue 평균 밝기
        feature_4 = img_RGB[:,:,2].mean() # Green 평균 밝기
        feature_5 = MostColor(img_RGB)
        feature_6 = Projection_COLUMN(img_RGB)
        
        #feature_6 = MostUseColors(img_RGB,K)
        #feature_6_flat= feature_6.reshape([1,feature_6.shape[0]*feature_6.shape[1]])
           
        feature_1_list.append(feature_1)
        feature_2_list.append(feature_2)
        feature_3_list.append(feature_3)
        feature_4_list.append(feature_4)
        feature_5_list.append(feature_5)
        feature_6_list.append(feature_6)
        #feature_6_list = np.vstack((feature_6_list, feature_6_flat))

    feature_1_list = np.array(feature_1_list)
    feature_2_list = np.array(feature_2_list)
    feature_3_list = np.array(feature_3_list)
    feature_4_list = np.array(feature_4_list)
    feature_5_list = np.array(feature_5_list)
    feature_6_list = np.array(feature_6_list)
    
    features = np.column_stack([feature_1_list,feature_2_list,feature_3_list,feature_4_list,feature_5_list, feature_6_list])
    return features, labels



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
        lr=LearningRate_cosine_annealing(epc,epoch,alpha, alpha*0.02)
        
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



directory = "C:\\Users\\USER\\Downloads\\train"
x,y=select_features(directory)
mats = np.column_stack([x,y])
train, validataion, test = DataDivide(mats,8,0,2)

train_x = train[:,:train.shape[1]-1]
train_y = train[:,train.shape[1]-1]

# -------- hyper parameter --------
init_space = 1.5
init_start = -0.7
hidden_layer_node = 52
hln=hidden_layer_node
batch_size = 64# 2^n
epoch = 1000
learning_rate = 0.003
# ---------------------------------
w_his, v_his, mse_his, acc_his=Two_Layer_NN(train_x, train_y, hln, learning_rate ,batch_size, epoch, init_start,init_space) # Neural network

test_x = test[:,:train.shape[1]-1]
test_y = test[:,train.shape[1]-1]

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
    headers[0,i] = "pred Class" + str(i)
    fcolumns[i+1,0] = "real Class" + str(i)
    
headers[0,-1] = "Recall"
fcolumns[-1,0] = "Precision"
fcolumns[0,0] = "\\"


conf_mat=np.concatenate((headers,conf_mat),axis=0)
conf_mat=np.concatenate((fcolumns,conf_mat),axis=1)
