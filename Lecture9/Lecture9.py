import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def GradientDescent(x, y, alpha, epoch, init_start, init_space):
    """
    Gradient Descent Method Function
    
    x : (Matrix) input data shape(Number of Data by Number of Feature + 1)
    y : (Matrix) output real data shape(Number of Data by Q) 
    a : (float) learning rate
    epoch : (integer) training epoch
    
    Returns
    -------
    weight history, mse history
    """
    # initalize
    # weight, first CEE
    NumberOfData=x.shape[0]
    xwidth = x.shape[1]
    w_his = np.empty([0,xwidth])
    cee_his = np.empty(0)
    w_init = []
    for idx in range(0, xwidth):
        w_init.append((np.random.rand()*init_space)+init_start)
        
    w_his = np.append(w_his, [w_init], axis=0)
    #print('random init w0, w1, w2 ', w_his)
    w_init = np.reshape(w_init,[xwidth,1])
    z = x@w_init
    p = Sigmoid(z)
    #cee = -np.mean(y.T@np.log(p) + (1-y).T@np.log(1-p))
    cee = -np.mean(y.T @ np.log(p + 1e-10) + (1 - y).T @ np.log(1 - p + 1e-10))
    cee_his = np.append(cee_his, cee) # MSE store
    
    for epc in range(0,epoch-1):        
        cur_w = w_his[epc]              # load to current weight
        cur_w = cur_w.reshape([xwidth,1])
        # weight update
        # 얘 지금 4by4로 나옴
        z = x@cur_w
        p = Sigmoid(z)

        new_w = np.reshape(cur_w,[1,xwidth]) - alpha*((p-y).T@x)/NumberOfData
        w_his = np.append(w_his, new_w.reshape([1,xwidth]), axis=0) # new weight store
        #cee = -np.mean(y.T@np.log(p) + (1-y).T@np.log(1-p))
        cee = -np.mean(y.T @ np.log(p + 1e-10) + (1 - y).T @ np.log(1 - p + 1e-10))
        cee_his = np.append(cee_his, cee) # MSE store
    return w_his, cee_his


def MSEGradientDescent(x, y, alpha, epoch, init_start, init_space):
    """
    Gradient Descent Method Function
    
    x : (Matrix) input data shape(Number of Data by Number of Feature + 1)
    y : (Matrix) output real data shape(Number of Data by Q) 
    a : (float) learning rate
    epoch : (integer) training epoch
    
    Returns
    -------
    weight history, mse history
    """
    # initalize
    # weight, first MSE
    NumberOfData=x.shape[0]
    xwidth = x.shape[1]
    w_his = np.empty([0,3])
    mse_his = np.empty(0)
    w_init = []
    for idx in range(0, 3):
        w_init.append((np.random.rand()*init_space)+init_start)
        
    w_his = np.append(w_his, [w_init], axis=0)
    #print('random init w0, w1, w2 ', w_his)
    w_init = np.reshape(w_init,[3,1])
    z = x@w_init
    p = Sigmoid(z)
    mse = np.mean((p - y)**2) # Calculate MSE using new weights 
    mse_his = np.append(mse_his, mse)
   
    for epc in range(0,epoch-1):        
        cur_w = w_his[epc]              # load to current weight
        cur_w = cur_w.reshape([3,1])
        z = x@cur_w
        p = Sigmoid(z)
        
        new_w = np.reshape(cur_w,[1,3]) - alpha*(((p - y) * p * (1 - p)).T@x)/NumberOfData
        w_his = np.append(w_his, new_w.reshape([1,3]), axis=0) # new weight store
        mse = np.mean((p - y)**2) # Calculate MSE using new weights 
        mse_his = np.append(mse_his, mse)
  
    return w_his, mse_his


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
def StandardNormalization(dt):
    dt_mean = np.mean(dt)
    dt_std = np.std(dt)
    if dt_std == 0:
        return np.zeros_like(dt)  # 모든 값이 같으면 0으로 대체
    return (dt - dt_mean) / dt_std

# ----- Hyper Parameter ----- #
epoch = 10000
alpha = 0.01
init_start = -0.1
init_space = 1
K = 5
# --------------------------- #

file_path = "logistic_regression_data.csv"
open_file = pd.read_csv(file_path,index_col = 0)
df = pd.DataFrame(open_file)

widthOfData = df.shape[1] # column size of data 
data = df.values

train_set, validation_set, test_set=DataDivide(data,7,0,3)

print(train_set.shape)
print(test_set.shape)
numberOfData = train_set.shape[0] # row size of data


input_mat = np.ones([numberOfData,widthOfData]) # size initialize
for idx in range(widthOfData):
    if widthOfData-1 == idx:
        output_mat = train_set[:,idx] # initialize
    else :
        input_mat[:,idx] = train_set[:,idx]
    
output_mat = output_mat.reshape([numberOfData,1])

w_his, cee_his = GradientDescent(input_mat, output_mat, alpha, epoch, init_start, init_space)
w_his_mse, mse_his = MSEGradientDescent(input_mat, output_mat, alpha, epoch, init_start, init_space)
x0 = np.arange(-2, 8, 0.1)
decision_boundary_cee = -(w_his[epoch-1,0]/w_his[epoch-1,1])*x0-(w_his[epoch-1,2]/w_his[epoch-1,1])
decision_boundary_mse = -(w_his_mse[epoch-1,0]/w_his_mse[epoch-1,1])*x0-(w_his_mse[epoch-1,2]/w_his_mse[epoch-1,1])

# ----- Polynomial Decision Boundary ----- #
input_norm = np.empty((numberOfData,input_mat.shape[1]-1))
for i in range(widthOfData - 1):
    input_norm[:, i] = StandardNormalization(input_mat[:, i])
    
#basis_input = GeneratorBasis(K,input_norm)
#w_his_basis, cee_his_basis = GradientDescent(basis_input, output_mat, alpha, epoch, init_start, init_space)


# ----------------------------------------- #

# --------- weight, cee, mse ---------- #
"""
plt.figure(figsize=(12,6))
xxx = np.arange(0,epoch,1)
plt.plot(xxx, w_his[:,0], label ="w0 CEE history")
plt.plot(xxx, w_his[:,1], label ="w1 CEE history")
plt.plot(xxx, w_his[:,2], label ="w2 CEE history")
plt.plot(xxx, w_his_mse[:,0], label ="w0 MSE history")
plt.plot(xxx, w_his_mse[:,1], label ="w1 MSE history")
plt.plot(xxx, w_his_mse[:,2], label ="w2 MSE history")
plt.grid(True)
plt.title('Weight change according to epoch, epoch='+str(epoch)+', learning rate='+str(alpha),fontsize=22)
plt.rc('font',size=18)
plt.legend(fontsize=18)
xticks = np.arange(0,epoch+1,1000)
plt.xticks(xticks,fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('weight', fontsize=20)
plt.show()


plt.figure(figsize=(12,6))
xxx = np.arange(0,epoch,1)
plt.plot(xxx, cee_his,'r--', label ="CEE")
plt.plot(xxx, mse_his, 'b-.',label ="MSE")
plt.grid(True)
plt.title('Loss Function change according to epoch, epoch='+str(epoch)+', learning rate='+str(alpha),fontsize=22)
plt.rc('font',size=18)
plt.legend(fontsize=18)
xticks = np.arange(0,epoch+1,1000)
plt.xticks(xticks,fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.show()
"""
# --------------------------------------- #

# --------- accuracy ---------- #
"""
NumofTestSet = 150 
widths = K*2+1

accuracy_his = np.empty(0) # TP+TN / N
precision_his = np.empty(0) # TP/(TP+FP)
recall_his = np.empty(0) # TP/(TP+FN)
FPR_his = np.empty(0) # 1 - TN/(FP+TN)

test_y = test_set[:,2]
test_x = test_set[:,0:2]
test_y = np.reshape(test_y,[NumofTestSet,1]) 
#test_x = np.concatenate([test_x, np.ones((test_x.shape[0], 1))], axis=1) # non-basis
# basis normalization
for i in range(0,2):
    mean = np.mean(input_mat[:, i])
    std = np.std(input_mat[:, i])
    test_x[:, i] = (test_x[:, i] - mean) / std
test_y_flat = test_y.ravel()
test_x = GeneratorBasis(K,test_x)
print(test_x.shape)
for ep in range(0,epoch):
    #print(f'---epoch={ep}---')
    TP=TN=FP=FN =0
    cur_w=np.reshape(w_his_basis[ep], [widths,1])
    z = test_x@cur_w
    p = Sigmoid(z)
    y_hat = np.where(p>=0.5, 1 ,0)    
    compare = y_hat == test_y
    
    acc = (np.sum(compare == True) / NumofTestSet) * 100
    accuracy_his = np.append(accuracy_his, acc)
    
    y_hat = y_hat.ravel()
    
    TP = np.sum((test_y_flat == 1) & (y_hat == 1))
    TN = np.sum((test_y_flat == 0) & (y_hat == 0))
    FP = np.sum((test_y_flat == 0) & (y_hat == 1))
    FN = np.sum((test_y_flat == 1) & (y_hat == 0))
    precision = TP/(TP+FP) * 100
    recall = TP/(TP+FN) * 100
    fpr = (1 - TN/(FP+TN)) * 100
    
    precision_his = np.append(precision_his, precision)
    recall_his = np.append(recall_his, recall)
    FPR_his = np.append(FPR_his, fpr)


    #print(cur_w.shape)
print("Confusion Matrix")
print(f"TP: {TP}, FP: {FP}")
print(f"FN: {FN}, TN: {TN}")
print(f'acc : {acc}')
print(f'precision : {precision}')
print(f'recall : {recall}')
print(f'FPR : {fpr}')
plt.figure(figsize=(12,6))
xxx = np.arange(0,epoch,1)
plt.plot(xxx, accuracy_his, label ="accuracy")
plt.plot(xxx, precision_his ,label ="precision")
plt.plot(xxx, recall_his ,label ="recall")
plt.plot(xxx, FPR_his ,label ="FPR")
plt.grid(True)
plt.title('Test set classfication Accuracy(Polynomial Basis) K='+str(K),fontsize=22)
plt.rc('font',size=18)
plt.legend(fontsize=18)
xticks = np.arange(0,epoch+1,1000)
plt.xticks(xticks,fontsize=18)
yticks = np.arange(0,100+1,10)
plt.yticks(yticks,fontsize=18)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('Percent[%]', fontsize=20)
plt.show()
"""
# ----------------------------- #


# ------- decision boundary ------- #

# ----- Polynomial Decision Boundary ----- #
K = [4,5,6,7]
input_norm = np.empty((numberOfData,input_mat.shape[1]-1))
for i in range(widthOfData - 1):
    input_norm[:, i] = StandardNormalization(input_mat[:, i])

plt.figure(figsize=(12,6))
plt.scatter(data[0:250,0], data[0:250,1], marker='o',label="Class 0", s=10)
plt.scatter(data[250:,0], data[250:,1], marker='x',label="Class 1", s=10)
plt.plot(x0,decision_boundary_cee,color='grey',label='CEE Decision Boundary')
#plt.plot(x0,decision_boundary_mse,color='blue',label='MSE Decision Boundary')
plt.title('Decision Boundary',fontsize=22)
x1_range = np.linspace(np.min(input_mat[:, 0])-0.5, np.max(input_mat[:, 0])+0.5, 150)
x2_range = np.linspace(np.min(input_mat[:, 1])-0.5, np.max(input_mat[:, 1])+0.5, 150)
X1, X2 = np.meshgrid(x1_range, x2_range)

grid_points = np.c_[X1.ravel(), X2.ravel()]
grid_points_norm = np.empty_like(grid_points)
for i in range(grid_points.shape[1]):
    mean = np.mean(input_mat[:, i])
    std = np.std(input_mat[:, i])
    grid_points_norm[:, i] = (grid_points[:, i] - mean) / std
plt.legend(fontsize=16)    
legendhandles=[]
color = ['crimson', 'royalblue', 'seagreen', 'darkorange']
i = 0
for k in K :
    basis_input = GeneratorBasis(k,input_norm)
    w_his_basis, cee_his_basis = GradientDescent(basis_input, output_mat, alpha, epoch, init_start, init_space)
    grid_basis = GeneratorBasis(k, grid_points_norm)

    final_w_basis = w_his_basis[-1].reshape(-1, 1)
    z = grid_basis @ final_w_basis
    p = Sigmoid(z)
    p = p.reshape(X1.shape)

    contour = plt.contour(X1, X2, p, levels=[0.5], linewidths=2, linestyles='--',colors=[color[i]])
    proxy_contour = Line2D([0], [0], color = color[i],lw=2, linestyle='--', label='Polynomial Basis k='+str(k)+' CEE DB')
    i=i+1
    legendhandles.append(proxy_contour)

plt.grid(True)
plt.rc('font',size=18)

plt.legend(handles=legendhandles + plt.gca().get_legend_handles_labels()[0], fontsize=16)
#xtick = np.arange(np.min(data[:,0])-0.5,np.max(data[:,0])+0.5,0.5)
#ytick = np.arange(np.min(data[:,1])-0.5,np.max(data[:,1])+0.5,1)
plt.xlabel('x0',fontsize=18)
plt.ylabel('x1',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

