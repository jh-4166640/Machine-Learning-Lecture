

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def newGradientDescent(x, y, a, epoch, init_start, init_space, data_len):
    """
    Gradient Descent Method Function
    
    x : (Matrix) input data shape(Number of Data by Number of Feature + 1)
    y : (Matrix) output real data shape(Number of Data by Q) 
    a : (float) learning rate
    epoch : (integer) training epoch
    data_len : (interger) Number of Data
    """
    # initalize
    # weight, first MSE
    xsize = x.shape[1]
    w_his = np.empty([0,xsize])
    mse_his = np.empty(0)
    #w_init = np.empty([0,xsize])
    #print(w_init)
    w_init = []
    for i in range(0,xsize):
         w_init.append((np.random.rand()*init_space[i])+init_start[i])
    w_his = np.append(w_his, [w_init], axis=0)
    print('random init w0, w1, w2 ', w_his)
    w_init = np.reshape(w_init,[1,xsize])
    mse = np.mean((np.dot(w_init, np.transpose(x)) - np.transpose(y))**2) # Calculate MSE using new weights 
    mse_his = np.append(mse_his, mse)
    for epc in range(0,epoch-1):        
        cur_w = w_his[epc]              # load to current weight
        cur_w = cur_w.reshape([1,xsize])
        # weight update
        new_w=cur_w - a*((np.dot(np.transpose(np.transpose(np.dot(cur_w,np.transpose(x))) - y),x))/data_len)
        w_his = np.append(w_his, new_w, axis=0) # new weight store
        mse = np.mean((np.dot(new_w, np.transpose(x)) - np.transpose(y))**2) # Calculate MSE using new weights 
        mse_his = np.append(mse_his, mse) # MSE store
    return w_his, mse_his
#3) 

file_path = "lin_regression_data_02.csv"
open_file = pd.read_csv(file_path)
df = pd.DataFrame(open_file)

data_order = 0 # 0 : ascending, 1 : descending

## -- User enter Variables --
learning_rate = 0.01 # learning rate
random_init_start = [-10, -10, -20] # random start value # input size+1 by 1
random_init_space = [100, 100, 100] # random space       # input size+1 by 1
epoch = 4000 # training epoch
## --------------------------

numberOfData = df.shape[0] # row size of data
widthOfData = df.shape[1] # column size of data 


# --- Euclidean distance sort --- #
# Made by chatgpt
center_x0 = df['x0'].mean()
center_x1 = df['x1'].mean()
# column을 추가하여 데이터 정렬
# (vector - scalar)**2 + (vector - scalar)**2 -> vector 
df['distance'] = np.sqrt((df['x0'] - center_x0)**2 + (df['x1'] - center_x1)**2)
data = df.sort_values(by='distance').reset_index(drop=True) # 정렬하고 index를 reset 0부터 정렬
data = data.drop(columns='distance')
data = data.values

# ---------------------------------

input_mat = np.ones([numberOfData,widthOfData]) # size initialize
for idx in range(widthOfData):
    if widthOfData-1 == idx:
        output_mat = data[:,idx] # initialize
    else :
        input_mat[:,idx] = data[:,idx]

output_mat = output_mat.reshape([numberOfData,1]) # initialize

# 3-1)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2],c='#009900',label = "Measured Data",s=30)

weight_his, mse_his = newGradientDescent(input_mat, output_mat, learning_rate, epoch,random_init_start, random_init_space, numberOfData)

# 3-2)
create_point = 1000
x0_range = np.linspace(input_mat[:,0].min(), input_mat[:,0].max(), create_point)
x1_range = np.linspace(input_mat[:,1].min(), input_mat[:,1].max(), create_point)
x0_plot, x1_plot = np.meshgrid(x0_range, x1_range)
#print(weight_his[0][0], weight_his[0][1] , weight_his[0][2])
y_pred_first = weight_his[0][0]*x0_plot + weight_his[0][1]*x1_plot + weight_his[0][2]

z_mask = (y_pred_first > -15) & (y_pred_first < 20)
y_pred_first_masked = np.where(z_mask, y_pred_first, np.nan)

ax.plot_surface(x0_plot, x1_plot, y_pred_first_masked, alpha=0.5, color='#FFC9F9',label='First weight Regression plane')

plt.legend(fontsize=20)
plt.rc('font',size=12)
ax.set_xlabel('x0',fontsize=15)
ax.set_ylabel('x1', fontsize=15)
ax.set_zlabel('y', fontsize=16)
plt.title('First weight 3D Regression Plane',fontsize=24)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
ax.set_zlim(-15,20)
#ax.tick_params('z', labelsize=11)
ax.grid(True)
plt.show()

#3-3)
plot_step = 10
ticks = 400
step_start = 0
step_max = epoch
step = np.arange(step_start,step_max,plot_step)

plt.figure(figsize=(12,6))
plt.plot(step, weight_his[step_start:step_max:plot_step,0], 'r--',label='W0')
plt.plot(step, weight_his[step_start:step_max:plot_step,1], 'b--',label='W1')
plt.plot(step, weight_his[step_start:step_max:plot_step,2], 'g--',label='W2')

plt.rc('font',size=20)
plt.xlabel('epoch(step)',fontsize=20)
plt.ylabel('weight', fontsize=20)
plt.title('Gradient Descent Weight variation. alpha='+str(learning_rate)+' epoch='+str(epoch),fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
x_ticks = np.arange(step_start, step_max+plot_step,ticks) # x axis ticks interval
plt.xlim(step_start-plot_step,step_max) 
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()


plot_mse_step = 50
mse_ticks = 400
mse_step_start = 1000
mse_step_max = epoch
mse_step = np.arange(mse_step_start,mse_step_max,plot_mse_step)

plt.figure(figsize=(12,6))

plt.plot(mse_step,mse_his[mse_step_start:mse_step_max:plot_mse_step],'g--',label='MSE')
# 출처 : https://zephyrus1111.tistory.com/178 , 2025-04-07 22:46 접속
# 지수 형식 출력 없애기

current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.4f}'.format(x) for x in current_values])
                          
plt.rc('font',size=20)
plt.xlabel('step',fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.title('Gradient Descent MSE variation. alpha='+str(learning_rate)+' epoch='+str(epoch),fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
x_ticks = np.arange(mse_step_start, mse_step_max+plot_mse_step,mse_ticks) # x axis ticks interval
plt.xlim(mse_step_start-plot_mse_step,mse_step_max) 
plt.xticks(x_ticks,fontsize=14)
plt.yticks(fontsize=14)
plt.show()


#3-4)
mse_min = np.inf
mse_min_idx = 0
for ep in range(0,epoch):
    if mse_his[ep] < mse_min:
        mse_min_idx = ep
        mse_min = mse_his[ep]
        
print('result weight' ,weight_his[mse_min_idx])
print('mse min', mse_min)
create_point = 500
x0_range = np.linspace(input_mat[:,0].min(), input_mat[:,0].max(), create_point)
x1_range = np.linspace(input_mat[:,1].min(), input_mat[:,1].max(), create_point)
x0_plot, x1_plot = np.meshgrid(x0_range, x1_range)

y_pred_optimal_plane = weight_his[mse_min_idx][0]*x0_plot + weight_his[mse_min_idx][1]*x1_plot + weight_his[mse_min_idx][2]

#z_mask = (y_pred_first > -15) & (y_pred_first < 20)
#y_pred_optimal_plane = np.where(z_mask, y_pred_optimal, np.nan)

y_pred_point = weight_his[mse_min_idx][0]*input_mat[:,0] + weight_his[mse_min_idx][1]*input_mat[:,1] + weight_his[mse_min_idx][2]


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2],c='#009900',label = "Measured Data",s=30)
ax.scatter(input_mat[:,0],input_mat[:,1],y_pred_point,c='#6f4f28',label = "prediction point",s=30)
ax.plot_surface(x0_plot, x1_plot, y_pred_optimal_plane, alpha=0.4, color='#FFC9F9',label='y_prediction optimal plane')

plt.legend(fontsize=15)
plt.rc('font',size=12)
ax.set_xlabel('x0',fontsize=15)
ax.set_ylabel('x1', fontsize=15)
ax.set_zlabel('y', fontsize=15)

plt.title('Optimal solution Plane. alpha='+str(learning_rate)+' epoch='+str(epoch),fontsize=20)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax.set_zlim(-15,20)
ax.grid(True)
plt.show()
