# 2nd semester
# first lecture
# 2025-09-11
# Jiheon Choi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DeepNeuralNetwork():
    
    def DataDivide(self, data, train, validation, test):
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

    def OneHotEncoding(self,y):
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


    def ActivationFunc(self,z):
        #sigmoid
        p = 1 / (1+np.exp(-z))
        return p

    def DifferentialByActiveFunc(self, arg):
        #sigmoid
        return arg * (1-arg)

    def LossDifferentialFunc(self, y_hat,y_real):
        #MSE
        return 2 * (y_hat-y_real)

    def BiasAdd(self,data, bch):
        bias = np.ones((1, bch))
        newData = np.concatenate((data, bias), axis=0)
        return newData
    
    def ReadData(self, filepath):
        file_path = filepath
        open_file = pd.read_csv(file_path)
        df = pd.DataFrame(open_file)

        # -------- find input and output -------- #

        col = list(df.columns)
        temp = [col for col in col if 'y' in col]
        split_idx=col.index(temp[0]) # output 데이터의 시작 열 번호
        all_widht = df.shape[1] # 전체 데이터의 열의 갯수
        datas = df.values
        self.N = datas.shape[0] 
        self.N = int(self.N*0.7) # train ratio
        self.train_set, self.validation_set, self.test_set = self.DataDivide(datas, 7, 0, 3)


        self.input_mat = self.train_set[:,:split_idx] # 데이터 자동 분할
        self.output_mat = self.train_set[:,split_idx:] # 데이터 자동 분할
        bias = np.ones((self.N, 1)) # bias 추가용 # 0.7은 train set 70% 분할 해서
        self.input_mat = np.concatenate((self.input_mat, bias), axis=1)

        self.output_mat = np.reshape(self.output_mat, [self.train_set.shape[0],all_widht-split_idx]) # 전체데이터 by 전체 열개수 - y 시작열번호
        self.one_hot_y = self.OneHotEncoding(self.output_mat)

    def __init__(self, filepath, hdn_node,batch=64,epoch=100,alpha=0.005,init_interval=2,init_start=-1):
        """
        parameters
        * filepath : data file path
        * hdn_node : (array) hidden node 수 결정
        * batch : batch
        * epoch : epoch
        * alpha : learning rate
        * init_interval : parameter random 초기값 간격
        * init_start : parameter random 초기값 시작 숫자
        """
        self.ReadData(filepath)
        
        self.batch = batch
        self.epoch = epoch
        self.alpha = alpha
        
        self.step_max = round(self.N/batch)
        cnts = batch*self.step_max
        if cnts < self.N:
            if self.N-cnts <= batch :
                self.step_max = self.step_max + 1
            elif self.N-cnts > batch:
                self.step_max = self.step_max + 2
        
        self.input_size = np.shape(self.input_mat)[1]-1
        self.hdn_node = hdn_node # 배열 형태
        self.layer_size = len(hdn_node) + 1
        self.output_class = len(np.unique(self.output_mat))
        
        
        if self.layer_size != len(hdn_node)+1:
            print("size not match!")
        
        self.hdn_node.insert(0,self.input_size)
        self.hdn_node.append(self.output_class)
        
        MAX_NODE_SIZE = max(hdn_node)+1
        
        self.param_mat=np.zeros((self.layer_size,MAX_NODE_SIZE,MAX_NODE_SIZE))
        self.param_his=np.zeros((epoch, self.layer_size,MAX_NODE_SIZE,MAX_NODE_SIZE))
        self.loss_his =np.zeros(epoch)
        self.acc_his  =np.zeros(epoch)
        self.var_mat  =np.zeros((self.layer_size-1,MAX_NODE_SIZE, batch))
        # --- Initialize Parameters --- 
        for layer_num in range(0,self.layer_size,1):
            rand_weight = (np.random.rand(hdn_node[layer_num+1],hdn_node[layer_num]+1)*init_interval)+init_start
            self.param_mat[layer_num,0:hdn_node[layer_num+1],0:hdn_node[layer_num]+1] = rand_weight
        # -----------------------------     
    def MSEandACC(self, epc):
        cx = self.input_mat.T
        y_real = self.one_hot_y.T
        
        for mse_fr in range(0,self.layer_size,1):
            param = self.param_mat[mse_fr,0:self.hdn_node[mse_fr+1],0:self.hdn_node[mse_fr]+1]
            alp = param @ cx
            after_activeFn = self.ActivationFunc(alp)
            if mse_fr == self.layer_size-1:
                y_hat=after_activeFn
            else:
                after_activeFn = self.BiasAdd(after_activeFn, cx.shape[1])
                cx = after_activeFn
                
        mse = np.mean((y_real - y_hat)**2)
        one_hot_y_hat = np.zeros_like(y_hat.T)
        max_idx = np.argmax(y_hat.T,axis=1)
        one_hot_y_hat[np.arange(y_hat.shape[1]), max_idx] = 1

        compare_max = np.all(self.one_hot_y == one_hot_y_hat, axis=1)
        acc = (np.sum(compare_max == True) / self.one_hot_y.shape[0]) * 100
        self.loss_his[epc] = mse
        self.acc_his[epc] = acc
        
        
    def ModelLearning(self):
        x=self.input_mat.T
        for epc in range(0,self.epoch):
            x_shf = x[:,:]
            x_shf = np.concatenate((x_shf,self.one_hot_y.T),axis=0)
            x_shf = x_shf.T
            np.random.shuffle(x_shf)
            x_shf = x_shf.T # shuffled data
            y_shf = x_shf[self.input_size+1:,:] # shuffled y data
            x_shf = x_shf[0:self.input_size+1,:] # shuffled x data
            
            start_idx = 0
            for step in range(0, self.step_max):
                if (step+1)*self.batch >= self.N :
                    end_idx = self.N-1
                else:
                    end_idx = (step+1)*self.batch
                    
                cur_x = x_shf[:,start_idx : end_idx] # M+1 by batch
                cur_y = y_shf[:,start_idx : end_idx] # 1 by batch
                batchs = end_idx - start_idx
                start_idx = end_idx
                    
                # --- Forward Propagation ---
                cx = cur_x # input_node+1 by batch size
                for frd_lyr in range(0,self.layer_size,1):
                    param = self.param_mat[frd_lyr,0:self.hdn_node[frd_lyr+1],0:self.hdn_node[frd_lyr]+1]
                    alp = param@cx
                    after_activeFn = self.ActivationFunc(alp)
                    if frd_lyr == self.layer_size-1:
                        y_hat=after_activeFn
                    else:
                        after_activeFn = self.BiasAdd(after_activeFn, batchs)
                        self.var_mat[frd_lyr,0:after_activeFn.shape[0],0:batchs]=after_activeFn
                        cx = after_activeFn
                # ---------------------------
                
                # --- Back Propagation ---
                legacy = self.LossDifferentialFunc(y_hat, cur_y)
                legacy = self.DifferentialByActiveFunc(y_hat) * legacy
                for bck_lyr in range(self.layer_size, 0,-1):
                    param_old = self.param_mat[bck_lyr-1, 0:self.hdn_node[bck_lyr],0:self.hdn_node[bck_lyr-1]+1]
                    var = self.var_mat[bck_lyr-2,0:self.hdn_node[bck_lyr-1]+1,0:batchs] # L+1 by batch
                    self.param_mat[bck_lyr-1, 0:self.hdn_node[bck_lyr],0:self.hdn_node[bck_lyr-1]+1] -= self.alpha*(legacy @ var.T)
                    param_old = param_old[:,:-1]
                    legacy = param_old.T @ legacy
                    legacy = self.DifferentialByActiveFunc(var[:-1,:]) * legacy
                    #param_old 6by3 var 3by64 legacy 6 by 64
                # ------------------------
                    
                #1 step uptate 완료 된거임
                
            # 1 epoch 완료
            self.MSEandACC(epc)
            self.param_his[epc,:,:,:] = self.param_mat[:,:,:]
            
            


file_path = "C:\\Users\\user\\Downloads\\NN_data.csv"
interval = 2
start    = -1
batch = 64
epoch = 2000
alpha = 0.0025
hdn_node = [20,20,16,20,20,40]

myNN = DeepNeuralNetwork(file_path, hdn_node, epoch=epoch,batch=batch,alpha=alpha,init_interval=interval,init_start=start)
myNN.ModelLearning()

acc_his = myNN.acc_his
loss_his = myNN.loss_his
    
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
ytick = np.arange(0,101,10)
plt.xticks(xtick, fontsize=18)
plt.yticks(ytick, fontsize=18)
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
plt.ylabel('MSE',fontsize=18)
xtick = np.arange(0,epoch+1,epoch/10)
plt.xticks(xtick, fontsize=18)
plt.yticks(fontsize=18)
plt.show()
