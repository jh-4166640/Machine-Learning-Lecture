
"""
* Machine Learning 1
* 2025-03-13 
* Jiheon Choi 
"""
myPath = "C:\\Users\\USER\\.spyder-py3\\"

for file_name in range(1,10,1):
    my_dir = myPath + str(file_name) + ".csv"
    # csv_file_open(my_dir)
    
import math as ma
var1=ma.sqrt(3)

list_2d = [[1,2],[3,4]]
print(len(list_2d))
list_2d.append([5,6])    
# epoch 마다 weight&loss값을 저장하기 위해 append 사용
# indexing arr[round(len(arr)/2)] 중앙 인덱싱
# list는 step을 소수점으로 제어 못함


import numpy as np
var2 = np.array([1,2,3,4,5])
var2d = np.array([[1,2,3,4,5],[1,2,3,4,5]])
dtp_var = np.array([255,256,-128], dtype="uint8")

myA=np.arange(1, 1001, 1)
np.reshape(myA, [250,4])

vec_col = np.array([1,2,3,4,5,6])
vec_col=vec_col.reshape([6,1])

vec_row = np.array([1,2,3,4,5,6])
vec_row=vec_row.reshape([1,6])

rand_val=np.random.rand(1,1)
#rand_val=np.random.rand(1,1)*5 ....0~5
#rand_val=np.random.rand(1,1)+2 ....2~3
#rand_val=np.random.rand(1,1)*7+5 ....5~12 scale을 먼저하고 shift를 한다
# 5~12의 사이즈는 7이니까
#rand_val=np.random.rand(1,1)*2-1 ....-1~1 초기화할 때 이값을 씀

A = np.random.rand(10,7)
a = A[1,2:5]
a=np.reshape(a, [1,3])


temp = np.random.rand(100,100)
sel = temp[round(len(temp)/4):round(100-len(temp)/4),round(len(temp)/4):round(100-len(temp)/4)]

# index
B = np.random.rand(1,100)
idx = np.where(0.7<B) # 0.7보다 큰 값이 어디있니 0행에 행좌표, 1행에 열좌표
idx_pt = idx[1] # size가 30근처에서 노는 이유는 uniform random이기 때문이다.


matA=np.array([[1,2,3],[4,5,6]])
matB=np.array([[1,2],[3,4],[5,6]])
matC=np.dot(matA,matB)


trans_B=np.transpose(matB)
matD =matA*trans_B # 요소 곱

xx = np.random.rand(1,10)
xx_max = max(xx)

yy = np.random.rand(10,1)
yy_max = max(yy)

zz = np.random.rand(10,7)
#zz_max = max(zz) error
zz_npmax=np.max(zz)
zz_max_idx = np.where(zz==zz_npmax)


zz_sum = np.sum(zz,axis=0) # AXIS = 0이면 열마다 더하고 AXIS=1이면 행마다 더함


#[간단 실습]
myMat = np.random.rand(10,10)*10
# 1
mySum1 = np.sum(myMat,axis=1)
my_Sum1_max_idx = np.where(mySum1==np.max(mySum1))
#aa=np.argmax(mySum1) 요소에서 가장 큰 값 출력
# 2
mySum2 = np.sum(myMat,axis=0)
my_Sum2_min_idx = np.where(mySum2==np.min(mySum2))




