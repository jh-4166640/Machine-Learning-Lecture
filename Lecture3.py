
import matplotlib.pyplot as plt
import numpy as np

# x=np.arange(-10,10,1 )
# y=x**2+1

# plt.plot(x,y)
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.title('TEST')


# x=np.arange(-10,10,1 )
# y=x**2+1
# y2 = x**2+10
# plt.plot(x,y,'b-o')
# plt.plot(x,y2,'r-^')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.title('TEST')
# plt.grid(True, alpha=0.5)
# plt.legend(['y','y2'],loc='upper right')



# t_start = -0.1
# t_end = 0.1
# freq = 10 #10[Hz]
# t_step = 1/(30*freq) # Nyquist Theory
# Magni = 3 # [V]
# t=np.arange(t_start, t_end+t_step, t_step) # end + step
# signal_1 = Magni*np.cos(2*np.pi*freq*t)
# signal_2 = Magni*np.cos(2*np.pi*freq*t)+0.5
#plt.scatter(t,signal_1)
# ---------------basic plot----------------
# plt.plot(t,signal_1,'b-^')
# plt.plot(t,signal_2,'r-o')
# plt.xlabel('Time [sec]')
# plt.ylabel('Voltage [V]')
# plt.title('TEST')
# plt.grid(True, alpha=0.5)
# plt.legend(['y','y2'],loc='upper right')
# -----------------------------------------

# ---------------multi window figure----------------
# plt.figure(1) # num is number of window
# plt.plot(t,signal_1,'b-^')
# plt.xlabel('Time [sec]')
# plt.ylabel('Voltage [V]')
# plt.title('Signal1')
# plt.grid(True, alpha=0.5)
# plt.figure(2) # num is number of window
# plt.plot(t,signal_2,'r-o')
# plt.xlabel('Time [sec]')
# plt.ylabel('Voltage [V]')
# plt.title('Signal2')
# plt.grid(True, alpha=0.5)
# --------------------------------------------------


# 레포트에서 graph의 폰트사이즈도 중요 감점 요인이 될 수 있다.

# ---------------------monster-----------------------
# enemy_hp = 100
# cnt = 0

# while enemy_hp>0:
#     cnt+=1
#     damage_val = np.round(np.random.rand(1,1) * 4 + 12)
#     enemy_hp -= damage_val
#     if enemy_hp < 0 :
#         enemy_hp=0
#     print(f"my damage {damage_val}, cnt : {cnt}, enemy {enemy_hp}")
    
# # 확률    
# drop = np.random.rand(1,1)
# if drop < 0.01: # 1%
#     print('S급')    
# elif drop < 0.1: # 10%
#     print('A급')    
# else:
#     print('B급')    

# print("Complete")
# ----------------------------------------------------

# -----------------------arr-------------------------
# arr = np.arange(0,10,1)
# temp = np.zeros([6,10])
# for i in range(0,6,1):
#     temp[i,:] = arr + 20*i
    
    
# temp2 = np.zeros([6,10])
# for c in range(0,10,1):
#     for r in range(0,6,1):
#         temp2[r,c] = c + r*20
# ----------------------------------------------------


# ---- write and read ----
# import numpy as np
# import pandas as pd
# # a=np.arange(0,100,1)
# # a=np.reshape(a, [10,10])
# # save_a = pd.DataFrame(a)
# # save_a.to_csv("C:\\test\\test.csv",index = False, header=False)
# # save_a.to_csv("C:\\test\\test2.csv")

# path = "C:\\test"
# file_name = "test.csv"
# a=pd.read_csv("C:\\test\\"+file_name,header=None) # 특히 주의 헤더 있는지 분별
# a=a.to_numpy('dtype=int32')
# ---- ----- ----


# def CosWaveForm(mag, freq, t_start, t_end, t_step=1/(30*freq)):
#     """
#     Parameters
#     ----------
#     mag : Numerical
#         Magnification.
#     freq : Numerical
#         Cos Wave frequancy [Hz].
#     t_start : Numerical
#         time_domain start value.
#     t_end : Numerical
#         time_domain end value.
#     t_step : Numerical, optional
#         time_domain step size. The default is 1/(30*freq).

#     Returns
#     -------
#     signal : np.array
#         Cos signal.

#     """
#     t=np.arrange(t_start,t_end,t_step)
#     signal = mag*np.cos(2*np.pi*freq*t)
#     return 

# def plot_figure(signal,):
    

# """
# 2025-03-20
# 전자공학부 임베디드시스템 전공
# 2021146036
# 최지헌
# """
# # ----  실습 ----
# """
#     1. -1~1 sec 사이의 정현파 신호 생성
#     크기: 10[V]
#     주파수: 60[Hz]
#     샘플링: 초당 30개
# """
# time_start = -1 # [sec]
# time_end = 1 # sec]

# CosWave(10,60,-1,1)
