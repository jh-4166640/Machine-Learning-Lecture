"""
* 2025-03-20
* 전자공학부 임베디드시스템 전공
* 2021146036
* 최지헌
* HomeWork #2
"""

"""
* Update
* 2025-03-21
* 최지헌
* Added Up sampling function
* If you want to find codes related to Upsampling, search '@Upsampling'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def SignalDownSampling(arrsignal, stdfreq, freqarr, mtime, numsignal):
    """
    Down Sampling Function
    Args:
        arrsignal (np.array): P by Q size by Signal data P is num of signal data, Q is num of signal
        stdfreq (float): Standard sampling frequancy
        freqarr (np.array): 1 by Q size by Signal sampling frequancy
        mtime (int): [sec] Measurement time
        numsignal (int): Q value
    Returns:
        np.array: Downsampled signal
        np.array: Time domain Values
    """
    t_domain = np.arange(0,mtime,1/stdfreq)       # Number divided by minimum sampling frequency per second
    new_signal = np.zeros([numsignal,len(t_domain)]) # Create space as many times as the number of time domains
    for signal in range(0,numsignal,1): 
        temp = arrsignal[:,signal] # Assign signal_n to temporary variable
        Fs=freqarr[0,signal] # To handle with a scalar
        new_Fs=Fs/stdfreq    # How many times the minimum frequency and the current frequency are different
        row = 0             # Variable for each of signal index data
        save_idx = 0        # Variables for indexing to the array to be stored
        while save_idx < len(t_domain): # down sampling
            new_signal[signal, save_idx] = temp[row]
            row += int(np.round(new_Fs)) # Move step by new_Fs
            save_idx+=1
    return new_signal, t_domain

def SignalUpSampling(arrsignal, stdfreq, freqarr, mtime, numsignal):
# @Up Sampling
    """
    Down Sampling Function
    Args:
        arrsignal (np.array): P by Q size by Signal data P is num of signal data, Q is num of signal
        stdfreq (float): Standard sampling frequancy
        freqarr (np.array): 1 by Q size by Signal sampling frequancy
        mtime (int): [sec] Measurement time
        numsignal (int): Q value
    Returns:
        np.array: Downsampled signal
        np.array: Time domain Values
    """
    t_domain = np.arange(0,mtime,1/stdfreq)       # Number divided by minimum sampling frequency per second
    new_signal_up = np.zeros([numsignal,len(t_domain)]) # Create space as many times as the number of time domains
    for signal in range(0,numsignal,1): 
        temp = arrsignal[:,signal] # Assign signal_n to temporary variable
        Fs=freqarr[0,signal] # To handle with a scalar
        new_Fs=Fs/stdfreq    # How many times the minimum frequency and the current frequency are different
        t_original = np.linspace(0, mtime, int(Fs*2))
        valid_data = temp[:len(temp)]
        # TypeError: Cannot cast array data from dtype('O') to dtype('float64')
        # UP Sampling을 하려면 np.nan이 필요하지만 과제 수행 내용은 NaN 데이터를 지우는 것이므로 사용할 수 없음
        # 해결 방법: ''을 0 또는 NaN으로 변경
        # --> 0을 사용할 수 없음 신호의 주기성으로 0도 신호의 중요한 데이터일 수 있으므로 NaN을 사용해야 함
        new_signal_up=[signal] = np.interp(t_domain,t_original,valid_data)
    return new_signal_up, t_domain


path = "problem_2_data.csv"
myfile = pd.read_csv(path)
df = pd.DataFrame(myfile)
df=df.fillna('') # Nan replaced to ''

df=df.values
Fs_min = math.inf
# Fs_max = 0
# @Up Sampling
Fs_arr = np.zeros([1,5]) # create Sampling Frequancy array
m_time = 2 # [sec] Measurement time
num_of_signal = 5

# find for min sampling frequancy
for sig in range(0,num_of_signal,1):    # sig is numbering of signal
    cur_signal = df[:,sig]
    idx =0                  # Variable for each of signal index data
    while cur_signal[idx] != '':
        idx+=1
        if idx>=len(df): break  # Prevent for ArrayIndexOutOfBounds error 
    Fs = idx/m_time         # Sampling Frequancy = Sampling Point / Measurement time
    Fs_arr[0,sig] = Fs      
    if(Fs < Fs_min):
        Fs_min = Fs
    # if(Fs > Fs_max):
    #    Fs_max = Fs
    # @Up Sampling
        
    
new_signal, t_domain = SignalDownSampling(df,Fs_min, Fs_arr, m_time, num_of_signal)
# new_signal_up, t_domain_up = SignalUpSampling(df,Fs_max,Fs_arr,m_time,num_of_signal)
# @Up Sampling

for idx in range(0,num_of_signal,1):
    plt.plot(t_domain,new_signal[idx],label='signal ' + str(idx+1))
    # plt.plot(t_domain_up,new_signal_up[idx],label='signal ' + str(idx+1))
    # @Up Sampling
    
plt.rc('font',size=20)
plt.xlabel('Time (s)',fontsize=20)
plt.ylabel('Value', fontsize=20)
plt.title('Signal Graphs',fontsize=24)
plt.grid(True)
plt.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
