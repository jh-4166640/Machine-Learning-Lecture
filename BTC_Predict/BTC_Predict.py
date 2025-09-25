# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 13:45:47 2025

@author: USER
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std
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

def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()
    
file_path = "C:\\Users\\USER\\Downloads\\BTC-USD.csv"
open_file = pd.read_csv(file_path)
df = pd.DataFrame(open_file)

titles = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

feature_keys = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
]

date_time_key = "Date"
show_raw_visualization(df)







