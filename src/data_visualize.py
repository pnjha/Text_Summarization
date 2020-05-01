import json
import os
import numpy as np
import sys
import pandas as pd
import seaborn as sns; sns.set(style="whitegrid")
import matplotlib.pyplot as plt
from pylab import savefig

test_original = os.getcwd() + "/data_bc/test.original"
train_original = os.getcwd() + "/data_bc/train.original"
test_compressed = os.getcwd() + "/data_bc/test.compressed"
train_compressed = os.getcwd() + "/data_bc/train.compressed"

def plot_graph(xlist,ylist,xname,yname,filename):
    x = np.array(xlist)
    y = np.array(ylist)
    d = {xname: x, yname: y}
    data = pd.DataFrame(d)
    sns_plot = sns.lineplot(x=xname, y=yname,data = data)
    figure = sns_plot.get_figure()    
    figure.savefig(filename, dpi=400)
    # sns_plot.savefig(filename)
    plt.show()
    plt.close()

def bar_plot(xlist,ylist,xname,yname,filename):
    x = np.array(xlist)
    y = np.array(ylist)
    d = {xname: x, yname: y}
    data = pd.DataFrame(d)
    sns.despine(left=True, bottom=True)
    sns_plot = sns.barplot(x=xname, y=yname, data=data)
    figure = sns_plot.get_figure()    
    figure.savefig(filename, dpi=400)
    # sns_plot.savefig(filename)
    plt.show()
    plt.close()

def get_data(filename):
    f = open(filename, "r")
    vocab = {}
    avg_len = 0
    len_dist = {}
    cnt = 0
    d = []
    y = []
    for x in f:
        cnt += 1
        y.append(cnt)
        l = x.split()
        avg_len += len(l)
        d.append(avg_len)
        for i in l:
            if i not in vocab:
                vocab[i] = 1
            else:
                vocab[i] += 1

    print("Number of words: ",avg_len)
    return vocab, avg_len/cnt, len(vocab), d , y

# print("train_original")
# v, avg_len, v_len, d, y = get_data(train_original)
# print(avg_len, v_len)
# plot_graph(d, y, "Sentence Length", "temp", "train_original_len_dist")
# print("train_compressed")
# v, avg_len, v_len, d, y = get_data(train_compressed)
# print(avg_len, v_len)
# plot_graph(d,y, "Sentence Length", "temp", "train_compressed_len_dist")
# print("test_original")
# v, avg_len, v_len, d, y = get_data(test_original)
# print(avg_len, v_len)
# plot_graph(d,y, "Sentence Length", "temp", "test_original_len_dist")
# print("test_compressed")
# v, avg_len, v_len, d, y = get_data(test_compressed)
# print(avg_len, v_len)
# plot_graph(d,y, "Sentence Length", "temp", "test_compressed_len_dist")


def line_info(filename):
    f = open(filename, "r")
    line = {}
    for x in f:
        length = len(x.split())
        if(length>500):
            print(x)
            print(x.split())
        if length not in line:
            line[length] = 1
        else:
            line[length] += 1

    a = []
    b = []

    for i in sorted(line.keys()):
        a.append(i)
        b.append(line[i])

    # print(a,b)
    return a,b

a,b = line_info(train_original)
print(a,b)
bar_plot(a,b,"Sentence Length","Frequency", "Sentence Length vs Frequency")
# plot_graph(a,b,"as","ass","asd")
