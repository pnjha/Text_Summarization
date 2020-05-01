import json
import os
import numpy as np
import sys
import pandas as pd

def load_data(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

f1 = "Ep_250_Ds_10000_Lr_0.01_Hs_256_Ml_0.2_Tf_0.5_GR_100_test_result.json"
f2 = "Ep_250_Ds_10000_Lr_0.01_Hs_256_Ml_0.3_Tf_0.5_GR_100_test_result.json"
f3 = "Ep_250_Ds_10000_Lr_0.001_Hs_256_Ml_0.5_Tf_0.5_GR_100_test_result.json"
f4 = "Ep_250_Ds_10000_Lr_0.01_Hs_256_Ml_0.08_Tf_0.5_GR_100_test_result.json"
f5 = "Ep_250_Ds_20000_Lr_0.001_Hs_256_Ml_0.4_Tf_0.5_GR_100_test_result.json"

d1 = load_data(f1)
d2 = load_data(f2)
d3 = load_data(f3)
d4 = load_data(f4)
d5 = load_data(f5)

g1 = {}
g2 = {}
g3 = {}
g4 = {}
g5 = {}

def get_data(d):
    a = []
    b = []
    g = {}
    c = {}
    for i in d:
        l = len(d[i]["Original_Text"].split())
        r1 = d[i]["Score"][0]["rouge-1"]["r"]
        if l < 75:
            a.append(r1)
            b.append(l)
    #     if l not in c:
    #         c[l] = 0
    #     c[l] += 1
    #     g[l] = r1
    # for i in sorted(g.keys()):
    #     g[i] = g[i]/c[i]
    #     a.append(i)
    #     b.append(g[i])
    return a,b


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from pylab import savefig

def plot_graph(xlist,ylist,xname,yname,filename):
    x = np.array(xlist)
    y = np.array(ylist)
    d = {xname: x, yname: y}
    data = pd.DataFrame(d)
    sns_plot = sns.lineplot(x=xname, y=yname,data = data)
    figure = sns_plot.get_figure()    
    figure.savefig(filename, dpi=400)
    # sns_plot.savefig(filename)
    # plt.show()
    plt.close()

a,b = get_data(d1)
plot_graph(b,a,'Sentence Length','ROUGE 1 Recal Score',"r1.png")

a,b = get_data(d2)
plot_graph(b,a,'Sentence Length','ROUGE 1 Recall Score',"r2.png")

a,b = get_data(d3)
plot_graph(b,a,'Sentence Length','ROUGE 1 Recall Score',"r3.png")

a,b = get_data(d4)
plot_graph(b,a,'Sentence Length','ROUGE 1 Recall Score',"r4.png")

a,b = get_data(d5)
plot_graph(b,a,'Sentence Length','ROUGE 1 Recall Score',"r5.png")

# plt.plot()
# plt.xlabel()
# plt.ylabel()
# plt.show()
# plt.savefig("Ep_250_Ds_10000_Lr_0.01_Hs_256_Ml_0.2_Tf_0.5_GR_100_test_result.png")