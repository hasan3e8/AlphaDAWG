#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras import models
from keras import layers
import pandas as pd
import numpy as np
from keras import regularizers
from sklearn.model_selection import KFold
import random

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description= 'Preprocess .ms files')
parser.add_argument('level', type=int, help= 'Level of wavelet decomposition')
parser.add_argument('chr_no', type=int, help= 'Chromosome number')





args = parser.parse_args()

Level =  args.level
chromo =  args.chr_no



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


kfold = KFold(5, shuffle = True, random_state=1)

lambdas = [1e-20, 1e-10, 1e-5, 1e-3, 1]
gammas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
levels = [Level] 

C_n = pd.read_csv(f"./Curvelets_neut_train_.csv", header = None)
C_s = pd.read_csv(f"./Curvelets_sweep_train_.csv", header = None)


dfs_C = pd.concat([C_n, C_s])
dfs_C = np.asarray(dfs_C)

for i in range(len(dfs_C)):
    dfs_C[i][abs(dfs_C[i]) < np.percentile(abs(dfs_C[i]),99)] = 0


C_n_ = pd.read_csv(f"./Curvelets_neut_test_.csv", header = None)
C_s_ = pd.read_csv(f"./Curvelets_sweep_test_.csv", header = None)

dfx_C = pd.concat([C_n_, C_s_])

dfx_C = np.asarray(dfx_C)

for i in range(len(dfx_C)):
    dfx_C[i][abs(dfx_C[i]) < np.percentile(abs(dfx_C[i]),99)] = 0

N = pd.read_csv(f"./Wavelets_neut_train_.csv", header = None)
N1 = np.asarray(N)
S = pd.read_csv(f"./Wavelets_sweep_train_.csv", header = None)
S1 = np.asarray(S)
dfs_W = pd.concat([N, S])


dfs_W = np.asarray(dfs_W)

for i in range(len(dfs_W)):
    dfs_W[i][abs(dfs_W[i]) < np.percentile(abs(dfs_W[i]),99)] = 0



W_n_ = pd.read_csv(f"./Wavelets_neut_test_.csv", header = None)
W_s_ = pd.read_csv(f"./Wavelets_sweep_test_.csv", header = None)
dfx_W = pd.concat([W_n_, W_s_])
W_s_=np.asarray(W_s_)
W_n_=np.asarray(W_n_)

dfx_W = np.asarray(dfx_W)

for i in range(len(dfx_W)):
    dfx_W[i][abs(dfx_W[i]) < np.percentile(abs(dfx_W[i]),99)] = 0
    


a = np.zeros(np.shape(N1)[0])
b = np.ones(np.shape(S1)[0])
y = np.concatenate([a,b])

a1 = np.zeros(np.shape(W_n_)[0])
b1 = np.ones(np.shape(W_s_)[0])
y1 = np.concatenate([a1,b1])

df = pd.read_csv("./Empirical/CRV"+str(chromo)+".csv", header = None)
dfx1_C=np.array(df)


df = pd.read_csv("./Empirical/WAV"+str(chromo)+".csv", header = None)
dfx1_W=np.array(df)

for j in range(len(dfx1_W)):
    dfx1_W[j][abs(dfx1_W[j]) < np.percentile(abs(dfx1_W[j]),99)] = 0
    
for j in range(len(dfx1_C)):
    dfx1_C[j][abs(dfx1_C[j]) < np.percentile(abs(dfx1_C[j]),99)] = 0


dfx1_ = pd.concat([pd.DataFrame(dfx1_W), pd.DataFrame(dfx1_C)],axis=1)
dfx1_=np.array(dfx1_)
idx1_ = np.argwhere(np.all(dfx1_[..., :] == 0, axis=0))




dfx = pd.concat([pd.DataFrame(dfx_W), pd.DataFrame(dfx_C)],axis=1)
dfx=np.array(dfx)
idx1 = np.argwhere(np.all(dfx[..., :] == 0, axis=0))


dfs = pd.concat([pd.DataFrame(dfs_W), pd.DataFrame(dfs_C)],axis=1)
dfs=np.array(dfs)
idx = np.argwhere(np.all(dfs[..., :] == 0, axis=0))


A = [i for i in range(14617) if i not in idx]
B = [i for i in range(14617) if i not in idx1_]
C = [i for i in range(14617) if i not in idx1]

D = list(set(A) & set(B) & set(C))

dfs=dfs[:,D]
dfx1_=dfx1_[:,D]
dfx=dfx[:,D]

N_=dfx[0:np.shape(W_n_)[0]]
S_=dfx[np.shape(W_n_)[0]:np.shape(W_n_)[0]+np.shape(W_s_)[0],]


med_test_neut=np.median(N_, axis=0)
med_emp = np.median(dfx1_, axis=0)
mad_test_neut= np.median(np.absolute(N_ - np.median(N_, axis=0)), axis=0)
mad_emp = np.median(np.absolute(dfx1_ - np.median(dfx1_, axis=0)), axis=0)

r=np.where(mad_emp!=0)
mad_emp1=[i for i in mad_emp if i !=0]

X_med0_mad1 = dfx1_[:,r[0]]-med_emp[r[0]]
X_med0_mad1 = X_med0_mad1/ mad_emp1

emp_adj_med = X_med0_mad1*mad_emp1
emp_adj_med = emp_adj_med+med_test_neut[r[0]]

emp_adj_med_mad = X_med0_mad1*mad_test_neut[r[0]]
emp_adj_med_mad = emp_adj_med_mad+med_test_neut[r[0]]

dfs=dfs[:,r[0]]
dfx=dfx[:,r[0]]

# for i in range(len(D)):
#     if i not in r[0]:
#         DFS_[:,i]=0

level_loss = []
for i in levels:
    print(np.shape(dfs))
    g_loss = []
    for g in gammas:
        l_loss = []
        for l in lambdas:
            for train, test in kfold.split(dfs):
                model = models.Sequential()
                model.add(layers.Dense(8, kernel_regularizer=regularizers.l1_l2(l1=(1-g)*l, l2=g*l), activation='sigmoid', input_shape=(np.shape(dfs)[1],)))
                model.add(layers.Dense(1, kernel_regularizer=regularizers.l1_l2(l1=(1-g)*l, l2=g*l),activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
                history = model.fit(dfs[train], y[train], epochs=100, batch_size=100, validation_data=(dfs[test], y[test]), verbose = 0)
            history_dict = history.history
            valoss = np.mean(history_dict['val_loss'])
            l_loss.append(valoss)
        g_loss.append(l_loss)
    level_loss.append(g_loss)

    

np.unravel_index(np.argmin(level_loss, axis=None), np.shape(level_loss))

c = np.unravel_index(np.argmin(level_loss, axis=None), np.shape(level_loss))

gamma =  gammas[c[1]]
lamda =  lambdas[c[2]]

std_scale = StandardScaler()
X_train_std = std_scale.fit_transform(dfs)
X_test_std_sim = std_scale.transform(dfx)

model = models.Sequential()
model.add(layers.Dense(8, kernel_regularizer=regularizers.l1_l2(l1=(1-gamma)*lamda, l2=gamma*lamda), activation='sigmoid', input_shape=(np.shape(dfs)[1],)))
model.add(layers.Dense(1, kernel_regularizer=regularizers.l1_l2(l1=(1-gamma)*lamda, l2=gamma*lamda),activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_std, y,validation_split = 0.1, epochs=50, batch_size=100,verbose=0)

results = model.evaluate(X_test_std_sim, y1)  # Performance on test data

X_test_std_emp1 = std_scale.transform(emp_adj_med_mad)
l=list(model.predict(X_test_std_emp1, verbose=0))

l1=list(model.predict(X_test_std_sim, verbose=0))

pd.DataFrame(l).to_csv("Probabilities of samples in chromosome_" +str(chromo)+".csv")
pd.DataFrame(l1).to_csv("Probabilities of samples in simulated data.csv")
