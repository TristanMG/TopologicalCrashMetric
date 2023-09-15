# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 20:15:11 2023

@author: tmade
"""

from ripser import ripser
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

def log_returns(M):
    # Compute the log returns from the adjusted close data
    R=np.log(M[1:,:]/M[:-1,:])
    return R

def time_window(data,w):
    """
    
    Create a list of windows from the log returns

    Parameters
    ----------
    data : Matrix
        list of the log returns of each index
    w : int
        size of the sliding window

    Returns
    -------
    points : array
        list of windows

    """
    points=[]
    for i in range(len(data)-w):
        points.append(data[i:i+w,:])
    return points

def norm_persistence(points):
    """
    Compute the persistent homology of degree 0 and 1 using the Rips filtration
    of a point cloud, here a window of log returns.
    Return the L1 norm of the topological features

    Parameters
    ----------
    points : array
        window of log returns

    Returns
    -------
    N : float
        L1 norm of the topological features

    """
    
    #Compute the persistent homology
    dgms = ripser(points,maxdim=1)['dgms']
    
    #Remove the topological feature of degree 0 with a death equal to infinity
    dgms[0]=dgms[0][:-1,:]
    N=0
    
    for bd_l in dgms:
        if len(bd_l)>=1:
            lifetime_l=bd_l[:,1]-bd_l[:,0]
            L=lifetime_l.sum()
            N+=L
    return N




window_size=50
folder="indices_europe/"

#List of indices to consider
indices=["^FTSE","^FCHI","^GDAXI","^N100","^SSMI"]

#Load the adjusted closed prices of the indices
df=pd.read_csv(folder+indices[0]+".csv",index_col="Date",parse_dates=True,usecols=['Date','Adj Close'])
for index in indices[1:]:
    df_t=pd.read_csv(folder+index+".csv",index_col="Date",parse_dates=True,usecols=['Date','Adj Close'])
    df_t=df_t.rename(columns={"Adj Close":f"Adj Close {index}"})
    # if df_t.index[0].timestamp() < datetime.fromisoformat("1990-12-31").timestamp():
    df=df.merge(df_t,on="Date",how="left")

df=df.dropna()

#Compute the log returns
data=log_returns(df.to_numpy())

#Extract the closed prices and log returns of FTSE
ftse=df.to_numpy()[:,0]
lr_ftse=data[:,0]



#Creations of the windows
windows=time_window(data, window_size)

#Computation of the topological crash metric for each window
TCM=np.zeros(len(windows))
for i,pts in enumerate(windows):
    TCM[i]=norm_persistence(pts)
    
    
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot( ftse[window_size+1:], 'g-',label="FTSE")
ax2.plot( TCM, 'b-',label="TCM")
fig.legend()
plt.title("FTSE adjusted closed and TCM")
plt.show()


#Maximum lag
lags_max=252

#Refining the log returns of the days of FTSE such that 
#the TCM has been computed from the last window_size days

lr_ftse=lr_ftse[window_size:]

corr=np.zeros(lags_max)
for i in range(len(corr)):
    if i != 0:
        corr[i]=stats.pearsonr(abs(lr_ftse[i:]), TCM[:-i])[0]
    else:
        corr[i]=stats.pearsonr(abs(lr_ftse), TCM)[0]
        
        
plt.plot(corr)
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.title("Correlation between the log returns of FTSE and TCM")
plt.show()
