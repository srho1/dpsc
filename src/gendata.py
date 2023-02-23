from unicodedata import name
import numpy as np
import pandas as pd
import math

from scipy.stats import uniform
from scipy.stats import truncnorm
from numpy.random import laplace
from numpy import random, linalg
import matplotlib.pyplot as plt

import os
import sys
sys.path.append("../")

from src.my_utils import *

# random walk
def generateData(T, N, M, std):
    ### random walk
    noise = truncnorm(-M, M, loc=0.0, scale=std)
    df = pd.DataFrame(columns=range(N), index=range(T))

    # initial
    df.iloc[0] = np.random.uniform(0, 1, N)
    # alhpa
    alphas = np.random.uniform(1, 2, N)

    print(alphas)
    for i in range(1, T):
        df.iloc[i] = df.iloc[i - 1] * alphas + noise.rvs(N)
    #         print(df)

    plt.plot(df)
    plt.show()

    return df


# X = M(signal) + Z(noise, [-s,s]), linear

def generateData2(T, N, s, std):
    ### linear model
    
    # noise
    noise = truncnorm(-s, s, loc=0.0, scale=std)
    Z = noise.rvs(N * T)
    Z.shape = (T, N)

    # time stamps
    t = np.arange(1, T + 1)
    t.shape = (T, 1)

    # alpha (slope)
    signal_slope = truncnorm(3, 5, loc=4.0, scale=1)
    alphas = signal_slope.rvs(N)
    alphas.shape = (1, N)

    # intercept
    intercept = np.random.uniform(-1, 1, N)

    df = pd.DataFrame(np.dot(t, alphas)) # signal

    df += intercept

    a = df.min().min()
    b = df.max().max()

    df = df - (a + b) / 2
    df = df / df.max().max()
    # df = df * (1 - s)

    return df, df+Z


# X = M(signal) + eps(noise), sinusoidal
def generateData3(T, N, M, std):
    ### sine graph
    noise = truncnorm(-M, M, loc=0.0, scale=std)

    # time stamps
    t = np.arange(1, T + 1)
    t.shape = (T, 1)

    # time shift
    shifts = np.random.uniform(0, 2, N * T)
    shifts.shape = (T, N)

    df = pd.DataFrame(np.dot(t, np.ones((1, N))))
    df += shifts
    df = np.sin(df)

    # amplitude
    amps = np.random.uniform(0.5, 1, N)
    amps.shape = (1, N)

    df = df * amps

    # noise
    eps = noise.rvs(N * T)
    eps.shape = (T, N)

    df += eps

    return df



if __name__=="__main__":
    
    for t in [10, 100]:
        for n in [10, 100]:
    
            T, N = t+3, n+1
            s, std = 1.0, 0.1
            unit = 0 # target unit

            name = "data_{}_{}".format(str(t), str(n))

            ############## Generate Data #################

            df_signal, df = generateData2(T, N, s, std)

            plt.plot(df.iloc[:,1:], color="grey", alpha=0.5) #donors
            plt.plot(df[0], color="red", marker='.', label="target") #target
            # plt.title("synthetic linear data, signal+noise (T={}, N={})".format(T, N))
            plt.axvline(x=t-0.5, color="grey", linestyle='--')
            plt.xticks(ticks=np.arange(0,T, int(t/10)), labels=np.arange(1,T+1, int(t/10)))
            plt.legend()
            plt.savefig("../data/synthetic/{}.png".format(name))
            plt.clf()

            plt.plot(df_signal.iloc[:,1:], color="grey", alpha=0.5) #donors
            plt.plot(df_signal[0], color="red", label="target") #target
            plt.axvline(x=t-0.5, color="grey", linestyle='--')
            plt.xticks(ticks=np.arange(0,T, int(t/10)), labels=np.arange(1,T+1, int(t/10)))
            # plt.title("synthetic linear data, signal only (T={}, N={})".format(T, N))
            plt.legend()
            plt.savefig("../data/synthetic/{}_signal.png".format(name))
            plt.clf()

            df_signal.to_csv("../data/synthetic/{}_signal.csv".format(name), index=False, header=None)
            df.to_csv("../data/synthetic/{}.csv".format(name), index=False, header=None)

