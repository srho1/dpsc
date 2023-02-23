import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import os
import sys
import argparse

sys.path.append("../")

from src.gendata import *
from src.matrix import Matrix
from src.synthetic_control import SyntheticControl
from src.private_synthetic_control import PrivateSC

##############################################
################### params ###################
##############################################
parser = argparse.ArgumentParser(description='Zipf')

parser.add_argument('--method', choices=["out", "obj", "non-private"], type=str, default="obj")

parser.add_argument('--T0', type=int, default=10, help='Intervention timepoint')
parser.add_argument('--n', type=int, default=10, help='number of donors')
parser.add_argument('--test_points', type=int, default=3, help='# of test timepoints')
parser.add_argument('--rank', type=int, default=3, help='approximate rank for HSVT, non-private SC')
parser.add_argument('--unit', type=int, default=0, help='target unit')

parser.add_argument('--lmbda', type=float, default=10.0, help='lmbda, regularization coefficient')
# parser.add_argument('--eps1', type=float, default=50.0, help='eps1, privacy budget for step 1')
# parser.add_argument('--eps2', type=float, default=50.0, help='eps2, privacy budget for step 2')
parser.add_argument('--delta', type=float, default=0, help='delta, privacy budget in case we allow delta (DPSC_obj)')
parser.add_argument('--thresh', type=float, default=0.1, help='threshold for rejection sampling')

args = parser.parse_args()

###################################################
################### import data ###################
###################################################


if args.T0==10:

    if args.n==10:
        #n=10
        df = pd.read_csv("../data/synthetic/data_10_10.csv", header=None)

    if args.n==100:
        #n=100
        df = pd.read_csv("../data/synthetic/data_10_100.csv", header=None)

elif args.T0==100:
    if args.n==10:
        #n=10
        df = pd.read_csv("../data/synthetic/data_100_10.csv", header=None)

    if args.n==100:
        #n=100
        df = pd.read_csv("../data/synthetic/data_100_100.csv", header=None)


# In this experiment, we fix eps1 = eps2 = 50
args.eps1 = 50
args.eps2 = 50

###################################################
################### non-private ###################
###################################################
M = Matrix(df, T0=args.T0, target_name=args.unit)
syc = SyntheticControl()

alpha = args.lmbda/2
syc.fit(M.pre_donor, M.pre_target, method="ridge", lmbda = alpha, fit_intercept=False)

f_reg = syc.model.coef_
pred = syc.predict(M.get_donor())

print("non-private", args.T0, args.n, args.test_points, args.rank, args.lmbda, args.eps1, args.eps2, args.delta, args.thresh, mse_pre, mse_post)


###################################################
################### obj perturb ###################
###################################################

M = Matrix(df, T0=args.T0, target_name=args.unit)

psc = PrivateSC()
psc.set_params(method= "obj", lmbda=args.lmbda, eps1=args.eps1, eps2=args.eps2, delta=args.delta)
psc.fit(M.pre_donor, M.pre_target, fit_intercept=False)
pred_priv = psc.predict(M.get_donor())

# error
mse_pre = psc.mse(M.pre_target, pred_priv[:args.T0])
mse_post = psc.mse(M.post_target[:args.test_points], pred_priv[args.T0:args.T0+args.test_points])

print("obj", args.T0, args.n, args.test_points, args.rank, args.lmbda, args.eps1, args.eps2, args.delta, args.thresh, mse_pre, mse_post)


###################################################
################### out perturb ###################
###################################################
psc = PrivateSC()
psc.set_params(method= "out", lmbda=args.lmbda, eps1=args.eps1, eps2=args.eps2, delta=args.delta, max_count=1)
psc.fit(M.pre_donor, M.pre_target, fit_intercept=False)
pred_priv = psc.predict(M.get_donor())
# error
mse_pre = psc.mse(M.pre_target, pred_priv[:args.T0])
mse_post = psc.mse(M.post_target[:args.test_points], pred_priv[args.T0:args.T0+args.test_points])

print("out", args.T0, args.n, args.test_points, args.rank, args.lmbda, args.eps1, args.eps2, args.delta, args.thresh, mse_pre, mse_post)
