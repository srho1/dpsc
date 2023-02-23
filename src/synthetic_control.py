import math

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.random import laplace
from numpy import random, linalg

from src.matrix import Matrix
from src.my_utils import *

class SyntheticControl:
    def __init__(self):
        self.target = None
        self.method = None

        self.model = None
        self.weights = None

    def fit(self, pre_donor, pre_target, method: str = 'linreg', lmbda=None, fit_intercept=False):
        self.target = pre_target.columns.values
        self.method = method
        if method == 'linreg':
            self.model = linear_model.LinearRegression(fit_intercept=fit_intercept)
            self.model.fit(pre_donor, pre_target)
        elif method == 'ridge':
            self.model = linear_model.Ridge(alpha=lmbda, fit_intercept=fit_intercept)
            self.model.fit(pre_donor, pre_target)
        elif method == 'lasso':
            self.model = linear_model.Lasso(alpha=lmbda, fit_intercept=fit_intercept)
            self.model.fit(pre_donor, pre_target)

    def predict(self, donor):
        if not self.method:
            print("Fit the model first.")
            # error handling msg
        else:
            pred = self.model.predict(donor)
        return pd.DataFrame(pred, index = donor.index, columns = self.target) 
    
    def mse(self, real_target, predicted_target):
        return mean_squared_error(real_target, predicted_target)

    def mape(self, real_target, predicted_target, multioutput='uniform_average'):
        return mean_absolute_percentage_error(real_target, predicted_target, multioutput=multioutput)

    def r2score(self, real_target, predicted_target):
        return r2_score(real_target, predicted_target)
    
    def score(self, donor, target):
        return self.model.score(donor, target)



'''
class PrivateSC(SyntheticControl):
    def __init__(self):
        super().__init__()

    def set_params(self, lmbda=1, eps1=5, eps2=5, delta=0, thresh=0.5, max_count=100):
        #for out, obj perturbation
        self.lmbda = lmbda
        self.eps1 = eps1
        self.eps2 = eps2
        self.delta = delta

        #for obj perturbation
        self.c = None

        #for rejection sampling
        self.thresh = thresh
        self.max_count = max_count

        # filled in later
        self.n = None
        self.T0 = None


    def fit(self, pre_donor, pre_target, method: str = 'obj', fit_intercept=True):
        self.n = pre_donor.shape[1]
        self.T0 = pre_donor.shape[0]

        if method == 'out':
            # lmbda = Ridge regression parameter
            # eps = Privacy parameter

            # thresh = 0.5 #parameter: boundary to assess "good" pre-interv fit
            # max_count = 100 #parameter: max number of iterations

            # Lap(a) to be sampled later
            a = (4 * self.T0 * np.sqrt(8+n) ) / (self.lmbda * self.eps1)

            # learn using Ridge regression parm lmbda/2T0
            alpha = self.lmbda/(2*self.T0)
            self.model = linear_model.Ridge(alpha=alpha, fit_intercept=fit_intercept)
            self.model.fit(pre_donor, pre_target)

            count = 0
            error = math.inf
            while error > thresh and count<max_count:
                count+=1
                # update
                self.weights = np.append(self.model.coef_, self.model.intercept_)
                noise = laplace_sample(len(self.weights), 1, a).flatten()
                self.weights += noise

                #error
                pred = self.predict(pre_donor)
                error = self.mape(pre_target, pred)

        elif method == 'obj':

            ## find c
            E = np.zeros((self.n, self.n))
            E[0, 0] = 2 * self.n  # ranges 0~2n
            E[1:, 0] = 4 * self.n  # ranges -4n~4n
            E[0, 1:] = 4 * self.n
            # u, s, vh = np.linalg.svd(E, full_matrices=True)
            w, v = np.linalg.eig(E)
            self.c = np.abs(w)[0]

            self.eps0 = eps1 - math.log(1 + (2 * c + c ** 2) / self.lmbda)

            if self.eps0 > 0:
                self.Delta = 0
            else:
                self.eps0 = self.eps1 / 2
                self.Delta = c / (math.e ** (self.eps1 / 4) - 1) - self.lmbda

            if delta > 0:
                beta = 4 * self.T0 * math.sqrt(8 + n) * math.sqrt(2 * math.log(2 / delta) + 2 * eps0) / eps0
                b = np.random.multivariate_normal(np.zeros( self.n ), np.dot(beta, np.eye(n)))
            else:
                beta = 4 * self.T0 * math.sqrt(8 + n) / self.eps0
                b = laplace_sample(d=n, n=1, b=beta)

            ## Construct the problem.
            f = cp.Variable(n)
            objective = cp.Minimize(
                cp.sum_squares(y - X @ f) + (lmd_real + delta) / 2 * cp.quad_form(f, np.eye(n)) + b.T @ f)
            # this objective is actually T0 * J^{obj}
            constraints = []
            prob = cp.Problem(objective, constraints)

            prob.solve()

            # Print result.
            print("\nThe optimal value is", prob.value)
            print("The optimal f is")
            print(f.value)

            f_obj = f.value

    def predict(self, donor):
        T = self.T0 + donor.shape[0]
        if not self.method:
            print("Fit the model first.")
            # error handling msg
        elif self.method == "out":
            donor[np.nan] = 1
            pred = np.dot(donor, self.weights.T)

            b = np.sqrt( 2* (T - self.T0) )
            b = b / self.eps2
            print("b=", b)
            print("b/sqrt(T-T0)=", b / np.sqrt(T - T0))

            noise = laplace_sample(T - T0, 1, b).flatten()
            pred = syc.predict(M.post_donor).values.flatten() + noise


        elif self.method == "obj":
            pass

        return pd.DataFrame(pred, index = donor.index, columns = self.target)
'''





# Test 
'''
X = np.sin(np.random.rand(10,5))
df = pd.DataFrame(X)
M = Matrix(df, T0 = 5, target_name = 0)

print('data\n', df)
print('\npre_target\n', M.pre_target)
print('\npre_donor\n', M.pre_donor)
print('\npost_donor\n', M.post_donor)
print('\npost_target\n', M.post_target)

syc = SyntheticControl()
syc.fit(M.pre_donor, M.pre_target)
print('\nweights:\n', syc.weights)
print('\npredict\n', syc.predict(M.donor))
print('\nR^2:\n', syc.score(M.post_donor, M.post_target))
'''