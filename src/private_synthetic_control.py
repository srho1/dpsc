import math
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.random import laplace
from numpy import random, linalg
import cvxpy as cp

from src.my_utils import *
from src.matrix import Matrix
from src.synthetic_control import SyntheticControl

class PrivateSC(SyntheticControl):
    def __init__(self):
        super().__init__()

    def set_params(self, method: str = 'out', lmbda=1, eps1=5, eps2=5, delta=0, c=None, thresh=0.5, max_count=1):
        self.method = method
        # for out, obj perturbation
        self.lmbda = lmbda
        self.eps1 = eps1
        self.eps2 = eps2
        self.delta = delta
        self.sens = None
        self.b = None
        self.fit_intercept = None

        # for output perturbation
        if self.method == "out":
            self.a = None

        # for obj perturbation
        if self.method == "obj":
            self.c = c
            self.beta = None
            self.eps0 = None
            self.f = None
            self.Delta = None

        # for rejection sampling
        self.thresh = thresh
        self.max_count = max_count

        # filled in later
        self.n = None
        self.T0 = None

    def fit(self, pre_donor, pre_target, fit_intercept=False):

        self.n = pre_donor.shape[1]
        self.T0 = pre_donor.shape[0]
        self.fit_intercept = fit_intercept

        self.sens = (4 * self.T0 * np.sqrt(8 + self.n)) / self.lmbda

        if self.method == 'out':
            # lmbda = Ridge regression parameter
            # eps = Privacy parameter

            # thresh = 0.5 #parameter: boundary to assess "good" pre-interv fit
            # max_count = 100 #parameter: max number of iterations

            # Lap(a) to be sampled later
            # self.a = (4 * self.T0 * np.sqrt(8 + self.n)) / (self.lmbda * self.eps1)
            self.a = self.sens/self.eps1

            # learn using Ridge regression parm lmbda/2
            alpha = self.lmbda / 2
            self.model = linear_model.Ridge(alpha=alpha, fit_intercept=self.fit_intercept)
            self.model.fit(pre_donor, pre_target)

            #### no rejection smapling #####
            # update
            if self.fit_intercept:
                self.weights = np.append(self.model.coef_, self.model.intercept_).copy()
            else:
                self.weights = self.model.coef_.copy()
            noise = laplace_sample(len(self.weights.flatten()), 1, self.a).flatten()
            self.weights += noise
            
            #### rejection smapling #####
            # count = 0
            # error = math.inf
            # while error > self.thresh and count < self.max_count:
            #     count += 1

            #     # update
            #     if self.fit_intercept:
            #         self.weights = np.append(self.model.coef_, self.model.intercept_).copy()
            #     else:
            #         self.weights = self.model.coef_.copy()
            #     noise = laplace_sample(len(self.weights.flatten()), 1, self.a).flatten()
            #     self.weights += noise

            #     # error
            #     if self.fit_intercept:
            #         pred = self.predict_design_mat(pre_donor)
            #     else:
            #         pred = np.dot(pre_donor, self.weights.T)
            #     error = self.mape(pre_target, pred)


        elif self.method == 'obj':

            ## find c
            # #TODO: are these T0 instead of n?
            # E = np.zeros((self.n, self.n))
            # E[0, 0] = 2 * self.n  # ranges 0~2n
            # E[1:, 0] = 4 * self.n  # ranges -4n~4n
            # E[0, 1:] = 4 * self.n
            # # u, s, vh = np.linalg.svd(E, full_matrices=True)
            # w, v = np.linalg.eig(E)
            # c = np.abs(w)[0]

            if not self.c:
                self.c = (1 + np.sqrt(16* self.n - 15) ) * self.T0

            # prepare X, y, n
            X = pre_donor.copy()
            y = pre_target.copy()
            n = self.n
            # if setting intercept, we make a design matrix X
            if fit_intercept:
                # todo: fix a bug when fit_intercept=True
                n += 1
                X[np.nan] = 1
                y[np.nan] = 1
            X = X.to_numpy()
            y = y.to_numpy().flatten()

            # first if loop
            condition = math.log(1 + (2 * self.c / self.lmbda ) + (self.c ** 2 / self.lmbda ** 2) )
            self.eps0 = self.eps1 - condition

            if self.eps0 > 0:
                self.Delta = 0
            else:
                self.eps0 = self.eps1 / 2
                self.Delta = self.c / (math.e ** (self.eps1 / 4) - 1) - self.lmbda

            # second if loop
            if self.delta > 0:
                self.beta = 4 * self.T0 * math.sqrt(8 + n) * math.sqrt(2 * math.log(2 / self.delta) + self.eps0) / self.eps0
                b = np.random.multivariate_normal(np.zeros(n), np.dot(self.beta, np.eye(n)))
                b = b.flatten()
            else:
                beta1 = 4 * self.T0 * math.sqrt(8 + n) / self.eps0
                beta2 = (self.c * math.sqrt(n) + 4 * self.T0) / self.eps0
                self.beta = min(beta1, beta2)
                b = laplace_sample(d=n, n=1, b=self.beta)

            ## Construct the problem.
            f = cp.Variable(n)
            self.f = f
            # print("f", f.shape)
            # this objective is actually T0 * J^{obj}
            objective = cp.Minimize(cp.sum_squares(y - X @ f) +  ((self.lmbda + self.delta) / 2) * cp.quad_form(f, np.eye(n)) + b @ f)
            constraints = []
            prob = cp.Problem(objective, constraints)
            prob.solve()
            self.weights = f.value.flatten()
            # # Print result.
            # print("\nThe optimal value is", prob.value)
            # print("The optimal f is")
            # print(f.value)

    def predict_design_mat(self, donor):
        # make a prediction with design matrix (with an additional all-1 column)
        temp_donor = donor.copy()
        temp_donor[np.nan] = 1
        pred = np.dot(temp_donor, self.weights.T)
        return pd.DataFrame(pred, index=donor.index, columns=self.target)

    def predict(self, donor):
        T_T0 = donor.shape[0]
        if not self.method:
            print("Fit the model first.")
            # error handling msg

        elif self.method == "out" or self.method == "obj":

            self.b = 2 * np.sqrt(T_T0) / self.eps2
            # print("b=", b)
            # print("b/sqrt(T-T0)=", b / np.sqrt(T_T0))

            ## noise
            W = laplace_sample( T_T0 * self.n, 1, self.b).reshape(T_T0, self.n)
            X_tilde = donor + W

            if self.fit_intercept:
                pred = self.predict_design_mat(X_tilde).values.flatten()
            else:
                pred = np.dot(X_tilde, self.weights.T)
        return pd.DataFrame(pred, index=donor.index, columns=self.target)

