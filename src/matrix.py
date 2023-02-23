import numpy as np
import pandas as pd


class Matrix:
    def __init__(
        self, data: pd.DataFrame, T0: int, target_name, donor_names: list = None
    ):
        """
        Initialize the data matrix for RSC
        Args:
            data: pandas dataframe; each column is a time series
            T0: int, the number of pre-intervention time points. 
            num_sv : number of singular values to keep
            target_name: name of series that we intend to predict (target unit)
            donor_names: list of non-target ids (donor pool)
        
        Notes:
            In the paper, T0 starts with 1, so first intervention point is T0+1
            Here time index starts with 0, so first treatment/post-intervention unit's index is T0
        """
        self.data = data
        self.T0 = T0
        self.target_name = target_name

        self.n = data.shape[0]
        self.T = data.shape[1]

        # if donors are not specified, then they are all column names in data excluding target
        if donor_names is None:
            self.donor_names = np.setdiff1d(self.data.columns.values, target_name)
        else:
            self.donor_names = donor_names

        self.pre_target = self.get_pre_target(self.data)
        self.post_target = self.get_post_target(self.data)
        self.pre_donor = self.get_pre_donor(self.data)
        self.post_donor = self.get_post_donor(self.data)

    def denoise(self, filter_method: str = "HSVT", num_sv: int = 2):
        if filter_method == "HSVT":
            # do hsvt for donor pool only
            denoised_donors = self.hsvt(
                self.data.drop(columns=self.target_name), num_sv
            )

            self.pre_donor = self.get_pre_donor(denoised_donors)
            self.post_donor = self.get_post_donor(denoised_donors)

        elif filter_method == "HSVT2":
            # do hsvt separately for pre-intervention, donor pool, and entire data matrix (for post-target)
            denoised_pre = self.hsvt(self.data[: self.T0], num_sv)
            denoised_donors = self.hsvt(
                self.data.drop(columns=self.target_name), num_sv
            )
            denoised_all = self.hsvt(self.data, num_sv)

            self.pre_target = self.get_pre_target(denoised_pre)
            self.post_target = self.get_post_target(denoised_all)
            self.pre_donor = self.get_pre_donor(denoised_donors)
            self.post_donor = self.get_post_donor(denoised_donors)

    def hsvt(self, df, rank: int = 2):
        """
        Input:
            df: matrix of interest
            rank: rank of output matrix
        Output:
            thresholded matrix
        """

        u, s, v = np.linalg.svd(df, full_matrices=False)
        s[rank:].fill(0)
        vals = np.dot(u * s, v)
        return pd.DataFrame(vals, index=df.index, columns=df.columns)

    def get_target(self):
        return pd.concat((self.pre_target, self.post_target), axis=0)

    def get_donor(self):
        return pd.concat((self.pre_donor, self.post_donor), axis=0)

    def get_pre_target(self, data: pd.DataFrame):
        return data[: self.T0][[self.target_name]]

    def get_post_target(self, data: pd.DataFrame):
        return data[self.T0 :][[self.target_name]]

    def get_pre_donor(self, data: pd.DataFrame):
        return data[: self.T0].drop(
            columns=self.target_name, errors="ignore"
        )  # ignores KeyError

    def get_post_donor(self, data: pd.DataFrame):
        return data[self.T0 :].drop(columns=self.target_name, errors="ignore")


# Test with random values
"""
X = np.arange(25).reshape((5,5))  
df = pd.DataFrame(X)
M = Matrix(df, T0 = 3)

print('data\n', df)
print('\npre_target\n', M.pre_target)
print('\npost_target\n', M.post_target)
print('\npre_donor\n', M.pre_donor)
print('\npost_donor\n', M.post_donor)
"""
