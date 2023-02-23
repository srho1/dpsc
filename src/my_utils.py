import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def singval_test(test_df):
    # test_df: df
    (U, s, Vh) = np.linalg.svd(test_df.values)
    s2 = np.power(s, 2)
    spectrum = np.cumsum(s2)/np.sum(s2)

    plt.plot(spectrum[:50])
    plt.grid()
    plt.title("Cumulative energy")
    plt.figure()
    #plt.rcParams["figure.figsize"] = [16,9]
    plt.plot(s2[:50])
    plt.grid()

    plt.xlabel("Ordered Singular Values")
    plt.ylabel("Energy")

    plt.title("Singular Value Spectrum")
    plt.show()
    return s2

def unif_sphere(d: int, n: int):
    # returns n samples of d dimensional random vector, uniform over unit sphere
    # output type numpy array n by d

    output = np.zeros(shape=(n, d))
    for i in range(n):
        x = np.random.normal(0, 1, size=d)
        norm = np.sqrt(np.sum(x ** 2))
        output[i] = x / norm

    return output

def laplace_sample(d: int, n: int, b: float):
    # Lap(b), n-samples in d-dimensional vector
    lap = np.random.laplace(loc=0.0, scale=b, size=n)
    uni = unif_sphere(d, n)
    return np.multiply(lap, uni.T).T