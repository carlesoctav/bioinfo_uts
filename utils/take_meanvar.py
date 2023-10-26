from scipy.io import loadmat
import numpy as np 
from pyprojroot import here

file_path = here("data/BRCA1View20000.mat")
data = loadmat(file_path)


# read and process data
x_input = data["data"].T
y_input = data["targets"]

x_mean = x_input.mean(axis=0, keepdims=True)
x_std = x_input.std(axis=0, keepdims=True)

np.save("data/x_mean.npy", x_mean)
print("wah")
np.save("data/x_std.npy", x_std)
print("wah")