from scipy.io import loadmat
import numpy as np


file_path = "data/BRCA1View20000.mat"
BRCA = loadmat(file_path)
print(BRCA["data"].shape)
print(f"==>> BRCA: {BRCA}")

print (np.unique(BRCA["targets"]))


