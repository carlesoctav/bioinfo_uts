from scipy.io import loadmat
import numpy as np


file_path = "data/BRCA1View20000_smote.mat"
BRCA = loadmat(file_path)
print(BRCA["data"].shape)
print(f"==>> BRCA: {BRCA}")

print(BRCA["targets"].shape)

print (np.unique(BRCA["targets"]))


