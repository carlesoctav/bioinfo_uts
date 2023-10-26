from scipy.io import loadmat
import numpy as np
import pandas as pd


file_path = "data/BRCA1View20000.mat"
BRCA = loadmat(file_path)


gene = BRCA["gene"].reshape(-1)
gene_name = [i[0] for i in gene]


df = pd.DataFrame(BRCA["data"].T, columns=gene_name)
df["targets"] = BRCA["targets"].reshape(-1)

print(df.head())    




