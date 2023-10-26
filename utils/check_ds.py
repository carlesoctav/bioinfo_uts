from scipy.io import loadmat
file_path = "data/BLCAMDAsmote.mat"
BRCA = loadmat("BRCA1View20000.mat")
print(BRCA["data"].shape)
print(BRCA["gene"].shape)
print(f"==>> BRCA: {BRCA}")



