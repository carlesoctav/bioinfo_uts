from imblearn.over_sampling import SMOTE
from scipy.io import loadmat, savemat
from pyprojroot import here
import numpy as np

def Smote_BRCA(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res



if __name__ == "__main__":
    file_path = here("data/BRCA1View20000.mat")
    save_file_path = here("data/BRCA1View20000_smote.mat")
    data = loadmat(file_path)

    x_input = data["data"].T
    y_input = data["targets"]

    x_res, y_res = Smote_BRCA(x_input, y_input)
    print(f"==>> x_res: {x_res.shape}")

    data["data"] = x_res.T
    data["targets"] = y_res

    savemat(save_file_path, data)

