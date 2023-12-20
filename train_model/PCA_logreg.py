from pyprojroot import here
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from keras import metrics
import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(7)
tf.random.set_seed(7)


PCA_DIM = 1000

def logreg():
    input = keras.Input((PCA_DIM,))
    x = keras.layers.Dense(6, activation="linear", kernel_regularizer=keras.regularizers.l2(0.01))(input)

    model = keras.Model(inputs=input, outputs=x)
    return model 


file_path = here("data/BRCA1View20000.mat")

data = loadmat(file_path)


# read and process data
x_input = data["data"].T
y_input = data["targets"].reshape(-1,1)
y_input = y_input - 1
y_input = keras.utils.to_categorical(y_input, num_classes=6)

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    x_input, y_input, test_size=0.2, random_state=7
)


standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)


#PCA 
pca = PCA(n_components=PCA_DIM)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print("cumulative explained variance ratio: ", np.sum(pca.explained_variance_ratio_))


# model arguments
model = logreg()
losses = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()


# # metrics
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1Score()
# CohenKappa = tfa.metrics.CohenKappa(num_classes=6, sparse_labels=True)
# HammingLoss = tfa.metrics.HammingLoss(mode="multilabel", threshold=0.5)

# # callbacks
csv_logger = tf.keras.callbacks.CSVLogger(here("logs/training_pca_logreg.csv"))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    patience=20,
    mode="max",
    restore_best_weights=True,
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=here("model_weight/best_smote.keras"),
    monitor="val_acc",
    save_best_only=True,
    mode="max",
    save_weights_only=True,
)


# Train arg
epochs = 100
batch_size = 256

# train
model.compile(
    loss=losses,
    optimizer=optimizer,
    metrics=["acc", precision, recall, f1],
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[csv_logger, early_stopping, model_checkpoint],
    verbose=1,
)
