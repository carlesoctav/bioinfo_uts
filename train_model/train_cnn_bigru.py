from model.CNNBiGRU import CNNBiGRU
from pyprojroot import here
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from keras import metrics

file_path = here("data/BRCA1View20000.mat")
data = loadmat(file_path)


# read and process data
x_input = data["data"].T
y_input = data["targets"]

x_feat = (x_input - x_input.mean(axis=0, keepdims=True)) / x_input.std(
    axis=0, keepdims=True
)


y_input = keras.utils.to_categorical(y_input, num_classes=10)

# model arguments
model = CNNBiGRU()
losses = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)


# metrics
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1Score()

# callbacks
csv_logger = tf.keras.callbacks.CSVLogger(here("logs/training.csv"))
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    patience=10,
    mode="min",
    restore_best_weights=True,
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=here("model_weight/best.keras"),
    monitor="val_acc",
    save_best_only=True,
    mode="max",
    save_weights_only=True,
)


# Train arg
epochs = 100
batch_size = 10
val_split = 0.1

# train
model.compile(
    loss=losses,
    optimizer=optimizer,
    metrics=["acc", precision, recall, f1],
)

model.fit(
    x_feat,
    y_input,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=val_split,
    callbacks=[csv_logger, early_stopping, model_checkpoint],
)
