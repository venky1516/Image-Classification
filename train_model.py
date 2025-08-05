from data_load import load_data
from preprocessing import preprocess_data
from model import build_custom_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

x_train, y_train, x_test, y_test = load_data()
x_train, x_test = preprocess_data(x_train, x_test)

model = build_custom_model()

early_stop = EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=25,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)
model.summary()

model.save("cnn_model.h5")
