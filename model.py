from tensorflow import keras
from keras import layers


def pose_estimation_model(output_classes=7):
    model = keras.Sequential([
        layers.LSTM(units=128, input_shape=(None, 34)),
        layers.Dropout(0.4),
        layers.Dense(units=64),
        layers.Dropout(0.4),
        layers.Dense(units=output_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    return model
