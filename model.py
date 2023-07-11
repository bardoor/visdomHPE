from tensorflow import keras
from keras import layers


def pose_estimation_model(output_classes=8):
    model = keras.Sequential([
        layers.LSTM(units=16, input_shape=(None, 34), return_sequences=True),
        layers.LSTM(units=8),
        layers.Dense(units=16),
        layers.Dropout(0.2),
        layers.Dense(units=8),
        layers.Dropout(0.2),
        layers.Dense(units=output_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    return model
