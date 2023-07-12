import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import dataset


class ActionEstimationModel:

    def __init__(self, weights=None, train_dataset=None):
        self.model = keras.Sequential([
            layers.GRU(units=64, input_shape=(None, 34), return_sequences=True),
            layers.GRU(units=32),
            layers.Dropout(0.2),
            layers.Dense(units=16),
            layers.Dropout(0.2),
            layers.Dense(units=dataset.ACTIVITY_CLASSES_NUMBER, activation='softmax')
        ])

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        if weights is not None:
            self.load_weights(weights)
        
        if train_dataset is not None:
            self.history = self.train(train_dataset)

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def train(self, dataset_file, output_weights="best_model.h5"):
        Xs, ys = dataset.create_dataset(dataset_file)

        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(filepath=output_weights, monitor='val_loss', save_best_only=True)
        ]

        history = self.model.fit(
            Xs, ys,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            shuffle=False,
            callbacks=callbacks
        )

        return history

    def predict(self, path_to_video):
        predicted_labels = []

        for keypoints_seq in dataset.generate(path_to_video):
            predicted = self.model(keypoints_seq)
            predicted_labels.append(dataset.ACTIVITY_LABELS[tf.math.argmax([predicted])])
        
        return predicted_labels