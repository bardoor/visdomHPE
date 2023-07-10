import argparse

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import dataset
import model


MAX_EPOCHS_COUNT = 100

parser = argparse.ArgumentParser(description="Для чего-то...")
parser.add_argument("-g", "--generate", type=str, default=None,
                    help="Путь к директории, содержащей папки с видео")
parser.add_argument("-t", "--train", type=str, default=None,
                    help="Путь к csv файлу, содержащему датасет")
args = parser.parse_args()

if args.generate is not None:
    dataset.to_csv(args.generate, "generated")
elif args.train is not None:
    Xs, ys = dataset.create_dataset(args.train)

    callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    model.fit(
        Xs, ys,
        epochs=MAX_EPOCHS_COUNT,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        callbacks=callbacks
    )
