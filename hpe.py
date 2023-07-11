import argparse

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import dataset
import model


MAX_EPOCHS_COUNT = 200

parser = argparse.ArgumentParser(description="Для чего-то...")
parser.add_argument("-g", "--generate", type=str, default=None,
                    help="Путь к директории, содержащей папки с видео")
parser.add_argument("-t", "--train", type=str, default=None,
                    help="Путь к csv файлу, содержащему датасет")
args = parser.parse_args()

if args.generate is not None:
    dataset.to_csv(args.generate, "generated")
elif args.train is not None:
    Xs, ys, a, b = dataset.create_dataset(args.train)
    m = model.pose_estimation_model()

    callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history = m.fit(
        Xs, ys,
        epochs=MAX_EPOCHS_COUNT,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        callbacks=callbacks
    )
    print(history.history.keys())

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()