import argparse
import sys
from pprint import pprint as pp

import matplotlib.pyplot as plt

import dataset
import model


parser = argparse.ArgumentParser(description="Для чего-то...")
parser.add_argument("-g", "--generate", type=str, default=None,
                    help="Путь к директории, содержащей папки с видео")
parser.add_argument("-t", "--train", type=str, default=None,
                    help="Путь к csv файлу, содержащему датасет")
parser.add_argument("-w", "--weights", type=str, default=None,
                    help="Начальные веса модели")
parser.add_argument("-p", "--predict", type=str,
                    help="Предсказание модели по видео")
args = parser.parse_args()

if args.generate is not None:
    dataset.to_csv(args.generate, "dataset")
elif args.predict is not None:
    if args.weights is None:
        print("Не указаны веса модели")
        sys.exit(1)
    
    aem = model.ActionEstimationModel(weights=args.weights)
    pp(aem.predict(args.predict))
elif args.train is not None:
    aem = model.ActionEstimationModel(weights=args.weights, train_dataset=args.train)

    history = aem.history
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()