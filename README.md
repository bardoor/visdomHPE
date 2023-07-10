# visdomHPE
human pose estimation and labeling by V.I.S.D.O.M.

Made by Dvoryankin inspirated club (DIC)

# Как скачать датасет?
Очень просто.
## Windows:
1. Переходим в папку downloader
2. Запускаем get_dataset.bat
3. Ждём окончания загрузки
## Linux, macOS: 
1. Просто скачайте папку с датасетами отсюда https://cloud.mail.ru/public/hC9L/u9tGmGkig и киньте в корень.

# Как подготовить данные для тренировки модели?
Для генерации csv файла, необходимого для тренировки модели, нужно запустить следующую команду:
python hpe.py --generate <путь к директории, содержащей папки с видео>

Например:
python hpe.py --generate dataset/

В результате будет сгенерирован csv файл, необходимый для тренировки модели

# Как тренировать модель?
Для тренировки модели при помощи созданного csv файла, необходимо запустить следующую команду:
python hpe.py --train <путь к csv файлу>

Например:
python hpe.py --train generated.csv

В результате будет сгенерирован файл "best_model.h5", содержащий лучшие веса для сети
