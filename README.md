# С чего начать?
С некоторых разъяснений, разумеется.

### Принцип работы
Всего есть три ступени:
- Подготовка входных данных:
  - Сначала каждый кадр видео преобразуется в массив ключевых точек человеческого тела: локоть, плечо, глаза, колено, ступни, и.т.д. с помощью библиотеки [YOLOv8-pose](https://docs.ultralytics.com/tasks/pose/).
  - Затем каждому кадру сопоставляется физическое упражнение, совершаемое в видео, из которого мы достали этот кадр - так мы и получаем csv файл для обучения нашей нейронной сети.
- Обучение модели
  - С помощью библиотеки [keras](https://github.com/keras-team/keras) компилируем и обучаем нейронную сеть, подавая ей кадры, полученные на предыдущем шаге.
  - Получаем файл (.h5) с весовой моделью нейросети
- Предсказание
  - На основе полученного файла с весами предсказываем физическое упражнение в произвольном видео


### Архитектурные тонкости
Датасет - совокупность папок с именами, совпадающими с названиями физических упражнений, совершаемых во всех видео внутри этой папки.

Например:

- dataset
  - jumps_right
      - video_1.mp4    
      - video_2.mp4  
      - video_3.mov
  - tilts_body
      - video_4.mp4
      - video_5.mp4

и так далее...

Структура нейросети:
|   Layer   |   Output Shape   |   Param #  |
| --------- | ---------------- | ---------- |
| LSTM      | (None, None, 16) | 3264       |
| LSTM      | (None, 8)        | 800        |
| Dence     | (None, 16)       | 144        |
| Dropout   | (None, 16)       | 0          |
| Dence     | (None, 8)        | 136        |
| Dropout   | (None, 8)        | 0          |
| Dence     | (None, 7)        | 63         |


# Как скачать датасет?
Очень просто:shipit:
Windows:
1. Переходим в папку downloader
2. Запускаем get_dataset.bat
3. Ждём окончания загрузки
Linux, macOS: 
1. Просто скачайте папку с датасетами отсюда https://cloud.mail.ru/public/hC9L/u9tGmGkig и киньте в корень.

# Как подготовить данные для обучения модели?
Для генерации csv файла, необходимого для тренировки модели, нужно запустить следующую команду:
```
python hpe.py --generate <путь к директории, содержащей папки с видео>
```
Например:
```
python hpe.py --generate dataset/
```

В результате будет сгенерирован csv файл, необходимый для тренировки модели

# Как обучить модель?
Для тренировки модели при помощи созданного csv файла, необходимо запустить следующую команду:
```
python hpe.py --train <путь к csv файлу>
```

Например:
```
python hpe.py --train generated.csv
```
В результате будет сгенерирован файл "best_model.h5", содержащий лучшие веса для сети

# Как классифицировать видео?
**ВАЖНО!**

**Если вы классифицируете видео используя наш датасет, то крайне важно чтобы человек был полностью в кадре**

Для классификации необходимо для начала получить .h5 файл с весами, а затем исполнить следующую команду:
```
python hpe.py --weights <путь к весовой модели> --predict <путь к видео>
```
В качестве весовой модели вы можете использовать нашу pretrained_model.h5 - точность около 90%.


>:electron:	***Made by Dvoryankin inspirated club (DIC)***   :electron:	
