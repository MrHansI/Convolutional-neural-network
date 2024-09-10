Convolutional neural network
==============================

This is a neural network for detecting objects in images

Коды сегментации : 

0 0 0 - фон
255 0 0 - здания
178 178 178 - дороги
128 64 0 - грунтовые дороги
0 255 0 - деревья и леса
255 255 0 - лесостепи
255 128 0 - сельхоз поля
0 0 255 - вода
0 255 255 - лед на воде
196 128 0 - горы



Convolutional neural network:
Нейронная сеть предназначена для много-классовой сегментации изображений , для реализации этой задачи использовалась модель сверточной нейронной сети UNet . 

requirements: 
txt файл с актуальными версиями библиотек 

augmentation.py : 
Блок в котором происходит аугментация изображения : поворот , зум , отзеркаливание - это всё нужно для расширения датасета .

creating_dataset.py : 
Блок в котором просходит разделение на чегментационные классы , связывание маски и оригинала .

eva_val_test_train.py: 
Тут находятся функции , которые отвечают за эвалюцию , т.е за то , как модель обучается , за валидацию , тест и траин .

filters.py: 
В процессе написания нейронной сети было необходимо использовать некотые фильтры для лучшего распозниния изображений , все имеющиеся фильтры находятся в этом блоке .

image_classification.py : 
В данном блоке мы распознаем какой класс представлен на изображении . 


plot.py : 
построение графиков обучения модели . 

predictions.py : 
предсказания , которые связаны с моделью . 

UNet.py : 
Тут представленна сама нейронная сеть , со всеми своими слоями . 

main.py : 
вызов всех необходимых функций и запуск модели . 

Так же в проекте существуют папки :
test : оригиналы тестовых изображений 
testannot : маски тестовых изображений
train : оригиналы траин изображений 
trainannot : маски траин изображений
val : оригиналы валидационных изображений 
valannot : маски валидационных изображений


Project Organization
------------

    ├── test
    ├── testannot  
    ├── train
    ├── trainannot
    ├── val
    ├── valannot
    ├── augmentation.py
    ├── creating_dataset.py
    ├── eva_val_test_train.py
    ├── filters.py
    ├── image_classification.py
    ├── main.py
    ├── plot.py
    ├── predictions.py
    ├── README.md
    ├── Unet.py
    ├── requirements.txt
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

To start project : 
1. Subscribe on my github
2. Install it from github 
3. install dependens 
4. python main.py 