import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split as tts
from keras import backend as K

# Глобальные переменные
rawdata = []
labels = [] 
data = [] 

Num = 100000    # Размер словаря для токенизации
Dim = 200       # Размерность векторного пространства признаков
Length = 26     # Максимальное количество слов в тексте для токенизации
Model = None    # Модель сверточной нейронной сети
tkz = None      # Токенизатор

# Чтение помеченных данных + создание массива меток
def Read(pathpos, pathneg):
    global rawdata, labels
    
    cols = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw',
            'stcount', 'foll', 'frien', 'listcount']
    dpos = pd.read_csv(pathpos, sep=';', error_bad_lines=False, names=cols, usecols=['text'])
    dneg = pd.read_csv(pathneg, sep=';', error_bad_lines=False, names=cols, usecols=['text'])

    # Создаем сбалансированный датасет
    size = min(dpos.shape[0], dneg.shape[0]);
    rawdata = np.concatenate((dpos['text'].values[:size], dneg['text'].values[:size]), axis = 0)
    labels = [1] * size + [0] * size
#--------------------------------------------------------------------------------   



def Preprocess(text):
    # Перевод в нижний регистр + замена ё на е
    text = text.lower().replace("ё", "е")

    # Замена ссылок на строку "URL"
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)

    # Замена имени пользователя на строку "USER"
    text = re.sub('@[^\s]+', 'USER', text)

    # Все кроме букв и цифр заменяем пробелами
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)

    # Несколько пробелов заменяем на один 
    text = re.sub(' +', ' ', text)

    # Удаляем пробелы в начале и в конце и возвращаем итоговую строку
    return text.strip()
#--------------------------------------------------------------------------------



# Записываем в файл неразмеченные данные из базы
def TweetsToFile(basepath):
    import sqlite3    
    base = sqlite3.connect(basepath)
    curs = base.cursor()
    with open('Tweets.txt', 'w', encoding='utf-8') as f:
        for row in curs.execute('SELECT ttext FROM sentiment'):
            if (row[0]): # Проверяем, не пустой ли твит
                t = Preprocess(row[0])
                print(t, file = f)
    print('finish');
#--------------------------------------------------------------------------------



# Записываем в файл неразмеченные данные из базы
def Embedding():
    import multiprocessing
    import gensim
    from gensim.models import Word2Vec
    
    # Считываем файл с твитами  
    Words = gensim.models.word2vec.LineSentence('Tweets.txt')
    
    model = Word2Vec(Words, size=Dim, window=5, min_count=3, workers=multiprocessing.cpu_count())

    # Сохраняем готовую модель
    model.save("model.w2v") 
#--------------------------------------------------------------------------------



def Tokenize(train, test):    
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from gensim.models import Word2Vec

    global tkz
    
    # Cоздаем токенизатор
    tkz = Tokenizer(num_words=Num)
    tkz.fit_on_texts(train)
    
    # Переводим тексты в массив идентификаторов токенов
    train_seq = pad_sequences(tkz.texts_to_sequences(train), maxlen=Length)
    test_seq = pad_sequences(tkz.texts_to_sequences(test), maxlen=Length)

    model_w2v = Word2Vec.load('model.w2v')

    # Инициализируем матрицу Embedding-слоя нулями
    emb_matrix = np.zeros((Num, Dim))
    
    # Добавляем Num наиболее часто встречающихся слов из обучающей выборки в Embedding-слой
    for word, i in tkz.word_index.items():
        if i >= Num:
            break
        if word in model_w2v.wv.vocab.keys():
            emb_matrix[i] = model_w2v.wv[word]
            
    return train_seq, test_seq, emb_matrix
#--------------------------------------------------------------------------------



# Определяем метрики
# Функция точности
def precision(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_pos / (pred_pos + K.epsilon())
    return precision

# Функция полноты
def recall(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_pos / (possible_pos + K.epsilon())
    return recall

def F(y_true, y_pred):
    def precision(y_true, y_pred):
        true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_pos / (pred_pos + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_pos / (possible_pos + K.epsilon())
        return recall
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
#--------------------------------------------------------------------------------



def BuildCNN(emb_matrix):
    from keras.layers import Input
    from keras.layers.embeddings import Embedding
    from keras import optimizers
    from keras.layers import Dense, concatenate, Activation, Dropout
    from keras.models import Model
    from keras.layers.convolutional import Conv1D
    from keras.layers.pooling import GlobalMaxPooling1D

    Input_Layer = Input(shape=(Length,), dtype='int32')

    # Определяем необучаемый Embedding-слой на основе ранее созданной матрицы весов
    Embedding_Layer = Embedding(Num, Dim, input_length=Length,
                          weights=[emb_matrix], trainable=False)(Input_Layer)
    # Добавляем dropout-регуляризацию
    x = Dropout(0.2)(Embedding_Layer)
    
    branches = []           # Слои сети
    filters = [2, 3, 4, 5]  # Высота фильтров

    for height in filters:
        for i in range(10):
            # Добавляем слой свертки. Не применяем паддинг
            branch = Conv1D(filters=1, kernel_size=height, padding='valid', activation='relu')(x)            
            # Добавляем слой субдискретизации
            branch = GlobalMaxPooling1D()(branch)
            # Добавляем в общую сеть
            branches.append(branch)

    # Конкатенируем карты признаков
    x = concatenate(branches, axis=1)
    # Добавляем dropout-регуляризацию
    x = Dropout(0.2)(x)
    # Устанавливаем полносвязный слой 
    x = Dense(30, activation='relu')(x)
    x = Dense(1)(x)
    Output = Activation('sigmoid')(x)

    model = Model(inputs=[Input_Layer], outputs=[Output])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision, recall, F])
    return model
#--------------------------------------------------------------------------------



def TrainCNN(train_seq, label):
    from keras.callbacks import ModelCheckpoint
    global Model
    checkpoint = ModelCheckpoint("cnn-{epoch:02d}-{val_F:.2f}.hdf5", monitor='val_F',
                                 save_best_only=True, mode='max', period=1)
    history = Model.fit(train_seq, label, batch_size = 32, epochs = 10, verbose = 2,
                        validation_split = 0.25, callbacks = [checkpoint])
#--------------------------------------------------------------------------------


def TestCNN(test_seq, label):
    from sklearn.metrics import classification_report  
    pred = np.round(Model.predict(test_seq))
    print(classification_report(label, pred, digits = 5))
#--------------------------------------------------------------------------------


def MyTest():
    from keras.preprocessing.sequence import pad_sequences
    global tkz, Model    
    f = open('!MyTest.txt', 'r');
    rd = f.readlines();
    d = [Preprocess(txt) for txt in rd]
    print(d);
    d_seq = pad_sequences(tkz.texts_to_sequences(d), maxlen=Length)
    print(np.round(Model.predict(d_seq)))
    print(Model.predict(d_seq))
#--------------------------------------------------------------------------------
    

#Основной код    
Read('positive.csv', 'negative.csv')

# Предобработка исходных текстов
data = [Preprocess(mess) for mess in rawdata]

TweetsToFile('TBSqlite3.db')
Embedding()

# Формирование обучающей и тестовой выборки
data_train, data_test, label_train, label_test = tts(data, labels, test_size=0.2, random_state=1)

train_seq,test_seq, emb_matrix = Tokenize(data_train, data_test)

Model = BuildCNN(emb_matrix)

TrainCNN(train_seq, label_train)

TestCNN(test_seq, label_test)

MyTest();
