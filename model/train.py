import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, TextVectorization, Dropout
from tensorflow.keras.models import Model

df = pd.read_csv('model/WELFake_Dataset.csv')

df.drop(['Unnamed: 0'], axis = 1, inplace = True)

df['text'] = df['text'].fillna('')
df['title'] = df['title'].fillna('')

# Combine title and text column into a single column and concatenate the string from title with text
df['news'] = (df['title']) + ' ' + (df['text'])
df['news'] = df['news'].fillna('')

y = df['label'].values
news = df['news'].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(news, y, test_size=0.1, random_state=42, stratify=y)

MAX_WORDS = 50000
BATCH_SIZE = 64
MAX_LEN = 500

vectorize = TextVectorization(max_tokens=MAX_WORDS, output_sequence_length=MAX_LEN)

text_ds = tf.data.Dataset.from_tensor_slices(xtrain).batch(BATCH_SIZE)
vectorize.adapt(text_ds)

len(vectorize.get_vocabulary())

K = 128

i = Input(shape=(1,), dtype=tf.string)
x = vectorize(i)
x = Embedding(MAX_WORDS, K)(x)
x = LSTM(128)(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((xtest, ytest)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model.fit(train_ds, epochs=5, validation_data=test_ds)

model.predict(tf.constant([["WATCH JUDGE ORDER PUNK WEARING “POLICE LIE” T-SHIRT TO LEAVE COURTROOM…Or Face Contempt Charges The hate for our law enforcement is at an all time high. The blame for the injuries suffered and murders of innocent law enforcement officers falls squarely on the shoulders of Eric Holder, Eric Sharpton and Barack Hussein Obama "]]))

model.save('fake_news_model.keras')