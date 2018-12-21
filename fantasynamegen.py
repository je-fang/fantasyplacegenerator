#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import sys
import io

fplaces = pd.read_csv('allnames.csv', usecols = ['fnames'], encoding = 'latin-1')

fplaces.head(20)


# In[2]:



fnum = len(fplaces)
fchars = len(' '.join(fplaces['fnames']))

print('Number of fantasy place examples: ', fnum)
print('Total characters: ', fchars)


# In[3]:


fplaces = ' '.join(fplaces['fnames']).lower()

fplaces[:100]


# In[4]:


chars = sorted(list(set(fplaces)))
print('Count of unique characters/features: ', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[5]:


maxlen = 15
step = 3
sentences = []
next_chars = []
for i in range(0, len(fplaces) - maxlen, step):
    sentences.append(fplaces[i: i + maxlen])
    next_chars.append(fplaces[i + maxlen])
print('Number of sentences:', len(sentences), "\n")

print('First 20 sentences: ', sentences[:20], "\n")
print(next_chars[:10])


# In[6]:


x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[7]:


model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[8]:


def sample(preds, temperature=0.5):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    #if epoch+1 == 1 or epoch+1 == 15:
    
    print()
    print('----- Generating text after Epoch: %d' % epoch)


    start_index = random.randint(0, len(fplaces) - maxlen - 1)
    for diversity in [0.5]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = fplaces[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        if epoch < 15:
            rangenum = 250
        else:
            rangenum = 500

        for i in range(rangenum):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

        print()

generate_text = LambdaCallback(on_epoch_end=on_epoch_end)


# In[9]:


# define the checkpoint
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

# fit model using our gpu
with tf.device('/gpu:0'):
    model.fit(x, y,
              batch_size=256,
              epochs=16,
              verbose=2,
              callbacks=[generate_text, checkpoint])


# In[ ]:




