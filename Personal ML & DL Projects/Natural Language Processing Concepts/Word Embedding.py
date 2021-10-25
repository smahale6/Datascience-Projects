import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

### One Hot Representation

voc_size=10000
onehot_repr=[one_hot(words,voc_size)for words in sent] 
print(onehot_repr)

### Padding of sentences.
sent_length=8
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

### Word Embedding Represntation
dim = 10

model = Sequential()
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')
model.summary()


print(model.predict(embedded_docs))
word_embed = model.predict(embedded_docs)

