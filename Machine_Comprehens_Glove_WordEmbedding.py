import numpy as np
import h5py

from keras.models import Sequential, Model
from keras.layers import Embedding, Dropout, Dense, Activation
from keras.layers import LSTM, Bidirectional, Merge, Input
from keras.layers import concatenate

# loading data
with h5py.File('context.h5', 'r') as hf:
    context_array = hf['context'][:]
with h5py.File('questions.h5', 'r') as hf:
    question_array = hf['questions'][:]
with h5py.File('begin.h5', 'r') as hf:
    begin_span = hf['begin'][:]
with h5py.File('end.h5', 'r') as hf:
    end_span = hf['end'][:]

# loading Glove embeddings
with h5py.File('embeddings_50.h5', 'r') as hf:
    embedding_matrix = hf['embed'][:]

# loding vocabulary
word_index = np.load('word_to_indx.npy').item()

print context_array.shape
print question_array.shape
print begin_span.shape
print end_span.shape

vocab_size = len(word_index) + 1
embedding_vector_length = 50
max_span_begin = np.amax(begin_span)
max_span_end = np.amax(end_span)
batch = 64
# slice of data to be used as one epoch training on full data is expensive
slce = 1000

# model1
context_input = Input(shape=(700, ), dtype='int32', name='context_input')
x = Embedding(input_dim=vocab_size, output_dim=50, weights=[embedding_matrix],
              input_length=700, trainable=False)(context_input)
lstm_out = Bidirectional(LSTM(256, return_sequences=True, implementation=2), merge_mode='concat')(x)
drop_1 = Dropout(0.5)(lstm_out)

# model2
ques_input = Input(shape=(50, ), dtype='int32', name='ques_input')
x = Embedding(input_dim=vocab_size, output_dim=50, weights=[embedding_matrix],
              input_length=50, trainable=False)(ques_input)
lstm_out = Bidirectional(LSTM(256, return_sequences=True, implementation=2), merge_mode='concat')(x)
drop_2 = Dropout(0.5)(lstm_out)

# merger model
merge_layer = concatenate([drop_1, drop_2], axis=1)
biLSTM = Bidirectional(LSTM(512, implementation=2), merge_mode='mul')(merge_layer)
drop_3 =  Dropout(0.5)(biLSTM)
softmax_1 = Dense(max_span_begin, activation='softmax')(biLSTM)
softmax_2 = Dense(max_span_end, activation='softmax')(biLSTM)

model = Model(inputs=[context_input, ques_input], outputs=[softmax_1, softmax_2])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model_history = model.fit([context_array[:slce], question_array[:slce]], [begin_span[:slce], end_span[:slce]], verbose=2, batch_size=batch, epochs=10)
