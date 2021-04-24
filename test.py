from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, BatchNormalization
from tensorflow.keras.layers import Dense, Input, Dropout
from keras.datasets import imdb
from keras.preprocessing import sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

seqnc_lngth =  128
embddng_dim = 64
vocab_size = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size, skip_top=20)
x_train = sequence.pad_sequences(x_train, maxlen=seqnc_lngth)
x_test = sequence.pad_sequences(x_test, maxlen=seqnc_lngth)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# now with batch norm
inpt_vec = Input(shape=(seqnc_lngth,))
l1 = Embedding(vocab_size, embddng_dim, input_length=seqnc_lngth)(inpt_vec)
l2 = Dropout(0.3)(l1)
l3 = LSTM(32)(l2)
l4 = BatchNormalization()(l3)
l5 = Dropout(0.2)(l4)
output = Dense(1, activation='sigmoid')(l5)

# model that takes input and encodes it into the latent space
lstm = Model(inpt_vec, output)

lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                              min_delta=1e-4, mode='min', verbose=1)

stop_alg = EarlyStopping(monitor='val_loss', patience=7,
                         restore_best_weights=True, verbose=1)

hist = lstm.fit(x_train, y_train, batch_size=100, epochs=1000,
                   callbacks=[stop_alg, reduce_lr], shuffle=True,
                   validation_data=(x_test, y_test))

lstm.save_weights("lstm.hdf5")

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,6))
plt.plot(hist.history['loss'], color='#785ef0')
plt.plot(hist.history['val_loss'], color='#dc267f')
plt.title('Model Loss Progress')
plt.ylabel('Binary Cross-Entropy Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='upper right')
plt.savefig('ch.13.lstm.imdb.loss.png', dpi=350, bbox_inches='tight')
plt.show()