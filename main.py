import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers import LSTM, BatchNormalization, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

_TEMP_DIR = "tmp"
_OUTPUT_DIR = "output"


def main():
    # Create working directories if they don't exit
    os.makedirs(_TEMP_DIR, exist_ok=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # This is the file we'll load
    current_file = "5a0546857ecc773753327266_1000"
    tempfile_x = f'{_TEMP_DIR}/{current_file}_x_tmp.npy'
    tempfile_y = f'{_TEMP_DIR}/{current_file}_y_tmp.npy'

    if not os.path.exists(tempfile_x):

        # Read training data
        print('Reading data...')
        train = pd.read_csv(f'data/train/{current_file}_train.csv')
        # test = pd.read_csv('data/test/5a0546857ecc773753327266_1000_test.csv')

        # Split data into X & y
        print ('Processing data...')
        y_columns = ['x', 'y', 'f']
        x_columns = [x for x in train.columns if x not in y_columns
                     and x is not train.columns[0]
                     and x != 'path']
        x = train.loc[:, x_columns]
        y = train.loc[:, y_columns]

        values = np.array([], dtype=int)
        for x_row in x.items():
            values = np.append(x_row[1].unique(), values)

        # Get list of unique values from the data
        unique_values = np.unique(values)
        num_unique_values = len(unique_values)

        # Convert list of unique values to dictionary of incremental values (ex. value1: 0, value2: 1)
        # https://stackoverflow.com/questions/31575675/how-to-convert-numpy-array-to-python-dictionary-with-sequential-keys
        # https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping
        value_mapping = {value: key for key, value in dict(enumerate(unique_values)).items()}

        # One-hot encode x
        x = x.to_numpy()
        x = [[[1.0 if value_mapping[v_i] == i else 0.0 for i in range(num_unique_values)] for v_i in v] for v in x]
        np.save(tempfile_x, x)
        np.save(tempfile_y, y)

    else:
        print('Processed data exists, reading from temp directory...')

    x = np.load(tempfile_x)
    y = np.load(tempfile_y)

    # Build the model
    print('Building model...')
    lstm = build_model(len(x[0]), len(x[0][0]))
    print(lstm.summary())

    # Cross-validation (taken from Rivas CSI 5325 H5)
    print('Beginning cross validation...')
    kf = KFold(n_splits=5)
    i = 0
    for train_split, val_split in kf.split(x):
        x_train, x_val, y_train, y_val = x[train_split], x[val_split], y[train_split], y[val_split]
        results = train_model(lstm, x_train, x_val, y_train, y_val)
        graph_results(results, f'output/lstm-{i}.pdf')
        i += 1


def train_model(lstm, x_train, x_val, y_train, y_val):
    """Trains the given LSTM model on the given data

    Trains the LSTM on the given data for a number of
    epochs. Returns the results for further analysis.

    Args:
        lstm (Model): the compiled LSTM model
        x_train (ndarray): the data matrix of the training data
        x_val (ndarray): the data matrix of the validation data
        y_train (ndarray): the target vector of the training data
        y_val (ndarray): the target vector of the validation data

    Returns:
        The result data
    """

    # Taken from DLFB Chapter 13: RNNs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=4,
                                  min_delta=1e-4, mode='min', verbose=1)
    stop_alg = EarlyStopping(monitor='val_loss', patience=8,
                             restore_best_weights=True, verbose=1)

    return lstm.fit(x_train, y_train, batch_size=1, epochs=5,
                    shuffle=True, callbacks=[stop_alg, reduce_lr],
                    validation_data=(x_val, y_val))


def build_model(sequence_length, num_unique_values):
    """Constructs the LSTM

    Constructs and compiles an LSTM model using the Keras API.
    The model consists of multiple layers.

    Args:
        sequence_length (int): the length of a given sequence
        num_unique_values (int): the size of the one-hot encoded array

    Returns:
        The compiled model
    """

    lstm = Sequential()
    lstm.add(Input(shape=(sequence_length, num_unique_values)))
    lstm.add(LSTM(256))
    lstm.add(Dropout(0.1))
    lstm.add(BatchNormalization())
    lstm.add(Dense(3, activation='sigmoid'))

    lstm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return lstm


def graph_results(hist, filename):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(hist.history['loss'], color='#785ef0')
    plt.plot(hist.history['val_loss'], color='#dc267f')
    plt.title('Model Loss Progress')
    plt.ylabel('Mean Squared Error Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Test Set'], loc='upper right')
    plt.savefig(filename, dpi=350, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
