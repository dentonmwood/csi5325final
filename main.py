import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers import LSTM, BatchNormalization, Dropout, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# The data file directory
_DATA_DIR = "data"
# The working temporary file directory
_TEMP_DIR = "tmp"
# The output file directory
_OUTPUT_DIR = "output"
# The number of splits to use for cross-validation
_NUM_CV_SPLITS = 5


def main():
    """main function for the program

    Runs the main logic of the program.

    Returns: NoneType
    """
    # Check for data directory and throw exception if not there
    if not os.path.exists(_DATA_DIR):
        raise Exception("No data files found")

    # Create working directories if they don't exit
    os.makedirs(_TEMP_DIR, exist_ok=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # This is the file we'll load
    current_file = "5a0546857ecc773753327266_1000"
    tempfile_x = f'{_TEMP_DIR}/{current_file}_x_tmp.npy'
    tempfile_x_enc = f'{_TEMP_DIR}/{current_file}_x_mlp_tmp.npy'
    tempfile_y = f'{_TEMP_DIR}/{current_file}_y_tmp.npy'

    if not os.path.exists(tempfile_x):
        # Read training data
        print('Reading data...')
        train = pd.read_csv(f'{_DATA_DIR}/train/{current_file}_train.csv')
        # test = pd.read_csv(f'{_DATA_DIR}}/test/5a0546857ecc773753327266_1000_test.csv')

        # Split data into X & y
        print('Processing data...')
        y_columns = ['x', 'y', 'f']
        x_columns = [x for x in train.columns if x not in y_columns
                     and x is not train.columns[0]
                     and x != 'path']
        x = train.loc[:, x_columns]
        y = train.loc[:, y_columns]

        np.save(tempfile_x_enc, x)

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
        # We've already processed the data once - just load the saved data
        print('Processed data exists, reading from temp directory...')

    x = np.load(tempfile_x)
    x_enc = np.load(tempfile_x_enc)
    y = np.load(tempfile_y)

    # Build the model
    print('Building models...')
    lstm = build_lstm_model(len(x[0]), len(x[0][0]))
    lstm.summary()
    encoder = build_autoencoder_model(len(x_enc[0]))
    encoder.summary()

    # # Autoencoder
    # print('Beginning Encoder')
    # cross_validate_model(encoder, 'encoder', x_enc, y, enc=True)

    # LSTM
    print('Beginning LSTM')
    cross_validate_model(lstm, 'lstm', x, y)


def cross_validate_model(model, filename, x, y, enc=False):
    """Runs cross-validation on the given model

    Splits the data into NUM_CV_SPLITS number of pieces and
    runs the given model on each test/train split. Graphs
    the results.

    Args:
        model: the model on which to run CV
        filename: the prefix of the filename to use when saving the graphs
        x: the data matrix
        y: the target vector
        enc: whether the model is an autoencoder

    Returns: None
    """
    # Cross-validation (taken from Rivas CSI 5325 H5)
    print('Beginning cross validation...')
    kf = KFold(n_splits=_NUM_CV_SPLITS)
    i = 0
    loss = np.ndarray(shape=(_NUM_CV_SPLITS, ))
    val_loss = np.ndarray(shape=(_NUM_CV_SPLITS, ))
    accuracy = np.ndarray(shape=(_NUM_CV_SPLITS, ))
    for train_split, val_split in kf.split(x):
        print(f'Cross Validation Run # {i + 1}')
        x_train, x_val, y_train, y_val = x[train_split], x[val_split], y[train_split], y[val_split]
        results = train_model(model, x_train, x_val, y_train, y_val, enc=enc)
        loss[i] = np.average(results.history['loss'])
        print(f'Loss: {loss[i]}')
        val_loss[i] = np.average(results.history['val_loss'])
        print(f'Validation Loss: {val_loss[i]}')
        accuracy[i] = np.average(results.history['accuracy'])
        print(f'Accuracy: {accuracy[i]}')
        graph_results(results, f'{_OUTPUT_DIR}/{filename}-{i}.pdf')
        i += 1
    calculate_results(loss, val_loss, accuracy, f'{_OUTPUT_DIR}/{filename}-results.txt')


def train_model(model, x_train, x_val, y_train, y_val, enc=False):
    """Trains the given model model on the given data

    Trains the model on the given data for a number of
    epochs. Returns the results for further analysis.

    Args:
        model (Model): the compiled model
        x_train (ndarray): the data matrix of the training data
        x_val (ndarray): the data matrix of the validation data
        y_train (ndarray): the target vector of the training data
        y_val (ndarray): the target vector of the validation data
        enc (bool): whether or not an autoencoder is being trained

    Returns:
        The result data
    """
    # Taken from DLFB Chapter 13: RNNs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=4,
                                  min_delta=1e-4, mode='min', verbose=1)
    stop_alg = EarlyStopping(monitor='val_loss', patience=8,
                             restore_best_weights=True, verbose=1)

    return model.fit(x_train, x_train if enc else y_train, batch_size=1, epochs=1000 if enc else 1,
                     shuffle=True, callbacks=[stop_alg, reduce_lr],
                     validation_data=(x_val, x_val if enc else y_val))


def build_lstm_model(sequence_length, num_unique_values):
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
    lstm.add(LSTM(20))
    lstm.add(BatchNormalization())
    lstm.add(Dropout(0.1))
    lstm.add(Dense(3, activation='sigmoid'))

    lstm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return lstm


def build_autoencoder_model(data_size):
    """Constructs an autoencoder model

    Uses the Keras API to construct a semi-deep autoencoder.
    Compiles the model and returns the resulting model for
    training.

    Args:
        data_size (int): the size of a single row of data

    Returns:
        The compiled model
    """
    encoder = Sequential()
    encoder.add(Input(shape=(data_size, )))
    encoder.add(Dense(np.round(data_size / 2)))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.1))
    encoder.add(Dense(32))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.1))
    encoder.add(Dense(3, activation='sigmoid'))

    encoder.add(Dense(32))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.1))
    encoder.add(Dense(np.round(data_size / 2)))
    encoder.add(Dropout(0.1))
    encoder.add(Dense(data_size, activation='sigmoid'))

    encoder.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return encoder


def graph_results(hist, filename):
    """Graphs the results of training

    Constructs a graph using the MatPlotLib API of the
    given training results. Saves the graph using the
    given filename.

    Args:
        hist: the results of training
        filename (str): the filename prefix to use when saving the file

    Returns:
        NoneType
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(hist.history['loss'], color='#785ef0')
    plt.plot(hist.history['val_loss'], color='#dc267f')
    plt.title('Model Loss Progress')
    plt.ylabel('Mean Squared Error Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.savefig(filename, dpi=350, bbox_inches='tight')
    plt.show()


def calculate_results(loss, val_loss, accuracy, filename):
    """Calculates the results of the run

    Calculates the results of the cross-validation run. Writes
    the results to an output file for later use.

    Args:
        loss (ndarray): the array of loss values
        val_loss (ndarray): the array of validation loss values
        accuracy (ndarray): the array of accuracy values
        filename (str): the name to use to save the file

    Returns:
        NoneType
    """
    with open(filename, 'w') as outfile:
        print('Loss:', loss)
        print(f'Loss Bias: {np.mean(loss)}')
        print(f'Loss Variance: {np.std(loss)}')
        print('Validation Loss:', val_loss)
        print(f'Validation Loss Bias: {np.mean(val_loss)}')
        print(f'Validation Loss Variance: {np.std(val_loss)}')
        print('Accuracy:', accuracy)
        print(f'Accuracy Bias: {np.mean(accuracy)}')
        print(f'Accuracy Variance: {np.std(accuracy)}')

        outfile.write('Loss: ' + loss)
        outfile.write(f'Loss Bias: {np.mean(loss)}')
        outfile.write(f'Loss Variance: {np.std(loss)}')
        outfile.write('Validation Loss: ' + val_loss)
        outfile.write(f'Validation Loss Bias: {np.mean(val_loss)}')
        outfile.write(f'Validation Loss Variance: {np.std(val_loss)}')
        outfile.write('Accuracy: ' + accuracy)
        outfile.write(f'Accuracy Bias: {np.mean(accuracy)}')
        outfile.write(f'Accuracy Variance: {np.std(accuracy)}')


if __name__ == '__main__':
    main()
