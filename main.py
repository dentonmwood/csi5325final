import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import LSTM, Embedding, BatchNormalization, Dropout


def main():
    # Read training data
    print('Reading data...')
    train = pd.read_csv('data/train/5a0546857ecc773753327266_1000_train.csv')
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
    x = x.apply(lambda v: [[1 if value_mapping[v_i] == i else 0 for i in range(num_unique_values)] for v_i in v])
    print(x)

    # # Build the model
    # print('Building model...')
    # lstm = build_model(len(x_columns), num_unique_values)
    #
    # # Cross-validation (taken from Rivas CSI 5325 H5)
    # print('Beginning cross validation...')
    # kf = KFold(n_splits=5)
    # for train_split, val_split in kf.split(x):
    #     x_train, x_val, y_train, y_val = x.iloc[train_split], x.iloc[val_split], y.iloc[train_split], y.iloc[val_split]
    #     results = train_model(lstm, x_train, x_val, y_train, y_val)
    #     print(results)


def one_hot_encode(x):
    print(x)
    return x


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

    # TODO add reduce learning rate and early stopping callbacks

    return lstm.fit(x_train, y_train, batch_size=1, epochs=1, shuffle=True,
                    validation_data=[x_val, y_val])


def build_model(sequence_length, num_unique_values):
    """Constructs the LSTM

    Constructs and compiles an LSTM model using the Keras API.
    The model consists of multiple layers.

    Args:
        sequence_length (int): the length of the initial sequence
        num_unique_values (int): the number of unique values in the sequence

    Returns:
        The compiled model
    """
    embedding_length = 50

    input_vector = Input(shape=(sequence_length,))
    layer = Embedding(num_unique_values, embedding_length, input_length=sequence_length)(input_vector)
    layer = Dropout(0.1)(layer)
    layer = LSTM(64)(layer)
    output_vector = Dense(3, activation='sigmoid')(layer)

    lstm = Model(input_vector, output_vector)
    lstm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return lstm


if __name__ == '__main__':
    main()
