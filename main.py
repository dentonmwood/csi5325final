import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import LSTM, Embedding, BatchNormalization


def main():
    # Read training data
    print('Reading data...')
    train = pd.read_csv('data/train/5a0546857ecc773753327266_1000_train.csv')
    # test = pd.read_csv('data/test/5a0546857ecc773753327266_1000_test.csv')

    # Split data into X & y
    print ('Processing data...')
    y_columns = ['x', 'y', 'f']
    x_columns = [x for x in train.columns if x not in y_columns]
    x = train.loc[:, x_columns]
    y = train.loc[:, y_columns]

    # Build the model
    print('Building model...')
    lstm = build_model(len(x_columns))

    # Cross-validation (taken from Rivas CSI 5325 H5)
    print('Beginning cross validation...')
    kf = KFold(n_splits=5)
    for train_split, val_split in kf.split(x):
        x_train, x_val, y_train, y_val = x[train_split, :], x[val_split, :], y[train_split], y[val_split]
        results = train_model(lstm, x_train, x_val, y_train, y_val)
        print(results)


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


def build_model(sequence_length):
    """Constructs the LSTM

    Constructs and compiles an LSTM model using the Keras API.
    The model consists of multiple layers.

    Args:
        sequence_length (int): the length of the initial sequence

    Returns:
        The compiled model
    """
    input_vector = Input(shape=(sequence_length,))
    layer = LSTM(100)(input_vector)
    output_vector = Dense(1, activation='sigmoid')(layer)

    lstm = Model(input_vector, output_vector)
    lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lstm


if __name__ == '__main__':
    main()
