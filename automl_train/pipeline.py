from tensorflow.train import cosine_decay, AdamOptimizer
from tensorflow.contrib.opt import AdamWOptimizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, CuDNNLSTM, GRU, CuDNNGRU, concatenate, Dense, BatchNormalization, Dropout, AlphaDropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
import csv
import sys
import warnings
from datetime import datetime
from math import floor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


def build_model(encoders):
    """Builds and compiles the model from scratch.

    # Arguments
        encoders: dict of encoders (used to set size of text/categorical inputs)

    # Returns
        model: A compiled model which can be used to train or predict.
    """

    # Unnamed: 0
    input_unnamed_0 = Input(shape=(10,), name="input_unnamed_0")

    # Month
    input_month_size = len(encoders['month_encoder'].classes_)
    input_month = Input(
        shape=(input_month_size if input_month_size != 2 else 1,), name="input_month")

    # Year
    input_year_size = len(encoders['year_encoder'].classes_)
    input_year = Input(
        shape=(input_year_size if input_year_size != 2 else 1,), name="input_year")

    # sher
    input_sher = Input(shape=(10,), name="input_sher")

    # Combine all the inputs into a single layer
    concat = concatenate([
        input_unnamed_0,
        input_month,
        input_year,
        input_sher
    ], name="concat")

    # Multilayer Perceptron (MLP) to find interactions between all inputs
    hidden = Dense(64, activation="relu", name="hidden_1",
                   kernel_regularizer=None)(concat)
    hidden = BatchNormalization(name="bn_1")(hidden)
    hidden = Dropout(0.5, name="dropout_1")(hidden)

    for i in range(4-1):
        hidden = Dense(256, activation="relu", name="hidden_{}".format(
            i+2), kernel_regularizer=None)(hidden)
        hidden = BatchNormalization(name="bn_{}".format(i+2))(hidden)
        hidden = Dropout(0.5, name="dropout_{}".format(i+2))(hidden)

    output = Dense(encoders['sales_encoder'].classes_.shape[0],
                   activation="softmax", name="output", kernel_regularizer=l2(1e-2))(hidden)

    # Build and compile the model.
    model = Model(inputs=[
        input_unnamed_0,
        input_month,
        input_year,
        input_sher
    ],
        outputs=[output])
    model.compile(loss="categorical_crossentropy",
                  optimizer=AdamWOptimizer(learning_rate=0.0001,
                                           weight_decay=0.025))

    return model


def build_encoders(df):
    """Builds encoders for fields to be used when
    processing data for the model.

    All encoder specifications are stored in locally
    in /encoders as .json files.

    # Arguments
        df: A pandas DataFrame containing the data.
    """

    # Unnamed: 0
    unnamed_0_enc = df['Unnamed: 0']
    unnamed_0_bins = unnamed_0_enc.quantile(np.linspace(0, 1, 10+1))

    with open(os.path.join('encoders', 'unnamed_0_bins.json'),
              'w', encoding='utf8') as outfile:
        json.dump(unnamed_0_bins.tolist(), outfile, ensure_ascii=False)

    # Month
    month_counts = df['Month'].value_counts()
    month_perc = max(floor(0.1 * month_counts.size), 1)
    month_top = np.array(month_counts.index[0:month_perc], dtype=object)
    month_encoder = LabelBinarizer()
    month_encoder.fit(month_top)

    with open(os.path.join('encoders', 'month_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(month_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Year
    year_counts = df['Year'].value_counts()
    year_perc = max(floor(0.1 * year_counts.size), 1)
    year_top = np.array(year_counts.index[0:year_perc], dtype=object)
    year_encoder = LabelBinarizer()
    year_encoder.fit(year_top)

    with open(os.path.join('encoders', 'year_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(year_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # sher
    sher_enc = df['sher']
    sher_bins = sher_enc.quantile(np.linspace(0, 1, 10+1))

    with open(os.path.join('encoders', 'sher_bins.json'),
              'w', encoding='utf8') as outfile:
        json.dump(sher_bins.tolist(), outfile, ensure_ascii=False)

    # Target Field: Sales
    sales_encoder = LabelBinarizer()
    sales_encoder.fit(df['Sales'].values)

    with open(os.path.join('encoders', 'sales_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(sales_encoder.classes_.tolist(), outfile, ensure_ascii=False)


def load_encoders():
    """Loads the encoders built during `build_encoders`.

    # Returns
        encoders: A dict of encoder objects/specs.
    """

    encoders = {}

    # Unnamed: 0
    unnamed_0_encoder = LabelBinarizer()
    unnamed_0_encoder.classes_ = list(range(10))

    with open(os.path.join('encoders', 'unnamed_0_bins.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        unnamed_0_bins = json.load(infile)
    encoders['unnamed_0_bins'] = unnamed_0_bins
    encoders['unnamed_0_encoder'] = unnamed_0_encoder

    # Month
    month_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'month_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        month_encoder.classes_ = json.load(infile)
    encoders['month_encoder'] = month_encoder

    # Year
    year_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'year_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        year_encoder.classes_ = json.load(infile)
    encoders['year_encoder'] = year_encoder

    # sher
    sher_encoder = LabelBinarizer()
    sher_encoder.classes_ = list(range(10))

    with open(os.path.join('encoders', 'sher_bins.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        sher_bins = json.load(infile)
    encoders['sher_bins'] = sher_bins
    encoders['sher_encoder'] = sher_encoder

    # Target Field: Sales
    sales_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'sales_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        sales_encoder.classes_ = np.array(json.load(infile))
    encoders['sales_encoder'] = sales_encoder

    return encoders


def process_data(df, encoders, process_target=True):
    """Processes an input DataFrame into a format
    sutable for model prediction.

    This function loads the encoder specifications created in
    `build_encoders`.

    # Arguments
        df: a DataFrame containing the source data
        encoders: a dict of encoders to process the data.
        process_target: boolean to determine if the target should be encoded.

    # Returns
        A tuple: A list containing all the processed fields to be fed
        into the model, and the processed target field.
    """

    # Unnamed: 0
    unnamed_0_enc = pd.cut(
        df['Unnamed: 0'].values, encoders['unnamed_0_bins'], labels=False, include_lowest=True)
    unnamed_0_enc = encoders['unnamed_0_encoder'].transform(unnamed_0_enc)

    # Month
    month_enc = df['Month'].values
    month_enc = encoders['month_encoder'].transform(month_enc)

    # Year
    year_enc = df['Year'].values
    year_enc = encoders['year_encoder'].transform(year_enc)

    # sher
    sher_enc = pd.cut(df['sher'].values, encoders['sher_bins'],
                      labels=False, include_lowest=True)
    sher_enc = encoders['sher_encoder'].transform(sher_enc)

    data_enc = [unnamed_0_enc,
                month_enc,
                year_enc,
                sher_enc
                ]

    if process_target:
        # Target Field: Sales
        sales_enc = df['Sales'].values

        sales_enc = encoders['sales_encoder'].transform(sales_enc)

        return (data_enc, sales_enc)

    return data_enc


def model_predict(df, model, encoders):
    """Generates predictions for a trained model.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.
        encoders: a dict of encoders to process the data.

    # Returns
        A numpy array of predictions.
    """

    data_enc = process_data(df, encoders, process_target=False)

    headers = encoders['sales_encoder'].classes_
    predictions = pd.DataFrame(model.predict(data_enc), columns=headers)

    return predictions


def model_train(df, encoders, args, model=None):
    """Trains a model, and saves the data locally.

    # Arguments
        df: A pandas DataFrame containing the source data.
        encoders: a dict of encoders to process the data.
        args: a dict of arguments passed through the command line
        model: A compiled model (for TensorFlow, None otherwise).
    """
    X, y = process_data(df, encoders)

    split = StratifiedShuffleSplit(
        n_splits=1, train_size=args.split, test_size=None, random_state=123)

    for train_indices, val_indices in split.split(np.zeros(y.shape[0]), y):
        X_train = [field[train_indices, ] for field in X]
        X_val = [field[val_indices, ] for field in X]
        y_train = y[train_indices, ]
        y_val = y[val_indices, ]

    meta = meta_callback(args, X_val, y_val)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs,
              callbacks=[meta],
              batch_size=256)


class meta_callback(Callback):
    """Keras Callback used during model training to save current weights
    and metrics after each training epoch.

    Metrics metadata is saved in the /metadata folder.
    """

    def __init__(self, args, X_val, y_val):
        self.f = open(os.path.join('metadata', 'results.csv'), 'w')
        self.w = csv.writer(self.f)
        self.w.writerow(['epoch', 'time_completed'] +
                        ['log_loss', 'accuracy', 'precision', 'recall', 'f1'])
        self.in_automl = args.context == 'automl-gs'
        self.X_val = X_val
        self.y_val = y_val

    def on_train_end(self, logs={}):
        self.f.close()
        self.model.save_weights('model_weights.hdf5')

    def on_epoch_end(self, epoch, logs={}):
        y_true = self.y_val
        y_pred = self.model.predict(self.X_val)

        y_pred_label = np.zeros(y_pred.shape)
        y_pred_label[np.arange(y_pred.shape[0]), y_pred.argmax(axis=1)] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logloss = log_loss(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred_label)
            precision = precision_score(y_true, y_pred_label, average='micro')
            recall = recall_score(y_true, y_pred_label, average='micro')
            f1 = f1_score(y_true, y_pred_label, average='micro')

        metrics = [logloss,
                   acc,
                   precision,
                   recall,
                   f1]
        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        self.w.writerow([epoch+1, time_completed] + metrics)

        # Only run while using automl-gs, which tells it an epoch is finished
        # and data is recorded.
        if self.in_automl:
            sys.stdout.flush()
            print("\nEPOCH_END")
