"""All CNN arquitectures"""
from math import ceil, sqrt
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    precision_score, accuracy_score, recall_score,
    f1_score, matthews_corrcoef, mean_squared_error,
    mean_absolute_error, r2_score, roc_auc_score, confusion_matrix
)
from scipy.stats import (kendalltau, pearsonr, spearmanr)
from keras.utils.layer_utils import count_params

class CnnA(tf.keras.models.Sequential):

    def __init__(self, x_train):
        super().__init__()
        
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2,
            activation="relu", input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))

        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))

        self.add(tf.keras.layers.Flatten())
        
        self.add(tf.keras.layers.Dense(units=64,
            activation="tanh"))
        
        self.add(tf.keras.layers.Dense(units=1,
            activation="linear"))
        self.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.MeanSquaredError())

class CnnB(tf.keras.models.Sequential):

    def __init__(self, x_train):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))


        self.add(tf.keras.layers.Dense(units=1,
            activation="linear"))
        self.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.MeanSquaredError())

class CnnC(tf.keras.models.Sequential):
    def __init__(self, x_train):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
    
        self.add(tf.keras.layers.Flatten())

        self.add(tf.keras.layers.Dense(units=1,
            activation="linear"))
        self.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.MeanSquaredError())

class CnnD(tf.keras.models.Sequential):

    def __init__(self, x_train):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Flatten())

        self.add(tf.keras.layers.Dense(units=1,
            activation="linear"))
        self.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.MeanSquaredError())


class Models:
    """Organize CNN objects, train and validation process"""
    def __init__(self, x_train, y_train, x_test, y_test, arquitecture):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.arquitecture = arquitecture

        if self.arquitecture in ("E", "F", "G", "H"):
            self.x_train, self.x_test = self.__reshape()

        if self.arquitecture == "A":
            self.cnn = CnnA(x_train=self.x_train)
        elif self.arquitecture == "B":
            self.cnn = CnnB(x_train=self.x_train)
        elif self.arquitecture == "C":
            self.cnn = CnnC(x_train=self.x_train)
        elif self.arquitecture == "D":
            self.cnn = CnnD(x_train=self.x_train)
        else:
            print("Wrong arquitecture for this dataset")
            exit()

    def __reshape(self):
        dim = self.x_train.shape[1]
        sq_dim = sqrt(dim)
        square_side = ceil(sq_dim)
        resized_x_train = np.resize(self.x_train, (self.x_train.shape[0], square_side*square_side))
        resized_x_test = np.resize(self.x_test, (self.x_test.shape[0], square_side*square_side))
        squared_x_train = np.reshape(resized_x_train, (-1, square_side, square_side))
        squared_x_test = np.reshape(resized_x_test, (-1, square_side, square_side))
        return squared_x_train, squared_x_test

    def fit_models(self, epochs, verbose):
        """Fit model"""
        self.cnn.fit(self.x_train, self.y_train, epochs = epochs, verbose = verbose)

    def save_model(self, folder, prefix = ""):
        """
        Save model in .h5 format, in 'folder' location
        """
        self.cnn.save(f"{folder}/{prefix}-{self.arquitecture}-{self.mode}.h5")

    def get_metrics(self):
        """
        Returns classification performance metrics.

        Accuracy, recall, precision, f1_score, mcc.
        """
        trainable_count = count_params(self.cnn.trainable_weights)
        non_trainable_count = count_params(self.cnn.non_trainable_weights)
        result = {}
        result["arquitecture"] = self.arquitecture
        result["trainable_params"] = trainable_count
        result["non_trainable_params"] = non_trainable_count

        y_train_predicted = self.cnn.predict(self.x_train)
        y_test_predicted = self.cnn.predict(self.x_test)

        try:
            train_metrics = {
                "mse": mean_squared_error(y_true = self.y_train, y_pred = y_train_predicted),
                "mae": mean_absolute_error(y_true = self.y_train, y_pred = y_train_predicted),
                "r2_score": r2_score(y_true = self.y_train, y_pred = y_train_predicted),
                "kendalltau": kendalltau(self.y_train, y_train_predicted)[0],
                "pearsonr": pearsonr(self.y_train, y_train_predicted)[0][0],
                "spearmanr": spearmanr(self.y_train, y_train_predicted)[0]
            }
        except:
            train_metrics = {
                "mse": None,
                "mae": None,
                "r2_score": None,
                "kendalltau": None,
                "pearsonr": None,
                "spearmanr": None
            }

        try:
            test_metrics = {
                "mse": mean_squared_error(y_true = self.y_test, y_pred = y_test_predicted),
                "mae": mean_absolute_error(y_true = self.y_test, y_pred = y_test_predicted),
                "r2": r2_score(y_true = self.y_test, y_pred = y_test_predicted),
                "kendalltau": kendalltau(self.y_test, y_test_predicted)[0],
                "pearsonr": pearsonr(self.y_test, y_test_predicted)[0][0],
                "spearmanr": spearmanr(self.y_test, y_test_predicted)[0]
            }
        except:
            test_metrics = {
                "mse": None,
                "mae": None,
                "r2_score": None,
                "kendalltau": None,
                "pearsonr": None,
                "spearmanr": None
            }

        result["train_metrics"] = train_metrics
        result["test_metrics"] = test_metrics


        return result