import string
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""
This class should return the data which can be directly given as an
input(argument) to LSTM network.
what are all the pre-processing steps that should be done to the data?
1. Each letter in each name should be converted to a tensor of size 26 (for now lets do one-hot encoding to the letters)
2. Each training dataset should be a list of tensors with shape ---> [batch size, input_size] with length = num_time steps.
3. Padding function should also be written such that all the names in that batch will be padded with zeros.(for this use
   tf.train.batch refer the wildml blog)
4. Should also create a seq_length for each batch which is used for training.
5. So, padding and sequence length functions should be in LSTM network or training class, so that they can be directly
   preprocessed befor training.
6. I have to convert the output of this class into [batchsize,time_steps , input_size]

"""

all_letters = string.ascii_lowercase
n_letters = len(all_letters)
max_len = 0
batch_size = 100
test_size = 0.2

"""
INPUT - A Dataframe with two columns - "FirstName"  and "Gender" 



"""


class preprocessing:

    def __init__(self, df, sequence_len=None, n_samples=None, val_samples=100, test_size=0.2, random_state=0,
                 ensure_preprocessed=False):
        """

        :param df:
        :param sequence_len:
        :param n_samples:
        :param val_samples:
        :param random_state:
        :param ensure_preprocessed:
        tensors --> names will be converted to tensors.
        """
        self.df = df
        self._target = df['Gender']  # to be mapped to 1 and 0
        self._seq_lengths = df['Name'].apply(len)
        self._series_names = self.__tensors(df)
        self.append_zeros_series = self.__append_zeros()
        # self.final_tensor  =self.__concatenating()
        indices = np.arange(len(self._target))
        x_train_plus_val, self._x_test, y_train_plus_val, self._y_test, train_plus_val_ind, test_ind = train_test_split(
            self.append_zeros_series,
            self._target,
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=self._target)

        self._x_train, self._x_val, self._y_train, self._y_val, train_ind, val_ind = train_test_split(
            x_train_plus_val,
            y_train_plus_val,
            train_plus_val_ind,
            random_state=random_state,
            test_size=val_samples,
            stratify=y_train_plus_val)
        self._val_indices = val_ind
        self._train_indices = train_ind
        self._test_indices = test_ind
        self._train_lengths = self._seq_lengths[train_ind]
        self._val_lengths = self._seq_lengths[val_ind]
        self._test_lengths = self._seq_lengths[test_ind]
        self._current_index = 0
        self._epoch_completed = 0

    def __letter_to_index(self, letter):

        return all_letters.find(letter)

    def __letter_to_tensor(self, letter):
        array = np.zeros(shape=[1, n_letters], dtype=float)
        array[0][self.__letter_to_index(letter)] = 1

        return array

    def __name_to_tensor(self, name):
        array = np.zeros(shape=[len(name), n_letters])
        for i, t in enumerate(name):
            array[i] = self.__letter_to_tensor(t)
        # tensor = tf.convert_to_tensor(array)
        return array

    def __tensors(self, df):
        series_array = df['Name'].apply(self.__name_to_tensor)

        return series_array

    """
    INPUT - No additional  input, directly calling on the instance(input- the series names which are created in the above function).  
    OUTPUT - Vectors(names) with equal length - done by appending zeros.  
    """

    def __append_zeros(self):
        max_len = 0
        for l in self._series_names:
            if len(l) > max_len:
                max_len = len(l)
            else:
                max_len = max_len

        nparray = np.zeros([len(self.df), max_len, 26])

        for i, row in enumerate(self._series_names):
            for j, val in enumerate(row):
                nparray[i][j] = val
        return nparray

    """
    INPUT - No additional(input instance method), but the series which is appended with zeros in the above function. 
    OUTPUT - Concatenated vector.
    """

    def __concatenating(self):
        final_array = np.zeros([len(self.df), 1, 468])
        for i in range(len(self.append_zeros_series)):
            final_array[i] = np.concatenate(self.append_zeros_series[i], axis=None)
        train_tensor = tf.convert_to_tensor(final_array)
        return train_tensor

    def next_batch(self, batch_size):
        start = self._current_index
        self._current_index += batch_size
        if self._current_index > len(self._y_train):
            self._epoch_completed += 1
            ind = np.arange(len(self._y_train))
            np.random.shuffle(ind)
            self._x_train = self._x_train[ind]
            self._y_train = self._y_train[ind]
            self._train_lengths = self._train_lengths[ind]
            start = 0
            self._current_index = batch_size
        end = self._current_index
        return self._x_train[start:end], self._y_train[start:end], self._train_lengths[start:end]

    # validation data
    def get_val_data(self, original_text=False):
        # if original_text:
        #   data = df
        #  samples = data.as_matrix(columns = ['FirstName'])[:,0]
        # return samples[self._val_indices], self._x_val, self._y_val, self._val_lengths
        return self._x_val, self._y_val, self._val_lengths

    """
    preparing the testing set
    """

    def get_test_data(self, original_text=False):
        # if original_text:
        #   data = df
        #   samples = data.as_matrix(columns = ['FirstName'])[:, 0]
        #  return samples[self._test_indices], self._x_test, self._y_test, self._test_lengths
        return self._x_test, self._y_test, self._test_lengths


data_lstm  =  preprocessing( df = uk_train, val_samples=100, test_size=0.2, random_state=0,ensure_preprocessed=False)