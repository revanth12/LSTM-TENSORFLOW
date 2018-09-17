from data import preprocessing
import tensorflow as tf
import numpy as np
import datetime
import time
import os

# Directories where Tensorflow summaries will be stored
summaries_dir = 'logs/'

hidden_size = 128
input_size = 26 # input size at each time step
learning_rate = 0.01

dropout_keep_pobability = 0.5
sequence_len = None # to be given
validate_every =100  # step frequency to validate


"""
prepare summaries to be used by the tensor board
"""
summaries_dir = '{0}/{1}'.format(summaries_dir, datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')




"""
LSTM CLASS WHICH CONSISTS OF THE WHOLE PROCESS
UNTIL FEED-FORWARD 
df --> will be the dataframe which wil have the names and gender. pre-processing  also will be done here.   

"""
class AVENGER_LSTM:
    def __init__(self,hidden_size,input_size,max_length,n_classes=2, learning_rate = 0.01):

        self.input = self.__input()
        self.seq_len = self.__seq_len()
        self.target = self.__target(2)
        self.dropout_keep_prob = self.__dropout_keep_prob()
        self.scores  = self.__scores(self.input,hidden_size,n_classes,self.dropout_keep_prob,random_state=None)
        self.predict = self.__predict(self.scores)



    def __input(self):
        return tf.placeholder(tf.float64,[None,max_length,26], name = 'input')


    def __seq_len(self):
        return tf.placeholder(tf.float64,[None], name = 'lengths')

    def __target(self,n_classes):
        return tf.placeholder(tf.float64, [ None, n_classes], name = 'target')
    """
    dropout_keep_prob()
    returns a placeholder holding the dropout keep probability
    to reduce ovefitting
    """
    def __dropout_keep_prob(self):
        return tf.placeholder(tf.float32, name = 'dropout_keep_prob')

    """
    __cell() 
    LSTM CELL WITH A DROP OUT WRAPPER
    """

    def __cell(self,hidden_size,dropout_keep_prob,seed =None):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size,state_is_tuple = True)
        dropout_cell  = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob, seed=seed)
        return dropout_cell

    """
    rnn layer which is above LSTM CELL.  
    """
    def __rnn_layer(self, hidden_size,x,dropout_keep_prob, variable_scope = None,random_state=None):
        with tf.variable_scope(variable_scope, default_name='rnn_layer'):
            lstm_cell = self.__cell(hidden_size,dropout_keep_prob, random_state)
            output, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype = tf.float32)
        return output


    """
    SCORE METHOD CALCULATES THE VALUES AT EACH STAGE
    """
    def __score(self, x,hidden_size,n_classes,dropout_keep_prob, random_state =None):
        input = x
        for h in hidden_size:
            output = self.__rnn_layer(h, input,dropout_keep_prob)

        output = tf.reduce_mean(output, axis=[1])
        with tf.name_scope('final_layer/weights'):
            w = tf.get_variable("w", shape=[hidden_size[-1], n_classes], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)
            self.variable_summaries(w,'final_layer/weights')

        with tf.name_scope('final_layer/biases'):
            b = tf.get_variable("b", shape = [n_classes], dtype= tf.float32, initializer= None, regularizer=None, trainable=True, collections= None )
            self.variable_summaries(b,'final_layer/biases')
            with tf.name_scope('final_layer/wx_plus_b'):
                scores = tf.nn.xw_plus_b(output,w,b, name = 'scores')
                tf.summary.histogram('final_layer/wx_plus_b', scores)
            return scores


    """
    predict() method
    """

    def __predict(self,scores):
        with tf.name_scope('final_layer/softmax'):
            softmax = tf.nn.softmax(scores, name = 'predictions')
            tf.summary.histogram('final_layer/softmax', softmax)
        return softmax

    def __losses(self,scores,target):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits= scores, labels = target, name = 'cross_entropy')
        return cross_entropy


    def __loss(self,losses):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(losses, name = 'loss')
            tf.summary.scalar('loss',loss)
        return loss

    def __train_step(self,learning_rate, loss):
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    def __accuracy(self, predict, target):
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'accuracy')
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def initialize_all_variables(self):
        return tf.global_variables_initializer()



    @staticmethod
    def variable_summaries(var, name):
        with tf.name_scope('summaries'):
            mean  = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

sess  =tf.session()

#initialize the variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

saver  = tf.train.saver()










