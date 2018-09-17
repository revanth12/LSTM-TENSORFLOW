train_steps = 1000 #number of training steps
validate_every = 100 # step frequency to validate
batch_size = 100
hidden_size  = 100
input_size = 26
dropout_keep_prob = 0.5


from data import preprocessing
from model import AVENGER_LSTM
"""
creating empty lists to hold the training loss, validation loss
and the steps, to see them graphically.

"""

train_loss_list = []
val_loss_list = []
step_list = []
sub_step_list = []
step = 0

data_lstm  =  preprocessing( df = uk_train, val_samples=100, test_size=0.2, random_state=0,ensure_preprocessed=False)
lstm_model = AVENGER_LSTM(hidden_size,input_size,18,n_classes=2, learning_rate = 0.01)
train_writer.add_graph(lstm_model.input.graph)


for i in range(train_steps):
    x_train, y_train, train_seq_len = data_lstm.next_batch(batch_size)
    train_loss, _, summary = sess.run([lstm_model.loss, lstm_model.train_step,
                                       lstm_model.merged], feed_dict = {
        lstm_model.input:x_train,
        lstm_model.target:y_train,
        lstm_model.seq_len :train_seq_len,
        lstm_model.dropout_keep_prob : dropout_keep_prob})

    train_writer.add_summary(summary, i)  # write train summary for tensorboard visualization
    train_loss_list.append(train_loss)
    step_list.append(i)
    print('{0}/{1} train loss: {2:4f}'. format(i+1, FLAGS.train_steps, train_loss))

    if(i+1) %validate_every ==0:
        val_loss, accuracy, summary = sess.run([lstm_model.loss, lstm_model.accuracy, lstm_model.merged],
                                               feed_dict = {lstm_model.input: x_val,
                                                            lstm_model.target: y_val,
                                                            lstm_model.seq_len: val_seq_len,
                                                            lstm_model.dropout_keep_prob: 1})

        validation_writer.add_summary(summary, i)
        print('   validation loss: {0:.4f} (accuracy {1:.4f})'.format(val_loss, accuracy))
        step = step + 1
        val_loss_list.append(val_loss)
        sub_step_list.append(step)




