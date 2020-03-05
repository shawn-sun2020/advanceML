import tensorflow as tf
import numpy as np


hidden_layer = 512
clip_margin = 4
learning_rate = 0.0001
batch_size = 10
window_size = 10

# weights and implementation of LSTM cell

# Weights for the input gate
weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_input = tf.Variable(tf.zeros([hidden_layer]))

# weights for the forgot gate
weights_forgot_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_forgot_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_forgot = tf.Variable(tf.zeros([hidden_layer]))

# weights for the output gate
weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_output = tf.Variable(tf.zeros([hidden_layer]))

# weights for the memory cell
weights_memory_cell = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))

# Output layer weights
weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))


def input_func(input, output):
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden)
                            + bias_input)
    return input_gate


def output_func(input, output):
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(output, weights_output_hidden)
                             + bias_output)
    return output_gate


def forgot_func(input, output):
    forgot_gate = tf.sigmoid(tf.matmul(input, weights_forgot_gate) + tf.matmul(output, weights_forgot_hidden)
                             + bias_forgot)
    return forgot_gate


def lstm_cell(input, output, state):
    # Ft = Wf*Xt + Uf*Ht-1 + b
    input_gate = input_func(input, output)

    forgot_gate = forgot_func(input, output)

    output_gate = output_func(input, output)

    # Ct' = tanh(Wc*Xt + Uc*ht-1 + b)
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) + tf.matmul(output, weights_memory_cell_hidden)
                          + bias_memory_cell)

    # Ct = Ft*Ct-1 + It*Ct'
    state = state * forgot_gate + input_gate * memory_cell

    # Ct = Ot*tanh(Ct)
    output = output_gate * tf.tanh(state)

    return state, output


def lstm_loop(inputs):
    outputs = []
    for i in range(batch_size):
        batch_state = np.zeros([1, hidden_layer], dtype=np.float32)
        batch_output = np.zeros([1, hidden_layer], dtype=np.float32)

        for j in range(window_size):
            batch_state , batch_output = lstm_cell(tf.reshape(inputs[i][j], (-1, 1)), batch_state, batch_output)

        outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)

    return outputs


def loss(outputs, labels):
    losses = []
    for i in range(len(outputs)):
        losses.append(tf.losses.mean_squared_error(tf.reshape(labels[i], (-1, 1)), outputs[i]))

    return tf.reduce_mean(losses)


def train(loss_val):
    gradients = tf.gradients(loss_val, tf.trainable_variables())
    clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    return trained_optimizer


def evaluator(outputs, labels):
    return labels


