import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lstm_testing import *
import tensorflow as tf
import numpy as np

batch_size = 10
window_size = 10
epochs = 500


def window_data(data, window_size):
    X = []
    y = []

    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])

        i += 1
    assert len(X) == len(y)
    return X, y


if __name__ == "__main__":
    data_to_use = pd.read_csv("diskdata.csv")
    # data_to_use = data_to_use[data_to_use['raid_type'] == 6][data_to_use['is_parity_sum_disk'] == 0][data_to_use['use_type'] == 1]
    # normalize power on hours
    data_to_use['pwr_on_hours'] = data_to_use['pwr_on_hours'].map(lambda x: int((x - 168) / 24))
    # data_to_use = data_to_use[['pwr_on_hours', 'erase_count']][1::2]
    # scaler = StandardScaler()
    # data_to_use['erase_count'] = scaler.fit_transform(data_to_use['erase_count'].values.reshape(-1, 1))
    data_to_use = data_to_use.groupby('serial_no')
    # print([y for x, y in data_to_use])
    plt.figure(figsize=(16, 7))
    plt.title('erase count for each disk given days')
    plt.xlabel('Days')
    plt.ylabel('Erase Counts')
    train_data = dict()
    test_data = dict()
    for serial_no, df_region in data_to_use:
        if len(df_region['pwr_on_hours'].unique()) >= 199:
            # print(df_region[['pwr_on_hours', 'erase_count']])
            df_1 = df_region.sort_values(by=['pwr_on_hours'])[::2]['erase_count']
            df_2 = df_region.sort_values(by=['pwr_on_hours'])[1::2]['erase_count']
            plt.plot(df_1.values, label=serial_no)
            print(serial_no)
            test_data[serial_no] = df_2
            train_data[serial_no] = df_1
            # plt.plot(df_2, label=serial_no)
            # plt.scatter(data_to_use['pwr_on_hours'], data_to_use['erase_count'], label='Scatter')
            plt.legend()
    plt.show()
    data = train_data['MN98DFSB'].values
    print(data)

    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = window_data(data, window_size)

    X_train = np.array(X[:140])
    y_train = np.array(y[:140])

    X_test = np.array(X[140:])
    y_test = np.array(y[140:])

    with tf.device('/device:GPU:0'):
        inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
        labels = tf.placeholder(tf.float32, [batch_size, 1])
        weights = lstm_loop(inputs)
        loss = loss(weights, labels)
        trained_optimizer = train(loss)
        evaluator = evaluator(weights, labels)

    # config = tf.ConfigProto(log_device_placement=True)
    session = tf.Session()
    session.run(tf.global_variables_initializer())


    for i in range(epochs):

        j = 0
        train_scores = []
        epoch_loss = []
        while(j + batch_size) <= len(X_train):
            X_batch = X_train[j:j+batch_size]
            y_batch = y_train[j:j+batch_size]

            prediction, loss_value, _ = session.run([weights, loss, trained_optimizer],
                                                    feed_dict={inputs: X_batch, labels: y_batch})

            epoch_loss.append(loss_value)
            train_scores.append(prediction)

            j += batch_size

        if (i % 30) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))

    train_scores = np.reshape(train_scores, (-1, 1))
    test_scores = []
    test_loss = []
    i = 0
    while i+batch_size <= len(X_test):
        o, loss_value = session.run([weights, loss], feed_dict={inputs: X_test[i:i+batch_size], labels:y_test[i:i+batch_size]})
        i += batch_size
        test_loss.append(loss_value)
        test_scores.append(o)
        if i % 25 == 0:
            print('Test Epoch {}'.format(i), ' Current loss: {}'.format(np.mean(test_loss)))

    test_scores = np.reshape(test_scores, (-1, 1))
    print(test_scores.shape)

    test_results = []
    for i in range(190):
        if i >= 141:
            test_results.append(test_scores[i - 141])
        else:
            test_results.append(None)

    plt.figure(figsize=(16, 7))
    plt.title('Erase Count for disk type 510 given days ')
    plt.xlabel('Days')
    plt.ylabel('Scaled Erased Count')
    plt.plot(data, label='Original data')
    plt.plot(train_scores, label='Training data')
    plt.plot(test_results, label='Testing data')
    plt.legend()
    plt.show()
