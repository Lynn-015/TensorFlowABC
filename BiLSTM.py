import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist_data", one_hot=True)

learning_rate = 0.01 #学习率
training_steps = 10000 #训练步数
batch_size = 128 #每个batch里样本数
display_step = 10 #显示屏幕信息步数间隔

n_input = 28 # 每个时间步输入数据的维度
time_steps = 28 # 时间步数。与n_input共同组成一张图像的维度
n_hidden = 256 # 隐藏层维度
n_classes = 10 # 分类数0~9

x = tf.placeholder("float", [None, time_steps, n_input])
y_ = tf.placeholder("float", [None, n_classes])

weights = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def BiRNN(x, weights, biases):
    #将(batch_size, time_steps, num_input)的数据转换成形状为(batch_size, num_input), 长度为time_steps的列表
    x = tf.unstack(x, time_steps, 1)

    # 向右传播的lstm
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 向左传播的lstm
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # 此处不使用激活函数，会在交叉熵部分使用softmax
    return tf.matmul(outputs[-1], weights) + biases

#计算预测值
pred = BiRNN(x, weights, biases)

#损失函数和优化器
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#计算准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#创建会话并开始训练
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    for step in range(training_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, time_steps, n_input))
        #将输入数据reshape成(batch_size, time_steps, n_input)形状
        sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})
        if step % display_step == 0:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x, y_: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

    print("Optimization Finished!")

    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, time_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y_: test_label}))