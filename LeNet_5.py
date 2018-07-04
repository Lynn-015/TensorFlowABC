import tensorflow as tf

Image_Size = 28 #图片尺寸为28x28
Num_Channels = 1 #灰度图像
Conv1_Size = 5 #第一个卷积层kernel尺寸
Conv1_Kernel_Num = 32 #第一个卷积层kernel数量
Conv2_Size = 5 #第二个卷积层kernel尺寸
Conv2_Kernel_Num = 64 #第二个卷积层kernel数量
FC_Size = 512 #全连接层维度
Output_Node = 10 #输出维度

def get_weight(shape, regularizer_rate):
    '''获取权重'''
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer_rate != None: 
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(w))
    return w

def get_bias(shape):
    '''获取偏置'''
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, w):
    '''卷积层'''
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    '''池化层'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def forward(x, train, regularizer_rate):
    '''前向传播'''
    #第一个卷积层
    conv1_w = get_weight([Conv1_Size, Conv1_Size, Num_Channels, Conv1_Kernel_Num], regularizer_rate)
    conv1_b = get_bias([Conv1_Kernel_Num])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool(relu1)

    #第二个卷积层
    conv2_w = get_weight([Conv2_Size, Conv2_Size, Conv1_Kernel_Num, Conv2_Kernel_Num], regularizer_rate)
    conv2_b = get_bias([Conv2_Kernel_Num])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool(relu2)

    #改变形状以便传入全连接层
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])

    #第一个全连接层，也可以用tf.nn.dense()实现
    fc1_w = get_weight([nodes, FC_Size], regularizer_rate)
    fc1_b = get_bias([FC_Size])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5) #在训练过程使用dropout

    #第二个全连接层
    fc2_w = get_weight([FC_Size, Output_Node], regularizer_rate)
    fc2_b = get_bias([Output_Node])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    
    return y