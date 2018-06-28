import tensorflow as tf

Input_Node = 784
Output_Node =10
Layer1_Node = 500

def get_weight(shape, regularizer_rate):
    w = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer_rate != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(w))
    return w
    
def get_bias(shape):
    b = tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.0))
    #b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer_rate):
    with tf.variable_scope('layer1'):
        w = get_weight([Input_Node, Layer1_Node], regularizer_rate)
        b = get_bias([Layer1_Node])
        y1 = tf.nn.relu(tf.matmul(x, w) + b)
    
    with tf.variable_scope('layer2'):
        w = get_weight([Layer1_Node, Output_Node], regularizer_rate)
        b = get_bias([Output_Node])
        y2 = tf.matmul(y1, w) + b

    return y2
    
    
    
