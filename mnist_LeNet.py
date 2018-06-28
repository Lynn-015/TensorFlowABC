import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'): 
        conv1_weights=tf.get_variable(
