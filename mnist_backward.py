import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

Batch_Size = 100 #一个batch中的样本个数
Learning_Rate_Base = 0.8 #初始学习率
Learning_Rate_Decay =0.99 #学习率衰减率
Regularizer_Rate = 0.0001 #正则化系数
Steps = 30000 #训练步数
Moving_Average_Decay = 0.99 #滑动平均衰减率
Model_Save_Path = './model/FC/' #模型存储地址
Model_Name = 'mnist_model.ckpt' #模型名称

def backward(mnist):
    '''反向传播'''
    x = tf.placeholder(tf.float32, [None, mnist_forward.Input_Node], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.Output_Node], name='y_input')
    y = mnist_forward.forward(x, Regularizer_Rate)
    global_step = tf.Variable(0, trainable=False)

    #滑动平均变量
    ema = tf.train.ExponentialMovingAverage(Moving_Average_Decay, global_step)
    ema_op = ema.apply(tf.trainable_variables()) 

    #损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, axis=1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    
    #学习率
    learning_rate = tf.train.exponential_decay(
        Learning_Rate_Base,
        global_step,
        mnist.train.num_examples / Batch_Size,
        Learning_Rate_Decay,
        staircase=True)
        
    #优化器    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
        
    #创建会话并开始训练    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op) 
        for i in range(Steps):
            xs, ys = mnist.train.next_batch(Batch_Size)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0: #每1000步打印一次损失函数值
                print('After %d step(s), loss is %g.' %(step, loss_value))
                saver.save(sess, os.path.join(Model_Save_Path, Model_Name), global_step=global_step)
                    
def main(argv=None):
    mnist = input_data.read_data_sets('./data/mnist_data', one_hot=True)
    backward(mnist)
    
if __name__ == '__main__':
    tf.app.run() #主程序入口