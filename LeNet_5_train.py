import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet_5
import os
import numpy as np

Batch_Size = 100 #一个batch的样本数量
Learning_Rate_Base = 0.01 #初始学习率
Learning_Rate_Decay =0.99 #学习率衰减率
Regularizer_Rate = 0.0001 #正则化系数
Steps = 30000 #训练步数
Moving_Average_Decay = 0.99 #滑动平均衰减率
Model_Save_Path = './model/LeNet_5/' #模型保存地址
Model_Name = 'mnist_model.ckpt' #模型名称

def backward(mnist):
    '''反向传播'''
    x = tf.placeholder(tf.float32, [None, LeNet_5.Image_Size, LeNet_5.Image_Size, LeNet_5.Num_Channels], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, LeNet_5.Output_Node], name='y_input')
    y = LeNet_5.forward(x, True, Regularizer_Rate)
    global_step = tf.Variable(0, trainable=False)

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

    #滑动平均变量
    ema = tf.train.ExponentialMovingAverage(Moving_Average_Decay, global_step)
    ema_op = ema.apply(tf.trainable_variables()) 
   
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
        
    #创建会话并开始训练
    saver = tf.train.Saver() 
    with tf.Session() as sess:
        #如果存在模型文件则直接读取
        ckpt = tf.train.get_checkpoint_state(Model_Save_Path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            init_op = tf.global_variables_initializer()
            sess.run(init_op) 
            
        for i in range(Steps):
            xs, ys = mnist.train.next_batch(Batch_Size)
            #xs是一维数据，需要reshape成二维以便卷积层处理
            reshaped_xs = np.reshape(xs, (Batch_Size, LeNet_5.Image_Size, LeNet_5.Image_Size, LeNet_5.Num_Channels))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            
            if i % 1000 == 0:
                print('After %d step(s), loss is %g.' %(step, loss_value))
                saver.save(sess, os.path.join(Model_Save_Path, Model_Name), global_step=global_step)
        
def test(mnist):
    '''测试函数'''
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, LeNet_5.Image_Size, LeNet_5.Image_Size, LeNet_5.Num_Channels])
        y_ = tf.placeholder(tf.float32, [None, LeNet_5.Output_Node])
        y = LeNet_5.forward(x, False, 0.0)

        #滑动平均变量
        ema = tf.train.ExponentialMovingAverage(Moving_Average_Decay)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        #计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        test_feed = {x: mnist.test.images.reshape(-1, LeNet_5.Image_Size, LeNet_5.Image_Size, LeNet_5.Num_Channels),\
                            y_: mnist.test.labels}
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(Model_Save_Path)
            if ckpt and ckpt.model_checkpoint_path:
                #从模型中恢复滑动平均变量
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=test_feed)
                print('test accuracy = %g' %(accuracy_score))
            else:
                print('No checkpoint file found')
                return          
        
def main(argv=None):
    mnist = input_data.read_data_sets('./data/mnist_data', one_hot=True)
    backward(mnist)
    test(mnist)
    
if __name__ == '__main__':
    tf.app.run()
    
