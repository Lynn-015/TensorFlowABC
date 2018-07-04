import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

Test_Interval_Secs = 10 #测试时间间隔为10秒

def test(mnist):
    '''测试函数'''
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.Input_Node])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.Output_Node])
        y = mnist_forward.forward(x, None)

        #滑动平均变量
        ema = tf.train.ExponentialMovingAverage(mnist_backward.Moving_Average_Decay)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        #计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3) #分配显存
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
        while True:   
            ckpt = tf.train.get_checkpoint_state(mnist_backward.Model_Save_Path)
            if ckpt and ckpt.model_checkpoint_path:
                #从模型中恢复滑动平均变量
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print('After %s training step(s), test accuracy = %g' %(global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return
            time.sleep(Test_Interval_Secs)
        sess.close()
            
def main():
    mnist = input_data.read_data_sets('./data/mnist_data', one_hot=True)
    test(mnist)
    
if __name__ == '__main__':
    main()