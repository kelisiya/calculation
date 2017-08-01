#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:06:49 2017

@author: liupeng
"""

import numpy as np  
import tensorflow as tf  
import cv2
#from attention import attention


image_file = 'image_pathALL.txt'
image_path = []
with open(image_file) as f:
    for line in f: 
        line = line.split('\n')[0] 
        image_path.append(line)


lex = ['_','0','1','2','3','4','5','6','7','8','9','+','-','*','/','(',')']
CHAR_SET_LEN = len(lex)
MAX_CAPTCHA = 7

org_file = 'labelsALL.txt'
labels = []
with open(org_file) as f:
    for line in f: 
        line = line.split(' ')[0] 
        
        if(len(line)==5):
            line = line + '__'
        if(len(line)==6):
            line = line + '_'
        
        lab = np.zeros(CHAR_SET_LEN*MAX_CAPTCHA)
        
        for i,w in enumerate(line):
            idx = i*CHAR_SET_LEN + lex.index(w)
            lab[idx] = 1  
        labels.append(lab)

print ('------image_path, label--------')


# 图像大小  
IMAGE_HEIGHT = 60  
IMAGE_WIDTH = 180   
batch_size = 128
print("验证码文本最长字符数", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐  

def get_next_batch(pointer):
    batch_x = np.zeros([batch_size, IMAGE_WIDTH*IMAGE_HEIGHT])  
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN]) 
    for i in range(batch_size):  
        image = cv2.imread(image_path[i+pointer*batch_size])
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        
        m = image.mean()
        s = image.std()
        min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
        std = max(min_s, s)
        image = (image-m)/std
        
        batch_x[i,:] = image.flatten()
        # print labels[i+pointer*batch_size]
        batch_y[i,:] = labels[i+pointer*batch_size]
    return batch_x, batch_y


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])  
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])  
keep_prob = tf.placeholder(tf.float32) # dropout  
   
# 定义CNN  
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):  
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  
    # w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
    w_c1 = tf.get_variable(
            name='W1',
            shape=[3, 3, 1, 32],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))  
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))  
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv1 = tf.nn.dropout(conv1, keep_prob)  
   
    # w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64])) 
    w_c2 = tf.get_variable(
            name='W2',
            shape=[3, 3, 32, 64],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))  
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))  
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv2 = tf.nn.dropout(conv2, keep_prob)  
   
    # w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))  
    w_c3 = tf.get_variable(
            name='W3',
            shape=[3, 3, 64, 64],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))  
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))  
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv3 = tf.nn.dropout(conv3, keep_prob)  
   
    # Fully connected layer  
    w_d = tf.Variable(w_alpha*tf.random_normal([23*8*64, 1024]))  
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))  
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])  
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))   

    return dense  

#一张图片是28*28,FNN是一次性把数据输入到网络，RNN把它分成块  
chunk_size = 32  
chunk_n = 32  
rnn_size = 256  
attention_size = 50
n_output_layer = MAX_CAPTCHA*CHAR_SET_LEN   # 输出层  

# 定义待训练的神经网络  
def recurrent_neural_network(): 
    data = crack_captcha_cnn()  
    layer = {'w_':tf.Variable(tf.random_normal([rnn_size, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}  
   
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)  
   
    data = tf.reshape(data, [-1, chunk_size, chunk_size])  
    data = tf.transpose(data, [1,0,2])  
    data = tf.reshape(data, [-1, chunk_size])  
    data = tf.split(data,chunk_n)
    outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)  
    
    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])  
   
    # attention_output = attention(np.asarray(outputs), attention_size)   
    # Fully connected layer
    # W = tf.Variable(tf.truncated_normal([attention_output.get_shape()[1].value, n_output_layer], stddev=0.1))
    # b = tf.Variable(tf.constant(0., shape=[n_output_layer]))
    # ouput = tf.add(tf.matmul(attention_output[-1], W), b)  
   
    return ouput 

   
# 训练  
def train_crack_captcha_cnn():  
    output = recurrent_neural_network()  
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = output))  
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  
   
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  
    max_idx_p = tf.argmax(predict, 2)  
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    correct_pred = tf.equal(max_idx_p, max_idx_l)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
   
    sess = tf.Session()
    saver = tf.train.Saver(tf.trainable_variables())
    sess.run(tf.initialize_all_variables())
    
    #saver.restore(sess, 'model/model_net-600')
    for j in range(2000):
        for i in range(2343): # 10w对应781 
            
            imgs, labels = get_next_batch(i)
            # keep_prob: 0.75
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: imgs, Y: labels, keep_prob: 0.75})  
            print (i, loss_)  
            
            if i%100==0 and i!=0:

                saver.save(sess, 'model/model_net', global_step=i, write_meta_graph=False)
            
            if i%1==0:
                
                img, label = get_next_batch(700+i%50)
                acc = sess.run(accuracy, feed_dict={X: img, Y: label, keep_prob: 1.})  
                print(i, acc)  
        
    sess.close()
    
   
train_crack_captcha_cnn()  





