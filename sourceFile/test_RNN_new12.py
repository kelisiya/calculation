#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 21:18:44 2017

@author: liupeng
"""
import numpy as np  
import tensorflow as tf  
import cv2

import re

lex = ['_','0','1','2','3','4','5','6','7','8','9','+','-','*','/','(',')']
CHAR_SET_LEN = len(lex)
MAX_CAPTCHA = 7

def get_image():
    image_file = '2image_path_validate.txt'
    image_path = []
    train_image = []
    with open(image_file) as f:
        for line in f: 
            line = line.split('\n')[0] 
            image_path.append(line)
        
            image = cv2.imread(line)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        
            m = image.mean()
            s = image.std()
            min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
            std = max(min_s, s)
            image = (image-m)/std
        
            train_image.append(image.flatten())
        return train_image

IMAGE_HEIGHT = 60  
IMAGE_WIDTH = 180  


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
   
    data = tf.reshape(data, [-1, chunk_size, chunk_size])  
    data = tf.transpose(data, [1,0,2])  
    data = tf.reshape(data, [-1, chunk_size])  
    data = tf.split(data,chunk_n)
    
    layer = {'w_':tf.Variable(tf.random_normal([rnn_size, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))} 
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size) 
    outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)  
    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])     
   
    return ouput 

o='[\+,\-,\*]' #运算符
lb='[\(]' #左括弧
rb='[\)]' #右括弧
n='[0-9]' #数字
exp71=re.compile(lb+n+o+n+rb+o+n)
exp72=re.compile(n+o+lb+n+o+n+rb)
exp5=re.compile(n+o+n+o+n)
	
def reCheck(exp):
    """
    reCheck:校验输入的表达式是否符合特定的四则运算表达式规则,
    正确就返回True,错误就返回False.
    exp:待校验的表达式字符串.
    """
    correct=True
    if re.fullmatch(exp71,exp) is None:
        if re.fullmatch(exp72,exp) is None:
            if re.fullmatch(exp5,exp) is None:
                correct=False
    return correct


def test(captcha_image):  
    f=open("2result.txt",'w')
    
    output = recurrent_neural_network()  
   
    saver = tf.train.Saver()  
    sess = tf.Session()  
    # saver.restore(sess, tf.train.latest_checkpoint('/Users/liupeng/Desktop/anaconda/i_code', 'checkpoint')) 
    sess.run(tf.initialize_all_variables())
    
    print ('start restore model')
    saver.restore(sess, 'model5/model_net-100')
    print ('ok')    
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    
    count = 0
    for img in captcha_image:
        text_list = sess.run(predict, feed_dict={X: [img], keep_prob: 1})  
        text = text_list[0].tolist()              
        
        res = [lex[i] for i in text]
        res = "".join(res)
        res = res.replace('_','')
        
        rex = reCheck(res)
        if rex==True:
            try:
                res1 = eval(res)
            except: #不能计算 应该不存在此类   
                res1 = "12345x"
                count = count + 1;
                print ('I get the error0',count)
        else: #不符合表达式
            res1 = "12345y"
            count = count + 1;
            print ('I get the error1',count)
               
        resx = ("{r0} {r1}".format(r0=res,r1=res1))
        print (resx)
        f.write(resx)
        f.write('\n')  
    print ('number of  error is :',count)
    f.close()       
    return text  


train_image = get_image()
test(train_image)


