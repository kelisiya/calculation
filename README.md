# 四则混合运算识别 深度学习应用大赛
*** 
###  数据下载地址：[http://baidudeeplearning.bj.bcebos.com/image_contest_level_1.tar.gz](http://baidudeeplearning.bj.bcebos.com/image_contest_level_1.tar.gz)
###  参考paper：Recursive Recurrent Nets with Attention Modeling for OCR in the Wild
###  方法：cnn + rnn 混合卷积网络
### 椒盐噪声处理方法：
* 中值滤波法，因为中值滤波对椒盐噪声效果好、
* trick：若噪声点大的就加大窗口，必要时可再加个判断，比如当前像素值灰度大于某个值时就采用中值或均值
### 用cnn+rnn，cnn，cnn+rnn+attention 和降噪模型共六个模型进行投票处理，选择票数多的答案进行更改
### 准确值：0.999805
