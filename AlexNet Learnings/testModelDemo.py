# coding=utf-8
import os
from alexnet import alexNet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes
import matplotlib.pyplot as plt


if __name__=="__main__":
    # step1:参数设置
    dropoutPro = 1
    classNum = 1000
    skip = []

    # step2:测试图像加载
    testPath = "testImage"  # 测试图像路径
    testImg = []
    for f in os.listdir(testPath):
        testImg.append(testPath + "/" + f)

    # step3:加载模型
    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])
    model = alexNet(x, dropoutPro, classNum, skip)
    score = model.fc3
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)  # 加载模型

        for i, path in enumerate(testImg):
            
            img = plt.imread(path)
            #img = cv2.imread(path)
            test = cv2.resize(img.astype(np.float), (227, 227))  # resize成网络输入大小
            test -= imgMean  # 去均值
            test = test.reshape((1, 227, 227, 3))  # 拉成tensor
            predictions = sess.run(softmax, feed_dict={x: test})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-5:][::-1]

            for node_id in top_k:     
                #获取分类名称
                class_name = caffe_classes.class_names[node_id]
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (class_name, score))
            
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            print(path)
            print("-"*30)
            

