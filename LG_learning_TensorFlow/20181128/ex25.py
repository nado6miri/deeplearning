import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Linear Regression (선형회귀) Example -  경사 하강법
'''
 y = ax + b
 y = a1x1 + a2x2 + a3x3 + ..... + anxn --> 여러가지 factor 가 존재함. - 이럴경우에는 최소 제곱법으로 풀지 못함.
 임의의 선을 긋고 오차가 최소화 되도록 상수(기울기, 절편)을 조정하여 오차를 최소화 하는 방식을 사용하면
 여러가지 factor 가 있어도 계산이 가능하다. 인공지능에서 쓰는 방식임.
 정답이 존재를 한다면 입력값에 따른 오차를 구하고 오차를 최소화 하는 과정을 통해서 상수값을 구할 수 있다.
'''

tf.set_random_seed(777)

x_data = [1, 2, 3]
y_data = [1, 2, 3]


W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W*X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(tf.random_normal([1])))

for step in range(21) :
    sess.run(update, feed_dict={ X: x_data, Y : y_data })
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))