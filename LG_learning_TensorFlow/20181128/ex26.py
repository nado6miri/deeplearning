import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Linear Regression (선형회귀) Example -  경사 하강법 - 자동으로  Gradient decent 실행 (tf lib)
'''
 y = ax + b
 y = a1x1 + a2x2 + a3x3 + ..... + anxn --> 여러가지 factor 가 존재함. - 이럴경우에는 최소 제곱법으로 풀지 못함.
 임의의 선을 긋고 오차가 최소화 되도록 상수(기울기, 절편)을 조정하여 오차를 최소화 하는 방식을 사용하면
 여러가지 factor 가 있어도 계산이 가능하다. 인공지능에서 쓰는 방식임.
 정답이 존재를 한다면 입력값에 따른 오차를 구하고 오차를 최소화 하는 과정을 통해서 상수값을 구할 수 있다.
'''
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
'''
# 위 코드를 tf lib로 실행하는 예제

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

#기울기 a와 y절편 b의 값을 임의로 정한다.
# 단, 기울기의 범위는 0 ~ 10 사이이며, y절편은 0~100사이이다.
a = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

#y에 대한 일차 방정식 ax+b의 식을 세운다.
y=a*x_data+b

#텐서플로우 RMSE함수
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

learning_rate = 0.1

#RMSE 값을 최소로 하는 값 찾기 (Gradient Decent)
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize((rmse))

#tensorflow learning
with tf.Session() as sess :
    tf.summary.FileWriter('./logs', sess.graph)
    #변수 초기화
    sess.run(tf.global_variables_initializer())
    #2001번 실행
    for step in range (2001) :
        sess.run(gradient_decent)
        if(step % 100 == 0) :
            print("epoch: %.f, RMSE = %.4f, 기울기 = %.4f, y절편 = %.4f" % (step, sess.run(rmse), sess.run(a), sess.run(b)))
