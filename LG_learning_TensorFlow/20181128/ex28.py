import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Logistic Regression (로지스틱 회귀) Example

data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

#기울기 a와 y절편 b의 값을 임의로 정한다.
a = tf.Variable(tf.random_uniform([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64, seed=0))

#y에 대한 sigmoid 함수 방정식 세움.
y = 1/(1 + np.e**(a * x_data + b))

#loss를 구함
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))

learning_rate = 0.5

#RMSE 값을 최소로 하는 값 찾기 (Gradient Decent)
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize((loss))

#tensorflow learning
with tf.Session() as sess :
    tf.summary.FileWriter('./logs', sess.graph)
    #변수 초기화
    sess.run(tf.global_variables_initializer())
    #2001번 실행
    for step in range (60001) :
        sess.run(gradient_decent)
        if(step % 6000 == 0) :
            print("epoch: %.f, loss = %.4f, 기울기 = %.4f, y절편 = %.4f" % (step, sess.run(loss), sess.run(a), sess.run(b)))
