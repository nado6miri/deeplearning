import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 신경망(Perceptron) =  Linear Regression + Logistic Regression
# XOR를 학습시킨다. 단층일 경우 매우 부정확함을 확인 --> 은닉층을 추가하여 정확도를 높여보자..!! 정확도 1.0으로 향상

tf.set_random_seed(777) # 잘 되는 case
#tf.set_random_seed(1234) # 잘 안되는 case -->  어떤 seed table을 쓰더라도 정확도 높도록 설계를 잘 해야 함.
learning_rate = 0.1

#XOR 진리표
x_data = [[0,0], [0, 1], [1, 0], [1,1]]
y_data = [[0], [1], [1], [0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 2], name='weight1'))
b1 = tf.Variable(tf.random_normal([2], name='bias1'))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1], name='weight2'))
b2 = tf.Variable(tf.random_normal([1], name='bias2'))

hypo = tf.sigmoid(tf.matmul(layer1, W2) + b2)

#cost / loss function
cost = -tf.reduce_mean(Y * tf.log(hypo) + (1 - Y) * tf.log(1 - hypo))

train = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(10001) :
        sess.run(train, feed_dict={ X : x_data, Y : y_data })
        if step % 100 == 0 :
            print(step, sess.run(cost, feed_dict={ X : x_data, Y : y_data}), sess.run(W2))

    h, c, a = sess.run([hypo, predicted, accuracy], feed_dict={ X : x_data, Y : y_data })
    print("hypo = ", h, "\nCorrect =  ", c, "\nAccuracy = ", a)