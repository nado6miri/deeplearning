import tensorflow as tf
import numpy as np

Data_set = np.loadtxt('../dataset/ThoraricSurgery.csv', delimiter=',')

learning_rate = 0.1

print(Data_set.shape)
print(Data_set)
x_data = Data_set[:, 0:17]
y_data = Data_set[:, 17]
print(x_data)
print(y_data)

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([17, 30], name='weight1'))
b1 = tf.Variable(tf.random_normal([30], name='bias1'))
layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.random_normal([30, 10], name='weight2'))
b2 = tf.Variable(tf.random_normal([10], name='bias2'))
layer2 = tf.sigmoid(tf.matmul(layer1, W2)+b2)

W3 = tf.Variable(tf.random_normal([10, 1], name='weight3'))
b3 = tf.Variable(tf.random_normal([1], name='bias3'))
hypo = tf.sigmoid(tf.matmul(layer2, W3)+b3)

cost = -tf.reduce_mean(Y * tf.log(hypo) + (1 - Y) * tf.log(1 - hypo))
train = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost) # foward / back propagation

predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(1001) :
        sess.run(train, feed_dict={ X : x_data, Y : y_data })
        if step % 100 == 0 :
            print(step, sess.run(cost, feed_dict={ X : x_data, Y : y_data}), sess.run([W1, W2, W3]))

    h, c, a = sess.run([hypo, predicted, accuracy], feed_dict={ X : x_data, Y : y_data })
    print("hypo = ", h, "\nCorrect =  ", c, "\nAccuracy = ", a)