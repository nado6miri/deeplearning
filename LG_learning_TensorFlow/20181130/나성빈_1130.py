import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

'''
1. UCI에서 Data 다운로드
2. hidden층은 최소 3개 이상
3. tensorboard에서 graph를 표현하기 위해 log 저장
4. 각 weight의 random 초기값을 최적화 한다.
5. 각 층과 optimizer는 그래프에서 보기 수월하게 block으로 나누어 표시 (layer 1/2/3...)
6. 모델링할때 trainData:80, testData:20%비율로 사용
7. 모델에서 생성되는 loss값(cost) accuracy(정확도)를 tensorboard에 보기위해 scalar로 저장한다.
8. 각 층의 weight값을 histogram에 저장하여 histogram을 그래프로 표시한다.
9. 각 모델을 재 사용하기 위해 학습된 모델은 저장하고 실행시 저장된 기존 모델이 있으면 불러와서 재사용한다.
'''

# tensorboard --logdir=C:\Users\user\PycharmProjects\nsbdeeplearning\20181130\logs --host=127.0.0.1

tf.set_random_seed(777)  # reproducibility

#data = input_data.read_data_sets("wine-quality/", one_hot=True)
data = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32)
global_step = tf.Variable(0, trainable=False, name='global_step')

x_data = data[:, 0:-1]
print("\nx_data = ", x_data)
y_data = data[:, [-1]]
print("\ny_data = ", y_data)

# input place holders
X = tf.placeholder(tf.float32, [None, 12])
Y = tf.placeholder(tf.float32, [None, 1])

with tf.name_scope('layer1'):
    W1 = tf.get_variable("W1", shape=[12, 12], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", shape=[12], initializer=tf.contrib.layers.xavier_initializer())
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    tf.summary.histogram("X", X)
    tf.summary.histogram("W1", W1)

with tf.name_scope('layer2'):
    W2 = tf.get_variable("W2", shape=[12, 12], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", shape=[12], initializer=tf.contrib.layers.xavier_initializer())
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    tf.summary.histogram("W2", W2)

with tf.name_scope('layer3'):
    W3 = tf.get_variable("W3", shape=[12, 12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", shape=[12], initializer=tf.contrib.layers.xavier_initializer())
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    tf.summary.histogram("W3", W3)

with tf.name_scope('OUT'):
    W4 = tf.get_variable("W4", shape=[12, 1], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
    hypothesis = tf.nn.softmax(tf.matmul(L3, W4) + b4)
    tf.summary.histogram("model", hypothesis)

# Cross entropy cost/loss
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost, global_step=global_step)
tf.summary.scalar('cost', cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess :
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    for step in range(2001) :
        sess.run(optimizer, feed_dict={ X : x_data, Y : y_data })
        if step % 100 == 0 :
            print(step, sess.run(cost, feed_dict={ X : x_data, Y : y_data}))

        summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=sess.run(global_step))

    saver.save(sess, './model/mydnn.ckpt', global_step=global_step)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={ X : x_data, Y : y_data })
    print("hypo = ", h, "\nCorrect =  ", c, "\nAccuracy = ", a)
