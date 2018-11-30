import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#정확도 높이는 방법
# hidden layer 추가
# relu 추가 (sigmoid는 깊어지면 안좋음... 따라서 relu나 다른걸 사용해야 함)
# 가중치 변수의 초기화 함수 잘 사용 - tf.contrib.layers.xavier_initializer()
# 노드수 증가 256 -> 512
# 과적합 방지를 위해 Drop Out 적용 (학습시 70%만, 테스트시 100% 사용)

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

keep_prob = tf.placeholder(tf.float32)
learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W = tf.get_variable("W", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypo = tf.add(tf.matmul(L4, W), b)

model = tf.nn.softmax(hypo)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypo, labels=Y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize((cost))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = { X: batch_xs, Y: batch_ys, keep_prob: 0.7  }  # 학습할때는 70% Node 만 Random On 해서 활용
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')

prediction = tf.argmax(hypo, 1)
target = tf.argmax(Y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target), tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={ X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1 })) # 테스트 할때는 full node 활용

r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypo, 1), feed_dict={ X: mnist.test.images[r: r+1], keep_prob: 1 }))  # 테스트 할때는 full node 활용

plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys')
plt.show()

