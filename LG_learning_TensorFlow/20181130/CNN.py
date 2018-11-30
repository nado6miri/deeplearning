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

# 20181130 CNN 은 영상처리에 강함. 어제 실습했던 mnist를 convolution 전처리 하고 어제와 동일하게 처리해 주면 높아짐

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

keep_prob = tf.placeholder(tf.float32)
learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])  # imag 28 x 28 x 1 (black/white) 형태로 만든다.
Y = tf.placeholder(tf.float32, [None, 10])

#L1 ImgIn shape(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# conv --> (?, 28, 28, 32)
# Pool --> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape(?, 14, 14, 32), dtype=float32)
'''

#L2 ImgIn shape(?, 14, 14, 32)
W1 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# conv --> (?, 14, 14, 64)
# Pool --> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W1, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7*7*64])

'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape(?, 3136), dtype=float32)
'''

#Final FC 7x7x64 inputs --> 10 outputs
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypo = tf.add(tf.matmul(L2_flat, W3), b)

model = tf.nn.softmax(hypo)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypo, labels=Y)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=hypo, labels=Y)
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

