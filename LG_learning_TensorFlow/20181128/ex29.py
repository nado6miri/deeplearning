import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Logistic Regression (로지스틱 회귀) Example -  param 이 여러개일경우 (공부시간, 과외횟수)

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# x, y input data
x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]]) # 공부시간, 과외횟수
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7,1)

# 입력값을 placeholder에 저장
X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# 기울기 a 와  bias b의 값을 임의로 정함
a = tf.Variable(tf.random_uniform([2, 1], dtype=tf.float64)) # [2,1]의 의미는 2는 입력개수, 1은 출력개수
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

#y 시그모이드 함수 방정식 세움
y = tf.sigmoid(tf.matmul(X, a) + b)
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))
learning_rate = 0.1

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize((loss))

#tensorflow learning
with tf.Session() as sess :
    #변수 초기화
    sess.run(tf.global_variables_initializer())
    #3001번 실행
    for step in range (3001) :
            a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={ X : x_data, Y : y_data })
            #if (step % 300 == 0):
            #    print("step: %.f, a1 = %.4f, a2 = $.4f, b = %.4f, loss = %.4f" % (step, a_[0], a_[1], b_, loss_))

    # 위에서 학습을 시켰고 아래에서 실제 data를 넣어 예측하는 부분임.
    new_x = np.array([7, 6]).reshape(1, 2) #[7, 6] 은 공부시간가 과외수업 회수
    new_y = sess.run(y, feed_dict= { X : new_x })
    print("공부시간 : %d,  개인과외 수 : %d" % (new_x[:,0], new_x[:, 1]))
    print("합격 가능성 : %6.2f %%" % (new_y * 100))