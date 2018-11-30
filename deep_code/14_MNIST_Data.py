#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils

import numpy
import sys
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# MNIST데이터셋 불러오기
(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()

print("#### X_train ####")
print(X_train)
print("#### Y_class_train ####")
print(Y_class_train)

print("학습셋 이미지 수 : {0} 개 (shape {1} - {2}, {3}, {4})".format(X_train.shape[0], X_train.shape, X_train.shape[0], X_train.shape[1], X_train.shape[2]))
print("테스트셋 이미지 수 : {0} 개 (shape {1} - {2}, {3}, {4})".format(X_test.shape[0], X_test.shape, X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# 그래프로 확인
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='Greys')
plt.show()

# 코드로 확인
for x in X_train[0]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

# 차원 변환 과정
X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float64')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

print("#### X_train - reshape[0] / 255 ####")
print(X_train[0])

# 클래스 값 확인
print("class : %d " % (Y_class_train[0]))

# 바이너리화 과정
Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)

print(Y_train[0])
