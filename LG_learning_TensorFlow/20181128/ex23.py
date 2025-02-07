import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Linear Regression (선형회귀) Example - 최소 제곱범
'''
 y = ax + b
 y = a1x1 + a2x2 + a3x3 + ..... + anxn --> 여러가지 factor 가 존재함. - 이럴경우에는 최소 제곱법으로 풀지 못함.
 따라서 1차로 factor 1개만 놓고 판단하자...
'''

x = [ 2, 4, 6, 8 ]
y = [ 81, 93, 91, 97 ]

mx = np.mean(x)
my = np.mean(y)

print("x의 평균값 : ", mx)
print("y의 평균값 : ", my)

#기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x])

def top(x, mx, y, my) :
    d = 0
    for i in range(len(x)) :
        d += ((x[i] - mx) * (y[i] - my))
    return d

dividend = top(x, mx, y, my)

print("분모 : ", divisor)
print("분자 : ", dividend)

#기울기와 Y절편 구하기
a = dividend / divisor
b = my - (mx*a)

print("기울기", a)
print("Y절편", b)

