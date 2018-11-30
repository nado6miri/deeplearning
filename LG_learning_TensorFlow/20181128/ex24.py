import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Linear Regression (선형회귀) Example - 평균 제곱근 오차 (RMSE)
'''
 y = ax + b
 y = a1x1 + a2x2 + a3x3 + ..... + anxn --> 여러가지 factor 가 존재함. - 이럴경우에는 최소 제곱법으로 풀지 못함.
 임의의 선을 긋고 오차가 최소화 되도록 상수(기울기, 절편)을 조정하여 오차를 최소화 하는 방식을 사용하면
 여러가지 factor 가 있어도 계산이 가능하다. 인공지능에서 쓰는 방식임.
 정답이 존재를 한다면 입력값에 따른 오차를 구하고 오차를 최소화 하는 과정을 통해서 상수값을 구할 수 있다.
'''

#기울기 a와 y절편 b
ab = [3, 76] # 임의로 상수값 지정함.. 이 값을 기준으로 오차를 계산하여 보정해 나감.

#x, y의 data값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#y = ax + b에 a와 b값을 대입하여 결과를 출력하는 함수
def predict(x) :
    return ab[0]*x + ab[1]

#RMSE 함수
def rmse(p, a) :
    return np.sqrt(((p-a)**2).mean())

#rmse 함수를 각 y값에 대입하여 최종값을 구하는 함수
def rmse_val(predict_result, y) :
    return rmse(np.array(predict_result), np.array(y))

#예측값이 들어갈 빈 list
predict_result = []

#모든 x값을 한번씩 대입하여 Predict List를 완성
for i in range(len(x)) :
    predict_result.append(predict(x[i]))
    print("공부시간=%.f, 실제점수=%.f, 예측점수 = %.f" % (x[i], y[i], predict(x[i])))

print("RMSE 최종값 = " +  str(rmse_val(predict_result, y)))