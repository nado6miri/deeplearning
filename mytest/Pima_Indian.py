# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

# 필요한 라이브러리를 불러옵니다.
import numpy
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

print("ata_set = numpy.loadtxt(ThoraricSurgery.csv)")
print(Data_set)


# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 불러올 때 각 컬럼에 해당하는 이름을 지정합니다.
df = pd.read_csv('../dataset/pima-indians-diabetes.csv',
               names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

# 처음 5줄을 봅니다.
print(df.head(5))

# 데이터의 전반적인 정보를 확인해 봅니다.
print(df.info())

# 각 정보별 특징을 좀더 자세히 출력합니다.
print(df.describe())

# 데이터 중 임신 정보와 클래스 만을 출력해 봅니다.
print(df[['pregnant', 'class']])

a = df[['pregnant', 'class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True)
print(a)


# 데이터 간의 상관관계를 그래프로 표현해 봅니다.

colormap = plt.cm.gist_heat  #그래프의 색상 구성을 정합니다.
plt.figure(figsize=(12,12))   #그래프의 크기를 정합니다.
print("df.corr()")
print(df.corr())
# 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시되게 합니다.
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
plt.show()

grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma',  bins=20)

plt.show()

grid2 = sns.FacetGrid(df)
grid2.map(plt.hist, 'plasma',  bins=20)

plt.show()


seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

dataset = numpy.loadtxt('../dataset/pima-indians-diabetes.csv', delimiter=',')
print("dataset = ", dataset)
x = dataset[:, 0:8]
y = dataset[:,8]
print("x = ", x)
print("y = ", y)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
'''
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, input_dim=12, activation='relu'))
model.add(Dense(1, input_dim=8, activation='sigmoid'))
'''

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=200, batch_size=10)

print("\n Accuracy: %.4f" % (model.evaluate(x, y)[1]))
