import numpy as np
np.random.seed(12345)

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print('\n',names,"\n")
print('\n',data)

print('\n',names == 'Bob')
print('\n',data[names == 'Bob'])

print('\n',data[names == 'Bob', 2:])
print('\n',data[names == 'Bob', 3])

print("\n",names != 'Bob')
print('\n',data[~(names == 'Bob')])

mask = (names == 'Bob') | (names == 'Will')
print('\n',mask)
print('\n',data[mask])

data[data < 0] = 0
print('\n',data)


data[names != 'Joe'] = 7
print('\n',data)

