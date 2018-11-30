import numpy as np

points = np.arange(-5, 5, 0.01)
print('\n',points)
xs, ys = np.meshgrid(points, points)
print(xs,'\n')
print('\n',ys)


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

result = np.where(cond, xarr, yarr)
print('\nresult:',result)
