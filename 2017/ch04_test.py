# coding=utf-8
# 第一個範例
x = [1, 2, 3]
y = [4, 5, 6]
print x + y

import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
print a + b

# 第二個範例
import numpy as np

a = np.array([1, 3, 2])
b = np.array([-2, 1, -1])

la = np.sqrt(a.dot(a))
lb = np.sqrt(b.dot(b))

print (la, lb)
cos_angle = a.dot(b) / (la * lb)
print (cos_angle)
angle = np.arccos(cos_angle)
print (angle)
angle2 = angle * 360 / 2 / np.pi
print (angle2)

# 第三個範例
import numpy as np

a = np.array([[3, 4], [2, 3]])
b = np.array([[1, 2], [3, 4]])
c = np.mat([[3, 4], [2, 3]])
d = np.mat([[1, 2], [3, 4]])
e = np.dot(a, b)
f = np.dot(c, d)
print (a * b)
print (c * d)
print (e)
print (f)

# 第四個範例
import numpy as np
a = np.random.randint(1, 10, (3, 5))
print (a)

# 第五個範例
from numpy import *
a = mat([[1, 2, -1], [3, 0, 1], [4, 2, 1]])
print linalg.det(a)

# 第六個範例
import numpy as np
from matplotlib import pyplot
x = np.arange(0, 10, 0.1)
y = np.sin(x)
pyplot.plot(x, y)
pyplot.show()
