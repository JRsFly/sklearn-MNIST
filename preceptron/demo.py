import numpy as np
#label =[0 for i in range(10)]
#print(label
"""a = np.arange(9).reshape((3, 3))
b = a[1,:]
print(b)
b = b /2
print(b)
print(a)
b[2] = b[2] /2
print(b)
print(a)"""

a = np.arange(9).reshape((3, 3))
print(a)
a = np.sum(a, axis=0)
print(a)


