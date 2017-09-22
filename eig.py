import numpy as np 
import numpy.linalg as la

a = np.array([[0, -1],[2 ,3]])
eigenValue , eigenVector = la.eig(a)

print(eigenValue)
print(eigenVector)
