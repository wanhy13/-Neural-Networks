import preprocess
import numpy as np
import forward_pass
x1 = np.array([[2,2,3,4,5],
                 [1,3,3,4,5],
                 [1,2,4,4,5],
                 [1,2,3,5,5]])
x2 = np.array([[1,2,3,4,5],
                 [1,2,3,4,5],
                 [1,2,3,4,5],
                 [1,2,3,4,5]])
wst = np.array([1,20,20,20,13])
# px1 , px2,u,m = preprocess.preprocess(x1,x2)
entry = np.cumsum(wst[0:-1] * wst[1:] + wst[0:-1])
print(entry)
# print(px1,px2,u,m)