import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def RE(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1],R[2,2])
        y = math.atan2(-R[2,0],sy)
        z = math.atan2(R[1,0],R[0,0])
    else:
        x = math.atan2(-R[1,2],R[1,1])
        y = math.atan2(-R[2,0],sy)
        z = 0
    return np.array([x,y,z])

cameratorobot = np.array([[-0.08822639, -0.86044569,  0.50184591], [-0.99609533,  0.07782632, -0.04167926], [-0.00319408, -0.50356358, -0.86395227]])

r = np.array([[-8.493e-01, 5.1216e-01, -1.275e-01], [5.00e-01, 7.037e-01, -5.0468e-01], [-1.687e-01, -4.924e-01, -8.538e-01]])

D = np.dot(cameratorobot, r)

c = RE(D)
print(c)

