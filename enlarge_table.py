import math
import pandas as pd 
import numpy as np
import scipy.interpolate

def unit_vector(vector):
    # Returns the unit vector of the vector. 
    return vector / np.linalg.norm(vector)

def AZ2V(A,Z):
    A = A*np.pi/180
    Z = Z*np.pi/180
    x = -np.sin(A)*np.cos(Z)
    y = np.sin(A)*np.sin(Z)
    z = -np.cos(A)
    return np.array([x, y, z])

def angle_between(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        z = 1 - (i / float(samples - 1)) * 2  # z goes from 1 to -1
        radius = math.sqrt(1 - z * z)  # radius at z
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        y = math.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points).T

#================ main ===============
#init
num_points = 41168                 #number of sample points in output table
file_name = "typical2_response"

sensor_name = ['NN','NP','NT','NB','PN','PP','PT','PB']
df = pd.read_csv("./table/" + file_name + ".csv")
Data = np.array(df)
print("============raw data START============")
print(Data)
print("============raw data END============")

tmp = AZ2V(df['theta'],df['phi']).T
print("============AZ2V START============")
print(tmp)
print("============AZ2V END============")
df['X'] = tmp[:,0]
df['Y'] = tmp[:,1]
df['Z'] = tmp[:,2]
pointss2 = fibonacci_sphere(num_points)
x, y, z = tmp[:,0:3].T
X, Y, Z = pointss2
big_table = np.array([X,Y,Z]).T
print(big_table.shape)
for ii in range(8):
    r_interp = df[sensor_name[ii]]
    print(r_interp.shape)
    interpolator = scipy.interpolate.Rbf(x, y, z, r_interp)
    r = np.array([interpolator(X, Y, Z)]).T
    print(r.shape)
    big_table = np.hstack((big_table,r))
    print(big_table.shape)
np.savetxt("./table/big_" + file_name + ".csv", big_table, delimiter=',',header='x,y,z,NN,NP,NT,NB,PN,PP,PT,PB')