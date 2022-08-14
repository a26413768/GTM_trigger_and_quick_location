import numpy as np
import pandas as pd 
from datetime import datetime

def V2AZ(x,y,z):
    theta = np.mod(np.arctan2(y,-x)/np.pi*180,360)-180
    phi = np.arctan2(np.sqrt(x**2 + y**2),-z)/np.pi*180-90
    return phi,theta


table = np.array(pd.read_csv("./table/" + "big_typical1_response_20220120.csv"))
total_duration = 1000       #second
BG_count_rate = 42.6       #photon/second from 2021.06.09_CountRate_v3.pdf
GRB_duration = 10         #second     long
fluence = 4E-6
GRB_start = np.random.randint(total_duration)             #second
index = np.random.randint(len(table))
source_direction = table[index,0:3]
source_phi_theta = V2AZ(source_direction[0],source_direction[1],source_direction[2])
print(source_direction)
print(source_phi_theta)

source_count_rate = (fluence/4E-4)*table[index,3:11]*245.96

sensor_data = np.array([[10000,10000]]) # just some number would be deleted later

# back ground signal
BG_count_rate_with_time = BG_count_rate*(0.95 + 0.05*np.square(np.arange(0,1,1/total_duration)))

for i in range(8):
    s = np.random.poisson(BG_count_rate_with_time, total_duration)
    for s,time in zip(s,range(0,total_duration)): 
        a = np.random.uniform(0+time,1+time,s)
        b =  np.ones(s)*(i+1)
        tmp = (np.vstack((a,b))).T
        sensor_data = np.concatenate( (sensor_data, tmp), axis=0)

#source signal
for time in range(GRB_start,GRB_start+GRB_duration):
    for i in range(8):
        num = np.random.poisson((source_count_rate[i]))
        a = np.random.uniform(time, time+1, num)
        b =  np.ones(num)*(i+1)
        tmp = (np.vstack((a,b))).T
        sensor_data = np.concatenate( (sensor_data, tmp), axis=0)

#other info
sensor_data = np.delete(sensor_data, (0), axis=0)
sensor_data = sensor_data[np.argsort(sensor_data[:, 0])]
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S.%f")
result = pd.DataFrame(sensor_data)
result[1] = result[1].astype(int)
# print(result[1])
result["pixel"] = np.random.uniform(0,15,len(sensor_data)).astype(int)
result["energy"] = np.random.uniform(50,300,len(sensor_data))
print(date_time)


result2 = pd.DataFrame()
result2["time"] = np.linspace(0,1000,300)
result2["qw"] = np.ones(300)*np.sqrt(0.5)
result2["qx"] = np.ones(300)*np.sqrt(0.5)
result2["qy"] = np.zeros(300)
result2["qz"] = np.zeros(300)
result2["ECIx"] = np.ones(300)*-12.32691165665816
result2["ECIy"] = np.ones(300)*4215.534296188868
result2["ECIz"] = np.ones(300)*5318.530945682411

header_str = ['time','detector','pixel','energy',]
location_str = ",S%.2f,ES%.2f,XS%.2f,YS%.2f,ZS%.2f,PS%.2f,TS%.2f"\
    % (GRB_start,GRB_start+GRB_duration,source_direction[0],source_direction[1],source_direction[2],source_phi_theta[0],source_phi_theta[1])
result.to_csv('sample.csv',index = False,header = header_str)

header_str = ['time','qw','qx','qy','qz','ECIx','ECIy','ECIz']
location_str = ",S%.2f,ES%.2f,XS%.2f,YS%.2f,ZS%.2f,PS%.2f,TS%.2f"\
    % (GRB_start,GRB_start+GRB_duration,source_direction[0],source_direction[1],source_direction[2],source_phi_theta[0],source_phi_theta[1])
result2.to_csv('sample_sc.csv',index = False,header = header_str)

print('Done.')