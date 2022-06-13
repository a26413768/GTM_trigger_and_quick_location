import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from astropy.coordinates import get_sun, SkyCoord
from astropy import units as u
from pyquaternion import Quaternion
from skyfield.api import Distance, load, wgs84
from skyfield.positionlib import Geocentric
from datetime import datetime
from astropy.time import Time
import os


def unit_vector(vector):
    # Returns the unit vector of the vector. 
    return vector / np.linalg.norm(vector)

def V2AZ(x,y,z):
    theta = np.mod(np.arctan2(y,-x)/np.pi*180,360)-180
    phi = np.arctan2(np.sqrt(x**2 + y**2),-z)/np.pi*180-90
    return theta,phi

def AZ2V(A,Z):
    A = A*np.pi/180
    Z = Z*np.pi/180
    x = -np.sin(A)*np.cos(Z)
    y = np.sin(A)*np.sin(Z)
    z = -np.cos(A)
    return np.array([x, y, z])

def xyz2RD(x,y,z):
    RA = np.mod(np.arctan2(y,x)/np.pi*180,360)
    Dec = np.arctan2(z,np.sqrt(x**2 + y**2))/np.pi*180
    return RA,Dec

def RD2xyz(Ra,Dec):
    x = np.cos(Dec)*np.cos(Ra)
    y = np.cos(Dec)*np.sin(Ra)
    z = np.sin(Dec)
    return np.array([x, y, z])

def angle_between(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
#================================================= main ===============================================
#init

table_name = ['middle','hard','soft']
table1 = np.array(pd.read_csv("./table/" + "big_typical1_response_20220120.csv")) #file name of middle energy big table
table2 = np.array(pd.read_csv("./table/" + "big_typical2_response_20220120.csv")) #file name of hard energy big table
table3 = np.array(pd.read_csv("./table/" + "big_typical3_response_20220120.csv")) #file name of soft energy big table

filename = "result_sample"
trigger_data = pd.read_csv("./output/" + filename + ".csv")

if not os.path.exists("./location_output/"):
    os.makedirs("./location_output/")

bkg_count = np.array(trigger_data.loc[0,['NN_BG_count','NP_BG_count','NT_BG_count','NB_BG_count','PN_BG_count','PP_BG_count','PT_BG_count','PB_BG_count']])
print(bkg_count)
source_count = np.array(trigger_data.loc[0,['NN_source_count','NP_source_count','NT_source_count','NB_source_count','PN_source_count','PP_source_count','PT_source_count','PB_source_count']])
print(source_count)
total_count = bkg_count + source_count
#==================================== calculate chi square distribution =================================
bkg_count_tmp,tmp = np.meshgrid(bkg_count,np.arange(0,len(table1)))
source_count_tmp,tmp = np.meshgrid(source_count,np.arange(0,len(table1)))
total_count_tmp,tmp = np.meshgrid(total_count,np.arange(0,len(table1)))

N1 = np.sum(table1[:,3:11]*source_count_tmp/total_count_tmp,1) / np.sum( table1[:,3:11]**2 / total_count_tmp ,1)
Chi_square1 = np.sum(source_count_tmp**2/total_count_tmp,1) \
    - 2*np.sum(table1[:,3:11]*source_count_tmp/total_count_tmp,1)*N1 \
    + np.sum( table1[:,3:11]**2 / total_count_tmp ,1)*N1**2

N2 = np.sum(table2[:,3:11]*source_count_tmp/total_count_tmp,1) / np.sum( table2[:,3:11]**2 / total_count_tmp ,1)
Chi_square2 = np.sum(source_count_tmp**2/total_count_tmp,1) \
    - 2*np.sum(table2[:,3:11]*source_count_tmp/total_count_tmp,1)*N2 \
    + np.sum( table2[:,3:11]**2 / total_count_tmp ,1)*N2**2

N3 = np.sum(table3[:,3:11]*source_count_tmp/total_count_tmp,1) / np.sum( table3[:,3:11]**2 / total_count_tmp ,1)
Chi_square3 = np.sum(source_count_tmp**2/total_count_tmp,1) \
    - 2*np.sum(table3[:,3:11]*source_count_tmp/total_count_tmp,1)*N3 \
    + np.sum( table3[:,3:11]**2 / total_count_tmp ,1)*N3**2

print(np.array([np.min(Chi_square1),np.min(Chi_square2),np.min(Chi_square3)]))
which_table = np.argmin(np.array([np.min(Chi_square1),np.min(Chi_square2),np.min(Chi_square3)]))
if which_table==0:
    index_min_chi = np.argmin(Chi_square1)
    Chi_square = Chi_square1
    location_xyz_sc = table1[index_min_chi,0:3]
elif which_table==1:
    index_min_chi = np.argmin(Chi_square2)
    Chi_square = Chi_square2
    location_xyz_sc = table2[index_min_chi,0:3]
elif which_table==2:
    index_min_chi = np.argmin(Chi_square3)
    Chi_square = Chi_square3
    location_xyz_sc = table3[index_min_chi,0:3]

location_theta, location_phi  =  V2AZ(location_xyz_sc[0],location_xyz_sc[1],location_xyz_sc[2])

print("location_xyz_sc: " + str(location_xyz_sc))
print("location_theta_phi: " + str((location_theta,location_phi)))
print("which_table: " + str(which_table+1))

q_sc = Quaternion(trigger_data['qw'],trigger_data['qx'],trigger_data['qy'],trigger_data['qz'])
# q_sc = Quaternion(0,1,0,0)
q_location_sc = Quaternion(0,location_xyz_sc[0],location_xyz_sc[1],location_xyz_sc[2])
q_location_sky = q_sc*q_location_sc*q_sc.inverse
location_xyz_sky = [q_location_sky[1],q_location_sky[2],q_location_sky[3]]
print("location_xyz_sky: " + str(location_xyz_sky))

location_RA, location_Dec  =  xyz2RD(location_xyz_sky[0],location_xyz_sky[1],location_xyz_sky[2])
print("location_RA_Dec: " + str((location_RA,location_Dec)))

#================== position of earth and sun and milkyway =================
# earth horizon
start_time_obj = datetime.strptime(trigger_data['Trigger_time_UTC'][0], '%m_%d_%Y_%H_%M_%S.%f')
UTC = [start_time_obj.year, start_time_obj.month, start_time_obj.day, start_time_obj.hour, start_time_obj.minute, start_time_obj.second]
ts = load.timescale()
t = ts.utc(UTC[0],UTC[1],UTC[2],UTC[3],UTC[4],UTC[5])
d = Distance(m=[trigger_data['ECIx']*1000,trigger_data['ECIy']*1000,trigger_data['ECIz']*1000]) # "*1000" for km to m
p = Geocentric(d.au, t=t)
g = wgs84.subpoint(p)
sc_height = g.elevation.m
earth_R = np.sqrt(np.sum([(trigger_data['ECIx']*1000)**2,(trigger_data['ECIy']*1000)**2,(trigger_data['ECIz']*1000)**2]))
print('earth_R= ' +str(earth_R))
earth_V = np.array([-trigger_data['ECIx'],-trigger_data['ECIy'],-trigger_data['ECIz']])
earth_V = earth_V/np.linalg.norm(earth_V)
earth_RA, earth_Dec = xyz2RD(earth_V[0],earth_V[1],earth_V[2])
print('earth ECI: '+str((earth_V[0],earth_V[1],earth_V[2])))
print('earth RA Dec: '+str((earth_RA, earth_Dec)))
earth_r = 90-np.arccos(earth_R/(earth_R+sc_height))/np.pi*180
t_angle = np.arange(0,np.pi*2,np.pi/100)
horizon_x =  (np.sin(earth_r)*np.cos(-earth_Dec/180*np.pi+np.pi*0.5)*np.cos(earth_RA/180*np.pi))*np.cos(t_angle)\
            -(np.sin(earth_r)*np.sin( earth_RA /180*np.pi))*np.sin(t_angle)\
            +(np.cos(earth_r)*np.sin(-earth_Dec/180*np.pi+np.pi*0.5)*np.cos(earth_RA/180*np.pi))
horizon_y =  (np.sin(earth_r)*np.cos(-earth_Dec/180*np.pi+np.pi*0.5)*np.sin(earth_RA/180*np.pi))*np.cos(t_angle)\
            -(np.sin(earth_r)*np.cos( earth_RA /180*np.pi))*np.sin(t_angle)\
            +(np.cos(earth_r)*np.sin(-earth_Dec/180*np.pi+np.pi*0.5)*np.sin(earth_RA/180*np.pi))
horizon_z = -(np.sin(earth_r)*np.sin(-earth_Dec/180*np.pi+np.pi*0.5))*np.cos(t_angle)\
            + np.cos(earth_r)*np.cos(-earth_Dec/180*np.pi+np.pi*0.5)
horizon_RA = np.array([])
horizon_Dec = np.array([])
for ii in range(len(horizon_x)):
    horizon_RA_, horizon_Dec_ = xyz2RD(horizon_x[ii],horizon_y[ii],horizon_z[ii])
    horizon_RA = np.append(horizon_RA,horizon_RA_)
    horizon_Dec = np.append(horizon_Dec,horizon_Dec_)

if np.count_nonzero(np.abs(np.diff(horizon_RA))>300)==1:
    is_horizon_split = False
    horizon_Dec = horizon_Dec[np.argsort(horizon_RA)]
    horizon_RA  = horizon_RA[np.argsort(horizon_RA)]
    horizon_RA = np.append(horizon_RA,np.ones(50)*359.99)
    horizon_RA = np.append(horizon_RA,np.ones(50)*0.01)
    if earth_Dec<0:
        horizon_Dec = np.append(horizon_Dec,np.linspace(horizon_Dec[-1],-90,num=50))
        horizon_Dec = np.append(horizon_Dec,np.linspace(-90,horizon_Dec[0],num=50))
    else:
        horizon_Dec = np.append(horizon_Dec,np.linspace(horizon_Dec[-1],90,num=50))
        horizon_Dec = np.append(horizon_Dec,np.linspace(90,horizon_Dec[0],num=50))
elif np.count_nonzero(np.abs(np.diff(horizon_RA))>300)==2:
    is_horizon_split = True
    print( np.where(np.abs(np.diff(horizon_RA))>300) )
    split_index_1 = np.where(np.abs(np.diff(horizon_RA))>300)[0][0]
    split_index_2 = np.where(np.abs(np.diff(horizon_RA))>300)[0][1]
    horizon_RA_1  = np.roll(horizon_RA,-split_index_1-1)[0:(split_index_2-split_index_1)]
    horizon_RA_2  = np.roll(horizon_RA,-split_index_2-1)[0:len(horizon_RA)-(split_index_2-split_index_1)]
    horizon_Dec_1 = np.roll(horizon_Dec,-split_index_1-1)[0:(split_index_2-split_index_1)]
    horizon_Dec_2 = np.roll(horizon_Dec,-split_index_2-1)[0:len(horizon_RA)-(split_index_2-split_index_1)]
    horizon_RA_1  = np.append(horizon_RA_1,np.ones(50)*359.99)
    horizon_RA_2  = np.append(horizon_RA_2,np.ones(50)*0.01)
    horizon_Dec_1 = np.append(horizon_Dec_1,np.linspace(horizon_Dec_1[-1],horizon_Dec_1[0],num=50))
    horizon_Dec_2 = np.append(horizon_Dec_2,np.linspace(horizon_Dec_2[-1],horizon_Dec_1[0],num=50))
else:
    is_horizon_split = False
    
# sun
trigger_time = Time(start_time_obj, scale='utc')
coords = get_sun(trigger_time)
sun_RA = coords.ra.degree
sun_Dec = coords.dec.degree
print("sun_RA_Dec " + str((sun_RA, sun_Dec)))

# milkyway
lon = np.linspace(0, 360, 100)
lat = np.zeros(100)
ecl = SkyCoord(lon, lat, unit=u.deg, frame='galactic')
ecl_gal = ecl.transform_to('icrs')
milkyway_RA, milkyway_Dec = ecl_gal.ra.degree, ecl_gal.dec.degree
milkyway_RA_C = milkyway_RA[0]
milkyway_Dec_C = milkyway_Dec[0]
milkyway_Dec = milkyway_Dec[np.argsort(milkyway_RA)]
milkyway_RA  = milkyway_RA[np.argsort(milkyway_RA)]

#======================init plot parameter===============
colormap_RGB = np.ones((256, 4))
colormap_RGB[:, 0] = np.linspace(138/256, 1, 256)
colormap_RGB[:, 1] = np.linspace(43/256, 1, 256)
colormap_RGB[:, 2] = np.linspace(226/256, 1, 256)
purple = ListedColormap(colormap_RGB)

RA = np.array([])
Dec = np.array([])
for ii in range(len(table1)):
    q_table = Quaternion(0, table1[ii,0], table1[ii,1], table1[ii,2])
    q_table_sky = q_sc*q_table*q_sc.inverse
    RA_, Dec_ = xyz2RD(q_table_sky[1],q_table_sky[2],q_table_sky[3])
    RA = np.append(RA,RA_)
    Dec = np.append(Dec,Dec_)

phi, theta = V2AZ(table1[:,0],table1[:,1],table1[:,2])

color_map_max = np.min(Chi_square)+9.6
color_map_min = np.min(Chi_square)

#==================== plot all sky map =======================
plt.figure(figsize=(16,8))
ax1 = plt.subplot(111, projection="mollweide")
plot_index = Chi_square<color_map_max
#earth
plt.scatter(-earth_RA/180*np.pi+np.pi, earth_Dec/180*np.pi, marker='x', c = 'tab:blue')
if is_horizon_split==True:
    plt.plot(-horizon_RA_1/180*np.pi+np.pi, horizon_Dec_1/180*np.pi, c = 'tab:blue', alpha = 0.3)
    plt.fill(-horizon_RA_1/180*np.pi+np.pi, horizon_Dec_1/180*np.pi, c = 'cyan', alpha = 0.3)
    plt.plot(-horizon_RA_2/180*np.pi+np.pi, horizon_Dec_2/180*np.pi, c = 'tab:blue', alpha = 0.3)
    plt.fill(-horizon_RA_2/180*np.pi+np.pi, horizon_Dec_2/180*np.pi, c = 'cyan', alpha = 0.3)
    print('test')
else:
    plt.plot(-horizon_RA/180*np.pi+np.pi, horizon_Dec/180*np.pi, c = 'tab:blue', alpha = 0.3)
    plt.fill(-horizon_RA/180*np.pi+np.pi, horizon_Dec/180*np.pi, c = 'cyan', alpha = 0.3)
#sun
plt.scatter(-sun_RA/180*np.pi+np.pi, sun_Dec/180*np.pi, marker='o', s = 100, c = 'gold')
#milkyway
plt.plot(-milkyway_RA/180*np.pi+np.pi,milkyway_Dec/180*np.pi,c = 'lightgray', lw=3.5, alpha = 0.3)
plt.plot(-milkyway_RA/180*np.pi+np.pi,milkyway_Dec/180*np.pi,c = 'gray', lw=0.5, alpha = 0.5)
plt.scatter(-milkyway_RA_C/180*np.pi+np.pi,milkyway_Dec_C/180*np.pi, c='gray', marker='o', s=100, alpha = 0.3,zorder=1)
plt.scatter(-milkyway_RA_C/180*np.pi+np.pi,milkyway_Dec_C/180*np.pi, c='k', marker='o', s=20, alpha = 0.5,zorder=1)
# location
plt.scatter(-RA[plot_index]/180*np.pi+np.pi, Dec[plot_index]/180*np.pi, marker='o', s=20, c=Chi_square[plot_index], cmap=purple, vmin=color_map_min, vmax=color_map_max, edgecolors='none')
plt.colorbar(fraction=0.05, orientation='horizontal', boundaries=np.linspace(color_map_min,color_map_max,100), ticks=np.linspace(color_map_min,color_map_max,5)).set_label(r'chi square', size='large')
plt.scatter(-location_RA/180*np.pi+np.pi, location_Dec/180*np.pi, marker='x', c='k')
cs = ax1.tricontour(-RA/180*np.pi+np.pi, Dec/180*np.pi, np.array(Chi_square, dtype = 'float'), levels = [2.3, 4.6, 9.6]+color_map_min, colors = 'k', linewidths=1)


yticks = np.array([-75,-60,-45,-30,-15,0,15,30,45,60,75])*np.pi/180.0
yticklabels = [r'-75$^{\circ}$',r'-60$^{\circ}$',r'-45$^{\circ}$',r'-30$^{\circ}$',r'-15$^{\circ}$',r'0$^{\circ}$',r'15$^{\circ}$',r'30$^{\circ}$',r'45$^{\circ}$',r'60$^{\circ}$',r'75$^{\circ}$']
plt.yticks(yticks,yticklabels,fontsize=12)
xticks = np.array([-150,-120,-90,-60,-30,0,30,60,90,120,150])*np.pi/180.0
xticklabels = [r'330$^{\circ}$',r'300$^{\circ}$',r'270$^{\circ}$',r'240$^{\circ}$',r'210$^{\circ}$',r'180$^{\circ}$',r'150$^{\circ}$',r'120$^{\circ}$',r'90$^{\circ}$',r'60$^{\circ}$',r'30$^{\circ}$']
plt.xticks(xticks,xticklabels,fontsize=12)
plt.xlabel('RA(J2000.0)')
plt.ylabel('Dec(J2000.0)')
plt.title('all sky map', y=1.05, size='large')
plt.grid(True)
plt.savefig("./location_output/allsykmap.png", dpi=300)
#============================ plot zoom in sky map ======================
plt.figure(figsize=(16,8))
if location_Dec>70 or location_Dec<-70:
    ax2 = plt.subplot(111, projection="polar")
else:
    ax2 = plt.subplot(111)
cs2 = ax2.tricontour(RA, Dec, np.array(Chi_square, dtype = 'float'), levels = [2.3, 4.6, 9.6]+color_map_min, colors = 'gray', linewidths=1)
plt.scatter(location_RA, location_Dec, marker='x', c = 'k')
ax2.set_xlim((np.min((np.array(cs2.collections[2].get_paths()[0].to_polygons())[0][:,0]))), np.max(((cs2.collections[2].get_paths()[0].to_polygons())[0][:,0])))
ax2.set_ylim((np.min(((cs2.collections[2].get_paths()[0].to_polygons())[0][:,1]))), np.max(((cs2.collections[2].get_paths()[0].to_polygons())[0][:,1])))
plt.xlabel('RA(J2000.0) degrees')
plt.ylabel('Dec(J2000.0) degrees')
plt.title('zoom in sky map', y=1.05, size='large')
plt.grid(True)
plt.savefig("./location_output/zoominsykmap.png", dpi=300)

#============================ result CSV ============================
contour_RA  = -np.array(cs.collections[0].get_paths()[0].to_polygons())[0][:,0]+np.pi
contour_Dec = np.array(cs.collections[0].get_paths()[0].to_polygons())[0][:,1]
contour_x, contour_y, contour_z = RD2xyz(contour_RA, contour_Dec)
coutour_dist = []
for contour_x_, contour_y_, contour_z_ in zip(contour_x, contour_y, contour_z):
    coutour_dist.append(angle_between(location_xyz_sky,[contour_x_, contour_y_, contour_z_])/np.pi*180)
One_Sigma_Radius = np.mean(coutour_dist)

best_fit_table = table_name[which_table]

result = np.array([
    location_RA,
    location_Dec,
    best_fit_table,
    One_Sigma_Radius
])
print(result)
header_str = ['RA','Dec','Best_fit_table','One_sigma_radius']
pd.DataFrame(result.reshape(-1, len(result))).to_csv("./location_output/result_"+ filename+ ".csv",index = False,header = header_str)
