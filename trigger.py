import numpy as np
import pandas as pd 
from scipy.stats import poisson
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import datetime
from datetime import timedelta
from astropy.io import fits
import os

def fit_BG(x,y,test_x):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x[..., None])
    y_train = y_scaler.fit_transform(y[..., None])
    # fit model
    # model = make_pipeline(PolynomialFeatures(2), HuberRegressor(epsilon=1))
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(x_train, y_train.ravel())
    # do some predictions
    predictions = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(test_x[..., None]))
    )
    return predictions

filename = "sample"  # the level 1 file name 
filename_sc =  "sample-sc" 
min_hits_intervel = 2e-6
Bin_size_array = [0.001, 0.002, 0.005 ,0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
number_of_detector = 8
sensor_name = ['NN','NP','NT','NB','PN','PP','PT','PB','P','N']

if not os.path.exists("./output/"):
    os.makedirs("./output/")

df = pd.read_csv("./input/" + filename + '.csv', skiprows=[0])
print('=================================== Date Read ==================================')
print(filename)
print(df)
print('================================ END of Date Read ==============================')
df_time = pd.read_csv("./input/" + filename + '.csv', nrows=1)
start_time_UTC_str = df_time.columns[1]
print(start_time_UTC_str)
start_time_obj = datetime.strptime(start_time_UTC_str, '%m_%d_%Y_%H_%M_%S.%f')
print('start_time: ' + str(start_time_obj))


#======================== combine hits within 2 mirco sec and filter energy 50-300 ========================
sensor_data = []
sector = df.groupby('detector')
for i in range(1,number_of_detector+1):
    raw_hit_time = np.array(sector.get_group(i)['time'])
    hits_intervel = np.diff(raw_hit_time)
    hits_intervel = np.insert(hits_intervel,0,10)  #10 is just a number bigger than min_hits_intervel
    raw_hit_time = np.where(hits_intervel > min_hits_intervel, raw_hit_time, -1)
    combined_energy = np.array(sector.get_group(i)['energy'])
    for j in range(len(combined_energy)):
        if raw_hit_time[j]==-1:
            combined_energy[least_non_combined_bin] += combined_energy[j]
            combined_energy[j] = -1
        else:
            least_non_combined_bin = j
    combined_filtered_hit_time = raw_hit_time[np.logical_and(np.logical_and((raw_hit_time!=-1),(combined_energy<=300)),(combined_energy>=50))]
    sensor_data.append(combined_filtered_hit_time)
sensor_data.append(np.concatenate((sensor_data[4],sensor_data[5],sensor_data[6],sensor_data[7])))
sensor_data.append(np.concatenate((sensor_data[0],sensor_data[1],sensor_data[2],sensor_data[3])))
print(sensor_data)


# =================== bin and trigger ===================
total_duration = np.max(df['time'])
trigger_bin = np.array([[0,0,0,0]])
for is_shift in range(2):
    for bin_size in Bin_size_array:
        shifter = is_shift*bin_size*(1/2)
        # print(shifter)
        for i in range(len(sensor_data)):
            R = np.array([])
            hist, bin_edges = np.histogram(sensor_data[i], bins=np.arange(shifter, total_duration+bin_size+shifter, bin_size) , density=False)
            if np.mean(hist)>3:
                background = fit_BG(np.arange(shifter, total_duration+shifter, bin_size),hist,np.arange(shifter, total_duration+shifter, bin_size))
                threshold = np.nan_to_num(poisson.ppf(1-0.001/(total_duration/bin_size),background))+1
            else:
                background = np.ones(len(hist))*np.mean(hist)
                threshold = poisson.ppf(1-0.001/(total_duration/bin_size),background)+1
            trigger_index = np.nonzero(hist>threshold)
            trigger_bin_count = hist[trigger_index]
            for trigger_single_bin_count,index in zip(trigger_bin_count,trigger_index):
                R = np.append(R,poisson.sf(trigger_single_bin_count,background[index]))
            trigger = (trigger_index[0]+is_shift*(0.5))*bin_size
            print("T:"+str(trigger))
            print("R:"+str(R))
            if len(trigger) != 0:
                tmp2 = (np.vstack((trigger, bin_size*np.ones(len(trigger)), (i+1)*np.ones(len(trigger)).astype(int),R))).T
                trigger_bin = np.concatenate( (trigger_bin,tmp2) ,axis=0) 
trigger_bin = np.delete(trigger_bin, (0), axis=0)
print('=====================================trigger====================================')
print('time, time bin, dedector, R')
print(trigger_bin)
print('==================================END OF trigger================================')


if len(trigger_bin) == 0:     #if NO trigger
    #============================== plot 2sec ================================
    bin_size = 2
    fig , ax = plt.subplots()
    for i in range(len(sensor_data)):
        hist, bin_edges = np.histogram(sensor_data[i], bins=np.arange(0, total_duration+bin_size, bin_size) , density=False)
        plt.subplot(2, 5, i+1)
        plt.plot(np.arange(0, total_duration, bin_size),hist)
        plt.title(sensor_name[i])
        plt.xlabel('time')
        plt.ylabel('count')
    fig.suptitle('2s bin light curve')
    
    #============================== plot 0.1msec ================================
    bin_size = 0.1
    fig , ax = plt.subplots()
    for i in range(len(sensor_data)):
        hist, bin_edges = np.histogram(sensor_data[i], bins=np.arange(0, total_duration+bin_size, bin_size) , density=False)
        plt.subplot(2, 5, i+1)
        plt.plot(np.arange(0, total_duration, bin_size),hist)
        plt.title(sensor_name[i])
        plt.xlabel('time')
        plt.ylabel('count')
    fig.suptitle('100ms bin light curve')

else:  #if trigger 
    #============================== plot min bin ================================
    min_trigger_bin = np.min(trigger_bin[:,1])
    backward_time = 30
    forward_time = 30
    forward_gap = 10
    first_trigger = np.min(trigger_bin[:,0])
    last_trigger = np.max(trigger_bin[:,0])+ trigger_bin[np.argmax(trigger_bin[:,0]),1]
    window_front = first_trigger - backward_time
    window_back =  last_trigger + forward_time
    print('first_trigger = ' + str(window_front+backward_time))
    print('last_trigger = ' + str(window_back-forward_time))
    window_duration = window_back - window_front
    sensor_data_for_plot = [sensor_data[0][np.logical_and(sensor_data[0]>window_front,sensor_data[0]<window_back)],
                            sensor_data[1][np.logical_and(sensor_data[1]>window_front,sensor_data[1]<window_back)],
                            sensor_data[2][np.logical_and(sensor_data[2]>window_front,sensor_data[2]<window_back)],
                            sensor_data[3][np.logical_and(sensor_data[3]>window_front,sensor_data[3]<window_back)],
                            sensor_data[4][np.logical_and(sensor_data[4]>window_front,sensor_data[4]<window_back)],
                            sensor_data[5][np.logical_and(sensor_data[5]>window_front,sensor_data[5]<window_back)],
                            sensor_data[6][np.logical_and(sensor_data[6]>window_front,sensor_data[6]<window_back)],
                            sensor_data[7][np.logical_and(sensor_data[7]>window_front,sensor_data[7]<window_back)],
                            sensor_data[8][np.logical_and(sensor_data[8]>window_front,sensor_data[8]<window_back)],
                            sensor_data[9][np.logical_and(sensor_data[9]>window_front,sensor_data[9]<window_back)],
                            ]

    fig , ax = plt.subplots(figsize=(16, 8))
    ii = 0
    for i in [0,1,2,3,9,4,5,6,7,8]:
        hist, bin_edges = np.histogram(sensor_data_for_plot[i], bins=np.arange(window_front, window_front+window_duration, min_trigger_bin) , density=False)
        plt.subplot(2, 5, ii+1)
        ii+=1
        plt.step(np.arange(-backward_time,backward_time+window_duration,min_trigger_bin)[0:len(hist)],hist/min_trigger_bin)
        plt.vlines(x = 0,ymin = np.min(hist/min_trigger_bin),ymax = np.max(hist/min_trigger_bin),color='r',linestyles = 'dashed')
        plt.title(sensor_name[i])
        plt.xlabel('time')
        plt.ylabel('count rate(count/second)')
    fig.suptitle('min bin ('+str(min_trigger_bin)+'s) light curve')
    plt.tight_layout()
    fig.savefig('./output/'+'min bin ('+str(min_trigger_bin)+'s) light curve.png', dpi=300)
    #================================== plot 0.1s bin ================================
    plot_bin = 0.1
    fig , ax = plt.subplots(figsize=(16, 8))
    ii = 0
    for i in [0,1,2,3,9,4,5,6,7,8]:
        hist, bin_edges = np.histogram(sensor_data_for_plot[i], bins=np.arange(window_front, window_front+window_duration, plot_bin) , density=False)
        plt.subplot(2, 5, ii+1)
        ii+=1
        plt.step(np.arange(-backward_time,backward_time+window_duration,plot_bin)[0:len(hist)],hist/plot_bin)
        plt.vlines(x = 0,ymin = np.min(hist/plot_bin),ymax = np.max(hist/plot_bin),color='r',linestyles = 'dashed')
        plt.title(sensor_name[i])
        plt.xlabel('time')
        plt.ylabel('count rate(count/second)')
    fig.suptitle(str(plot_bin)+'(s)bin light curve')
    plt.tight_layout()
    fig.savefig('./output/'+str(plot_bin)+'(s)bin light curve.png', dpi=300)

    #=========================================== T50 T90 =========================================
    fig = plt.figure()
    MAX_R_detector = int(trigger_bin[np.argmin(trigger_bin[:,3]),2])-1
    print('MAX_R_detector'+str(MAX_R_detector))

    hist, bin_edges = np.histogram(sensor_data[MAX_R_detector], bins=np.arange(0, total_duration+min_trigger_bin, min_trigger_bin) , density=False)
    hist_for_BG_fit = np.concatenate((hist[int(window_front/min_trigger_bin):int(first_trigger/min_trigger_bin)], hist[int(last_trigger/min_trigger_bin):int(window_back/min_trigger_bin)]))
    hist_for_window = hist[int(window_front/min_trigger_bin):int(window_back/min_trigger_bin)]
    time_for_BG_fit = np.concatenate((np.arange(window_front, first_trigger, min_trigger_bin),np.arange(last_trigger, window_back, min_trigger_bin)))
    time_for_window = np.arange(window_front, window_front+window_duration, min_trigger_bin)[:len(hist_for_window)]

    background = fit_BG(time_for_BG_fit,hist_for_BG_fit,time_for_window)
    hist_source = hist_for_window - background
    hist_cum = np.cumsum(hist_source)
    zero_point = np.mean(hist_cum[:int(backward_time/min_trigger_bin)])
    hist_cum = hist_cum - zero_point
    total_flux = np.mean(hist_cum[int(len(hist_cum)-(forward_time-forward_gap)/min_trigger_bin):])
    print(total_flux)

    #result output
    plt.step(np.arange(-backward_time, window_duration, min_trigger_bin)[0:len(hist_cum)],hist_cum,where='post')
    T25 = (np.where(np.diff(np.sign(hist_cum - total_flux*0.25)))[0][-1]+1)*min_trigger_bin + window_front
    T75 = (np.where(np.diff(np.sign(hist_cum - total_flux*0.75)))[0][0] +1)*min_trigger_bin + window_front
    T05 = (np.where(np.diff(np.sign(hist_cum - total_flux*0.05)))[0][-1]+1)*min_trigger_bin + window_front
    T95 = (np.where(np.diff(np.sign(hist_cum - total_flux*0.95)))[0][0] +1)*min_trigger_bin + window_front
    T05_to_trigger = T05-first_trigger
    T25_to_trigger = T25-first_trigger
    T75_to_trigger = T75-first_trigger
    T95_to_trigger = T95-first_trigger
    T90 = T95-T05
    T50 = T75-T25
    plt.vlines(x = T05_to_trigger,ymin = 0,ymax = np.max(hist_cum),color='r',linestyles = 'dashed')
    plt.vlines(x = T25_to_trigger,ymin = 0,ymax = np.max(hist_cum),color='k',linestyles = 'dashed')
    plt.vlines(x = T75_to_trigger,ymin = 0,ymax = np.max(hist_cum),color='k',linestyles = 'dashed')
    plt.vlines(x = T95_to_trigger,ymin = 0,ymax = np.max(hist_cum),color='r',linestyles = 'dashed')
    print('T05: '+str(T05_to_trigger))
    print('T25: '+str(T25_to_trigger))
    print('T75: '+str(T75_to_trigger))
    print('T95: '+str(T95_to_trigger))

    plt.xlabel('time')
    plt.ylabel('count')
    fig.suptitle('cumulative light cruve ('+str(min_trigger_bin) +'s), detector '+sensor_name[MAX_R_detector])
    fig.savefig('./output/'+'cumulative light curve ('+str(min_trigger_bin)+'s), detector '+sensor_name[MAX_R_detector]+'.png')

    trigger_time_UTC = start_time_obj + timedelta(seconds = np.floor(first_trigger),microseconds = np.mod(first_trigger, 1))
    Min_trigger_bin = min_trigger_bin
    background_for_each_detector = []
    hist_source_for_each_detector = []
    for ii in range(8):
        hist, bin_edges = np.histogram(sensor_data[ii], bins=np.arange(0, total_duration+min_trigger_bin, min_trigger_bin) , density=False)
        hist_for_BG_fit = np.concatenate((hist[int(window_front/min_trigger_bin):int(first_trigger/min_trigger_bin)], hist[int(last_trigger/min_trigger_bin):int(window_back/min_trigger_bin)]))
        hist_for_window = hist[int(window_front/min_trigger_bin):int(window_back/min_trigger_bin)]
        time_for_BG_fit = np.concatenate((np.arange(window_front, first_trigger, min_trigger_bin),np.arange(last_trigger, window_back, min_trigger_bin)))
        time_for_window = np.arange(window_front, window_front+window_duration, min_trigger_bin)[:len(hist_for_window)]

        background = fit_BG(time_for_BG_fit,hist_for_BG_fit,time_for_window)
        background_for_each_detector.append(background)
        hist_source_for_each_detector.append(hist_for_window - background)

    if T90<10:
        NN_source_count = np.sum(hist_source_for_each_detector[0][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        NP_source_count = np.sum(hist_source_for_each_detector[1][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        NT_source_count = np.sum(hist_source_for_each_detector[2][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        NB_source_count = np.sum(hist_source_for_each_detector[3][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        PN_source_count = np.sum(hist_source_for_each_detector[4][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        PP_source_count = np.sum(hist_source_for_each_detector[5][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        PT_source_count = np.sum(hist_source_for_each_detector[6][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        PB_source_count = np.sum(hist_source_for_each_detector[7][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        
        NN_BG_count = np.sum(background_for_each_detector[0][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        NP_BG_count = np.sum(background_for_each_detector[1][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        NT_BG_count = np.sum(background_for_each_detector[2][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        NB_BG_count = np.sum(background_for_each_detector[3][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        PN_BG_count = np.sum(background_for_each_detector[4][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        PP_BG_count = np.sum(background_for_each_detector[5][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        PT_BG_count = np.sum(background_for_each_detector[6][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
        PB_BG_count = np.sum(background_for_each_detector[7][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T95_to_trigger+backward_time)/min_trigger_bin)+1])
    else:
        NN_source_count = np.sum(hist_source_for_each_detector[0][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        NP_source_count = np.sum(hist_source_for_each_detector[1][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        NT_source_count = np.sum(hist_source_for_each_detector[2][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        NB_source_count = np.sum(hist_source_for_each_detector[3][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        PN_source_count = np.sum(hist_source_for_each_detector[4][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        PP_source_count = np.sum(hist_source_for_each_detector[5][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        PT_source_count = np.sum(hist_source_for_each_detector[6][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        PB_source_count = np.sum(hist_source_for_each_detector[7][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        
        NN_BG_count = np.sum(background_for_each_detector[0][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        NP_BG_count = np.sum(background_for_each_detector[1][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        NT_BG_count = np.sum(background_for_each_detector[2][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        NB_BG_count = np.sum(background_for_each_detector[3][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        PN_BG_count = np.sum(background_for_each_detector[4][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        PP_BG_count = np.sum(background_for_each_detector[5][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        PT_BG_count = np.sum(background_for_each_detector[6][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
        PB_BG_count = np.sum(background_for_each_detector[7][int((T05_to_trigger+backward_time)/min_trigger_bin):int((T05_to_trigger+backward_time+10)/min_trigger_bin)+1])
    
    df2 = pd.read_csv("./input/" + filename_sc + '.csv',skiprows=[0])
    trigger_mid_time = ((T25+T75)/2)
    trigger_mid_time_index = (np.abs(df2['time'] - trigger_mid_time)).argmin()
    qx = df2['qx'][trigger_mid_time_index]
    qy = df2['qy'][trigger_mid_time_index]
    qz = df2['qz'][trigger_mid_time_index]
    qw = df2['qw'][trigger_mid_time_index]
    ECIx = df2['ECIx'][trigger_mid_time_index]
    ECIy = df2['ECIy'][trigger_mid_time_index]
    ECIz = df2['ECIz'][trigger_mid_time_index]

    result = np.array([datetime.strftime(trigger_time_UTC,'%m_%d_%Y_%H_%M_%S_%f'),
                        Min_trigger_bin,
                        round(T50,3),
                        round(T90,3),
                        round(T05_to_trigger,3),
                        round(T25_to_trigger,3),
                        round(T75_to_trigger,3),
                        round(T95_to_trigger,3),
                        NN_source_count,
                        NP_source_count,
                        NT_source_count,
                        NB_source_count,
                        PN_source_count,
                        PP_source_count,
                        PT_source_count,
                        PB_source_count,
                        NN_BG_count,
                        NP_BG_count,
                        NT_BG_count,
                        NB_BG_count,
                        PN_BG_count,
                        PP_BG_count,
                        PT_BG_count,
                        PB_BG_count,
                        qw,
                        qx,
                        qy,
                        qz,
                        ECIx,
                        ECIy,
                        ECIz
                        ])
    header_str = ['Trigger_time_UTC','Min_trigger_bin','T50','T90','T05_to_trigger','T25_to_trigger','T75_to_trigger','T95_to_trigger',
    'NN_source_count','NP_source_count','NT_source_count','NB_source_count','PN_source_count','PP_source_count','PT_source_count','PB_source_count',
    'NN_BG_count','NP_BG_count','NT_BG_count','NB_BG_count','PN_BG_count','PP_BG_count','PT_BG_count','PB_BG_count',
    'qw','qx','qy','qz','ECIx','ECIy','ECIz']
    pd.DataFrame(result.reshape(-1, len(result))).to_csv("./output/result_"+ filename+".csv",index = False,header = header_str)
