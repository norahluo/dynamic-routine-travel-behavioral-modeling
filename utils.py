#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 20),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


# def tpd(string):
#     try:
#         return datetime.fromisoformat(string)
#     except ValueError:
#         return datetime.strptime(string, '%m/%d/%Y %H:%M')
#     raise ValueError('Format not match!')

    
    
def extract(string):
    try:
        return string.split('_')
    except:
        return [""]
    
def deal_w_time(df):
    """
    Convert the arrival time from string to datetime
    Extract year, month, day, hour, and the corresponding week, weekday 
    """
    df = df.copy()

    df['arrival_time'] = df['arrival_local'].apply(lambda x: datetime.fromisoformat(x).replace(tzinfo = None))
    
    # To extract COVID information
#     df['arrival_date'] = df['arrival_time'].apply(lambda x: x.date())  
#     df['arrival_year'] = df['arrival_time'].apply(lambda x: x.year)
#     df['arrival_month'] = df['arrival_time'].apply(lambda x: x.month)
#     df['arrival_day'] = df['arrival_time'].apply(lambda x: x.day)
    #df['arrival_hour'] = df['arrival_time'].apply(lambda x: x.hour) 
    
    # For habit formation model to infer on xx weekday of the xxth week, agent commute or not 
#     df['arrival_week'] = df['arrival_time'].apply(lambda x: x.isocalendar()[1])
#     df['arrival_weekday'] = df['arrival_time'].apply(lambda x: x.weekday())
    
    df['departure_time'] = df['departure_local'].apply(lambda x: datetime.fromisoformat(x).replace(tzinfo=None))
    #df['departure_weekday'] = df['departure_time'].apply(lambda x: x.weekday())
    #df['departure_hour'] = df['departure_time'].apply(lambda x: x.hour)
    
    return df


def time_missing(data):
##### return the longest continuous days without records #####
    return data.sort_values().diff().max()


def missing_per(data):
##### return percentage of days without records #####    
    first = data.sort_values().iloc[0]
    last = data.sort_values().iloc[-1]
    return 1 - len(data.unique()) / (last - first).days


def extract_trajectories(record):
    traj = []
    for user in np.sort(record['third_party_user_id'].unique()):
        work = pd.DataFrame(record[record.third_party_user_id == user].groupby(['year', 'month', 'week', 'arrival_weekday', 'arrival_date'])['location_name'].apply(set(['work']).issubset))
        work['commute'] = work['location_name'].astype(int)
        work.reset_index(inplace=True)
        trajectories = np.array(work[['arrival_date', 'arrival_weekday', 'commute']])
        traj.append(trajectories)
    
    return np.array(traj)


def plot_logger(logger):
    fig = plt.figure(figsize = (16, 24))
    ax1 = fig.add_subplot(521)
    ax2 = fig.add_subplot(522)
    ax3 = fig.add_subplot(523)
    ax4 = fig.add_subplot(524)
    ax5 = fig.add_subplot(525)
    ax6 = fig.add_subplot(526)
    ax7 = fig.add_subplot(527)
    ax8 = fig.add_subplot(528)
    ax9 = fig.add_subplot(529)
    ax10 = fig.add_subplot(5,2,10)
    
    pd.DataFrame(logger['losses']).rename({0: 'loss'}, axis = 1).plot(ax=ax1)
    pd.DataFrame(logger['grads'])[['dHDP','dHGP']].plot(ax=ax2)
    pd.DataFrame(logger['grads'])[['dtheta_h', 'dtheta_d']].plot(ax=ax3)
    pd.DataFrame(logger['grads'])[['db1','db0']].plot(ax=ax4)
    pd.DataFrame(logger['grads'])[['dwh','dw0']].plot(ax=ax5)
    pd.DataFrame(logger['params'])[['HDP']].plot(ax=ax6)
    pd.DataFrame(logger['params'])[['HGP']].plot(ax=ax7)
    pd.DataFrame(logger['params'])[['theta_h', 'theta_d']].plot(ax=ax8)
    pd.DataFrame(logger['params'])[['b1','b0']].plot(ax=ax9)
    pd.DataFrame(logger['params'])[['wh','w0']].plot(ax=ax10)
    
def plot_H(Hs, days):
    
    fig = plt.figure(figsize = (16, 20))
    ax1 = fig.add_subplot(421, ylabel = 'Habit Strength')
    ax2 = fig.add_subplot(422, ylabel = 'Habit Strength')
    ax3 = fig.add_subplot(423, ylabel = 'Habit Strength')
    ax4 = fig.add_subplot(424, ylabel = 'Habit Strength')
    ax5 = fig.add_subplot(425, ylabel = 'Habit Strength')
    ax6 = fig.add_subplot(426, ylabel = 'Habit Strength')
    ax7 = fig.add_subplot(427, ylabel = 'Habit Strength')

    df = pd.DataFrame(Hs).rename({0:'Home', 1:'Office'}, axis = 1)
    df['date'] = days[:,0]                  
    df['wday'] = days[:,1]
    df[''] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    df[df['wday'] == 0].plot('', ['Home', 'Office'], ax=ax1, title = 'Monday', rot = 20)
    df[df['wday'] == 1].plot('', ['Home', 'Office'], ax=ax2, title = 'Tuesday', rot = 20)
    df[df['wday'] == 2].plot('', ['Home', 'Office'], ax=ax3, title = 'Wednesday', rot = 20)
    df[df['wday'] == 3].plot('', ['Home', 'Office'], ax=ax4, title = 'Thursday', rot = 20)
    df[df['wday'] == 4].plot('', ['Home', 'Office'], ax=ax5, title = 'Friday', rot = 20)
    df[df['wday'] == 5].plot('', ['Home', 'Office'], ax=ax6, title = 'Saturday', rot = 20)
    df[df['wday'] == 6].plot('', ['Home', 'Office'], ax=ax7, title = 'Sunday', rot = 20)
    fig.tight_layout()
    plt.tick_params(bottom=False)
                      
    
def plot_p_and_w(x, days):
   
    fig = plt.figure(figsize = (16, 20))
    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)
    
    df = pd.DataFrame(x).rename({0:'x'}, axis = 1)
    df['date'] = days[:,0]                  
    df['wday'] = days[:,1]
    df[''] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    df[df['wday'] == 0].plot('', 'x', ax=ax1, title = 'Monday', legend=False, rot = 20)
    df[df['wday'] == 1].plot('', 'x', ax=ax2, title = 'Tuesday', legend=False, rot = 20)
    df[df['wday'] == 2].plot('', 'x', ax=ax3, title = 'Wednesday', legend=False, rot = 20)
    df[df['wday'] == 3].plot('', 'x', ax=ax4, title = 'Thursday', legend=False, rot = 20)
    df[df['wday'] == 4].plot('', 'x', ax=ax5, title = 'Friday', legend=False, rot = 20)
    df[df['wday'] == 5].plot('', 'x', ax=ax6, title = 'Saturday', legend=False, rot = 20)
    df[df['wday'] == 6].plot('', 'x', ax=ax7, title = 'Sunday', legend=False, rot = 20)
    fig.tight_layout()
    plt.tick_params(bottom=False)