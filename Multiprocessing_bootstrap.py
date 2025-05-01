#!/usr/bin/env python
# coding: utf-8

import multiprocessing
from multiprocessing import Process
from datetime import datetime
import random
import numpy as np
import pandas as pd
from optim import adam
from Trainer import Trainer
from agents import Agent
import os, glob, re

def bootstrapEstimate(agents, params, i):
    
    start_time = datetime.now()
    print('Core {} starts bootstrap estimate at {}'.format(i, start_time))
    
    num_b = agents[0].X.shape[1]
    # param_dict = {
    #               'HDP': [0.01, 0.1, 0.2, 0.5],
    #               'HGP': [0.02, 0.2, 0.35, 0.5],
    #               'hs': [3/10, 5/10, 7/10], 
    #               'B' : dict(zip(['b'+str(i) for i in range(num_b)], 
    #                              [[-0.01, -0.05, -0.1, -0.5] for _ in range(num_b)])),
    #               'wh': [20, 25, 30],     
    #               'w0': [1, 2, 3]        
    #             }    

    # init_params = dict(zip(param_dict.keys(), [ # param_dict['HDP'][random.randint(0, len(param_dict['HDP'])-1)],
    #                                             np.log(-1 + 1/0.127925), 
    #                                             0.3,
    #                                             # param_dict['hs'][random.randint(0, len(param_dict['hs'])-1)],
    #                                             3.531039, 
    #                                             np.array([param_dict['B'][key][random.randint(0, len(value)-1)] for key, value in param_dict['B'].items()]),
    #                                             # param_dict['wh'][random.randint(0, len(param_dict['wh'])-1)],
    #                                             20.378311, 
    #                                             # param_dict['w0'][random.randint(0, len(param_dict['w0'])-1)]
    #                                             1.013918, 
    #                                             ]))
    # 0.127925 	0.3 	3.531039 	20.378311 	1.013918 	0.904213 	-0.009434 	-0.046232 	-0.455488 	-0.017448 	0.071218 	-0.581072 	-1.393972 	254425.379267
    # 0.083404 	0.278888 	3.682959 	20.490208 	0.860153 	0.902616 	-0.008324 	-0.051583 	-0.580761 	-0.006841 	0.125830 	-0.440056 	-1.463603 	254110.434450
    init_params = dict(zip(['HDP', 'HGP', 'hs', 'B', 'wh', 'w0'], [0.083, 0.28, 3.68, np.array([0.902616, -0.008324*10, -0.051583, np.log(0.580761), np.log(0.006841), 0.125830, -0.440056, -1.463603]), 20.490208, 0.860153]))

    print('Initial parameters for core {} are {}'.format(i, init_params))
    np.random.seed(2023 + i)
    for b in range(10):
        start_time_ = datetime.now()
        print('{} bootstrap estimate start at {}'.format(b, start_time_))
        bootstrapSample = np.random.choice(agents, size = 573, replace = True)
        trainer = Trainer(bootstrapSample, i)
        trainer.train(optim = adam, init_params = init_params, 
                      directory = '../Output/Bootstrap/',
                      expname = params['exp_name'],
                      n_epoch = params['n_epoch'], 
                      tol = params['tol'],
                      batch_size = params['batch_size'],
                      logger = params['logger'])
        print('{} bootstrap estimate ends at {}. Time elapse: {}'.format(b, datetime.now(), datetime.now() - start_time_))
        
    end_time = datetime.now()
    print('Core {} ends at {}. Time elapse: {}'.format(i, end_time, end_time - start_time))


def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_run', type=int, default=5)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--tol', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=573)
    parser.add_argument('--logger', type=bool, default=True)
    args = parser.parse_args()
    params = vars(args)
    
    #######################################
    #####      Initialize agents
    #######################################
    agents = []
    len_ = 0
    init_habit_strength = np.array([[0.0, 0.57],
                                    [0.0, 0.57],
                                    [0.0, 0.57],
                                    [0.0, 0.57],
                                    [0.0, 0.57],
                                    [0.57, 0.0],  
                                    [0.57, 0.0]])
    
    csv_files = glob.glob(os.path.join('//global/scratch/users/norahluo/trajectory', '*.csv'))
    demo = pd.read_csv('//global/scratch/users/norahluo/demo.csv')
    
    for f in csv_files:
        df = pd.read_csv(f, parse_dates = ['date'], usecols = lambda x: 'Unnamed' not in x)
        df = df.loc[(df.date >= '2020-01-06')]
        df.reset_index(drop = True, inplace = True)
        demo_ = demo[demo.panelist_id == re.split(r'_|\.',f)[1]]
        df['dfrw'] = demo_['dfrw_cap_mean'].values[0]
        df['income'] = demo_['hh_income_mid'].values[0]

        constraint = [3, 4]
        agent = Agent(init_habit_strength, df, constraint)
        agents.append(agent)
        
        len_ += len(df)
        
    print("Total length of trajectory for {} agents is {}".format(len(agents), len_))    
        
    ########################################
    #####     Execute multiprocessing
    ########################################
    print('The number of cpu using:', multiprocessing.cpu_count())
    processes = [Process(target = bootstrapEstimate, args = (agents, params, i)) for i in range(multiprocessing.cpu_count())]
    
    # start all processes
    for process in processes:
        process.start()
        
    # wait for all child processes to terminate before starting the command in the main process
    for process in processes:
        process.join()        
        
if __name__ == '__main__':
    
    main()
    
    print('Optimization finished!')

