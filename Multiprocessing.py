#!/usr/bin/env python
# coding: utf-8


from multiprocessing import Process
from datetime import datetime
import random
import numpy as np
import pandas as pd
from optim import adam, sgd_momentum, rmsprop, sgd
from Trainer import Trainer
from agents import Agent
import os
import glob
import re

def findOptim(agents, params, i):

    start_time = datetime.now()
    print('Trainer {} starts at {}'.format(i, start_time))

    ############################################
    #####      Get the starting points     #####
    ############################################
    num_b = agents[0].X.shape[1]
    param_dict = {
                  'HDP': [0.01, 0.1, 0.2, 0.5],
                  'HGP': [0.1, 0.2, 0.3, 0.4, 0.5],
                  'hs': [1, 5, 10],    # ~4
                  'B' : dict(zip(['b'+str(i) for i in range(num_b)], 
                                 [[-0.01, -0.05, -0.1, -0.5] for _ in range(num_b)])),
                  'wh': [10, 40, 70],     
                  'w0': [1, 5, 10]        
                }    

    init_params = dict(zip(param_dict.keys(), [
                                                param_dict['HDP'][random.randint(0, 3)],
                                                param_dict['HGP'][random.randint(0, 3)],
                                                param_dict['hs'][random.randint(0, 2)],
                                                np.array([param_dict['B'][key][random.randint(0, len(value)-1)] for key, value in param_dict['B'].items()]),
                                                param_dict['wh'][random.randint(0, 2)],
                                                param_dict['w0'][random.randint(0, 2)]
                                                ]))
    
    # Try to make a set of params with relatively low loss converge   
    init_params = dict(zip(['HDP', 'HGP', 'hs', 'B', 'wh', 'w0'], [0.081, 0.175, 6.984, np.array([0.488, -0.004*10, -0.042, np.log(-(-0.380)), np.log(-(-0.041)), 0.154, -0.194, -1.106]), 4.151, 1.287])) 
    
    print('Initial parameters for trainer {} are {}'.format(i, init_params))

    ############################################################################################
    #####      Train the model to find the optimum that yields the highest likelihood     #####
    ############################################################################################
    trainer = Trainer(agents, i)
    trainer.train(optim = adam, init_params = init_params, 
                  directory = '../Output/', 
                  expname = params['exp_name'], 
                  n_epoch = params['n_epoch'], 
                  tol = params['tol'],
                  batch_size = params['batch_size'],
                  logger = params['logger'])
    
    end_time = datetime.now()
    print('Trainer {} ends at {}. Time elapse: {}'.format(i, end_time, end_time - start_time))

def main():
    
    ##############################################
    #####      Set experiment parameters     #####
    ##############################################
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'todo')
    parser.add_argument('--n_run', type=int, default = 5)
    parser.add_argument('--n_epoch', type=int, default=800)
    parser.add_argument('--tol', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=573)
    parser.add_argument('--logger', type = bool, default = True)
    args = parser.parse_args()
    params = vars(args)
    
    ######################################
    #####      Initialize agents     #####
    ######################################
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
        df['college'] = 1 if demo_['edu_level'].values[0] in ['University/College', 'Postgraduate Education'] else 0

        constraint = [3, 4]
        agent = Agent(init_habit_strength, df, param_constraint = constraint)
        agents.append(agent)
        
        len_ += len(df)
        
    print("Total length of trajectory for {} agents is {}".format(len(agents), len_))
    
    ###########################################
    #####     Execute multiprocessing     #####
    ###########################################

    # seeds = [42, 31, 628, 92, 299]
    processes = [Process(target = findOptim, args = (agents, params, i)) for i in range(params['n_run'])]
    
    # start all processes
    for process in processes:
        process.start()
        
    # wait for all child processes to terminate before starting the command in the main process
    for process in processes:
        process.join()
        
if __name__ == '__main__':   
    
    main()
    
    print('Optimization finished!')

