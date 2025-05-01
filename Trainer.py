#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from datetime import datetime
import os

class Trainer(object):
    def __init__(self, agents, trainer_id):
        
        self.agents = agents
        self.constraint = agents[0].constraint
        self.id = trainer_id
    
    def loss(self, params, batch_agents):
        
        """
        Compute the total loss and gradients for all agents under current parameters
        """
        
        losses = 0.0
        grads = {'dHDP': 0.0, 'dHGP': 0.0, 'dhs': 0.0, 'dB': np.zeros(len(params['B'])), 'dwh': 0.0, 'dw0': 0.0}
        ws, Ps = [], []
        
        batch_size = len(batch_agents)
        
        for n in range(batch_size):
            
            agent = batch_agents[n]
            agent.HS = agent.init_habit_strength.copy()   # re-initialize habit strength
            agent.params = params.copy()
            
            # 
            loss, grad, w, P = agent.compute_loss()
            losses += loss

            grads['dB'] += grad['dB']
            grads['dHDP'] += grad['dHDP']
            grads['dHGP'] += grad['dHGP']
            grads['dhs'] += grad['dhs']
            grads['dwh'] += grad['dwh']
            grads['dw0'] += grad['dw0']
            
            ws.append(w) 
            Ps.append(P[:,1])

        # for key in grads:
        #     grads[key] /= batch_size

        return losses, grads, ws, Ps
    
    def train(self, optim, init_params, directory, expname, n_epoch = 500, tol = 1e-3, batch_size = 573, logger = True):
        """
        Perform gradient descent to optimize the parameters.
        """
        params = init_params.copy()
        optim_config = {'HDP': None, 'HGP': None, 'hs': {'learning_rate': 1e-0}, 'B': None, 
                        'wh': {'learning_rate': 1e-1}, 
                        'w0': {'learning_rate': 1e-1}}
        self.logger = {'losses':[], 'grads':[], 'params':[], 'ws': [], 'Ps': []} 

        n_samples = len(self.agents)
        indices = np.arange(n_samples)
        
        start_time = datetime.now()
        file_path = directory+ expname + '_' + 'Train{}_{}.csv'.format(self.id, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
        self.converge = False
        
        for epoch in range(n_epoch):
            epoch_loss = 0
            np.random.shuffle(indices)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i+batch_size, n_samples)]
                losses, grads, ws, Ps = self.loss(params, np.array(self.agents)[batch_indices])
            
                if np.isnan(losses) or any(np.isnan(g).any() if isinstance(g, np.ndarray) else np.isnan(g) for g in grads.values()): 
                    print('NaN encountered during epoch {}. Stopping training'.format(epoch))
                    break
                
                self.logger['losses'].append(losses)
                self.logger['grads'].append(grads)
                self.logger['params'].append(params.copy())
                self.logger['ws'].append(ws)
                self.logger['Ps'].append(Ps)

                if all(np.abs(g).max() <= tol for g in grads.values()) or (len(self.logger['losses']) > 1 and np.abs(losses - self.logger['losses'][-2]) <= tol): # converge
                    print('Training converged at epoch {}.'.format(epoch))
                    print('Final Loss: %.2f' %(losses))
                    print('Final Gradients: {}'.format(grads))
                    print('Final Parameters: {}'.format(params))
                    print('Time elapse: {}'.format(datetime.now() - start_time))
                    self.converge = True
                    break
                     
                # Add to the total epoch loss
                epoch_loss += losses
                # Update the parameters using the optimizer after each mini-batch
                for key in ['B', 'HDP', 'HGP', 'hs', 'wh', 'w0']:
                    # print('Parameter to update is: ', key)
                    params[key], optim_config[key] = optim(params[key], grads[f'd{key}'], optim_config[key])
                    
            if self.converge:
                break

            if ((epoch % 10 == 9) or self.converge or (epoch == n_epoch-1)):

                if logger:
                    # self.logEpoch(epoch, epoch_loss, grads, params, directory+ 'Epochlogger_' + expname+'.txt')
                    # save results after every epoch or when converging
                    self._save_results(file_path, epoch) 
                    
                print('Trainer {} Epoch {} ends at {}'.format(self.id, epoch, datetime.now()))
                print('-------------------------------------------------------')
                print('Loss for epoch %d: %.2f' % (epoch, epoch_loss))
                print('Gradients: {}'.format(grads))
                
                print(f'Parameters after epoch {epoch}: {params}. ' + f'HDP is {1 / (1 + np.exp(params["HDP"])):.3f}. ' + (f'w0 is {np.exp(params["w0"]):.3f}. ' if 'w0' in self.constraint else '') + ''.join([f'b{i} is {-np.exp(params["B"][i]):.3f}. ' for i in self.constraint if i != 'w0']))
                    
                print('Time elapse: {}'.format(datetime.now() - start_time))
                print("-------------------------------------------------------") 
        
        
        
    def logEpoch(self, epoch, epoch_loss, epoch_grads, epoch_params, filename):
        """
        Log epoch results
        """
        f = open(filename, 'a')
        train_time = f'Trainer {self.id} Epoch {epoch} ends at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        epochLoss = f'Loss for epoch {epoch}: {epoch_loss:.3f}'
        epochGrads = f'Gradients: {epoch_grads}'
        epochEst = f'Parameters after epoch {epoch}: {epoch_params}. ' + f'HDP is {1 / (1 + np.exp(epoch_params["HDP"])):.3f}. ' + (f'w0 is {np.exp(epoch_params["w0"]):.3f}. ' if 'w0' in self.constraint else '') + ''.join([f'b{i} is {-np.exp(epoch_params["B"][i]):.3f}. ' for i in self.constraint if i != 'w0'])
        f.write(f'{train_time} \n {epochLoss} \n {epochGrads} \n {epochEst} \n')
        f.close()
        
    def _save_results(self, file_path, epoch):
        """
        Save training results and intermediate outputs to files
        """
        index = -(epoch%10+1)
        print(index)
        grads_df = pd.DataFrame(self.logger['grads'][index:])
        grads_df[['db{}'.format(i) for i in range(len(self.logger['params'][-1]['B']))]] = pd.DataFrame(grads_df.dB.to_list(), index = grads_df.index)     
        params_df = pd.DataFrame(self.logger['params'][index:])
        params_df[['b{}'.format(i) for i in range(len(self.logger['params'][-1]['B']))]] = pd.DataFrame(params_df.B.to_list(), index = params_df.index)
        output = pd.concat([grads_df, params_df], axis = 1)
        output['loss'] = self.logger['losses'][index:]
        output.drop(columns = ['dB','B'], inplace = True)

        # convert back to the right scale:
        # if 'w0' not in self.constraint:
        # output['w0'] = output['w0'] * 10
        # output['wh'] = output['wh'] * 100
        # output['hs'] = output['hs'] * 10
        output['b1'] = output['b1'] / 10

        if not os.path.exists(file_path):
            output.to_csv(file_path, index = False, mode = 'w', header = True)
        else:
            output.to_csv(file_path, index = False, mode = 'a', header = False)

        print('Results for epoch {} saved to {}'.format(epoch, file_path))
                                                                       

        # if 'Bootstrap' not in directory:
                            
        #     min_idx = np.argmin(self.logger['losses'])    
        #     prob = pd.DataFrame(self.logger['Ps'][min_idx])
        #     weight = pd.DataFrame(self.logger['ws'][min_idx])
        #     prob.to_csv(directory + 'prob/' + '_' + 'Train{}_{}_prob.csv'.format(self.id, timestamp), index = False)
        #     weight.to_csv(directory + 'weight/' + '_' + 'Train{}_{}_weight.csv'.format(self.id, timestamp), index = False)          
        
    def cal_std(self):
        
        """
        Calculate the varaince matrix for B
        """
        k = self.agents[0].X.shape[1]
        V = np.zeros((k, k))
        
        for i in range(len(self.agents)):
            traj = self.agents[i].traj
            pc = self.logger['Ps'][i]
            w = self.logger['ws'][i]
            
            W = traj * (1/pc**2) + (1-traj) * 1/(1-pc)**2
            X = self.agents[i].X
            var = np.dot(np.dot(X.T, np.diag(W) * (1-w)**2), X)              
            
            V += var

        return (V / (self.L * self.L))**0.5   
    