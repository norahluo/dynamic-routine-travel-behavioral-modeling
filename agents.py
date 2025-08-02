import numpy as np
np.seterr(all='raise')

class Agent(object):

    def __init__(self, init_habit_strength, info, param_constraint): 
        """
        Initialize agents with initial habit strength, individual-level features, and parameter constraints.

        Parameters
        ----------
        init_habit_strength : ndarray of shape (7, 2)
            Each row contains the initial habit strength of notCommute/Commute [HS_nc, HS_c] for a specific weekday.
            The first to the last row corresponds to Monday to Sunday.

        info : pandas.DataFrame
            A DataFrame where each row contains information including the individual's commute choice on a specific day, alternative
            - 'weekday': weekday indicator (0=Monday, ..., 6=Sunday)
            - 'dfrw': distance from home to workplace (in meters)
            - 'income': individual income
            - 'shelter-in-place': binary indicator for SIP period
            - 'pre-vax': binary indicator for pre-vaccination period
            - 'during': binary indicator for during-vaccination period
            - 'post-vax': binary indicator for post-vaccination period
            - 'is_weekend': binary indicator for weekend
            - 'is_holiday_extra': binary indicator for holidays
            - 'commute_filled_diff_dist': binary indicator of commute choice. 0 indicates the individual did not go to the workplace; 1 indicates he/she visited the workplace.

        param_constraint : list of int or str
            A list of indices (for utility parameters) and/or parameter names (for weight function parameters) to be constrained.
            For example: [3, 4, 'w0'] means utility parameters b3 and b4, and weight parameter 'w0' are constrained.
            A negative exponential transformation will be applied to the constrained utility parameters to enforce negative effects.
        """
        
        self.init_habit_strength = init_habit_strength.copy()
        
        self.HS_hdpplus, self.HS_hdpminus, self.HS_hgpplus, self.HS_hgpminus = init_habit_strength.copy(), init_habit_strength.copy(), init_habit_strength.copy(), init_habit_strength.copy()

        self.HS = init_habit_strength.copy()
        self.weekday = info['weekday']      
        
        self.X = np.array([[1]*len(info), info['dfrw']/1000, np.log(info['income']), 
                           info['shelter-in-place'], info['pre-vax'], (info['during'] + info['post-vax']), 
                           info['is_weekend'], info['is_holiday_extra']]).T
        
        self.constraint = param_constraint
        self.traj = info['commute_filled_diff_dist']
        self.epsilon = 1e-6
        
    def Arbiter(self, HS, X):
        """
        Compute alternative choice probabilities based on alternative utility and habit strength.
        
        Parameters
        ----------
        HS: ndarray of shape (2,)
            current habit strength for not commute/ commute [HS_nc, HS_c]

        X: ndarray of shape (n_features,)
            Attributes vector containing both individual characteristics and alternative attributes (e.g., intercept, distance from work to home, is weekend,etc.).

        Returns
        -------
        h : float
            Overall habituation.

        w : float
            System-balancing weight [0, 1]. 
            0 means the deciison-making is fully driven by the deliberative system; 1 means the decision is fully driven by habit system.
            
        P : ndarray of shape (2,)
            Probability of choosing each alternative [P_nc, P_c].
        """
        # Utility calculation
        B_adjusted = [-np.exp(self.params['B'][i]) if i in self.constraint else self.params['B'][i] * 0.1 if i == 1 else self.params['B'][i] for i in range(len(self.params['B']))]
        U = np.array([0, np.dot(X, np.array(B_adjusted))]).T

        # Overall habituation and system balancing weight
        h = np.sqrt(np.sum((HS - np.mean(HS))**2))
        if 'w0' in self.constraint:
            w = 1/(1 + np.exp(-self.params['wh'] * h + np.exp(self.params['w0'])))  
        else:
            w = 1/(1 + np.exp(-self.params['wh'] * h + self.params['w0']))

        # Drive and probabilities
        D = w * self.params['hs'] * HS + (1-w) * U
        P = np.exp(D)/np.sum(np.exp(D))

        return h, w, P
        
    def update(self, choice, wday):
        """
        Update habit strength for a given choice and weekday 
        """
        self.HS_hdpplus[wday] = self.HS_hdpplus[wday] - self.HS_hdpplus[wday] * (self.params['HDP'] + self.epsilon) + (1 - self.HS_hdpplus[wday]) * self.params['HGP'] * np.array([1 if i == choice else 0 for i in range(2)])
        self.HS_hdpminus[wday] = self.HS_hdpminus[wday] - self.HS_hdpminus[wday] * (self.params['HDP'] - self.epsilon) + (1 - self.HS_hdpminus[wday]) * self.params['HGP'] * np.array([1 if i == choice else 0 for i in range(2)])
        self.HS_hgpplus[wday] = self.HS_hgpplus[wday] - self.HS_hgpplus[wday] * self.params['HDP'] + (1 - self.HS_hgpplus[wday]) * (self.params['HGP'] + self.epsilon) * np.array([1 if i == choice else 0 for i in range(2)])
        self.HS_hgpminus[wday] = self.HS_hgpminus[wday] - self.HS_hgpminus[wday] * self.params['HDP']  + (1 - self.HS_hgpminus[wday]) * (self.params['HGP'] - self.epsilon) * np.array([1 if i == choice else 0 for i in range(2)])
        
        self.HS[wday] = self.HS[wday] - self.HS[wday] * self.params['HDP'] + (1 - self.HS[wday]) * self.params['HGP'] * np.array([1 if i == choice else 0 for i in range(2)])
  
    def forward(self):
        """
        Calculate the probability of choosing the observed alternative 
        """
        T = len(self.traj)
        HSs, hs, ws, Ps = np.zeros((T, 2)), np.zeros(T), np.zeros(T), np.zeros((T, 2)) 

        HSs_hdpplus, HSs_hdpminus, HSs_hgpplus, HSs_hgpminus = np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2))
        
        for t in range(T): 
            wday = self.weekday[t]
            HSs_hdpplus[t] = self.HS_hdpplus[wday]
            HSs_hdpminus[t] = self.HS_hdpminus[wday]
            HSs_hgpplus[t] = self.HS_hgpplus[wday]
            HSs_hgpminus[t] = self.HS_hgpminus[wday]
            HSs[t] = self.HS[wday]
            hs[t], ws[t], Ps[t] = self.Arbiter(HSs[t], self.X[t])  
            
            self.update(self.traj[t], wday)
            
                
        return HSs, hs, ws, Ps, HSs_hdpplus, HSs_hdpminus, HSs_hgpplus, HSs_hgpminus
    
    def compute_loss(self):
        """
        Compute the loss and gradients for the agent.
        """
        grad = {}
        T = len(self.traj)
        HS, h, w, P, HSs_hdpplus, HSs_hdpminus, HSs_hgpplus, HSs_hgpminus = self.forward()

        # Utility calculation
        B_adjusted = [-np.exp(self.params['B'][i]) if i in self.constraint else self.params['B'][i] * 0.1 if i == 1 else self.params['B'][i] for i in range(len(self.params['B']))]
        U = np.array([[0]*T, np.dot(self.X, np.array(B_adjusted))]).T

        # Loss computation
        loss = - np.sum(self.traj * np.log(P[:,1] + self.epsilon) + (1 - self.traj) * np.log(1 - P[:,1] + self.epsilon)) 

        # Gradient computations
        dlossdpc = - (self.traj / P[:,1] - (1 - self.traj) / (1 - P[:,1])) # An array of shape (T,)
        dpcdD = P[:,1] * (1 - P[:,1]) 
        
        
        dDdw = (HS[:,1] - HS[:,0]) * self.params['hs'] - (U[:,1] - U[:,0])
        dDdhs = (HS[:,1] - HS[:,0]) * w 

        dwdb = -w * (1-w) 
        dwdh = dwdb * (-self.params['wh'])        
        dhdHS = 1/(2**0.5) * np.sign(HS[:,1] - HS[:,0])   
        dDdHS = w * self.params['hs'] + dDdw * dwdh * dhdHS 

        # LET HS_t = HS_t(C) - HS_t(WFH) # calculate  numerical gradients
        dHSdHDP = ((HSs_hdpplus[:, 1] - HSs_hdpplus[:, 0]) - (HSs_hdpminus[:, 1] - HSs_hdpminus[:, 0])) / (2 * self.epsilon)
        dHSdHGP = ((HSs_hgpplus[:, 1] - HSs_hgpplus[:, 0]) - (HSs_hgpminus[:, 1] - HSs_hgpminus[:, 0])) / (2 * self.epsilon)

        grad['dB'] = np.dot(dlossdpc * dpcdD * (1 - w), self.X) * np.array([-np.exp(self.params['B'][i]) if i in self.constraint else 0.1 if i == 1 else 1 for i in range(len(self.params['B']))])  
        grad['dHDP'] = np.sum(dlossdpc * dpcdD * dDdHS * dHSdHDP) 
        grad['dHGP'] = np.sum(dlossdpc * dpcdD * dDdHS * dHSdHGP) 
        grad['dwh'] = np.sum(dlossdpc * dpcdD * dDdw * dwdb * (-h)) 
        grad['dw0'] = np.sum(dlossdpc * dpcdD * dDdw * dwdb * (np.exp(self.params['w0']) if 'w0' in self.constraint else self.params['w0'])) 
        grad['dhs'] = np.sum(dlossdpc * dpcdD * dDdhs) 

        
        return loss, grad, w, P

