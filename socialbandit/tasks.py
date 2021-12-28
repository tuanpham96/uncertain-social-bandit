import numpy as np
import numpy.random as npr
import pandas as pd

from itertools import product as iterprod

from socialbandit.utils import * 

class TaskSetting:
    _rho_is_const = True
    _rho_fn = None # arm availability function

    def __init__(self, mean, var, rho=None):
        # mean: list of task mean reward
        # var: list of task uncertainty
        # rho: current availability (either list or function (-> array) or None)

        self.K = len(mean) # number of tasks

        if len(mean) != len(var):
            raise ValueError('The two vectors "mean" and "var" must have similar lengths')
        if min(var) < 0:
            raise ValueError('All values of "var" vector needs to be non-negative')

        # might need to convert `mean` and `var` to column vec
        self.mean = np.array(mean)[:,None]
        self.var = np.array(var)[:,None]
        self.std = np.sqrt(self.var) # some funs need var directly, some need std
        self.set_rho(rho)

    def set_rho(self, rho):
        if rho is None: # by default, every arm is available
            rho = [1] * self.K

        if callable(rho): # if passing a function f(t)
            self._rho_is_const = False
            if len(rho(1)) != self.K: # testing for f(t=1)
                raise ValueError('The generated "rho" vector from the function must have same length as "mean" and "var" vectors')
            self._rho_fn = rho

        if self._rho_is_const: # assign const vector
            if len(rho) != self.K:
                raise ValueError('The "rho" vector must have same length as "mean" and "var" vectors')
            self.rho = np.array(rho)[:,None]

    set_availability = set_rho

    def get_rho(self, t):
        if not self._rho_is_const:
            self.rho = self._rho_fn(t)[:,None]
        return self.rho

    get_availability = get_rho

    def sample(self, action):
        chosen_mean = self.mean.T @ action
        chosen_std  = self.std.T @ action
        reward = action * npr.normal(chosen_mean, chosen_std)
        return reward

    sample_reward = sample

    def __setitem__(self, ind, data):
            self.mean[ind,0] = data['mean']
            self.var[ind,0] = data['var']

    def __getitem__(self, ind):
            return dict(mean=self.mean[ind,0],
                        var=self.var[ind,0])

class ChildDevelopmentEnvironment(TaskSetting):
    def __init__(self, env = dict(mean = [-100, 100], var = [1, 80], levels = 12),
                 child = dict(mean = [-50, 50], var = [0, 40], time = [0, 400])):

        unq_mu = np.linspace(env['mean'][0], env['mean'][1], env['levels'])
        unq_s2 = np.linspace(env['var'][0], env['var'][1], env['levels'])
        df = pd.DataFrame(list(iterprod(unq_mu, unq_s2)), columns=['mu', 's2'])
        child_indices = df.query("mu > @child['mean'][0]" +
                                 "and mu < @child['mean'][1]" +
                                 "and s2 > @child['var'][0]" +
                                 "and s2 < @child['var'][1]").index.tolist()
        super().__init__(
            mean = df.mu.tolist(),
            var  = df.s2.tolist(),
            rho  = None # override "get_rho" method instead 
        )
        
        self.env = env 
        self.child = dict(indices = child_indices, **child)
    
    def get_rho(self, t):
        child_time = self.child['time']
        num_tasks = self.K
        if t >= child_time[0] and t <= child_time[1]:
            rho = np.zeros((num_tasks,1))
            rho[self.child['indices'],0] = 1.0
        else:
            rho = np.ones((num_tasks,1))
        self.rho = rho
        return self.rho


class ActionSampler:
    
    def sample(self, prev_states, rho):
        raise NotImplementedError

    def __call__(self, prev_states, rho, states=None):
        actions = self.sample(prev_states, rho)

        if states:
            states.update(actions)
            return states
        else:
            return actions

class SoftmaxSampler(ActionSampler):
    
    def __init__(self, tau):
        if tau == 0:
            raise ValueError('"tau" needs to be != 0')
        self.tau = tau

    def sample(self, prev_states, rho=1.0):
        Q = prev_states.Q
        Q = Q - Q.max(axis=0, keepdims=True) # to avoid overflow
        P = to_prob_per_col(np.exp(Q / self.tau) * rho)
        A = weighted_choice_matrix(P, norm=False)
        return dict(A=A, P=P)

class ArgmaxSampler(ActionSampler):

    def sample(self, prev_states, rho=1.0):
        A = argmax_matrix(prev_states.Q * rho)
        return dict(A=A)

class GreedyEpsilonSampler(ActionSampler):
    
    def __init__(self, epsilon):
        if epsilon < 0 or epsilon > 1:
            raise ValueError('"epsilon" needs to be within [0,1]')
        self.epsilon = epsilon

    def sample(self, prev_states, rho=1.0):
        loc_max     = argmax_matrix(prev_states.Q * rho)
        loc_others  = (np.ones_like(loc_max) - loc_max) * rho # need to mask again

        P_of_max    = (1.0 - self.epsilon) * loc_max
        P_others    = self.epsilon * to_prob_per_col(loc_others)

        P = P_of_max + P_others
        A = weighted_choice_matrix(P, norm=False)
        return dict(A=A, P=P)


class ThompsonSampler(ActionSampler):    
    def __init__(self):
        raise NotImplementedError

    