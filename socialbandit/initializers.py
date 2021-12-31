import numpy as np
import numpy.random as npr
import copy
from autoslot import Slots, SlotsPlusDict

from socialbandit.utils import * 

class BanditDimensions:
    def __init__(self, K, N):
        # K: number of tasks (or arms)
        # N: number of agents
        self.K = K
        self.N = N

    @property
    def num_tasks(self):
        return self.K

    @num_tasks.setter
    def num_tasks(self, val):
        self.K = val

    num_arms = num_tasks
    num_actions = num_tasks

    @property
    def num_agents(self):
        return self.N

    @num_agents.setter
    def num_agents(self, val):
        self.N = val


class Initializer:
    def __init__(self, *args, **kwargs):
        pass 
    
    def init_belief(self):
        raise NotImplementedError
        
    def init_prob(self):
        raise NotImplementedError

    def __call__(self, states=None):
        init_states = self.init_belief()
        init_states['P'] = self.init_prob()

        if states:
            states.update(init_states)
            return states
        else:
            return init_states

class InitializerWithEqualBeliefs(BanditDimensions,Initializer):
    def __init__(self, K, N, mu_0, sigma2_0, *args, **kwargs):
        # mu_0: initial (prior) belief
        # sigma2_0: initial (prior) belief
        self.mu_0 = mu_0
        self.sigma2_0 = sigma2_0
        super().__init__(K, N, *args, **kwargs)

    def init_belief(self):
        return dict(
            M = self.mu_0 * np.ones((self.K,self.N)),
            V = self.sigma2_0 * np.ones((self.K,self.N))
        )

    def init_prob(self):
        raise NotImplementedError

class InitEqualProb(InitializerWithEqualBeliefs):
    
    def init_prob(self):
        return np.ones((self.K, self.N)) / self.K

class InitNormUniformProb(InitializerWithEqualBeliefs):
    
    def init_prob(self):
        return norm_uniform(self.K, self.N)
    
class InitializerBanditAgnostic(InitializerWithEqualBeliefs):
    def __init__(self, mu_fn = np.max, sigma2_fn = np.mean, 
                 P_fn = lambda K,N: np.ones((K,N))):
        self.__fns = dict(
            mu = mu_fn,
            sigma2 = sigma2_fn,
            P = P_fn
        )
        
    def config_dim_and_priors(self, task_settings, num_agents):
        
        super().__init__(
            K = task_settings.K, 
            N = num_agents, 
            mu_0 = self.__fns['mu'](task_settings.mean),
            sigma2_0 = self.__fns['sigma2'](task_settings.var)
        )
        
    def init_prob(self):
        return to_prob_per_col(self.__fns['P'](self.K,self.N))