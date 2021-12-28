import numpy as np
import numpy.random as npr
from autoslot import Slots

from socialbandit.utils import * 


class ContentConstructor:
    def __init__(self, state_source='A', transform_fn=None, norm_fn=None):
        self.state_source = state_source

        if state_source != 'A' and transform_fn is None:
            # default of `transform_fn` if not using 'A(t-1)`
            transform_fn = argmax_matrix

        self.transform_fn = transform_fn

    def __call__(self, prev_states):
        state_source = self.state_source 
        source = getattr(prev_states, state_source)

        if state_source == 'A':
            return source

        prev_rho = prev_states.rho
        return self.transform_fn(source * prev_rho)

class SocialSetting:
    def __init__(self, N,
                 social_fn = 'all2all',
                 content_fn = ContentConstructor()):
        self.N = N
        
        # if either one is not available, do not use         
        self._use_social = True
        if social_fn is None or content_fn is None:
            self._use_social = False
            social_fn, content_fn = None, None
            
            
        if isinstance(social_fn, str):
            if social_fn.lower() != 'all2all':
                raise ValueError('The only acceptable string "social_fn" right now is "all2all". Otherwise use a subclass of SocialNetwork')
            social_fn = All2AllSocialNetwork(N)
            
        self.social_fn = social_fn
        self.content_fn = content_fn

    def __call__(self, prev_states, states=None):
        if not self._use_social:
            return None
        
        C = self.content_fn(prev_states)

        if self.social_fn.use_states:
            W = self.social_fn(prev_states)
        else:
            W = self.social_fn()

        social_states = dict(C = C, W = W)

        if states:
            states.update(social_states)
            return states
        else:
            return social_states

class SocialNetwork:
    def __init__(self, N, use_states=False):
        self.N = N
        self.use_states = use_states

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class StaticSocialNetwork(SocialNetwork,Slots):
    def __init__(self, W):
        super().__init__(N=W.shape[0], use_states=False)
        self.W = W

    def __call__(self):
        return self.W

class All2AllSocialNetwork(StaticSocialNetwork):
    def __init__(self, N):
        super().__init__(W=np.ones((N,N)))

class DynamicSocialNetwork(SocialNetwork):    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class Homophily(SocialNetwork):
    def __init__(self, beta,
                 content_fn = ContentConstructor(),
                 bin_weight = True,
                 use_ntask_subtract = False):
        super().__init__(N=None, use_states=True)

        self.beta = beta
        self.content_fn = content_fn
        self.use_ntask_subtract = use_ntask_subtract
        self.bin_weight = bin_weight

    def __call__(self, prev_states):
        C = self.content_fn(prev_states)
        num_tasks, num_agents = C.shape

        if self.N is None:
            self.N = num_tasks

        sim_mat = C.T @ C

        if self.use_ntask_subtract:
            X = np.power(num_tasks - sim_mat, -self.beta) # roughly like distance^{-beta}
        else:
            # might need to consider whether there's overflow to inf here
            X = np.power(sim_mat, self.beta) # like similarity^{beta}

        P = to_prob_per_col(X)

        if self.bin_weight:
            R = npr.rand(num_agents, num_agents)
            return 1.0 * (R < P)
        else:
            return P
