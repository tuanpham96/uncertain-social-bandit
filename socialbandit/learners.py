import numpy as np

from socialbandit.utils import * 
from socialbandit.tasks import *
from socialbandit.social import *

class ActionLearner:
    
    def learn(self, states):
        raise NotImplementedError

    def __call__(self, states, return_utility=True):
        utility = self.learn(states)

        if isinstance(utility, dict):
            raise ValueError("The utility output needs cannot be a dict")

        states.update(dict(Q_acl = utility))

        if return_utility:
            return utility
        else:
            return states

class UpperConfidenceBound(ActionLearner):
    def __init__(self, beta):
        self.beta = beta

    def learn(self, states):
        return states.M + self.beta * np.sqrt(states.V)

class MeanGreedyExploit(ActionLearner):
    def learn(self, states):
        return states.M

class VarianceGreedyExplore(ActionLearner):
    def learn(self, states):
        return np.sqrt(states.V)

class SocialLearner:
    def __init__(self, norm_social_fn=None, norm_weight_fn=None, norm_content_fn=None):
        self.norm_social_fn = norm_social_fn
        self.norm_weight_fn = norm_weight_fn
        self.norm_content_fn = norm_content_fn

    def learn(self, states):
        raise NotImplementedError

    def __call__(self, states, return_utility=True):
        utility = self.learn(states)

        if isinstance(utility, dict):
            raise ValueError("The utility output needs cannot be a dict")

        states.update(dict(Q_sol = utility))

        if return_utility:
            return utility
        else:
            return states

class SocialMassPower(SocialLearner):
    def __init__(self, alpha, eta = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.eta = eta

    def learn(self, states):
        C, W = states.C, states.W

        if self.norm_weight_fn:
            W = self.norm_weight_fn(W)
        if self.norm_content_fn:
            C = self.norm_content_fn(C)

        S = np.matmul(C, W)

        if self.norm_social_fn:
            S = self.norm_social_fn(S)

        Q_sol = self.eta * np.power(S, self.alpha)
        return Q_sol
    

class MeanFriendContentLearner(SocialLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(
            norm_weight_fn = col_mean,
            *args, **kwargs)
        
    def learn(self, states):
        C, W = states.C, states.W
        W = self.norm_weight_fn(W)
        
        if self.norm_content_fn:
            C = self.norm_content_fn(C)

        S = np.matmul(C, W)

        if self.norm_social_fn:
            S = self.norm_social_fn(S)
            
        return S