import numpy as np
import numpy.random as npr
import copy

def to_prob_per_col(X):
    # a bit faster than `col_norm(X,1)`, but has to make sure `all(X >= 0)`
    return X / np.sum(X, axis=0, keepdims=True)

def col_norm(X, p=1):
    return X / np.linalg.norm(X, ord=p, axis=0, keepdims=True)

def row_norm(X, p=1):
    return X / np.linalg.norm(X, ord=p, axis=1, keepdims=True)

def norm_uniform(m, n):
    return to_prob_per_col(npr.uniform(size=(m,n)))

def weighted_choice_matrix(P, norm=True):
    if norm: # if unsure then just to be safe, otherwise don't to save time
        P = to_prob_per_col(P)
    K, N = P.shape
    bounds = np.cumsum(np.vstack((np.zeros(N), P)), axis=0)
    x = npr.rand(N)[None,:]
    return (x - bounds[:-1,]) * (x - bounds[1:,]) < 0

def argmax_matrix(X):
    K, N = X.shape
    inds = np.argmax(X, axis=0)
    return np.eye(K)[inds].T

def nan_matrix(size):
    X = np.empty(size)
    X.fill(np.nan)
    return X

class ExperimentClock:
    def __init__(self, T,
                 T_action_sampling = 2,
                 T_belief_updating = 2,
                 T_action_learning = 2,
                 T_social_learning = 2):
        self.T = T # number of trials
        self.set_time('T_action_sampling', T_action_sampling) # to start using ActionSampler
        self.set_time('T_belief_updating', T_belief_updating) # to start using BeliefUpdater
        self.set_time('T_action_learning', T_action_learning) # to start using ActionLearner
        self.set_time('T_social_learning', T_social_learning) # to start using SocialLearner
        self.t = 0 # current trial 

    def set_time(self, name, value):
        # if value is None: value = np.inf
        if value < 2:
            raise ValueError('"%s" needs to be at least 2')
        setattr(self, name, value)

    def is_at(self, progress):
        _T = getattr(self, 'T_' + progress)
        if _T is None: 
            return False 
        return self.t >= _T

    def update(self):
        self.t += 1