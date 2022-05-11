import numpy as np
import numpy.random as npr
import copy
import re
import warnings
    
def to_prob_per_col(X):
    # a bit faster than `col_norm(X,1)`, but has to make sure `all(X >= 0)`
    # TODO: catch warning when divide by 0 
    P = X / np.sum(X, axis=0, keepdims=True)
    return P

def to_prob_per_col_with_div0_handling(X):
    # will be much slower especially when X contains all-zeros columns (which then could uniform)
    # TBD on whether to keep
    # catch divide_by_0 from: https://newbedev.com/how-do-i-catch-a-numpy-warning-like-it-s-an-exception-not-just-for-testing
    colsum = np.sum(X, axis=0, keepdims=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try: 
            return X / colsum
        except RuntimeWarning:
            pX = copy.deepcopy(X)
            pX[:,colsum.reshape(-1) == 0] = 1.0 
            return to_prob_per_col(pX)
        
def col_norm(X, p=1):
    return X / np.linalg.norm(X, ord=p, axis=0, keepdims=True)

def col_mean(X): 
    sX = np.sum(X, axis=0, keepdims=True)
    sX[sX == 0] = 1.0 # to avoid division by 0
    return X / sX

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

def kwargs2dict_overridevars(d, s, sep='_'):
    # e.g: d = dict(a_time = 2, b_val = 3, d_name=4), s = ['a', 'b', 'c'], sep = '_'
    # -> return dict(a = dict(time=2), b=dict(val=3), c=dict(), __undefined__=dict(d_name=4))
    undef_key = '__undefined__'
    d2 = {_k: dict() for _k in s + [undef_key]}
    re_pattern = re.compile('^(%s)%s*' %('|'.join(s), sep))
    for k, v in d.items():
        main_k2 = re_pattern.findall(k)
        if len(main_k2) == 0: 
            d2[undef_key] = v
        else:
            sub_k2 = re_pattern.sub('',k)
            d2[main_k2[0]][sub_k2] = v 
    return d2 

    
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