import pandas as pd
import numpy as np
import itertools

def create_variations(d, add_id=True, num_trials=1, file_prefix=None):
    keys = list(d.keys())
    values = list(d.values())
    
    if not add_id:
        df = pd.DataFrame(itertools.product(*values), columns=keys)
        return df.to_dict('records') * num_trials
    
    trial_ids = list(range(num_trials))
    
    combos = list(itertools.product(*values)) # first, combinations of values
    combos = [list(x) + [i] for i, x in enumerate(combos)] # then, add a variation ID
    combos = list(itertools.product(combos, trial_ids)) # duplicating with trials
    combos = [list(x[0]) + [x[1], i] for i, x in enumerate(combos)] # lastly, add unique ID
    
    df = pd.DataFrame(combos, columns=keys + ['_var_id', '_trial_id', '_unq_id'])
    
    if file_prefix:
        df['file_name'] = file_prefix + df['_unq_id'].apply(lambda x: '_%06d' %(x))
        
    return df.to_dict('records')

def unnest_dict(d): 
    # TODO: this only unnest one level, need to implement for further
    results = {}
    for k, v in d.items():
        if type(v) is dict:
            results.update({k + '::' + k1: v1 for k1, v1 in v.items()})
        else:
            results[k] = v
    return results

def is_prevchoice_diff(A_prev, A_curr):
    # A_*: binary matrices of size "num_tasks x num_agents"
    # A_prev could also be cumulative previous A
    return np.sum(A_prev * A_curr, axis=0) < 1.0
    
def get_task_choices(A):
    inds = np.where(A > 0)
    return inds[0][np.argsort(inds[1])]

def get_task_rewards(Y):
    return Y.sum(axis=0)

def aggregate_rewards(rewards, condition):
    num = np.mean(condition.sum(axis=0))
    mag = np.mean(np.abs(rewards[condition])) if num > 0 else 0.0
    return dict(num=num, mag=mag)

def calculate_entropy(X):
    _, n = np.unique(X, return_counts=True)
    P = n/sum(n)
    return np.sum(-P * np.log2(P))

class PseudoAverageCumTime:
    # this is a pseduo class to create a function for time average with a persistent time variable
    # needed for when using a function with only one argument and no internal place for time update
    _t = -1
    
    def __init__(self):
        self._t = 0
    
    def __call__(self, X):
        self._t += 1
        return X / self._t
