import numpy as np
import numpy.random as npr


def col_norm(X, p=1):
    return X / np.linalg.norm(X,ord=p,axis=0)

# slower
def wc1(p):
    K,N = p.shape
    inds = [npr.choice(K, p=pi) for pi in p.T]
    return np.eye(K)[inds].T

# faster
def wc2(p):
    K,N = p.shape
    bounds = np.cumsum(np.vstack((np.zeros(N), p)),axis=0)
    x = npr.rand(N)[None,:]
    return (x - bounds[:-1,]) * (x - bounds[1:,]) < 0

p = col_norm(npr.rand(500,1000))

%time X1 = np.mean([wc1(p) for _ in range(1000)], axis=0)
%time X2 = np.mean([wc2(p) for _ in range(1000)], axis=0)

np.sqrt(np.mean((p-X1)**2))
np.sqrt(np.mean((p-X2)**2))
np.sqrt(np.mean((X1-X2)**2))



import numpy as np
from autoslot import Slots, SlotsPlusDict

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

class States(SlotsPlusDict,BanditDimensions):
    def __init__(self, K, N, t, extra_states=[]):
        super().__init__(K, N)
        self.Q = np.zeros((K,N))  # utility
        self.A = np.zeros((K,N))  # action
        self.t = t

        allowable_extras = [
            'P',                    # action prob
            'G',                    # kalman gain
            'W',                    # social network
            'C',                    # content matrix
            'Q_alc',                # action learning utility
            'Q_sol'                 # social learning utility
        ]

        for _state in extra_states:
            if _state not in allowable_extras:
                raise ValueError('"%s" is not an allowable extra states' %(_state))
            dim_mat = (K,N) if _state != 'W' else (N,N)
            setattr(self, _state, np.zeros(dim_mat))

Z = States(3,7,1,['P','G'])

