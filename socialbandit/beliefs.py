import numpy as np

from socialbandit.utils import * 
from socialbandit.tasks import TaskSetting

class BeliefUpdater:
    _need_further_config = False
    def update(self, prev_states, states):
        raise NotImplementedError

    def __call__(self, prev_states, states):
        beliefs = self.update(prev_states, states)
        states.update(beliefs)
        return states

class BayesianMeanTracker(BeliefUpdater):
    
    def __init__(self, var_error=None):
        
        if var_error is None:
            self.var_error = var_error
            self._need_further_config = True
            return 
        
        self.set_var_error(var_error)
        
    def set_var_error(self, var_error):
        if isinstance(var_error, TaskSetting):
            var_error = var_error.var

        if np.array(var_error).min() <= 0:
            raise ValueError('"var_error" needs to be positive')

        if isinstance(var_error, list) or isinstance(var_error, np.ndarray):
            var_error = np.array(var_error).reshape(-1,1) # column-vectorize it

        self.var_error = var_error
        self._need_further_config = False

    def update(self, prev_states, states):
        G_t = prev_states.V / (prev_states.V + self.var_error) # kalman gain
        GA_t = states.G * states.A # only update the ones acted upon at t
        dM_t = GA_t * (states.Y - prev_states.M) # update posterior mean
        dV_t = GA_t * prev_states.V # update posterior var (uncertainty)
        M_t = prev_states.M + dM_t
        V_t = prev_states.V + dV_t

        return dict(
            M = M_t,
            V = V_t,
            G = G_t
        )