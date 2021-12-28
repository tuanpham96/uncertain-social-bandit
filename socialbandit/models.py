import pandas as pd
import numpy as np
import xarray as xr
import numpy.random as npr
import copy
from autoslot import Slots, SlotsPlusDict
import math

from tqdm import tqdm
from pathlib import Path

from socialbandit.utils import * 
from socialbandit.initializers import *
from socialbandit.tasks import *
from socialbandit.beliefs import *
from socialbandit.social import *
from socialbandit.learners import *


class SocialMultiArmedBandit(BanditDimensions):

    def __init__(self,
                 task_settings, social_settings, 
                 initializer = None, 
                 clock = ExperimentClock(T = 100),
                 action_sampler = SoftmaxSampler(tau = 1.0),
                 belief_updater = BayesianMeanTracker(var_error = 1.0),
                 action_learner = MeanGreedyExploit(),
                 social_learner = SocialMassPower(alpha = 0.8)
                 ):
        super().__init__(task_settings.K, social_settings.N)

        self.task_settings = task_settings
        self.social_settings = social_settings
        
        self.__set_initializer(initializer)
        
        self.clock = clock
        self.action_sampler = action_sampler
        
        self.belief_updater = belief_updater
        self.action_learner = action_learner
        self.social_learner = social_learner

        self.states = None
        self.prev_states = None

    def __set_initializer(self, initializer=None):
        if initializer:
            self.initializer = initializer
            return 
        
        # by default set optimistic priors
        self.initializer = InitEqualProb(
            K=self.K, N=self.N, 
            mu_0=self.task_settings.mean.max(), 
            sigma2_0=self.task_settings.var.mean()
        )
        
    def set_state_managers(self, saved_states = [], extra_states = [], save = False, save_args = dict()):
        self._save = save
        if save and ('file_prefix' not in save_args):
            raise ValueError('Need a "file_prefix" key in "save_args" if "save=True"')

        self.save_args = save_args

        self.states = States(
            K=self.K, N=self.N, t=0,
            saved_states = saved_states,
            extra_states = extra_states
        )
        self.states_are_set = True

    @property
    def t(self):
        return self.clock.t

    def initialize(self):
        self.initializer(states = self.states)
        self.cycle_state()

    def update_cum_reward(self):
        self.states.Yc = self.states.Y + self.states.Yc

    def cycle_state(self):
        self.prev_states = self.states.clone()
        self.states.step() # update t in states
        self.clock.update() # update t in clock

    def sample_action(self):
        t = self.states.t
        rho_t = self.task_settings.get_rho(t) # task availability
        self.states.set_rho(rho_t)

        if not self.clock.is_at('action_sampling'):
            self.states.A = weighted_choice_matrix(self.prev_states.P * rho_t)
            return

        self.action_sampler(
            prev_states = self.prev_states,
            rho = rho_t,
            states = self.states
        ) # states updated inside

    def sample_reward(self):
        self.states.Y = self.task_settings.sample(action=self.states.A)
        self.update_cum_reward()

    def update_belief(self):
        if not self.clock.is_at('belief_updating') or self.belief_updater is None:
            return
        
        self.belief_updater(
            prev_states = self.prev_states,
            states = self.states
        ) # states updated inside

    def learn_action(self):
        if not self.clock.is_at('action_learning') or self.action_learner is None:
            return self.prev_states.Q_acl

        return self.action_learner(states=self.states)

    def set_social(self):
        if self.social_settings is None: 
            return 
        
        self.social_settings(
            prev_states = self.prev_states,
            states = self.states
        )

    def learn_social(self):
        if not self.clock.is_at('social_learning') or self.social_learner is None:
            return self.prev_states.Q_sol
        
        self.set_social()
        return self.social_learner(states=self.states)

    def update_utility(self):
        Q_acl = self.learn_action()
        Q_sol = self.learn_social()
        self.states.Q = Q_acl + Q_sol

    def step(self):
        # this can also be overriden, but should keep order 
        self.sample_action()
        self.sample_reward()
        self.update_belief()
        self.update_utility()
        
        self.analyze()
        self.save()
        
        self.cycle_state()
    
    def save(self):
        # this can/should be overriden
        # go to States.save() and States.to_xarray() to know more
        if self._save: # save states first
            self.states.save(**self.save_args)
            
    def analyze(self):
        # this can/should be overriden
        pass         
        
    def run(self, dT=None, tqdm_fn=tqdm):
        # this can/should be overriden    
        # this is a template, could just have a customized loop
        # dT is number of trials to continue, if not provided, using clock.T
        T = dT if dT else self.clock.T

        if self.t == 0:
            self.initialize()

        if not self.states_are_set:
            raise ValueError('States are not set yet, please run "SocialMultiArmedBandit.set_state_managers"')

        for t in tqdm_fn(range(T)):
            self.step()


class States(SlotsPlusDict,BanditDimensions):
    __default_saved_states = {'Q','A','Y','Yc','M','V','W'} # default states to save
    __allowable_extras = {} # TBD
    __attr_desc = dict(
        t = dict(name='t', type='dim'),
        K = dict(name='number of tasks (arms, actions)', type='dim'),
        N = dict(name='number of agents', type='dim'),
        Q = dict(name='total utility matrix', type='state', dim=['task', 'agent']),
        Q_acl = dict(name='[ac]tion [l]earning utility matrix', type='state', dim=['task', 'agent']),
        Q_sol = dict(name='[so]cial [l]earning utility matrix', type='state', dim=['task', 'agent']),
        A = dict(name='action matrix', type='state',  dim=['task', 'agent']),
        P = dict(name='action probability matrix', type='state',  dim=['task', 'agent']),
        Y = dict(name='reward matrix', type='state',  dim=['task', 'agent']),
        Yc = dict(name='cumulative reward matrix', type='state',  dim=['task', 'agent']),
        M = dict(name='belief mean matrix', type='state',  dim=['task', 'agent']),
        V = dict(name='belief uncertainty matrix', type='state',  dim=['task', 'agent']),
        G = dict(name='kalman gain matrix', type='state',  dim=['task', 'agent']),
        C = dict(name='social content matrix', type='state',  dim=['task', 'agent']),
        W = dict(name='social network', type='state',  dim=['neighbor', 'agent']),
        rho = dict(name='task (arm) availability', type='state',  dim=['task', '_']),
    )

    def __init__(self, K, N, t=0, saved_states = [], extra_states=[]):
        # saved_states: states to save, the default is `__default_saved_states`
        # extra_states: extra states to track, TBD for now
        super().__init__(K, N)
        self.Q = np.zeros((K,N))        # utility
        self.A = nan_matrix((K,N))      # action
        self.Y = nan_matrix((K,N))      # reward
        self.Yc = nan_matrix((K,N))     # cumulative reward
        self.M = nan_matrix((K,N))      # belief mean
        self.V = nan_matrix((K,N))      # belief var

        self.P = nan_matrix((K,N))      # action prob
        self.G = nan_matrix((K,N))      # kalman gain
        self.W = nan_matrix((N,N))      # social network
        self.C = nan_matrix((K,N))      # content matrix

        self.Q_acl = np.zeros((K,N))    # action learning utility
        self.Q_sol = np.zeros((K,N))    # social learning utility
        self.t = t                      # time step
        self._done = False              # signal whether all states are done being used

        self.rho = nan_matrix((K,1))

        self.set_saved_states(saved_states)
        self.set_extra_states(extra_states)

    def set_rho(self, rho):
        self.rho = rho

    def set_saved_states(self, saved_states):
        state_attrs = set([
            k for k, v in self.__attr_desc.items()
            if v['type'].lower() == 'state'
        ])

        if len(saved_states) == 0:
            saved_states = copy.deepcopy(self.__default_saved_states)
        else:
            for _state in saved_states:
                if not hasattr(self, _state):
                    raise ValueError('"%s" is not an allowable state to save' %(_state))

        self.saved_states = set(saved_states).union(state_attrs)

    def set_extra_states(self, extra_states):
        if len(self.__allowable_extras) == 0 and len(extra_states) > 0:
            raise NotImplementedError('Currently cannot add any more extra states to track. Please set "extra_states=[]"')

        for _state in extra_states:
            if _state not in self.__allowable_extras:
                raise ValueError('"%s" is not an allowable extra state to track' %(_state))
            dim_mat = (K,N) if _state != 'W' else (N,N)
            setattr(self, _state, nan_matrix(dim_mat))

        self.extra_states = set(extra_states)

    def clone(self, done=True):
        s_clone = copy.deepcopy(self)
        s_clone._done = done
        return s_clone

    def update(self, data):
        # data is a dict
        # save items of data to self if matching key=attribute
        for k, v in data.items():
            if hasattr(self, k):
                state_dim = getattr(self, k).shape
                new_dim = v.shape
                if getattr(self, k).shape != v.shape:
                    raise ValueError('Dimension mismatch between current ({}) and new values ({}) of "{}" state'.format(state_dim, new_dim, k))
                setattr(self, k, v)

    def step(self):
        self.t += 1
        self._done = False # new time step, reset _done
    
    def describe(self, state=None):
        if state:
            return self.__attr_desc[state] 
        return self.__attr_desc
    
    def to_xarray(self, attrs=dict()):
        def _dims_and_data(state):
            dims = ['t'] + self.describe(state)['dim']
            data = [getattr(self, state)]
            return (dims, data)
        
        return xr.Dataset(
            data_vars = {_state: _dims_and_data(_state) for _state in self.saved_states},
            coords = dict(t = [self.t]),
            attrs = dict(state_desc = self.describe(), **attrs)
        )

    def to_sparse(self):
        raise NotImplementedError('Sparse saving is not currently available')

    def save(self, file_prefix, concat=True, save_every=1, desc=dict(), t_fmt = '%06d',
             ds_save_engine=None, ds_open_engine='h5netcdf', *args, **kwargs):
        # methods involving saving and to_xarray are still experimental 
        # consider not concat as would be more time and memory expensive
        # and also don't need to save all the time

        if self.t % save_every != 0:
            return

        # process file path
        file_obj = Path(file_prefix)
        file_suffix = '.nc'
        if not concat: # each t is a file
            file_suffix = '_' + t_fmt %(self.t) + file_suffix
        file_obj = file_obj.parent / (file_obj.name + file_suffix)
        file_exists = file_obj.exists()

        # get the data
        data = self.to_xarray(desc)

        # if concat and file already exists
        if concat and file_exists:
            prev_data = xr.open_dataset(file_obj, engine = ds_open_engine)
            data = xr.concat([prev_data, data], dim='t')

        # save
        data.to_netcdf(file_obj, engine = ds_save_engine)
        return str(file_obj)

