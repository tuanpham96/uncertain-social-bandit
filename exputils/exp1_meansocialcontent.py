import pandas as pd
import numpy as np
import networkx as nx 

import itertools
from functools import partial
from tqdm.contrib.concurrent import process_map
import yaml
import pickle 
import os

import socialbandit as sb
from exputils.utils import *

class SocialMAB_ChildDev(sb.models.SocialMultiArmedBandit):
    def __init__(self, 
                 num_agents=20, 
                 max_time=1200,
                 social_content = 'belief', 
                 social_network = None,
                 gamma = 1,
                 mu0 = 100.0, 
                 var0 = 40.0, 
                 softmax_tau = 1.0, 
                 BMT_error = 'use-tasks', 
                 agg_explore_step=50, 
                 agg_reward_step=400):
        
        if type(gamma) is list:
            if len(gamma) != 2: 
                raise ValueError("`gamma` can only either be a 2-element list or a scalar")
        else:
            if gamma < 0 or gamma > 1:
                raise ValueError("`gamma` needs to be in [0, 1] if it's a scalar input")
            gamma = [gamma, 1.0 - gamma]
            
        self.gamma_acl = gamma[0] # for action learning scaling
        self.gamma_sol = gamma[1] # for social learning scaling
        
        # Social content for social influence
        social_content_opts = {
            'belief': 'M',
            'action': 'A',
            'reward': 'Y', 
            'cum_reward': 'Yc',
            'mean_reward': ('Yc', PseudoAverageCumTime())
        }
        
        social_content = social_content.lower()
        if social_content not in social_content_opts: 
            raise ValueError("`social_content` cannot be {social_content}. Available options are {list(social_content_opts.keys())}.")

        content_fn = sb.social.ContentConstructor(*social_content_opts[social_content])
        
        # Social network constructor/functions
        social_fn = None
        if social_network is not None:                
            social_fn = sb.social.StaticSocialNetwork(W = social_network)
        
        # Social setting constructur from social content and network        
        social_settings = sb.social.SocialSetting(
            N = num_agents, 
            social_fn = social_fn,
            content_fn = content_fn
        )
        
        # Social learner constructor
        social_learner = None
        if social_network is not None:
            social_learner = sb.learners.MeanFriendContentLearner()
        
        # Childhood development settings
        task_settings = sb.tasks.ChildDevelopmentEnvironment(env_levels = 12)
        
        # Initializer
        initializer = sb.initializers.InitializerBanditAgnostic(
            mu_fn = lambda x: mu0,
            sigma2_fn = lambda x: var0
        )
        
        # Sampler
        action_sampler = sb.tasks.SoftmaxSampler(tau = softmax_tau)
        
        # Bayesian mean tracker to update belief
        if type(BMT_error) is str:
            if BMT_error.lower() == 'use-tasks':
                bmt_var_err = task_settings
            else: 
                try:
                    bmt_var_err = float(BMT_error)
                except:
                    raise ValueError("If `BMT_error` is a string, only 'use-tasks' or a string of a float is accepted at this point")
        elif type(BMT_error) in [float, int]:
            bmt_var_err = BMT_error
        else:
            raise ValueError("`BMT_error` can either be a str/float/int")
        
        belief_updater  = sb.beliefs.BayesianMeanTracker(var_error = bmt_var_err)
        
        # Initialize 
        super().__init__(
            task_settings   = task_settings,
            social_settings = social_settings,
            clock           = sb.utils.ExperimentClock(T = max_time),
            initializer     = initializer,
            action_sampler  = action_sampler,
            belief_updater  = belief_updater,
            action_learner  = sb.learners.MeanGreedyExploit(),
            social_learner  = social_learner
        )
        
        self.set_state_managers()
        
        # Set up analyses variables
        analysis_size = (max_time+1, num_agents)
        cum_choice_size = (self.num_tasks, num_agents)
        self.analysis = dict(
            params = dict(
                agg_explore_step = agg_explore_step, 
                agg_reward_step = agg_reward_step
            ),
            per_agent = dict(
                cum_choice = np.zeros(cum_choice_size),
                explore = np.zeros(analysis_size),
                choice = np.zeros(analysis_size),
                reward = np.zeros(analysis_size)
            ),
            time = dict(
                explore = range(0, max_time, agg_explore_step),
                reward = np.array(range(0, max_time, agg_reward_step)) + agg_reward_step,
            ),
            aggregate = dict(
                explore_num = [],
                unq_choices = [],
                explore_ent = [], 
                mean_reward = [],
                loss_num = [],
                loss_mag = [],
                gain_num = [],
                gain_mag = []                
            ),
            aux = dict(
                Q_acl = np.zeros(max_time+1), 
                Q_sol = np.zeros(max_time+1)
            )
        )

    def update_utility(self):
        Q_acl = self.gamma_acl * self.learn_action()
        Q_sol = self.gamma_sol * self.learn_social()
        self.states.Q = Q_acl + Q_sol

    def analyze(self):
        t = self.t
        
        # auxilary 
        aux_var = self.analysis['aux']
        aux_var['Q_acl'][t] = np.mean(self.states.Q_acl)
        aux_var['Q_sol'][t] = np.mean(self.states.Q_sol)
        
        # analysis
        anly_per_agent = self.analysis['per_agent']
        anly_params = self.analysis['params']
        agg_anly = self.analysis['aggregate']
        
        anly_per_agent['explore'][t,:] = is_prevchoice_diff(anly_per_agent['cum_choice'], self.states.A)
        anly_per_agent['cum_choice'] += self.states.A
        
        anly_per_agent['choice'][t,:] = get_task_choices(self.states.A)
        anly_per_agent['reward'][t,:] = get_task_rewards(self.states.Y)
        
        agg_explore_step = anly_params['agg_explore_step']
        if t % agg_explore_step == 0 and t > 1:
            agg_anly['explore_num'].append(
                np.mean(anly_per_agent['explore'][t-agg_explore_step:t,:].sum(axis=0))        
            )
            
            # choice_matrix = anly_per_agent['choice'][t-agg_explore_step:t,:]
            choice_matrix = anly_per_agent['choice'][:t,:]
            agg_anly['explore_ent'].append(
                np.mean([calculate_entropy(X) for X in choice_matrix.T])                
            )
            
            agg_anly['unq_choices'].append(
                np.mean([len(np.unique(X)) for X in choice_matrix.T])     
            )
            
            
            
        agg_reward_step = anly_params['agg_reward_step']
        if t % agg_reward_step == 0 and t > 0:
            reward_matrix = anly_per_agent['reward'][t-agg_reward_step:t]
            agg_anly['mean_reward'].append(np.mean(reward_matrix))
            agg_loss = aggregate_rewards(reward_matrix, reward_matrix < 0)
            
            agg_anly['loss_num'].append(agg_loss['num'])
            agg_anly['loss_mag'].append(agg_loss['mag'])
            
            agg_gain = aggregate_rewards(reward_matrix, reward_matrix > 0)
            
            agg_anly['gain_num'].append(agg_gain['num'])
            agg_anly['gain_mag'].append(agg_gain['mag'])
            
        if t == self.clock.T:
            for k, v in agg_anly.items():
                agg_anly[k] = np.array(v)

def _construct_sbm(N, k, p_max, p_min=0.01, *args, **kwargs):
    block_sizes = [round(N / k)] * (k-1) 
    block_sizes += [N - sum(block_sizes)]
    
    block_probs = np.eye(k) * (p_max - p_min) + p_min
    
    net = nx.stochastic_block_model(
        sizes = block_sizes,
        p = block_probs,
        *args, **kwargs
    )
    
    return net 

def construct_social_network(num_agents, net_kwargs):
    if net_kwargs is None:
        return None
    
    net_type = net_kwargs['type']
    if type(net_type) is str:
        net_type = net_type.upper()
    if net_type == 'NONE' or net_type is None:
        return None
    
    
    
    net_args = {k: v for k, v in net_kwargs.items() if k != 'type'}
    
    net_fn_maps = dict(
        ER = nx.erdos_renyi_graph,
        SBM = _construct_sbm, 
        BA = nx.barabasi_albert_graph
    )
    
    if net_type not in net_fn_maps:
        raise ValueError(f'Only one in these types {list(net_fn_maps.keys())} are accepted, not {net_type}')
    
    net_fn = net_fn_maps[net_type]
    net = net_fn(num_agents, **net_args)
    net = nx.to_numpy_array(net)
    return net 

def experiment_fn(num_agents, social_content, social_net_args, utility_gamma, BMT_error, mu0=100.0, *args, **kwargs):
    
    social_network = construct_social_network(num_agents, social_net_args)
    
    model = SocialMAB_ChildDev(
        num_agents = num_agents,
        social_content = social_content, 
        social_network = social_network,
        gamma = utility_gamma,
        BMT_error = BMT_error,
        mu0 = mu0
    )

    model.run(tqdm_fn=None)
    
    result_keys = dict(
        explore = ['explore_num', 'unq_choices', 'explore_ent'],
        reward = ['mean_reward', 'loss_num', 'loss_mag', 'gain_num', 'gain_mag']
    )
    
    agg_anly = model.analysis['aggregate']
    agg_time = model.analysis['time']
    
    results = {}
    for section_key, agg_keys in result_keys.items():
        results[section_key] = {k: v for k, v in agg_anly.items() 
                                if k in agg_keys}
        results[section_key].update({'time': agg_time[section_key]})
        results[section_key] = pd.DataFrame(results[section_key])
    
    file_name = kwargs.get('file_name', None)
    if file_name:
        with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(model.analysis, f)
            
    return results

def save_results(save_dir, results, variations, var_dict):
    results = {k: pd.concat([x[k] for x in results], ignore_index=True)
               for k in results[0].keys()}
    results.update({'variations': pd.DataFrame(variations)})
    
    for k, df in results.items(): 
        file_name = os.path.join(save_dir, k + '.parq')
        df.to_parquet(file_name, engine='fastparquet')
    
    with open(os.path.join(save_dir, 'variations.yaml'), 'w') as f: 
        yaml.safe_dump(var_dict, f)
    
    return results

def _run_1_exp(variation, fn, common_args=dict()):
    dfs = fn(**variation)
    unnest_var = unnest_dict(variation)
    dfs = {k: df.assign(**unnest_var) for k, df in dfs.items()}    
    return dfs

def _create_save_sub_dir(save_dir, sub_dir = 'sims'):
    sub_dir = os.path.join(save_dir, sub_dir) 
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir) 
    return sub_dir

def run_experiment(var_dict, save_dir,  
                   fn = experiment_fn,
                   num_trials=1, 
                   save_detailed_analysis=False,
                   common_args = dict(),
                   max_workers=2, chunksize=1): 
    
    fp4v = None 
    if save_detailed_analysis:
        fp4v = _create_save_sub_dir(save_dir, 'sims') + '/data'
        
    variations = create_variations(
        var_dict, 
        num_trials=num_trials,
        file_prefix=fp4v
    )
    
    print(f'Number of total simulations = {len(variations)}')
    
    results = process_map(
        partial(_run_1_exp, fn=fn, common_args=common_args),
        variations, 
        max_workers=max_workers,
        chunksize=chunksize
    )
    
    variations = list(map(unnest_dict, variations))
    results = save_results(save_dir, results, variations, var_dict)
    
    return results