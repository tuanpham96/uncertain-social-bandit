### 1. Brief description of change

For specific notations, see [pdf file](https://github.com/tuanpham96/uncertain-social-bandit/blob/main/docs/methods.pdf). In the paper and the [replication notebook](https://github.com/tuanpham96/uncertain-social-bandit/blob/main/notebooks/replication-attempts.ipynb), the social influence on utility of a given choice is from the number of agents with choosing the same choice on the previous time step, of an *all-to-all* network of size $N=20$. The social influence "factor" is controled by $\alpha$. The equation would be as followed:

$$
\mathbf{Q}_t =
    \mathbf{M}_t +
    \left(
        \mathbf{A}_{t-1}\mathbf{W}
    \right)^{\alpha}
$$

However, this does not seem to be a good choice of social influence integration when $N$ gets larger. A way to control this is using mean social content instead.

$$
\mathbf{Q}_t =
    \gamma \mathbf{M}_t +
    (1-\gamma) \mathbf{C}_{t-1} \psi_1\left(\mathbf{W}\right)
    \text{ where } \mathbf{C} \in \left\{ \mathbf{M}, \mathbf{Y} \right\}
    \text{ and } \psi_1 \text{ is a column } L^1 \text{-norm}
$$

The term $\mathbf{C}$ is called content, and can be either the belief ($\mathbf{M}$) or reward ($\mathbf{Y}$) from previous time step from the neighbors. The term $\psi_1\left(\mathbf{W}\right)$ is normalizing the neighborhood with $L^1$-norm, so the social influence on choice $i$ of agent $j$ would be the mean belief (or reward) on the same choice $i$ only from neighbors of $j$. What this modification of the social integration is attempting is to have the same units between the belief influence and the mean social influence on the final utility.

There are two variations to control social influence "factor" here. One is the explicit $\gamma$ factor to scale the mean belief and social influence. The other is the initial creation of $\mathbf{W}$ (so far still considering static social networks), in which $p$ controls the connectivity probability for an ER network or SBM communities. The experiment notebook can be accessed [here](https://github.com/tuanpham96/uncertain-social-bandit/blob/main/notebooks/changed-social-integration.ipynb)

### 2. Parameters

#### 2.1. Experiment variations

- Social network types and the connectivity prob `p` (colors)
    - `ER`: Erdos-Renyi network with prob of neighbor `p`
    - `SBM`: Stochastic block model with 4 communities, wither inter-block prob as 0.01, and varying intra-block prob `p`
- Factor between self- and social-derived beliefs: `utility_gamma`, i.e. $\gamma$, lower $\gamma$ is higher social influence
- Social content, i.e. $\mathbf{C}$, which is the content used for social influence integration, which is either neighbors' previous *belief* or *reward*

#### 2.2. Constants

- Number of agents: $N=100$
- Number of tasks: $K=144$
    - Childhood period: $t \in [0, 400]$
    - Adolescence period (increased possible options): $t \in [400, 800]$
    - Adult period (increased possible options): $t \in [800, 1200]$
- Initial optimistic mean belief: $\mu_0 = 100$
- Initial uncertainty: $\sigma^2_0 = 40$
- Softmax "temperature" for sampling: $\tau_s = 1.0$
- Bayes mean tracker error constant: $\sigma^2_{\epsilon} = 3600$

### 3. Result quantifications

#### 3.1. Exploration measures

- `explore_num`: this was first calculated bin by bin, to see whether the `choice[t]` is different from all previous `choice[:t]`, then average every 50 bins, non-overlappingly. This was an attempt to replicate the "Exploration" measure in paper.
- `unq_choices`: for every 50 bins, calculate how many unique choices, i.e. `len(unique(choice[t-50:t]))`
- `explore_ent`: for every 50 bins, calculate the entropy of the choice distribution, i.e. `entropy(choice[t-50:t])`

#### 3.2. Reward measures
- `mean_reward`: mean reward every development phase (400 bins), disregarding signs
- reward separations every development phase
    - `loss_*` is `reward < 0` and `gain_*` is `reward > 0`
    - `*_num` is the mean number of outcomes corresponding to reward signs
    - `*_mag` is the mean magnitude of outcomes corresponding to reward signs

