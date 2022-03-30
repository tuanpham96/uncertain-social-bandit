# Notes on MAB, thompson sampling, UCB

- source: https://speekenbrink-lab.github.io/modelling/2019/02/28/fit_kf_rl_1.html

```
rl_softmax_sim <- ...
    p <- exp(gamma*m[t,])
    p <- p/sum(p)
    choice[t] <- sample(1:4,size=1,prob=p)


rl_ucb_sim <- ...
    choice[t] <- which.is.max(m[t,] + beta*sqrt(v[t,]) + sigma_xi_sq)


rl_thompson_sim <- ...
    sim_r <- rnorm(4,mean=m[t,],sd=sqrt(v[t,] + sigma_xi_sq))
    choice[t] <- which.is.max(sim_r)
```
