# Basic Q-Learning
- Submit your logs of CartPole-v1, and a plot with environment steps on the x-axis and eval return on the y-axis.
![[CartPole0.0.1.png]]
- Run DQN with three different seeds on LunarLander-v2:
 ```python
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1 python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2 python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3
```
**Your code may not reach high return (200) on Lunar Lander yet; this is okay!** Your returns may go up for a while and then collapse in some or all of the seeds.
![[Figure_1.png]]
Not stable to get the reward.
- Run DQN on CartPole-v1, but change the learning rate to 0.05 (you can change this in the YAML config file). What happens to (a) the predicted Q-values, and (b) the critic error? Can you relate this to any topics from class or the analysis section of this homework?
![[CartPole_lr.png]]
Your would get larger q values and larger critic loss because of the max bias.

# Double Q-Learning

- Run three more seeds of the lunar lander problem:
```python
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml -- seed 1 
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml -- seed 2 
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml -- seed 3
```

You should expect a return of 200 by the end of training, and it should be fairly stable compared to your policy gradient methods from HW2. Plot returns from these three seeds in red, and the “vanilla” DQN results in blue, on the same set of axes. Compare the two, and describe in your own words what might cause this difference.
![[ddqn_lunar.png]]
Double deep q learning alleviates the max bias and get  a better result.(?)
- Run your DQN implementation on the MsPacman-v0 problem. Our default configuration will use doubleQ learning by default. You are welcome to tune hyperparameters to get it to work better, but the default parameters should work (so if they don’t, you likely have a bug in your implementation). Your implementation should receive a score of around 1500 by the end of training (1 million steps. This problem will take about 3 hours with a GPU, or 6 hours without, so start early!
```python
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yam
```
- Plot the average training return (train_return) and eval return (eval_return) on the same axes. You may notice that they look very different early in training! Explain the difference.
![[double_dqn_mapacman.png]]

e-greedy, epsilon is larger at start.

# SAC

## Actor with REINFORCE
![[Half_cheetah.png]]
You will get a better result for REINFORCE-10 because of the better estimation of gradient with low variance.
## Actor with REPARAMETERIZE
![[reprametrize.png]]
You get a better estimate of gradient and get a better result for this.
![[humanoid.png]]
Better performance compared to policy gradient.