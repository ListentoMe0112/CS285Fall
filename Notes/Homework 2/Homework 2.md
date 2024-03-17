# Experiment
## Experiment 1

Command to run experiment, convenient for copy.

```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000  --exp_name cartpole 
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000  -rtg --exp_name cartpole_rtg 
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000  -na --exp_name cartpole_na 
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000  -rtg -na --exp_name cartpole_rtg_na 
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000  --exp_name cartpole_lb 
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000  -rtg --exp_name cartpole_lb_rtg 
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000  -na --exp_name cartpole_lb_na 
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000  -rtg -na --exp_name cartpole_lb_rtg_na
```
![[Experiment1.png]]
Q:Which value estimator has better performance without advantage normalization: the trajectory- centric one, or the one using reward-to-go?
A: Reward-to-go seems to be better in small batch. 
Q:Did advantage normalization help?
A: More useful in minibatch, but not very useful for large batch.
Q: Did the batch size make an impact?
A: Yes, larger size and more stable.

## Experiment 2
```shell
# No baseline 
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4  -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01  --exp_name cheetah 
# Baseline 
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4  -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01  --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```
- Plot a learning curve for the baseline loss.
- Plot a learning curve for the eval return. You should expect to achieve an average return over 300 for the baselined version. (A little lower than 300, extend iterations num to achieve)
![[Experiment2.png]]

## Experiment 3
```shell
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0 --exp_name lunar_lander_lambda0
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name lunar_lander_lambda0.95
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.98 --exp_name lunar_lander_lambda0.98
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.99 --exp_name lunar_lander_lambda0.99
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 1 --exp_name lunar_lander_lambda1
```
- Provide a single plot with the learning curves for the LunarLander-v2 experiments that you tried. Describe in words how λ affected task performance. The run with the best performance should achieve an average score close to 200 (180+)
![[Experiment3.png]]
- Consider the parameter λ. What does λ = 0 correspond to? What about λ = 1? Relate this to the task performance in LunarLander-v2 in one or two sentences
λ = 0 means TD(0)。
λ = 1 means MC Sample.

## Experiment 4
```shell
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100  --exp_name pendulum_default_s1  -rtg --use_baseline -na  --batch_size 5000  --gae_lambda 0.98 --seed 1 
```
The default arguments work wells to reach max average evaluation return.
## Experiment 5
```shell
python cs285/scripts/run_hw2.py  --env_name Humanoid-v4 --ep_len 1000  --discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001  --baseline_gradient_steps 50  -na --use_reward_to_go --use_baseline --gae_lambda 0.97  --exp_name humanoid --video_log_freq 5
```

Although I don't reach return 300 by iterations which means I probably have a bug as material noted, I do reach a 600 average evaluation return within 10 hours.
# Analysis
