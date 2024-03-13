## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2023/blob/master/hw1/cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`. In particular, edit
 - [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
 - [infrastructure/utils.py](cs285/infrastructure/utils.py)
 - [scripts/run_hw1.py](cs285/scripts/run_hw1.py)

You have the option of running locally or on Colab using
 - [scripts/run_hw1.py](cs285/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](cs285/scripts/run_hw1.ipynb) (if running on Colab)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Section 1 (Behavior Cloning)
Command for problem 1:

```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```

If running on Colab, you will be using the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](cs285/scripts/run_hw1.ipynb) for more details.

# My Answer Start
# Analysis 
### Question 1
![[Snipaste_2024-03-13_21-49-14.jpg]]
![[Snipaste_2024-03-13_21-48-07.jpg]]
### Question 2
![[Pasted image 20240313215145.png]]
Simply replacing $$C_{t}\left( S_{t}\right)$$ with $$R_{max}$$ 
you will get your answer.
# Experiment
This is homework for imitation learning.
The Iterations used for Dagger is 10.
The eval batch size is 1000 and the length of ep is 1000 which would produce five trajectories to generate mean and std and avoid coincidence.

All the other hyperparameters  is same as the default.
## Ant Example

| PolicyName     | Average | Std   | Max     | Min     |
| -------------- | ------- | ----- | ------- | ------- |
| Expert         | 4681.89 | 30.70 | 4712.60 | 4651.18 |
| Behavior Clone | 4602.60 | 69.92 | 4692.87 | 4501.71 |
| Dagger         | 4602.60 | 69.92 | 4692.87 | 4501.71 |
The performance of Dagger and Behavior is same.

### HalfCheetah Example

| PolicyName     | Average | Std   | Max     | Min     |
| -------------- | ------- | ----- | ------- | ------- |
| Expert         | 4036.2  | 0     | 4036.2  | 4036.2  |
| Behavior Clone | 3921.44 | 72.41 | 4013.36 | 3792.21 |
| Dagger         | 4033.87 | 45.24 | 4084.67 | 3963.8  |
Dagger is a liitle better.

I cann't run Hopper or Walker due to the latest MuJoCo not support global coordinates.
So I can't try a task which behavior clone act really worse.
Do this homeword just for fun.