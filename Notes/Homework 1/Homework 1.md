# Analysis 
### Question 1
![[Snipaste_2024-03-13_21-49-14.jpg]]
![[Snipaste_2024-03-13_21-48-07.jpg]]
### Question 2
![[Pasted image 20240313215145.png]]
Simply replacing $$C_{t}\left( S_{t}\right)$$ with $$R_{max}$$you will get your answer.
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


