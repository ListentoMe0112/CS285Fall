# Problem 1
 Origin (layer = 1, hidden cells = 32)
!["L1-H32"][l1-h32.png]
Tuned
layer = 1, hidden cells = 64
 ![[l1-h64.png]]
layer = 2, hidden cells = 32
![[l2-h32.png]]
# Problem 2
![[obstacles_1_iter.png]]

The evaluation return is greater than -70 after one iteration.
# Problem 3
![[Problem3.png]]
You should expect rewards of around -25 to -20 for the obstacles env, rewards of around -300 to -250 for the reacher env, and rewards of around 250-350 for the cheetah env.

# Problem4
![[Problem4.png]]
The longer horizon is, the more error over time will be induces.
The ensemble size doesn't impact the performance too significant because of the difficulty of the task is too low.
The num of action sequences doesn't impact the performance too significant. It looks like 500 action sequences is still enough for this task to get a good estimate.
# Problem5
![[Problem5.png]]
You should expect rewards around 800 or higher when using CEM on the cheetah env. Try a cross entropy method iterations value of both 2 and 4, and compare results.
Include a plot comparing random shooting (from Problem 3) with CEM, as well as captions that describe how CEM affects results for different numbers of sampling iterations (2 vs. 4).

More iterations for cross entropy method would get a better estimation of action distribution and get a better performance.

# Problem6
![[Problem6.png]]