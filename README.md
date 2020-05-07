# Lunar-Lander

Train an agent to successfully land the 'Lunar Lander' in OpenAI gym. The problem is considered solved when achieving a score of 200 or higher on average over 100 consecutive runs.

Course Instructor has all the rights on course materials, homeworks, exams and projects. Please note that unauthorized use of any previous semester course materials, such as tests, quizzes, homework, projects, videos, and any other coursework, is prohibited in this course.


## Algorithms 

Three methods, Deep Q-learning Network(DQN), Double Deep Q-learning Network(DDQN) and DQN with soft update were applied and different settings of hyperparameters were compared. As Lunar Lander environment has already feature vectors, plain neural network with fully connected layers was used.

Deep Q-learning mainly proposes two modifications to make standard online Q-learning more stable:
1. Separate Target Network. The target network is kept frozen most of the time and updated every several steps to avoid the divergence of value estimates casued by frequent updates;

2. Experience Replay. The experience of the agent is stored in a pool. Later the experience is sampled from this pool to break the correlations among the samples and thus reducing the variance of updates and divergence of the parameters.

In DDQN, to alleviate the overestimation of Q value, the next action is selected by taking the one that maximize the value of policy network rather than the target network.

Compared with original DQN, DQN with soft update aims to avoid 'hard' updating the target network by keeping part of the parameters of the old target network.

The results of three algorithms shared the similar pattern, and all converged. However, DDQN and DQN with soft update didn't seem to have improved the learning process significantly. All algorithms reached desired rewards after 100 runs.

## Hyperparameters
Experiments with different hyperparameters were conducted with original DQN algorithm. 

#### 1. ε Decay Values
Three ε decay values, 0.995, 0.9999 and 0.9 were tested to see the impact on the performance. 
A large ε decay value encourages more exploration at initial phase. It turned out that with a value of 0.9999 the training process seemed to be just exploring rather than learning from previous experiences while a value of 0.9 demonstrated a slower learning pace at beginning but was able to improve and reached a desired average reward.

#### 2. Learning rate α
Three α values, 5e-4,1e-4, and 5e-5 were tested to see the impact on the performance. As expected, a smaller learning rate took more time to converge. However, the learning rate with 5e-5  was not able to pass the criteria and also yielded a negative rewards after 100 trials.


#### 3. Target Update Frequency
The frequency of updating target netwirk may impact the stability of the training process, hence three update frequency 5,10 and 100 were tested. The results showed that when updated every 100 steps, the agent failed to achieve the average rewards probably because the target network was too 'old' with such interval.



