import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pickle
from collections import deque
from collections import namedtuple
from statistics import mean


#### Neural Network ####
class NN(nn.Module):
    def __init__(self,fc1_size,fc2_size):
        super(NN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(N_STATES,fc1_size)
        self.fc2 = nn.Linear(fc1_size,fc2_size)
        self.out = nn.Linear(fc2_size,N_ACTIONS)         
  
    def forward(self, x):
        t = F.relu(self.fc1(x))
        x = F.relu(self.fc2(t))
        action_values = self.out(x)      
        return action_values
    
    
#### Replay Memory ####
Transition = namedtuple('transition',('state_cur','action','reward','state_next','done'))
class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)  
        self.memory = []
        self.position = 0
        
    def push(self,*args):
        self.memory.append(Transition(*args)) 
       
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)
    
env = gym.make('LunarLander-v2')
#### HYPERPARAMS ####
BATCH_SIZE = 64
ALPHA = 5e-4
EPS_DECAY = 0.995
MIN_EPSILON = 0.01
TARGET_UPDATE = 10
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
MEMORY_CAPACITY = 100000
GAMMA = 0.99
FC1_SIZE = 64
FC2_SIZE = 64
MAX_STEPS = 1000
EPISODES = 1500
TAU = 0.01
seed = 0


# DQN
env.seed(seed)
memory_DQN = ReplayMemory(MEMORY_CAPACITY)
loss_func = nn.MSELoss()
step = 1
epsilon = 1
scores_DQN = []

policy_net_DQN,target_net_DQN = NN(FC1_SIZE,FC2_SIZE),NN(FC1_SIZE,FC2_SIZE)
target_net_DQN.load_state_dict(policy_net_DQN.state_dict())
optimizer = torch.optim.Adam(policy_net_DQN.parameters(),lr = ALPHA)
    
for i in range(EPISODES):
    s = env.reset()
    total_score = 0
    done = False

    for t in range(MAX_STEPS):
        s = torch.from_numpy(s).float().unsqueeze(0)
        
        if np.random.uniform() <= epsilon:
            action = np.random.randint(0,N_ACTIONS)
        else:
            action_values = policy_net_DQN.forward(s)
            action = torch.max(action_values,1)[1].data.numpy()
            action = action[0]
        
        a = action
        s_next,r,done,_ = env.step(a)
        memory_DQN.push(s,a,r,s_next,done)
        
        s = s_next
        total_score += r
        
        if memory_DQN.__len__() < BATCH_SIZE:
            continue

        transitions = memory_DQN.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        b_s = torch.from_numpy(np.vstack([s for s in batch.state_cur if s is not None])).float()
        b_a = torch.from_numpy(np.vstack([a for a in batch.action if a is not None])).long()
        b_r = torch.from_numpy(np.vstack([r for r in batch.reward if r is not None])).float()
        b_s_n = torch.from_numpy(np.vstack([s_n for s_n in batch.state_next if s_n is not None])).float()
        b_d = torch.from_numpy(np.vstack([d for d in batch.done if d is not None])).float()

        q_policy = policy_net_DQN(b_s).gather(1,b_a)
        q_next = target_net_DQN(b_s_n).detach()
        
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)*(1-b_d) # DQN
        loss = loss_func(q_policy,q_target)
      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % TARGET_UPDATE == 0:           
            target_net_DQN.load_state_dict(policy_net_DQN.state_dict())
    
        step+=1
        if done:
            break

    epsilon = max(MIN_EPSILON, epsilon * EPS_DECAY)
    scores_DQN.append(total_score)

pickle.dump(scores_DQN,open("scores_DQN","wb"))
fig = plt.figure()
plt.plot(np.arange(len(scores_DQN)),scores_DQN)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()



agent_DQN = []

for i in range(100):
    total_score = 0
    state = env.reset()
    done = 0
    while not done:
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = np.argmax(policy_net_DQN(state).cpu().data.numpy())
        state, reward, done, _ = env.step(action)        
        total_score += reward 
    agent_DQN.append(total_score)

pickle.dump(agent_DQN,open("agent_DQN","wb"))

print("Average reward in 100 trials: ", mean(agent_DQN))  
env.close()

fig = plt.figure()
plt.plot(np.arange(len(agent_DQN)), agent_DQN)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()


# DDQN
env.seed(seed)
step = 1
epsilon = 1
scores_DDQN = []

memory_DDQN = ReplayMemory(MEMORY_CAPACITY)
loss_func = nn.MSELoss()
policy_net_DDQN,target_net_DDQN = NN(FC1_SIZE,FC2_SIZE),NN(FC1_SIZE,FC2_SIZE)
target_net_DDQN.load_state_dict(policy_net_DDQN.state_dict())
optimizer = torch.optim.Adam(policy_net_DDQN.parameters(),lr = ALPHA)

for i in range(EPISODES):
    s = env.reset()
    total_score = 0
    done = False

    for t in range(MAX_STEPS):
        s = torch.from_numpy(s).float().unsqueeze(0)
        
        if np.random.uniform() <= epsilon:
            action = np.random.randint(0,N_ACTIONS)
        else:
            action_values = policy_net_DDQN.forward(s)
            action = torch.max(action_values,1)[1].data.numpy()
            action = action[0]
        
        a = action
        
        s_next,r,done,_ = env.step(a)
        memory_DDQN.push(s,a,r,s_next,done)
        
        s = s_next
        total_score += r
        
        if memory_DDQN.__len__() < BATCH_SIZE:
            continue

        transitions = memory_DDQN.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        b_s = torch.from_numpy(np.vstack([s for s in batch.state_cur if s is not None])).float()
        b_a = torch.from_numpy(np.vstack([a for a in batch.action if a is not None])).long()
        b_r = torch.from_numpy(np.vstack([r for r in batch.reward if r is not None])).float()
        b_s_n = torch.from_numpy(np.vstack([s_n for s_n in batch.state_next if s_n is not None])).float()
        b_d = torch.from_numpy(np.vstack([d for d in batch.done if d is not None])).float()

        q_policy = policy_net_DDQN(b_s).gather(1,b_a)
        q_next = target_net_DDQN(b_s_n).detach()
        
        with torch.no_grad():
            _,next_action = policy_net_DDQN(b_s_n).max(1, keepdim=True)
        q_target = b_r + GAMMA * q_next.gather(1,next_action) *(1-b_d) 
        loss = loss_func(q_policy,q_target)
      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % TARGET_UPDATE == 0:    

            target_net_DDQN.load_state_dict(policy_net_DDQN.state_dict())
    
        step+=1
        if done:
            break
    epsilon = max(MIN_EPSILON, epsilon * EPS_DECAY)
    scores_DDQN.append(total_score)

pickle.dump(scores_DDQN,open("scores_DDQN","wb"))
fig = plt.figure()
plt.plot(np.arange(len(scores_DDQN)),scores_DDQN)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()

agent_DDQN = []

for i in range(100):
    total_score = 0
    state = env.reset()
    done = 0
    while not done:
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = np.argmax(policy_net_DDQN(state).cpu().data.numpy())
        state, reward, done, _ = env.step(action)
        total_score += reward 
    agent_DDQN.append(total_score)

    
pickle.dump(agent_DDQN,open("agent_DDQN","wb"))

print("Average reward in 100 trials: ", np.mean(agent_DDQN))  
env.close()

fig = plt.figure()
plt.plot(np.arange(len(agent_DDQN)),agent_DDQN)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()

# DQN - SOFT UPDATE
env = gym.make('LunarLander-v2')
env.seed(seed)

memory_SOFT = ReplayMemory(MEMORY_CAPACITY)
loss_func = nn.MSELoss()
step = 1
epsilon = 1
scores_DQN_SOFT = []
env.seed(seed)
policy_net_SOFT,target_net_SOFT = NN(FC1_SIZE,FC2_SIZE),NN(FC1_SIZE,FC2_SIZE)
target_net_SOFT.load_state_dict(policy_net_SOFT.state_dict())
optimizer = torch.optim.Adam(policy_net_SOFT.parameters(),lr = ALPHA)
    
for i in range(EPISODES):
    s = env.reset()
    total_score = 0
    done = False

    for t in range(MAX_STEPS):
        s = torch.from_numpy(s).float().unsqueeze(0)
        
        if np.random.uniform() <= epsilon:
            action = np.random.randint(0,N_ACTIONS)
        else:
            action_values = policy_net_SOFT.forward(s)
            action = torch.max(action_values,1)[1].data.numpy()
            action = action[0]
        
        a = action
        s_next,r,done,_ = env.step(a)
        memory_SOFT.push(s,a,r,s_next,done)
        
        s = s_next
        total_score += r
        
        if memory_SOFT.__len__() < BATCH_SIZE:
            continue

        transitions = memory_SOFT.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        b_s = torch.from_numpy(np.vstack([s for s in batch.state_cur if s is not None])).float()
        b_a = torch.from_numpy(np.vstack([a for a in batch.action if a is not None])).long()
        b_r = torch.from_numpy(np.vstack([r for r in batch.reward if r is not None])).float()
        b_s_n = torch.from_numpy(np.vstack([s_n for s_n in batch.state_next if s_n is not None])).float()
        b_d = torch.from_numpy(np.vstack([d for d in batch.done if d is not None])).float()

        q_policy = policy_net_SOFT(b_s).gather(1,b_a)
        q_next = target_net_SOFT(b_s_n).detach()
        
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)*(1-b_d) # DQN
        loss = loss_func(q_policy,q_target)
      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % TARGET_UPDATE == 0:
            for target_param,policy_param in zip(target_net_SOFT.parameters(),policy_net_SOFT.parameters()):
                target_param.data.copy_(TAU*policy_param.data + (1.0-TAU)*target_param.data)
        step+=1
        if done:
            break
    epsilon = max(MIN_EPSILON, epsilon * EPS_DECAY)
    scores_DQN_SOFT.append(total_score)

pickle.dump(scores_DQN_SOFT,open("scores_DQN_SOFT","wb"))
fig = plt.figure()
plt.plot(np.arange(len(scores_DQN_SOFT)),scores_DQN_SOFT)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()



agent_DQN_SOFT = []

for i in range(100):
    total_score = 0
    state = env.reset()
    done = 0
    while not done:
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = np.argmax(policy_net_SOFT(state).cpu().data.numpy())
        state, reward, done, _ = env.step(action)
        total_score += reward 
    agent_DQN_SOFT.append(total_score)

pickle.dump(agent_DQN_SOFT,open("agent_DQN_SOFT","wb"))

print("Average reward in 100 trials: ", np.mean(agent_DQN_SOFT))  
env.close()

fig = plt.figure()
plt.plot(np.arange(len(agent_DQN_SOFT)),agent_DQN_SOFT)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()

#### Function for ploting ####

def moving_average(scores):
    N = 20
    cumsum, avg = [0], []

    for i, x in enumerate(scores, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            avg.append(moving_ave)
    return avg


#### Plots ####
scores_DQN = pickle.load(open("scores_DQN","rb"))
scores_DDQN = pickle.load(open("scores_DDQN","rb"))
scores_DQN_SOFT = pickle.load(open("scores_DQN_SOFT","rb"))

fig = plt.figure()
plt.plot(np.arange(len(moving_average(scores_DQN))),moving_average(scores_DQN), 'b-', alpha=0.7)
plt.plot(np.arange(len(moving_average(scores_DDQN))),moving_average(scores_DDQN), 'g-', alpha=0.7)
plt.plot(np.arange(len(moving_average(scores_DQN_SOFT))),moving_average(scores_DQN_SOFT), 'r-', alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend(['DQN', 'DDQN', 'DQN_SOFT'], loc='upper left')
plt.show()


agent_DQN = pickle.load(open("agent_DQN","rb"))
agent_DDQN = pickle.load(open("agent_DDQN","rb"))
agent_DQN_SOFT = pickle.load(open("agent_DQN_SOFT","rb"))
    
fig = plt.figure()
plt.plot(np.arange(len(moving_average(agent_DQN))),moving_average(agent_DQN), 'b-', alpha=0.7)
plt.plot(np.arange(len(moving_average(agent_DDQN))),moving_average(agent_DDQN), 'g-', alpha=0.7)
plt.plot(np.arange(len(moving_average(agent_DQN_SOFT))),moving_average(agent_DQN_SOFT), 'r-', alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend(['DQN', 'DDQN', 'DQN_SOFT'], loc='upper left')
plt.show()