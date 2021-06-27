import gym
import gym_minigrid
import numpy as np

def make_env(env_key, seed=None, reshape = False, horizon = 60):
    env = gym.make(env_key)
    env.seed(seed)
    # Add PRD wrapper to modify the reward function
    if reshape:
        env = PRDWrapper(env)
    if horizon is not None:
        env = FixedHorizonWrapper(env, horizon)
    return env

class FixedHorizonWrapper(gym.Wrapper):
    def __init__(self, env, horizon = 60):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.horizon = horizon
        self.t = 0

    def reset(self):
        self.t = 1
        return self.env.reset()
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.t == self.horizon - 1:
            return obs, rew, True, info
        else:
            self.t += 1
            return obs, rew, done, info


class PRDWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.traj = []
        self.t_min = float('inf')
        self.carry = {}
        self.locked = {}
        self.reach = False
        self.doc = []
        self.sum_rew = 0.
        self.sum_reshaped_rew = 0.
        self.stat = np.zeros([10, 1])
    
    def reset_stat(self):
        self.stat = np.zeros([10, 1])

    def reset(self):
        if self.sum_rew > 0. or self.sum_reshaped_rew >= 5:
            print(self.doc)
        obs = self.env.reset()
        obs['rew'] = 10. if self.reach else 0.

        self.sum_rew = 0.
        self.sum_reshaped_rew = 0.
        self.doc = []
        self.carry = {}
        self.locked = {}
        self.reach = False
        self.t_min = float('inf')
        
        self.traj = [[0, obs, None]]
        self.stat = np.concatenate((self.stat, np.zeros([10, 1])), axis = 1)

        return obs

        
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        
        self.traj[-1][2] = np.asarray([action])
         
        t_, obs_, action_ = self.traj[-1]
        obs['rew'] = 0.
        self.stat[0, -1] += 1.
        if (obs_['image'][3, 5][0] == 8) and (action_ == 2).all():
            # Reach the goal state
            if t_ < self.t_min:
                # For the first time
                self.t_min = t_
                obs['rew'] = 1.
                self.stat[1, -1] += 1.
                self.traj.append([self.traj[-1][0] + 1, obs, None])
                self.doc.append("reach the goal state for the first time")
                self.doc.append(obs['rew'])
            else:
                # Not the first time
                self.doc.append("reach the goal state again")
                obs['rew'] = 0.8
                self.stat[2, -1] += 1.
            self.reach = True
        elif self.reach and obs_['image'][3, 5][0] == 1 and (action_ == 2).all():
            # Leave the goal state
            self.doc.append("leave the goal state")
            self.reach = False
        elif obs_['image'][3, 5][0] == 5 and (action_ == 3).all():       
            # Saw a key in front and pick up
            color = obs_['image'][3, 5][1]
            if (color in self.locked.keys()):
                if self.locked[color]:
                    if (color not in self.carry.keys()):
                        # pick up the key if locked
                        obs['rew']  = .1 
                        self.stat[3, -1] += 1.
                        self.doc.append("pick up a {} key".format(color))
                        self.carry[color] = True
                    elif not self.carry[color]:
                        # pick up the key again if locked
                        obs['rew']  = 0.05
                        self.stat[4, -1] += 1.
                        self.doc.append("pick up the {} key again".format(color))
                        self.carry[color] = True    
                    #self.traj = [self.traj[-1]]
                    # From now on, there is no need to repeat past action
        elif obs['image'][3, 5][0] == 5 and (action_ == 4).all():
            # Key was dropped and it is in front now
            color = obs['image'][3, 5][1]
            if color in self.carry.keys():
                self.carry[color] = False 
                self.doc.append("drop a {} key".format(color))
                if color in self.locked.keys():
                    # Don't drop the key if locked
                    if self.locked[color]:
                        obs['rew']  = -0.2
                        self.stat[5, -1] += 1.
        elif obs['image'][3, 5][0] == 4:
            # See a door
            color = obs['image'][3, 5][1]
            self.locked[color] = (obs['image'][3, 5][2] == 2)

        elif obs_['image'][3, 5][0] == 4 and (action_ == 5).all():
            # Seeing a door, and toggle it 
            color = obs_['image'][3, 5][1]
            if obs_['image'][3, 5][2] == 0 and obs['image'][3, 5][2] == 1:
                # If the door was open, then toggling it means closing the gate
               obs['rew']  = -0.1
               self.stat[6, -1] += 1.
            elif obs_['image'][3, 5][2] == 2 and obs['image'][3, 5][2] == 0:
                # If the door was locked and agent is carrying a key, then by toggling it the door is open
                    obs['rew']  = 0.5
                    self.stat[7, -1] += 1.
                    self.doc.append("unlock a {} door".format(color))
                    self.locked[color] = False
        elif obs_['image'][3, 5][0] == 2 and (action_ == 2).all():
            # Hit the wall
            obs['rew'] = -0.2
            self.stat[8, -1] += 1.
       
       
        self.traj.append([self.traj[-1][0] + 1, obs, None])
        self.sum_rew += rew
        self.sum_reshaped_rew += obs['rew']
        return obs, rew, done, info

