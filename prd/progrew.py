import numpy as np
import torch
import os, sys
from samplers import GenerativeSampler, sample, log_normal_pdf

class Rewarder(object):
    rewardfn = None
    sampler = None
    device = None
    logp = None
    logps = None
    rewardfn_state = {}
    obs = {}

    @classmethod
    def __init__(cls, rewardfn, sampler, device, obs = None):
        cls.rewardfn = rewardfn
        cls.sampler = sampler
        cls.device = device
        cls.obs = {'image': None}
        if obs is not None:
            cls.reset(obs)
        #cls.holes = []

    @classmethod
    def suspend(cls):
        cls.rewardfn_state = cls.rewardfn.__dict__.copy()
        return cls.rewardfn_state
    
    @classmethod
    def restore(cls, state = None, refill = True):
        #cls.rewardfn.reset()
        if state is not None:
            cls.rewardfn_state = state.copy()
        cls.rewardfn.__dict__ = cls.rewardfn_state.copy()
        cls.rewardfn_state = {}

        if refill:
            cls.rewardfn.holes = [h for h in cls.refill(True)]
    
    @classmethod
    def reset(cls, obs = None):
        cls.rewardfn.reset()
        if obs is not None:
            cls.obs['image'] = np.copy(obs['image']) if isinstance(obs['image'], np.ndarray) else np.copy(obs['image'].detach().cpu().numpy())

    @classmethod
    def __call__(cls, obs, act, reward, done):
        if isinstance(act, torch.Tensor):
            action = act.detach().cpu().numpy().item()
        if isinstance(act, np.ndarray):
            action = act.item()
        
        if cls.obs['image'] is None:
            holes = cls.refill(use_mean = True)
            cls.rewardfn.reset(holes, verbose = True)  
            cls.obs['image'] = np.copy(obs['image']) if isinstance(obs['image'], np.ndarray) else np.copy(obs['image'].detach().cpu().numpy())
            #cls.holes.append([holes])
            return 0.
        else:
            rew = cls.rewardfn.step(cls.obs, act, reward, done)
            #cls.holes[-1].append(cls.holes[-1][-1])     
            cls.obs['image'] = np.copy(obs['image']) if isinstance(obs['image'], np.ndarray) else np.copy(obs['image'].detach().cpu().numpy())
            if done:
                cls.rewardfn.reset(verbose = cls.rewardfn.reach)  
            return rew
    
    @classmethod
    def refill(cls, use_mean = True):
        seed = torch.ones([1, cls.sampler.input_size]).to(cls.device)
        with torch.no_grad():
            cls.sampler.eval()
            mean, logvar = cls.sampler(seed)
            if use_mean:
                print("means: {}".format(mean))
                print("logvar: {}".format(logvar))
                return mean.flatten().detach().cpu().numpy().tolist()
            holes = sample(mean, logvar, num_samples = 1).flatten().detach().cpu().numpy().tolist()
            #holes = mean.flatten().detach().cpu().numpy().tolist()
            return holes
    
class RewarderWithMemory(Rewarder):
    #rewardfn = None
    #sampler = None
    #device = None
    #logp = None
    #logps = None

    dcmodel = None
    memory = None
    memories = []
    preprocess_obss = lambda obs, device: obs

    @classmethod
    def __init__(cls, rewardfn, sampler, dcmodel, device, preprocess_obss = None, obs = None, ):
        super(cls, RewarderWithMemory).__init__(rewardfn, sampler, device, obs)
        #cls.rewardfn = rewardfn
        #cls.sampler = sampler
        #cls.device = device
        #cls.obs = None
        #cls.holes = []

        cls.dcmodel = dcmodel
        if preprocess_obss is not None:
            cls.preprocess_obss = preprocess_obss
        cls.memory = None
        cls.memories = []


    @classmethod
    def __call__(cls, obs, act, reward, done):
        if isinstance(act, torch.Tensor):
            action = act.detach().cpu().numpy().item()
        if isinstance(act, np.ndarray):
            action = act.item()       
        
        # Update dc_model memory
        state = cls.preprocess_obss([obs], device = cls.device).image.to(cls.device)
        action = torch.FloatTensor([action]).flatten().unsqueeze(1).to(cls.device)
        if cls.memory is None:
            cls.memory = torch.zeros([action.shape[0], cls.dcmodel.memory_size], device = cls.device).to(cls.device)
        #if len(cls.memories) == 0:
            #cls.memories = [[cls.memory[i]] for i in range(cls.memory.shape[0])]
        # else:
        #     cls.memories = cls.memories + [cls.memory[i] for i in range(cls.memory.shape[0])]
        with torch.no_grad():
            cls.dcmodel.eval()
            dc_rew, cls.memory = cls.dcmodel(state, action, cls.memory)
        dc_rew = dc_rew.cpu().numpy().item()
        mask = 1. - torch.FloatTensor([done]).to(cls.device)
        cls.memory = cls.memory * mask           

        # Let programmatic reward function output reward
        if cls.obs['image'] is None:
            holes = cls.refill(use_mean = True)
            cls.rewardfn.reset(holes, verbose = True)  
            cls.obs['image'] = np.copy(obs['image']) if isinstance(obs['image'], np.ndarray) else np.copy(obs['image'].detach().cpu().numpy())
            #cls.holes.append([holes])
            prog_rew = 0.
        else:
            prog_rew = cls.rewardfn.step(cls.obs, act, reward, done)
            #cls.holes[-1].append(cls.holes[-1][-1])     
            cls.obs['image'] = np.copy(obs['image']) if isinstance(obs['image'], np.ndarray) else np.copy(obs['image'].detach().cpu().numpy())
            if done:
                cls.rewardfn.reset(verbose = cls.rewardfn.reach)  
        rew = prog_rew #+ 1./2. * np.log(1./(1. + np.exp(dc_rew - prog_rew)) * 1./(1. + np.exp(prog_rew - dc_rew)))
        return rew 
        
    @classmethod
    def update_memory(cls, memory):
        idx = 0
        for i in range(len(cls.memories)):
            for j in range(len(cls.memories[i])):
                memory[idx, :] = cls.memories[i][j].flatten()[:]
                idx += 1
        cls.memories = []
        return memory 



class RewardFn(object):
    def __init__(self):
        self.traj = []
        self.t_min = float('inf')
        self.carry = True
        self.locked = True
        self.reach = False
        self.doc = []
        self.sum_rew = 0.
        self.init_rew = 0.
        self.num_holes = 8
        self.holes = [None for i in range(self.num_holes)]

    def reset(self, holes = None, verbose = False):
        if self.reach and verbose:
            print(self.doc)
        if self.holes[0] is not None:
            self.init_rew = len(self.traj) * self.holes[0] - self.sum_rew
        self.sum_rew = 0.
        self.traj = []
        self.doc = []
        self.carry = False
        self.locked = True
        self.reach = False
        self.t_min = float('inf')
        if holes is not None:
            self.holes = [h for h in holes]
        
        

    def step(self, obs, action, reward, done):
        #print("Analyzing obs\n pre_obs: {}\n action: {}\n reward: {}\n done: {}\n".format(obs, action, reward, done))
        if len(self.traj) < 1:
            ########## reward hole ##########
            #rew = self.holes[0]
            rew = 0. #- self.holes[0] - self.holes[1] - self.holes[3] - self.holes[4] - self.holes[5] - self.holes[6] - self.holes[7]
            ########## reward hole ##########
            self.traj.append([obs, action, rew, done, None])
            self.sum_rew += rew
            self.doc.append("initialize")
            return rew

        t = len(self.traj) + 1
        ########## reward hole ##########
        rew = 0. #- self.holes[1] - self.holes[3] - self.holes[4] - self.holes[5] - self.holes[6] - self.holes[7]
        #rew = self.holes[0]
        ########## reward hole ##########
        if (obs['image'][3, 5][0] == 8) and (action == 2):
            # Reach the goal state
            if t < self.t_min:
                # For the first time
                self.t_min = t
                ########## reward hole ##########
                rew = self.holes[1] #+ self.holes[0] 
                #rew =  - 1.0
                self.traj.append([obs, action, rew, done, 1])
                ########## reward hole ##########
                self.doc.append("reach the goal state for the first time. default reward {}".format(reward))
                #print("Reaching obs\n pre_obs: {}\n action: {}\n reward: {}\n done: {}\n".format(obs, action, reward, done))
            elif self.reach:
                #if done:
                #    raise ValueError("Repeat done: {} >= t_min = {}".format(reward, self.t_min))
                # Not the first time
                self.doc.append("reach the goal state again")
                ########## reward hole ##########
                #rew = self.holes[1]
                rew = self.holes[2]
                self.traj.append([obs, action, rew, done, 2])
                ########## reward hole ##########
            self.reach = True
        elif obs['image'][3, 5][0] == 5:       
            # See a key in front 
            if (action == 3) and self.locked and (not self.carry):
                # pick up the key if locked
                self.doc.append("pick up a key")
                ########## reward hole ##########
                #rew = 0.1 
                rew = self.holes[3]#+ self.holes[0] 
                self.traj.append([obs, action, rew, done, 3])
                ########## reward hole ##########
                self.carry = True
                # From now on, there is no need to repeat past action
        elif action == 4 and self.carry and self.locked:
            # Key is dropped
            self.carry = False
            # Don't drop the key if locked
            self.doc.append("drop the key without using it")
            ########## reward hole ##########
            #rew = -0.2 
            rew = self.holes[4]#+ self.holes[0] 
            self.traj.append([obs, action, rew, done, 4])
            ########## reward hole ##########
        elif obs['image'][3, 5][0] == 4 and (action == 5):
            # Seeing a door, and toggle it 
            if obs['image'][3, 5][2] == 0:
                # If the door was open, then toggling it means closing the gate
                self.doc.append("closed a door")
                ########## reward hole ##########
                #rew = -0.1 
                rew = self.holes[5] #+ self.holes[0]
                self.traj.append([obs, action, rew, done, 5])
                ########## reward hole ##########
            elif obs['image'][3, 5][2] == 2 and self.carry and self.locked:
                # If the door was locked and agent is carrying a key, then by toggling it the door is open
                self.doc.append("unlock a door")
                ########## reward hole ##########
                #rew = 0.5
                rew = self.holes[6] #+ self.holes[0] 
                self.traj.append([obs, action, rew, done, 6])
                ########## reward hole ##########
                self.locked = False
        elif obs['image'][3, 5][0] == 2 and (action == 2):
            # Hit the wall
            self.doc.append("hit a wall")
            ########## reward hole ##########
            #rew = -0.2 
            rew = self.holes[7] #+ self.holes[0]
            self.traj.append([obs, action, rew, done, 7])
            ########## reward hole ##########
        #elif not self.locked:
        #    rew = self.holes[2]
        else:
            rew = 0.
            self.traj.append([obs, action, rew, done, None])
        self.sum_rew +=  rew 
        return rew  
