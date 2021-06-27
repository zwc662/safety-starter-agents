import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList
import torch.distributions as dist
from samplers import GenerativeSampler, sample, log_normal_pdf


class HGAILAlgo(object): 
    def __init__(self, rewardfn, sampler, acmodel, dcmodel, device, batch_size = 100, adam_eps=1e-8, epochs=4, lr=0.001, max_grad_norm=0.5, recurrence=4, preprocess_obss = None):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        dcmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        """
        # Store parameters
        self.rewardfn = rewardfn
        self.sampler = sampler
        self.acmodel = acmodel
        self.dcmodel = dcmodel
        self.device = device
        self.gen_optimizer = torch.optim.RMSprop(self.sampler.parameters(), lr = 0.001,
                                             alpha=0.99, eps=1e-8)
        
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        #self.optimizer = torch.optim.Adam(self.dcmodel.parameters(), lr, eps=adam_eps)

        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss
        # Control parameters
        assert self.dcmodel.recurrent or self.recurrence == 1
        self.returns = None
         # Initialize log values    
        self.dc_optimizer = torch.optim.Adam(self.dcmodel.parameters(), lr, eps=adam_eps)

    def preprocess_demonstrations(self, demos_list):
        demos = DictList()
        with torch.no_grad():
            self.dcmodel.eval()
            
            tot_len = 0
            for i in range(len(demos_list)):
                for j in range(len(demos_list[i]['ob'])):
                    tot_len += 1
            
            demos.obs = []
            demos.action = torch.zeros((tot_len, *np.asarray(demos_list[0]['ac'][0]).shape))
            demos.reward = torch.zeros(tot_len)
            demos.log_prob = torch.zeros(demos.action.size())
            if self.dcmodel.recurrent:
                demos.mask = torch.ones([tot_len, 1]).to(self.device)
                demos.memory = torch.zeros([tot_len, self.dcmodel.memory_size]).to(self.device)

            idx = 0
            for i in range(len(demos_list)):
                assert len(demos_list[i]['ob']) == len(demos_list[i]['ac'])
                memory = torch.zeros((1, self.dcmodel.memory_size)).to(self.device)
                demos.mask[idx] = demos.mask[idx] * 0.
                for j in range(len(demos_list[i]['ob'])):
                    ob = demos_list[i]['ob'][j]
                    ac = demos_list[i]['ac'][j]
                    
                    demos.memory[idx] = memory.squeeze(0)
                    demos.obs.append(ob)
                    demos.action[idx] = ac
                    idx += 1
            
            demos.obs = self.preprocess_obss(demos.obs, device = self.device)
        return demos


    def update_samples(self, demos, exps, exps_state_prev = None, exps_memory_prev = None):
        exps_prog_rews = []
        demos_prog_rews = []
        exps_dc_rews = []
        demos_dc_rews = []
        exps_prog_means = exps
        demos_prog_means = demos

        seed = torch.ones([1, self.sampler.input_size]).to(self.device)
        with torch.no_grad():
            self.sampler.eval()
            mean, logvar = self.sampler(seed)
            mean = mean.detach()
            logvar = logvar.detach()

            """
            #idx = 0
            #for i in range(len(exps.obs)):
            #    if i == 0 or (not exps.mask[i]):
            #        exps_prog_means.append([])
                #exps_ac_dist, _, _ = self.acmodel(exps.obs[i:i+1], exps.memory[i:i+1])
                #exps_prog_means[-1].append(exps_ac_dist.log_prob(exps.action[i:i+1].to(self.device)).flatten())
            #    exps_prog_means[-1].append(exps.log_prob[i])

            
            for i in range(len(demos.obs)):
                if i == 0 or (not demos.mask[i]):
                    demos_prog_means.append([])
                    memory = torch.zeros((1, self.acmodel.memory_size)).to(self.device)
                demos_ac_dist, _, memory = self.acmodel(demos.obs[i:i+1], memory)
                #demos_prog_means[-1].append(demos_ac_dist.log_prob(demos.action[i:i+1].to(self.device)).flatten())
                demos.log_prob[i] =  demos_ac_dist.log_prob(demos.action[i:i+1].to(self.device)).flatten()#demos_prog_means[-1][-1]
            
            self.rewardfn.holes = [h for h in mean.flatten().detach().cpu().numpy().tolist()]
            for i in range(len(exps.obs)):
                if i == 0 or (not exps.mask[i]):
                    if i == 0 and (exps_state_prev is not None):
                        self.rewardfn.__dict__ = exps_state_prev.copy()
                    else: 
                        self.rewardfn.reset()
                    exps_prog_means.append([])
                exps_prog_mean = self.rewardfn.step({'image': exps.obs[i].image}, exps.action[i], float(not exps.mask[i+1]) if i < len(exps.obs) - 1 else 1., not exps.mask[i+1] if i < len(exps.obs) - 1 else False)
                #exps_prog_means[-1].append(exps_prog_mean)
                #exps_prog_means[-1].append(np.log(np.exp(exps_prog_mean)/np.sum(np.exp(self.rewardfn.holes))).item())
                exps_prog_means[-1].append(exps_prog_mean - self.rewardfn.holes[0])
            """

            #print(mean, logvar)
            exps_output = sample(mean, logvar, num_samples = int(exps.mask[0]) + int(torch.sum(1. - exps.mask) .item()))
            demos_output = sample(mean, logvar, num_samples = int(torch.sum(1. - demos.mask).item()))
            #exps_output = mean.repeat(1 + int(torch.sum(1 - exps.mask).item()), 1)
            #demos_output = mean.repeat(1 + int(torch.sum(1 - demos.mask).item()), 1)
           

            idx = 0
            for i in range(len(exps.obs)):
                exps.reward[i] = exps.reward[i] - mean.flatten()[0]
                if i == 0 or (not exps.mask[i]):
                    if i == 0 and (exps_state_prev is not None):
                        self.rewardfn.__dict__ = exps_state_prev.copy()
                        self.rewardfn.holes = [h for h in exps_output[0].flatten().detach().cpu().numpy().tolist()]
                    else:
                        self.rewardfn.reset(exps_output[idx].flatten().detach().cpu().numpy().tolist())
                    exps_prog_rews.append([])
                    exps_dc_rews.append([])
                    memory = torch.zeros((1, self.dcmodel.memory_size)).to(self.device)
                    if i == 0 and (exps_memory_prev is not None):
                        memory = exps_memory_prev.flatten().unsqueeze(0).to(self.device)
                    idx += 1
                    
                exps_prog_rew = self.rewardfn.step({'image': exps.obs[i].image}, exps.action[i], float(not exps.mask[i+1]) if i < len(exps.obs) - 1 else 1., not exps.mask[i+1] if i < len(exps.obs) - 1 else False)
                #exps_prog_rews[-1].append(exps_prog_rew - np.log(np.sum(np.exp([0.] + self.rewardfn.holes[1:]))).item())
                exps_prog_rews[-1].append(exps_prog_rew - self.rewardfn.holes[0])
                exps.reward[i] = 0. if self.rewardfn.traj[-1][-1] is None else mean[0, self.rewardfn.traj[-1][-1]]
                #exps.reward[i] = exps.reward[i] - np.log(1. + np.sum(np.exp(mean[0, 1:].cpu().numpy()))).item()
                exps.reward[i] = exps.reward[i] - mean[0, 0]

                exps.memory[i] = memory.squeeze(0)
                _, exps_dc_rew, memory = self.dcmodel(exps.obs[i].image.unsqueeze(0).to(self.device), \
                                                torch.FloatTensor([exps.action[i]]).flatten().unsqueeze(1).to(self.device), \
                                                memory.to(self.device))
                exps_dc_rews[-1].append(exps_dc_rew) #torch.exp(exps_prog_means[len(exps_dc_rews) - 1][len(exps_dc_rews[-1]) - 1]) * \
                    #(1./(1. - torch.exp(exps_dc_rew.flatten())) - 1.))

                
                

            idx = 0
            for i in range(len(demos.obs)):
                if i == 0 or (not demos.mask[i]): 
                    acmemory = torch.zeros((1, self.acmodel.memory_size)).to(self.device)
                    self.rewardfn.reset(demos_output[idx].flatten().detach().cpu().numpy().tolist())
                    demos_prog_rews.append([])
                    demos_dc_rews.append([])
                    memory = torch.zeros((1, self.dcmodel.memory_size)).to(self.device)
                    idx += 1
            
                demos_prog_rew = self.rewardfn.step({'image': demos.obs[i].image}, demos.action[i], float(not demos.mask[i+1]) if i < len(demos.obs) - 1 else 1., not demos.mask[i+1] if i < len(demos.obs) - 1 else True)
                #demos_prog_rews[-1].append(demos_prog_rew - np.log(np.sum(np.exp([0.] + self.rewardfn.holes[1:]))).item())
                demos_prog_rews[-1].append(demos_prog_rew - self.rewardfn.holes[0])
                demos.reward[i] = 0. if self.rewardfn.traj[-1][-1] is None else mean[0, self.rewardfn.traj[-1][-1]]
                #demos.reward[i] = demos.reward[i] - np.log(1. + np.sum(np.exp(mean[0, 1:].cpu().numpy()))).item()
                demos.reward[i] = demos.reward[i] - mean[0, 0]
                
                demos.memory[i] = memory.squeeze(0)
                _, demos_dc_rew, memory = self.dcmodel(demos.obs[i].image.unsqueeze(0).to(self.device), \
                                                torch.FloatTensor([demos.action[i]]).flatten().unsqueeze(1).to(self.device), \
                                                memory.to(self.device))                  
                demos_dc_rews[-1].append(demos_dc_rew) #torch.exp(demos_prog_means[len(demos_dc_rews) - 1][len(demos_dc_rews[-1]) - 1]) * \
                    #(1./(1. - torch.exp(demos_dc_rew.flatten())) - 1.))
                
                demos_ac_dist, _, acmemory = self.acmodel(demos.obs[i:i+1], acmemory)
                #demos_prog_means[-1].append(demos_ac_dist.log_prob(demos.action[i:i+1].to(self.device)).flatten())
                demos.log_prob[i] =  demos_ac_dist.log_prob(demos.action[i:i+1].to(self.device)).flatten()#demos_prog_means[-1][-1]

            return demos, demos_output, demos_prog_rews, demos_dc_rews, exps, exps_output, exps_prog_rews, exps_dc_rews, exps_prog_means


    def update_sampler(self, demos_output, demos_prog_rews, demos_dc_rews, exps_output, exps_prog_rews, exps_dc_rews, exps_prog_means):
        tot_loss = 0.
        n = 0.

        tot_exps_loss = 0.
        tot_demos_loss = 0.
        #batch_size = min(self.batch_size, min(len(exps_rews), len(demos_rews)))
        for i in range(self.epochs):
            demos_error = torch.zeros([len(demos_prog_rews)]) #torch.zeros([batch_size])
            exps_demos_error = torch.zeros([len(demos_prog_rews)])
            demos_idx = range(len(demos_prog_rews)) #np.random.choice(np.arange(len(demos_rews)), batch_size).tolist()
            for i_traj in demos_idx:
                """
                weight = torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device)
                exps_demos_error[i_traj] = exps_demos_error[i_traj] + 0.1 * F.mse_loss(torch.sum(weight, (0), keepdim = True).to(self.device), torch.ones([1]).to(self.device)) +  (1./len(demos_prog_rews[i_traj])) * F.binary_cross_entropy_with_logits(
                    torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device) - torch.tensor(demos_dc_rews[i_traj]).flatten().to(self.device).detach(), torch.zeros([len(demos_prog_rews[i_traj])]).to(self.device), weight)
                exps_demos_error[i_traj] = exps_demos_error[i_traj] + 0.1 * F.mse_loss(torch.sum(weight, (0), keepdim = True).to(self.device), torch.ones([1]).to(self.device)) +  (1./len(demos_prog_rews[i_traj])) * F.binary_cross_entropy_with_logits(
                    - torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device) + torch.tensor(demos_dc_rews[i_traj]).flatten().to(self.device).detach(), torch.ones([len(demos_prog_rews[i_traj])]).to(self.device), weight)
                """
                #demos_error[i_traj] = demos_error[i_traj] - torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device).mean()
                #demos_error[i_traj] = demos_error[i_traj] - (1./len(demos_prog_rews[i_traj])) * \
                #    torch.sum(torch.exp(torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device)) * torch.tensor(demos_dc_rews[i_traj]).flatten().to(self.device)).detach()
                #demos_error[i_traj] = demos_error[i_traj] + (1./len(demos_prog_rews[i_traj])) * \
                #    F.binary_cross_entropy_with_logits(torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device), torch.sigmoid(torch.tensor(demos_dc_rews[i_traj]).flatten()).to(self.device).detach())
                #demos_error[i_traj] = demos_error[i_traj] - torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device).sum()
                #demos_error[i_traj] = demos_error[i_traj] + (1./len(demos_prog_rews[i_traj])) * F.binary_cross_entropy_with_logits(
                #    torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device).detach() - torch.tensor(demos_dc_rews[i_traj]).flatten().to(self.device).detach(), torch.ones([len(demos_prog_rews[i_traj])]).to(self.device))
                #demos_error[i_traj] = demos_error[i_traj] + (1./len(demos_prog_rews[i_traj])) * F.binary_cross_entropy_with_logits(
                #    - torch.tensor(demos_prog_rews[i_traj]).flatten().to(self.device) + torch.tensor(demos_dc_rews[i_traj]).flatten().to(self.device).detach(), torch.zeros([len(demos_prog_rews[i_traj])]).to(self.device))
                demos_error[i_traj] = demos_error[i_traj] + F.mse_loss(
                    torch.tensor(demos_prog_rews[i_traj]).to(self.device), 
                    torch.tensor(demos_dc_rews[i_traj]).to(self.device).detach())
                #demos_error[i_traj] = demos_error[i_traj] + F.binary_cross_entropy_with_logits(
                #    torch.sum(torch.tensor(demos_prog_rews[i_traj]).flatten().unsqueeze(0), dim = -1).to(self.device) - torch.sum(torch.tensor(demos_dc_rews[i_traj]).flatten().unsqueeze(0), dim = -1).to(self.device),
                #    torch.ones([1]).to(self.device))
                #demos_error[i_traj] = demos_error[i_traj] + F.binary_cross_entropy_with_logits(
                #    - torch.sum(torch.tensor(demos_prog_rews[i_traj]).flatten().unsqueeze(0), dim = -1).to(self.device) + torch.sum(torch.tensor(demos_dc_rews[i_traj]).flatten().unsqueeze(0), dim = -1).to(self.device),
                #    torch.zeros([1]).to(self.device))
            
            exps_error = torch.zeros([len(exps_prog_rews)]) #torch.zeros([batch_size])
            #exps_rew = torch.zeros([len(exps_prog_rews)])
            exps_idx = range(len(exps_prog_rews)) #np.random.choice(np.arange(len(exps_rews)), batch_size).tolist()
            for i_traj in exps_idx:
                #exps_error[i_traj] = exps_error[i_traj] - (1./len(exps_prog_rews[i_traj])) * torch.tensor(exps_prog_rews[i_traj]).flatten().to(self.device).sum()
                #weight = torch.tensor(exps_prog_rews[i_traj]).flatten().to(self.device) - exps_prog_means.log_prob[t-len(exps_prog_rews[i_traj]):t].clone().detach().flatten().to(self.device)
                #exps_error[i_traj] = exps_error[i_traj] + F.binary_cross_entropy_with_logits(weight, torch.sigmoid(torch.tensor(exps_dc_rews[i_traj]).flatten().to(self.device).detach()))
                #weight = torch.exp(weight).detach()
                #exps_error[i_traj] = exps_error[i_traj] + F.binary_cross_entropy_with_logits(weight, torch.sigmoid(torch.tensor(exps_dc_rews[i_traj]).flatten().to(self.device).detach()))
                #exps_error[i_traj] = exps_error[i_traj] - (1./len(exps_prog_rews[i_traj])) * \
                #    torch.sum(torch.exp(torch.tensor(exps_prog_rews[i_traj]).flatten().to(self.device)) * torch.tensor(exps_dc_rews[i_traj]).flatten().to(self.device)).detach()
                
                #exps_error[i_traj] = exps_error[i_traj] - torch.tensor(exps_prog_rews[i_traj]).flatten().to(self.device).sum()
                #exps_error[i_traj] = exps_error[i_traj] + (1./len(exps_prog_rews[i_traj])) * \
                #    F.binary_cross_entropy_with_logits(torch.tensor(exps_prog_rews[i_traj]).flatten().to(self.device), torch.sigmoid(torch.tensor(exps_dc_rews[i_traj]).flatten()).to(self.device).detach())
                #exps_error[i_traj] = exps_error[i_traj] + (1./len(exps_prog_rews[i_traj])) * F.binary_cross_entropy_with_logits(
                #     torch.tensor(exps_prog_rews[i_traj]).flatten().to(self.device).detach() - torch.tensor(exps_dc_rews[i_traj]).flatten().to(self.device).detach(), torch.zeros([len(exps_prog_rews[i_traj])]).to(self.device))#, weight)
                #exps_error[i_traj] = exps_error[i_traj] + (1./len(exps_prog_rews[i_traj])) * F.binary_cross_entropy_with_logits(
                #     - torch.tensor(exps_prog_rews[i_traj]).flatten().to(self.device) + torch.tensor(exps_dc_rews[i_traj]).flatten().to(self.device).detach(), torch.ones([len(exps_prog_rews[i_traj])]).to(self.device))#, weight)
                
                #exps_rew[i_traj] = torch.tensor(exps_prog_rews[i_traj]).flatten().to(self.device).sum()
                exps_error[i_traj] = exps_error[i_traj] + F.mse_loss(
                    torch.tensor(exps_prog_rews[i_traj]).to(self.device), 
                    torch.tensor(exps_dc_rews[i_traj]).to(self.device).detach())
                #exps_error[i_traj] = exps_error[i_traj] + weight * F.binary_cross_entropy_with_logits(
                #    torch.sum(torch.tensor(exps_prog_rews[i_traj]).flatten().unsqueeze(0), dim = -1).to(self.device) - torch.sum(torch.tensor(exps_dc_rews[i_traj]).flatten().unsqueeze(0), dim = -1).to(self.device), 
                #    torch.zeros([1]).to(self.device))
                #exps_error[i_traj] = exps_error[i_traj] + weight * F.binary_cross_entropy_with_logits(
                #    - torch.sum(torch.tensor(exps_prog_rews[i_traj]).flatten().unsqueeze(0), dim = -1).to(self.device) + torch.sum(torch.tensor(exps_dc_rews[i_traj]).flatten().unsqueeze(0), dim = -1).to(self.device), 
                #    torch.ones([1]).to(self.device))
            
            with torch.enable_grad():
                self.sampler.train()
                mean, logvar = self.sampler(torch.ones([1, self.sampler.input_size]).to(self.device))
           
                demos_logps = log_normal_pdf(demos_output.to(self.device).detach(), mean, logvar).flatten()
                demos_loss = torch.mean(demos_logps * demos_error.to(self.device).detach()) 
                #demos_loss = demos_loss + F.binary_cross_entropy_with_logits(mean[0, 1:] - mean[0, 0], torch.zeros(mean[0, 1:].size()).to(self.device))
                exps_logps = log_normal_pdf(exps_output.to(self.device).detach(), mean, logvar).flatten()
                exps_loss = torch.mean(exps_logps * exps_error.to(self.device).detach())
                #exps_loss = exps_loss + F.binary_cross_entropy_with_logits(mean[0, 1:] - mean[0, 0], torch.zeros(mean[0, 1:].size()).to(self.device))
                
                
                
                #exps_demos_loss = torch.mean(demos_logps * exps_demos_error.to(self.device).detach())
                #exps_loss = 1./2. * (exps_loss + exps_demos_loss)
                
                #constr_loss = torch.mean(demos_logps * F.relu(-torch.abs(demos_output).to(self.device)[:, 5] + torch.abs(demos_output).to(self.device)[:, 6]) * F.binary_cross_entropy_with_logits(torch.abs(demos_output).to(self.device)[:, 5] - torch.abs(demos_output).to(self.device)[:, 6], torch.ones(demos_output.size()[0]).to(self.device)))
                #constr_loss = constr_loss + torch.mean(demos_logps * F.relu(-torch.abs(demos_output).to(self.device)[:, 4] + torch.abs(demos_output).to(self.device)[:, 3]) * F.binary_cross_entropy_with_logits(torch.abs(demos_output).to(self.device)[:, 4] - torch.abs(demos_output).to(self.device)[:, 3], torch.ones(demos_output.size()[0]).to(self.device)))
                #constr_loss = constr_loss + torch.mean(exps_logps * F.relu(-torch.abs(exps_output).to(self.device)[:, 5] + torch.abs(exps_output).to(self.device)[:, 6]) * F.binary_cross_entropy_with_logits(torch.abs(exps_output).to(self.device)[:, 5]- torch.abs(exps_output).to(self.device)[:, 6], torch.ones(exps_output.size()[0]).to(self.device)))
                #constr_loss = constr_loss + torch.mean(exps_logps * F.relu(-torch.abs(exps_output).to(self.device)[:, 4] + torch.abs(exps_output).to(self.device)[:, 3]) * F.binary_cross_entropy_with_logits(torch.abs(exps_output).to(self.device)[:, 4]- torch.abs(exps_output).to(self.device)[:, 3], torch.ones(exps_output.size()[0]).to(self.device)))
                

                pd_old = dist.multivariate_normal.MultivariateNormal(loc = mean.detach().flatten(), covariance_matrix = torch.diag(torch.exp(logvar.detach().flatten())))
                pd_new = dist.multivariate_normal.MultivariateNormal(loc = mean.flatten(), covariance_matrix = torch.diag(torch.exp(logvar.flatten())))
                kl_loss = dist.kl.kl_divergence(pd_new, pd_old)

                self.gen_optimizer.zero_grad()
                (exps_loss + demos_loss + 0.0 * kl_loss).backward()
                self.gen_optimizer.step()
                
                tot_loss += (exps_loss + demos_loss).item()
                tot_exps_loss += exps_loss.item()
                tot_demos_loss += demos_loss.item()
                n += 1

            return {"loss": tot_loss / n, "exps_loss": tot_exps_loss / n, "demos_loss": tot_demos_loss / n}

    def update_discriminator(self, demos, demos_prog_rews, demos_output, exps, exps_prog_rews, exps_output):
        # Flatten the programmatic rewards
        with torch.no_grad():
            mean, logvar = self.sampler(torch.ones([1, self.sampler.input_size]).to(self.device))
            demos_logps = log_normal_pdf(demos_output.to(self.device).detach(), mean, logvar).flatten()
            exps_logps = log_normal_pdf(exps_output.to(self.device).detach(), mean, logvar).flatten()

        demos_logps_flattened = []
        demos_prog_rews_flattened = []
        for i_traj in range(len(demos_prog_rews)):
            for t in range(len(demos_prog_rews[i_traj])):
                demos_prog_rews_flattened.append(demos_prog_rews[i_traj][t])
                demos_logps_flattened.append(demos_logps[i_traj])
        demos_prog_rews_flattened = torch.tensor(demos_prog_rews_flattened).to(self.device).unsqueeze(1).detach()
        demos_logps_flattened = torch.tensor(demos_logps_flattened).to(self.device).unsqueeze(1).detach()

        exps_logps_flattened = []
        exps_prog_rews_flattened = []
        for i_traj in range(len(exps_prog_rews)):
            for t in range(len(exps_prog_rews[i_traj])):
                exps_prog_rews_flattened.append(exps_prog_rews[i_traj][t])
                exps_logps_flattened.append(exps_logps[i_traj])
        exps_prog_rews_flattened = torch.tensor(exps_prog_rews_flattened).to(self.device).unsqueeze(1).detach()
        exps_logps_flattened = torch.tensor(exps_logps_flattened).to(self.device).unsqueeze(1).detach()

        with torch.enable_grad():
            self.dcmodel.train()
            # Compute starting indexes
            demos_inits = np.arange(0, len(demos) - self.recurrence, self.recurrence)
            exps_inits = np.arange(0, len(exps) - self.recurrence, self.recurrence)

            self.batch_size = min(demos_inits.shape[0], min(self.batch_size, exps_inits.shape[0]))
            
            loss = 0.
            tot_gail_loss = 0.
            tot_grad_pen = 0.
            tot_reg_loss = 0.
            n = 0
            for ep in range(self.epochs):
                demos_inds = np.random.permutation(demos_inits.shape[0])[:self.batch_size]
                exps_inds =  np.random.permutation(exps_inits.shape[0])[:self.batch_size]

                expert_loss = 0.
                agent_loss = 0.
                reg_loss = 0.

                if self.dcmodel.recurrent:
                    demos_memory = demos.memory[demos_inds].detach()
                    exps_memory = exps.memory[exps_inds].detach()
                    
                agent_ds = [None for i in range(self.recurrence)]
                expert_ds = [None for i in range(self.recurrence)]
                
                for i in range(self.recurrence):
                    
                    demos_prog_rews_batch = demos_prog_rews_flattened[demos_inds + i].detach()
                    demos_logps_batch = demos_logps_flattened[demos_inds + i]
                    exps_prog_rews_batch = exps_prog_rews_flattened[exps_inds + i].detach()
                    exps_logps_batch = exps_logps_flattened[exps_inds + i]

                    
                    demos_batch = demos[demos_inds + i]
                    exps_batch = exps[exps_inds + i]

                    _, agent_ds[i], exps_memory = self.dcmodel(exps_batch.obs.image.to(self.device), exps_batch.action.unsqueeze(1).to(self.device), exps_memory.to(self.device) * exps_batch.mask.to(self.device))
                    _, expert_ds[i], demos_memory = self.dcmodel(demos_batch.obs.image.to(self.device), demos_batch.action.unsqueeze(1).to(self.device), demos_memory.to(self.device) * demos_batch.mask.to(self.device))
                    
                    expert_loss = expert_loss + F.binary_cross_entropy_with_logits(
                            expert_ds[i] - demos_batch.log_prob.unsqueeze(1).to(self.device).detach(),
                            torch.ones(expert_ds[i].size()).to(self.device))
                    #expert_loss = expert_loss + F.binary_cross_entropy_with_logits(
                    #        - expert_ds[i] + demos_batch.log_prob.unsqueeze(1).to(self.device).detach(),
                    #        torch.zeros(expert_ds[i].size()).to(self.device))
                    agent_loss = agent_loss + F.binary_cross_entropy_with_logits(
                            agent_ds[i] - exps_batch.log_prob.unsqueeze(1).to(self.device).detach(), # - exps_prog_rews_batch,
                            torch.zeros(agent_ds[i].size()).to(self.device)) #, weight)
                    #agent_loss = agent_loss + F.binary_cross_entropy_with_logits(
                    #        - agent_ds[i] + exps_batch.log_prob.unsqueeze(1).to(self.device).detach(), # - exps_prog_rews_batch,
                    #        torch.ones(agent_ds[i].size()).to(self.device)) #, weight)
                    
                    """
                    weight = torch.exp(demos_prog_rews_batch).detach()
                    agent_loss =  1./2. * (agent_loss + F.binary_cross_entropy_with_logits(
                            expert_ds[i] - demos_prog_rews_batch,
                            torch.zeros(expert_ds[i].size()).to(self.device), weight))
                    agent_loss =  agent_loss + 1./2. * F.binary_cross_entropy_with_logits(
                            - expert_ds[i] + demos_prog_rews_batch,
                            torch.ones(expert_ds[i].size()).to(self.device), weight)
                    """
                    #reg_loss = reg_loss + F.binary_cross_entropy_with_logits(agent_ds[i], torch.sigmoid(exps_prog_rews_batch))
                    #reg_loss = reg_loss + F.binary_cross_entropy_with_logits(expert_ds[i], torch.sigmoid(demos_prog_rews_batch))
                    #reg_loss = reg_loss + F.binary_cross_entropy_with_logits(agent_ds[i], torch.exp(exps_prog_rews_batch.detach()))
                    #reg_loss = reg_loss + F.binary_cross_entropy_with_logits(expert_ds[i], torch.exp(demos_prog_rews_batch.detach()))
                    #reg_loss = reg_loss - torch.sum(agent_ds[i] * torch.exp(exps_batch.reward.unsqueeze(1).detach().to(self.device)))
                    #reg_loss = reg_loss - torch.sum(expert_ds[i] * torch.exp(demos_batch.reward.unsqueeze(1).detach().to(self.device)))
                    #reg_loss = reg_loss + F.mse_loss(agent_ds[i], exps_prog_rews_batch.detach().to(self.device))
                    #reg_loss = reg_loss + F.mse_loss(expert_ds[i], demos_prog_rews_batch.detach().to(self.device))
                    reg_loss = reg_loss + 0.01 * torch.sum(torch.exp(exps_logps_batch) * torch.square(agent_ds[i] - exps_prog_rews_batch.unsqueeze(1).detach().to(self.device)))
                    reg_loss = reg_loss + 0.01 * torch.sum(torch.exp(demos_logps_batch) * torch.square(expert_ds[i] - demos_prog_rews_batch.unsqueeze(1).detach().to(self.device)))

                gail_loss = expert_loss + agent_loss
                
                demos_batch = demos[demos_inds + self.recurrence]
                exps_batch = exps[exps_inds + self.recurrence]
                #grad_pen = self.compute_grad_pen(demos_batch.obs, demos_batch.action, demos_memory.to(self.device) * demos_batch.mask.to(self.device),
                #                                   exps_batch.obs, exps_batch.action, exps_memory.to(self.device) * exps_batch.mask.to(self.device))
                grad_pen = reg_loss * 0.
                self.dc_optimizer.zero_grad()
                (gail_loss + grad_pen + reg_loss).backward()
                self.dc_optimizer.step()

                tot_gail_loss = tot_gail_loss + gail_loss.item() 
                tot_grad_pen = tot_grad_pen + grad_pen.item()
                tot_reg_loss = tot_reg_loss + reg_loss.item()
                loss = loss + (gail_loss + grad_pen + reg_loss).item()
                n += self.recurrence

            return {"avg_l": loss / n, "avg_gail_l": tot_gail_loss / n, "avg_grad_pen": tot_grad_pen / n, "avg_reg_loss": tot_reg_loss / n}

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         expert_memory,
                         agent_state,
                         agent_action,
                         agent_memory,
                         lambda_=10):
        alpha = torch.rand(self.batch_size).to(self.device)

        mixup_state = DictList()
        mixup_state.image = alpha.view(expert_state.image.size()[0], *([1] * (len(expert_state.image.size()) - 1))) * expert_state.image.to(self.device) + \
            (1 - alpha).view(agent_state.image.size()[0], *([1] * (len(agent_state.image.size()) - 1))) * agent_state.image.to(self.device)
        mixup_state.image.requires_grad = True

        #mixup_state.text = DictList()
        #mixup_state.text = alpha.view(expert_state.text.size()[0], *([1] * (len(expert_state.text.size()) - 1))) * expert_state.text.to(self.device) + \
        #    (1 - alpha).view(agent_state.text.size()[0], *([1] * (len(agent_state.text.size()) - 1))) * agent_state.text.to(self.device)
        #mixup_state.text.requires_grad = True


        mixup_action = alpha.view(expert_action.size()[0], *([1] * (len(expert_action.size()) - 1))) * expert_action.to(self.device) + \
            (1 - alpha).view(agent_action.size()[0], *([1] * (len(agent_action.size()) - 1))) * agent_action.to(self.device)
        mixup_action = torch.floor(mixup_action)
        mixup_action.requires_grad = True
        
        mixup_memory = alpha.view(expert_memory.size()[0], *([1] * (len(expert_memory.size()) - 1))) * expert_memory.to(self.device) + \
            (1 - alpha).view(agent_memory.size()[0], *([1] * (len(agent_memory.size()) - 1))) * agent_memory.to(self.device)
        
        disc, memory = self.dcmodel(mixup_state.image.to(self.device), mixup_action.flatten().unsqueeze(1).to(self.device), mixup_memory.to(self.device))
        
        disc_ones = torch.ones(disc.size()).to(self.device)
        memory_ones = torch.ones(memory.size()).to(self.device)
        
        grad = autograd.grad(
            outputs=(disc, memory),
            inputs=(mixup_state.image, mixup_action, mixup_memory),
            grad_outputs=(disc_ones, memory_ones),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

def main():
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_recurrence{args.ac_recurrence}_seed{args.seed}_gail_{timestamp}"
    demo_dir = os.path.join(utils.get_model_dir(args.expert), "demo.pt")
    with open(demo_dir, "rb") as demo_file:
        demos_list = pickle.load(demo_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")
    print(device)

    rewardfn = self.RewardFn(device)
    batch_size = 100

    mlealgo = GAILAlgo(rewardfn, device, batch_size)
    mlealg.train(demos_list, random_list)