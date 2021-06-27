import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

import numpy as np

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class GenerativeSampler(nn.Module):
    def __init__(self, name, input_size = 5, output_size = 1, use_seed = False, hidden_size = 32):
        # Suggest use_seed = True because of faster training and better performance
        super().__init__()

        self.name = name
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.output_size * 2),
            #nn.Tanh()
        )
        
        self.use_seed = use_seed
        
        # Initialize parameters correctly
        self.apply(init_params)

       
    def forward(self, x = None):
        if not self.use_seed:
            x = torch.ones([1, self.input_size])
        elif x is None:
            x = torch.rand([1, self.input_size])
        else:
            assert x.size()[-1] == self.input_size
        
        mean_logvar = torch.split(self.fc(x), self.output_size, dim = 1)

        mean = mean_logvar[0]
        logvar =  mean_logvar[1]
        return mean, logvar

    
def sample(mean, logvar, num_samples):
    return torch.normal(mean.repeat(num_samples, 1), torch.exp(logvar).repeat(num_samples, 1))
        

def reparameterize(mean, logvar, num_samples = 1):
    eps = torch.rand_like(logvar.repeat(num_samples, 1))
    return eps * torch.exp(logvar * .5) + mean


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = np.log(2. * np.pi).item()
    return torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi), dim=raxis, keepdims = False)


class GenerativeSamplerUnitTest(object):
    def __init__(self, sampler, target, batch_size = 256, num_eps = 100):
        self.sampler = sampler
        self.input_size = self.sampler.input_size
        self.output_size = self.sampler.output_size
        self.target = target
        self.batch_size = batch_size
        self.num_eps = num_eps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler.to(self.device)
        

    def test(self):
        label = np.asarray([self.target])
        if label.shape[-1] != self.output_size:
            label = np.zeros([[self.output_size]])
            label[:, self.target] = 1.
        label = torch.tensor(label)
        batch_label = label.repeat(self.batch_size, 1).to(self.device)

        #if self.output_size > 1:
        #    loss = lambda tru, pred: tf.reduce_sum(tru * pred)
        #else:
        #    loss = lambda tru, pred: tf.reduce_mean(tf.keras.losses.MSE(tru, pred))
        loss = lambda tru, pred: nn.functional.mse_loss(tru, pred)
        optimizer = torch.optim.RMSprop(self.sampler.parameters(), lr = 0.001,
                                             alpha=0.99, eps=1e-8)
        #optimizer = torch.optim.Adam(self.sampler.parameters(), 0.01, eps=1e-8)
        for i_ep in range(self.num_eps):
            batch_loss = self.step(optimizer, loss, batch_label)
            if i_ep % 10 == 0:# and i_ep > 10:
                print("Episode {}, batch loss: {}".format(i_ep, batch_loss))
                mean, logvar, batch_error = self.step(optimizer, loss, batch_label, training = False)
                print("Mean: {} logvar: {}  Error: {}".format(mean, logvar, batch_error))


    def step(self, optimizer, loss, batch_label, training = True):
        seed = torch.ones([1, self.input_size]).to(self.device)
        with torch.no_grad():
            self.sampler.eval()
            mean, logvar = self.sampler(seed)
            mean = mean.detach()
            logvar = logvar.detach()
            #print(mean, logvar)
            batch_output = sample(mean, logvar, num_samples = self.batch_size)
            #print(batch_output)

            batch_error = torch.sum(torch.square(batch_label - batch_output), dim = 1, keepdim = False).flatten()
            batch_error.detach()
            #print(batch_error)
            assert batch_error.size()[0] == self.batch_size
        
        
        if training:           
            with torch.enable_grad():
                self.sampler.train()
                mean_, logvar_ = self.sampler(seed)
                #print(mean_, logvar_)
                batch_logps = log_normal_pdf(batch_output, mean_, logvar_).flatten()
                assert batch_logps.size()[0] == self.batch_size
                #print(batch_logps)
                #print(torch.exp(batch_logps))
                
                batch_loss = torch.sum(batch_logps * batch_error)
                optimizer.zero_grad()
                batch_loss.backward()
                #update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.sampler.parameters()) ** 0.5
                #torch.nn.utils.clip_grad_norm_(self.sampler.parameters(), 0.5)
                optimizer.step()
                return batch_loss
        else:
            return mean.detach().cpu().numpy(), logvar.detach().cpu().numpy(), batch_error.mean().cpu().numpy()


if __name__ == "__main__":
    sampler = GenerativeSampler("sampler", 20, 3, True, 64)
    samplertest = GenerativeSamplerUnitTest(sampler, [-1. , 0., 1.], 200, 100)
    samplertest.test()
