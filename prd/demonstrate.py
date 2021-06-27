import argparse
import time
import numpy
import torch
import pickle

import sys
import os
sys.path.append("{}".format(os.path.dirname(os.path.dirname(__file__))))
import utils
from datetime import datetime
timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import os



# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=bool, default=True,
                    help="store output as gif with the given filename")
parser.add_argument("--traj", type=bool, default=True, help="store trajectories")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes to visualize")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--use-best", action='store_true', default = False, help="use the highest return policy")



args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")
env_path = "{}/datasets".format(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# Load agent

model_dir = utils.get_model_dir(args.model)
use_memory = False
for str_i in args.model.split('_'):
    if str_i.startswith('recurrence'):
        if int(str_i.split('recurrence')[-1]) > 1:
            use_memory = True
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, use_memory=use_memory, use_text=args.text, best = args.use_best)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
#env.render('human')

if args.traj:
    traj_list =[]

# Static reward function 
from progrew import RewardFn, Rewarder, RewarderWithMemory
from samplers import GenerativeSampler
rewardfn = RewardFn()
sampler = GenerativeSampler("sampler", 20, rewardfn.num_holes, True, 64).to(device)
reshape_reward = Rewarder(rewardfn, sampler, device)

for episode in range(args.episodes):
    traj_list.append({"ob":[], "ac":[], "ep_ret":[]})
    obs = env.reset()
    i_step = 0
    while True:
        env.render('human')
        print("Step %d" % i_step)
        i_step += 1
        if i_step > 200:
            break
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = agent.get_action(obs)
        traj_list[-1]["ob"].append(obs)
        traj_list[-1]["ac"].append(action)
        
        obs, reward, done, _ = env.step(action)
        
        traj_list[-1]["ep_ret"].append(reward)
        agent.analyze_feedback(reward, done)
        
        reshape_reward(obs, action, reward, done)

        if done:# or env.window.closed:
            print("Done")
            break

    #if env.window.closed:
        #break


if args.gif:
    print("Saving gif... ", end="")
    
    traj_path = os.path.join(model_dir, "demo/")
    if not os.path.exists(traj_path):
        os.mkdir(traj_path)
    gif_path = os.path.join(traj_path, "{}_{}.gif".format(args.env, timestamp))
    write_gif(numpy.array(frames), gif_path, fps=1/args.pause)
    print("Done.")

if args.traj:
    print("Saving trajectories...", end="")
    traj_path = os.path.join(model_dir, "demo/")
    if not os.path.exists(traj_path):
        os.mkdir(traj_path)
    traj_path = os.path.join(traj_path, "{}_{}.pt".format(args.env, timestamp))
    with open(traj_path, "wb") as output_file:
        pickle.dump(traj_list, output_file)
