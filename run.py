#! /bin/python

from safe_rl import ppo_lagrangian
import gym, safety_gym

ppo_lagrangian(
    env_fn = lambda : gym.make('Safexp-PointGoal1-v0'),
        ac_kwargs = dict(hidden_sizes=(64,64))
            )

