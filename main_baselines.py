# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:31:21 2018

@author: ck
"""

from baselines import deepq
from Car2D import Car2DEnv

env = Car2DEnv()

model = deepq.models.mlp([32, 16], layer_norm=True)
act = deepq.learn(
    env,
    q_func=model,
    lr=0.01,
    max_timesteps=10000,
    print_freq=1,
    checkpoint_freq=1000
)

print('Finish!')
#act.save("mountaincar_model.pkl")

#act = deepq.load("mountaincar_model.pkl")
while True:
    obs, done = env.reset(), False
    episode_reward = 0
    while not done:
        env.render()
        obs, reward, done, _ = env.step(act(obs[None])[0])
        episode_reward += reward
    print([episode_reward, env.counts])
