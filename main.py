import argparse
import gym
import torch_rl
import torch
import babyai
# import levelsRR
import random
import time

env = gym.make('BabyAI-KeyCorridorS5R3-v0')

actions = [0, 1, 2, 3, 4, 5, 6]
rs = []

for _ in range(1000):
    env.render()
    time.sleep(0.0)
    _, r, _, _ = env.step(actions[random.randint(0, len(actions) - 1)])

    print(r, sum(rs), rs)
    if r > 0:
        time.sleep(1)
        rs.append(r)

