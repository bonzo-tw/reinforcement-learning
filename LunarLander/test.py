from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
import random
import gym

seed = 543 # Do not change this
def fix(env, seed):
    env.reset(seed=seed)# env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True) # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





env = gym.make('LunarLander-v2')
fix(env, seed)


env.reset()

img = plt.imshow(env.render(render_mode='rgb_array')) # img = plt.imshow(env.render(mode='rgb_array'))

terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncation, info = env.step(action)

    img.set_data(env.render(render_mode='rgb_array')) # img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)