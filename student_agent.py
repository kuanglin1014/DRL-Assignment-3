import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import cv2
import pickle
import os
import time
from collections import deque
from torchvision import transforms
from PIL import Image

class Features(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        c, h, w = input_shape
        self.convolution = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convolution(x)

class DuelingDQN(nn.Module):
    def __init__(self, features, input_shape, action_size):
        super().__init__()
        c, h, w = input_shape
        self.features = features
        self.action_size = action_size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w).to(next(features.parameters()).device)
            self.fc_input_dim = self.features(dummy).view(1, -1).size(1)

        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = 12
        self.width = 84
        self.height = 84
        self.state_shape = (4, 84, 84)
        self.frames = deque(maxlen=4)
        self.features = Features(self.state_shape).to(self.device)
        self.q_network = DuelingDQN(self.features, self.state_shape, self.action_size).to(self.device)
        checkpoint = torch.load("dqn_agent.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.q_network.eval()
        self.skip = 4
        self.start = 0
        self.step_count = 0
        self.last_action = 0

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height))
        frame = frame / 255.0
        return frame.astype(np.float32)

    def reset(self):
        self.frames.clear()
        self.start = 0
        self.step_count = 0
        self.last_action = 0


    def act(self, observation):
        #return self.action_space.sample()
        processed_obs = self.preprocess(observation)
        if not self.start:
            self.reset()
            for _ in range(4):
                self.frames.append(processed_obs)
            self.start = 1
        self.step_count += 1
        if self.step_count % self.skip == 0:
            self.frames.append(processed_obs)
            state = np.stack(self.frames, axis=0)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
                self.last_action = torch.argmax(q_values, dim=1).item()
        return self.last_action

        