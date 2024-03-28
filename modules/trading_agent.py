"""
Trading agent to manage several assets
"""
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from modules.q_network import Q_network
from modules.memory import Memory

class TradingAgent:

    def __init__(self, state_size, num_actions, assets,
                 qnet=Q_network, 
                 batch_size = 64, learning_rate=1e-3,
                 tau = 2e-3, gamma=0.95, device='cpu',
                 learning_freq=5, seed = 123) -> None:
        
        self.state_size = state_size
        self.num_actions = num_actions

        self.map_assets = {asset: id for id, asset in enumerate(assets)}

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.gamma = gamma
        self.tau = tau

        self.qnets = [qnet(state_size, num_actions) for asset in assets]
        self.qnets_target = [qnet(state_size, num_actions) for asset in assets]

        self.memory = [Memory(batch_size*2, seed=seed) for asset in assets]

        self.device = device
        self.seed = seed
        self.learning_freq = learning_freq

        self.step_count = 0

        self.loss_fun = nn.MSELoss()
        self.optimiser = [optim.Adam(qnet.parameters(), lr=learning_rate) for qnet in self.qnets]

    def step(self, asset, state, action, reward, next_state, done):

        # save tuple in memory
        self.memory[self.map_assets[asset]].add(state, action, reward, next_state, done)
        
        # learn at learning_freq (every learning_freq days)
        self.step_count += 1
        if self.step_count % self.learning_freq == 0:
            if len(self.memory[self.map_assets[asset]]) > self.batch_size:
                experiences = self.memory[self.map_assets[asset]].sample(self.batch_size)
                self.learn(asset, experiences)


    def act(self, asset, state, epsilon):
        """Selects action. 

        With proba (1-epsilon) select action having the highest score.
        With proba epsilon select random action

        Parameters
        ----------
        asset: str
            Asset name for which we should learn
        state: list
            List of observations constituting the state. List is composed of the 
            history and the predicted value for the current period.
        epsilon: float
            Must be such that 0 < epsilon < 1.

        return
        ------
        int:
            Action to perform at the current step.
        """
        state = torch.from_numpy(np.array(state, dtype=np.float32).reshape(1, -1))

        self.qnets[self.map_assets[asset]].eval() # set model in evaluation mode
        if np.random.rand() > epsilon:
            with torch.no_grad():
                scores_actions = self.qnets[self.map_assets[asset]](state)
                action = np.argmax(scores_actions.cpu().data.numpy())
        else:
            action = np.random.randint(self.num_actions)
        self.qnets[self.map_assets[asset]].train() # set model in evaluation mode
        
        return int(action)


    def learn(self, asset, experiences):

        # unpack states, actions, rewards, next_states and dones
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)


        pred_Q_vals = self.qnets[self.map_assets[asset]](states).gather(1, actions)

        # get max Q values for (s_{t+1}, a_{t+1}) from target model
        pred_Q_vals_next_target_scores = self.qnets_target[self.map_assets[asset]](next_states).detach()
        pred_Q_vals_next_target = pred_Q_vals_next_target_scores.max(1)[0].unsqueeze(1)

        # Q target for current states
        pred_Q_vals_target = rewards + (self.gamma * pred_Q_vals_next_target * (1 - dones))

        # compute loss + minimize it
        loss = self.loss_fun(pred_Q_vals, pred_Q_vals_target)

        self.optimiser[self.map_assets[asset]].zero_grad()
        loss.backward()
        self.optimiser[self.map_assets[asset]].step()

        # update target network
        self.soft_update(self.qnets[self.map_assets[asset]], self.qnets_target[self.map_assets[asset]], self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau*local_param + (1-tau)*target_param.data)

    def save_models(self, folder="/"):
        for asset in self.map_assets.keys():
            filename = os.path.join(folder, "trained_agent_model_"+asset+".pth")
            torch.save(self.qnets[self.map_assets[asset]].state_dict(), filename)