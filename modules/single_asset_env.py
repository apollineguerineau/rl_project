"""
Trading environment for a single asset
"""
from collections import deque

class SingleAssetEnv:
    
    def __init__(self, data, len_history=7):
        self.data = data
        self.len_history = len_history
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profit = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.len_history)]
        return [self.position_value] + self.history # obs
    
    def step(self, act):
        reward = 0
        
        # act = {0: stay, 1: buy, 2: sell}

        # BUY
        if act == 1:
            reward -= self.data.iloc[self.t]['Close']
            self.profit -= reward
            self.positions.append(self.data.iloc[self.t]['Close'])
        
        # SELL
        elif act == 2:
            if len(self.positions) == 0:
                reward = -100000
            else:
                reward += self.data.iloc[self.t]['Close']*len(self.positions)
                self.profit += reward
                self.positions = []

        # HOLD
        elif act == 0:
            reward = -1
        
        # predict next value
        self.position_value = sum(self.history)/self.len_history

        # update history
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t]['Close'])

        if (self.t==len(self.data)-1):
            self.done=True

        # set next time
        self.t += 1

        return [self.position_value] + self.history, reward, self.done # obs, reward, done