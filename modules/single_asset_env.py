"""
Trading environment for a single asset
"""
from collections import deque
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class SingleAssetEnv:
    
    def __init__(self, data, len_history=30, initial_balance = 100000):
        self.data = data
        self.len_history = len_history
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.reset()
        self.store = {"action_store": [],
                    "reward_store": [],
                    "running_capital": []}
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profit = 0
        self.positions = 0
        self.position_value = 0
        self.balance = self.initial_balance
        self.history = [0 for _ in range(self.len_history)]
        self.store = {"action_store": [],
                    "reward_store": [],
                    "running_capital": [self.initial_balance]}
        return self.history # obs
    
    def step(self, act):
        reward = 0
        
        # act = {0: stay, 1: buy, 2: sell}

        # BUY
        if act == 1:
            if self.balance >= self.data.iloc[self.t]['Close']:
                self.balance -= self.data.iloc[self.t]['Close']
                self.positions += 1
            else:
                reward = -100000

        # SELL (all)
        elif act == 2:
            if self.positions == 0:
                reward = -100000
            else:
                self.balance += (self.data.iloc[self.t]['Close'])*self.positions  
                self.positions = 0
        
        elif act == 0: # Hold
                reward -= 100

        reward += ((self.data.iloc[self.t]['Close'])*self.positions + self.balance) - self.initial_balance

        # update history
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t]['Close'])

        if (self.t==len(self.data)-1):
            self.done=True

        # set next time
        self.t += 1
        self.store["action_store"].append(act)
        self.store["reward_store"].append(reward)
        self.store["running_capital"].append(self.balance)

        return self.history, reward, self.done # obs, reward, done

    def render(self, asset_name):
        # Create a figure with one subplot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot running capital over time
        ax.plot(self.store["running_capital"], label="Running Capital")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Capital")
        ax.set_title("Running Capital Over Time")
        ax.legend()

        # Plot candlestick chart
        candlestick_data = {
            'x': self.data.index,
            'open': self.data['Open'],
            'high': self.data['High'],
            'low': self.data['Low'],
            'close': self.data['Close'],
            'name': asset_name
        }
        fig_candlestick = go.Figure(data=go.Candlestick(**candlestick_data))
        fig_candlestick.update_layout(title="Candlestick Chart of " + asset_name, xaxis_title="Date", yaxis_title="Price")
        # Add markers for actions
        buy_indices = [idx for idx, action in enumerate(self.store["action_store"]) if action == 1]
        sell_indices = [idx for idx, action in enumerate(self.store["action_store"]) if action == 2]
        hold_indices = [idx for idx, action in enumerate(self.store["action_store"]) if action == 0]

        buy_prices = [self.data.iloc[idx]['Low'] for idx in buy_indices]
        sell_prices = [self.data.iloc[idx]['High'] for idx in sell_indices]
        hold_prices = [self.data.iloc[idx]['Close'] for idx in hold_indices]

        fig_candlestick.add_trace(go.Scatter(x=self.data.iloc[buy_indices].index, y=buy_prices, mode='markers', name='Buy', marker_symbol='triangle-up', marker=dict(color='green', size=5)))
        fig_candlestick.add_trace(go.Scatter(x=self.data.iloc[sell_indices].index, y=sell_prices, mode='markers', name='Sell', marker_symbol='triangle-down', marker=dict(color='red', size=5)))
        fig_candlestick.add_trace(go.Scatter(x=self.data.iloc[hold_indices].index, y=hold_prices, mode='markers', name='Hold', marker_symbol='circle', marker=dict(color='blue', size=5)))

        # Show the candlestick chart using Plotly's iplot function
        #fig_candlestick.show()

        # Display the subplots
        #plt.tight_layout()
        #plt.show()
        return fig, fig_candlestick