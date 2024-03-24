STATE_SPACE = 28
ACTION_SPACE = 3

ACTION_LOW = -1
ACTION_HIGH = 1

GAMMA = 0.9995
TAU = 1e-3
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.9

MEMORY_LEN = 10000
MEMORY_THRESH = 500
BATCH_SIZE = 200

LR_DQN = 5e-4

LEARN_AFTER = MEMORY_THRESH
LEARN_EVERY = 3
UPDATE_EVERY = 9

COST = 3e-4
CAPITAL = 100000
NEG_MUL = 2

DEVICE = "cpu"

from collections import namedtuple
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DataGetter:
  """
  The class for getting data for assets.
  """

  def __init__(self, asset="BTC-USD", start_date=None, end_date=None, freq="1d", 
               timeframes=[1, 2, 5, 10, 20, 40]):
    self.asset = asset
    self.sd = start_date
    self.ed = end_date
    self.freq = freq

    self.timeframes = timeframes
    self.getData()

    self.scaler = StandardScaler()
    self.scaler.fit(self.data[:, 1:])


  def getData(self):
    
    asset = self.asset  
    if self.sd is not None and self.ed is not None:
      df =  yf.download([asset], start=self.sd, end=self.ed, interval=self.freq)
      df_spy = yf.download(["BTC-USD"], start=self.sd, end=self.ed, interval=self.freq)
    elif self.sd is None and self.ed is not None:
      df =  yf.download([asset], end=self.ed, interval=self.freq)
      df_spy = yf.download(["BTC-USD"], end=self.ed, interval=self.freq)
    elif self.sd is not None and self.ed is None:
      df =  yf.download([asset], start=self.sd, interval=self.freq)
      df_spy = yf.download(["BTC-USD"], start=self.sd, interval=self.freq)
    else:
      df = yf.download([asset], period="max", interval=self.freq)
      df_spy = yf.download(["BTC-USD"], interval=self.freq)
    
    # Reward - Not included in Observation Space.
    df["rf"] = df["Adj Close"].pct_change().shift(-1)

    # Returns and Trading Volume Changes
    for i in self.timeframes:
      df_spy[f"spy_ret-{i}"] = df_spy["Adj Close"].pct_change(i)
      df_spy[f"spy_v-{i}"] = df_spy["Volume"].pct_change(i)

      df[f"r-{i}"] = df["Adj Close"].pct_change(i)      
      df[f"v-{i}"] = df["Volume"].pct_change(i)
    
    # Volatility
    for i in [5, 10, 20, 40]:
      df[f'sig-{i}'] = np.log(1 + df["r-1"]).rolling(i).std()

    # Moving Average Convergence Divergence (MACD)
    df["macd_lmw"] = df["r-1"].ewm(span=26, adjust=False).mean()
    df["macd_smw"] = df["r-1"].ewm(span=12, adjust=False).mean()
    df["macd_bl"] = df["r-1"].ewm(span=9, adjust=False).mean()
    df["macd"] = df["macd_smw"] - df["macd_lmw"]

    # Relative Strength Indicator (RSI)
    rsi_lb = 5
    pos_gain = df["r-1"].where(df["r-1"] > 0, 0).ewm(rsi_lb).mean()
    neg_gain = df["r-1"].where(df["r-1"] < 0, 0).ewm(rsi_lb).mean()
    rs = np.abs(pos_gain/neg_gain)
    df["rsi"] = 100 * rs/(1 + rs)

    # Bollinger Bands
    bollinger_lback = 10
    df["bollinger"] = df["r-1"].ewm(bollinger_lback).mean()
    df["low_bollinger"] = df["bollinger"] - 2 * df["r-1"].rolling(bollinger_lback).std()
    df["high_bollinger"] = df["bollinger"] + 2 * df["r-1"].rolling(bollinger_lback).std()

    # SP500
    #df = df.merge(df_spy[[f"spy_ret-{i}" for i in self.timeframes] + [f"spy_sig-{i}" for i in [5, 10, 20, 40]]], 
    #              how="left", right_index=True, left_index=True)

    # Filtering
    for c in df.columns:
      df[c].interpolate('linear', limit_direction='both', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    self.frame = df
    self.data = np.array(df.iloc[:, 6:])
    return


  def scaleData(self):
    self.scaled_data = self.scaler.fit_transform(self.data[:, 1:])
    return


  def __len__(self):
    return len(self.data)


  def __getitem__(self, idx, col_idx=None):
    if col_idx is None:
      return self.data[idx]
    elif col_idx < len(list(self.data.columns)):
      return self.data[idx][col_idx]
    else:
      raise IndexError

class SingleAssetTradingEnvironment:
  """
  Trading Environment for trading a single asset.
  The Agent interacts with the environment class through the step() function.
  Action Space: {-1: Sell, 0: Do Nothing, 1: Buy}
  """

  def __init__(self, asset_data,
               initial_money=CAPITAL, trans_cost=COST, store_flag=1, asset_ph=0, 
               capital_frac=0.2, running_thresh=0.1, cap_thresh=0.3):

    self.past_holding = asset_ph
    self.capital_frac = capital_frac # Fraction of capital to invest each time.
    self.cap_thresh = cap_thresh
    self.running_thresh = running_thresh
    self.trans_cost = trans_cost

    self.asset_data = asset_data
    self.terminal_idx = len(self.asset_data) - 1
    self.scaler = self.asset_data.scaler    

    self.initial_cap = initial_money

    self.capital = self.initial_cap
    self.running_capital = self.capital
    self.asset_inv = self.past_holding

    self.pointer = 0
    self.next_return, self.current_state = 0, None
    self.prev_act = 0
    self.current_act = 0
    self.current_reward = 0
    self.current_price = self.asset_data.frame.iloc[self.pointer, :]['Adj Close']
    self.done = False

    self.store_flag = store_flag
    if self.store_flag == 1:
      self.store = {"action_store": [],
                    "reward_store": [],
                    "running_capital": [],
                    "port_ret": []}


  def reset(self):
    self.capital = self.initial_cap
    self.running_capital = self.capital
    self.asset_inv = self.past_holding

    self.pointer = 0
    self.next_return, self.current_state = self.get_state(self.pointer)
    self.prev_act = 0
    self.current_act = 0
    self.current_reward = 0
    self.current_price = self.asset_data.frame.iloc[self.pointer, :]['Adj Close']
    self.done = False
    
    if self.store_flag == 1:
      self.store = {"action_store": [],
                    "reward_store": [],
                    "running_capital": [],
                    "port_ret": []}

    return self.current_state


  def step(self, action):
    self.current_act = action
    self.current_price = self.asset_data.frame.iloc[self.pointer, :]['Adj Close']
    self.current_reward = self.calculate_reward()
    self.prev_act = self.current_act
    self.pointer += 1
    self.next_return, self.current_state = self.get_state(self.pointer)
    self.done = self.check_terminal()

    if self.done:
      reward_offset = 0
      ret = (self.store['running_capital'][-1]/self.store['running_capital'][-0]) - 1
      if self.pointer < self.terminal_idx:
        reward_offset += -1 * max(0.5, 1 - self.pointer/self.terminal_idx)
      if self.store_flag:
        reward_offset += 10 * ret
      self.current_reward += reward_offset

    if self.store_flag:
      self.store["action_store"].append(self.current_act)
      self.store["reward_store"].append(self.current_reward)
      self.store["running_capital"].append(self.capital)
      info = self.store
    else:
      info = None
    
    return self.current_state, self.current_reward, self.done, info


  def calculate_reward(self):
    investment = self.running_capital * self.capital_frac
    reward_offset = 0

    # Buy Action
    if self.current_act == 1: 
      if self.running_capital > self.initial_cap * self.running_thresh:
        self.running_capital -= investment
        asset_units = investment/self.current_price
        self.asset_inv += asset_units
        self.current_price *= (1 - self.trans_cost)

    # Sell Action
    elif self.current_act == -1:
      if self.asset_inv > 0:
        self.running_capital += self.asset_inv * self.current_price * (1 - self.trans_cost)
        self.asset_inv = 0

    # Do Nothing
    elif self.current_act == 0:
      if self.prev_act == 0:
        reward_offset += -0.1
      pass
    
    # Reward to give
    prev_cap = self.capital
    self.capital = self.running_capital + (self.asset_inv) * self.current_price
    reward = 100*(self.next_return) * self.current_act - np.abs(self.current_act - self.prev_act) * self.trans_cost
    if self.store_flag==1:
      self.store['port_ret'].append((self.capital - prev_cap)/prev_cap)
    
    if reward < 0:
      reward *= NEG_MUL  # To make the Agent more risk averse towards negative returns.
    reward += reward_offset

    return reward


  def check_terminal(self):
    if self.pointer == self.terminal_idx:
      return True
    elif self.capital <= self.initial_cap * self.cap_thresh:
      return True
    else:
      return False


  def get_state(self, idx):
    state = self.asset_data[idx][1:]
    state = self.scaler.transform(state.reshape(1, -1))

    state = np.concatenate([state, [[self.capital/self.initial_cap,
                                     self.running_capital/self.capital,
                                     self.asset_inv * self.current_price/self.initial_cap,
                                     self.prev_act]]], axis=-1)
    
    next_ret = self.asset_data[idx][0]
    return next_ret, state
  
  def render(self):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Plot running capital and portfolio returns over time
    axes[0].plot(self.store["running_capital"], label="Running Capital")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Capital")
    axes[0].set_title("Running Capital and Portfolio Returns Over Time")
    axes[0].legend()

    axes[1].plot(self.store["port_ret"], label="Portfolio Returns")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Returns")
    axes[1].set_title("Portfolio Returns Over Time")
    axes[1].legend()

    # Plot candlestick chart
    candlestick_data = {
        'x': self.asset_data.frame.index,
        'open': self.asset_data.frame['Open'],
        'high': self.asset_data.frame['High'],
        'low': self.asset_data.frame['Low'],
        'close': self.asset_data.frame['Close'],
        'name': self.asset_data.asset
    }
    fig_candlestick = go.Figure(data=go.Candlestick(**candlestick_data))
    fig_candlestick.update_layout(title="Candlestick Chart of " + self.asset_data.asset, xaxis_title="Date", yaxis_title="Price")
    # Add markers for actions
    buy_indices = [idx for idx, action in enumerate(self.store["action_store"]) if action == 1]
    sell_indices = [idx for idx, action in enumerate(self.store["action_store"]) if action == -1]
    hold_indices = [idx for idx, action in enumerate(self.store["action_store"]) if action == 0]

    buy_prices = [self.asset_data.frame.iloc[idx]['Low'] for idx in buy_indices]
    sell_prices = [self.asset_data.frame.iloc[idx]['High'] for idx in sell_indices]
    hold_prices = [self.asset_data.frame.iloc[idx]['Close'] for idx in hold_indices]

    fig_candlestick.add_trace(go.Scatter(x=self.asset_data.frame.iloc[buy_indices].index, y=buy_prices, mode='markers', name='Buy', marker_symbol='triangle-up', marker=dict(color='green', size=5)))
    fig_candlestick.add_trace(go.Scatter(x=self.asset_data.frame.iloc[sell_indices].index, y=sell_prices, mode='markers', name='Sell', marker_symbol='triangle-down', marker=dict(color='red', size=5)))
    fig_candlestick.add_trace(go.Scatter(x=self.asset_data.frame.iloc[hold_indices].index, y=hold_prices, mode='markers', name='Hold', marker_symbol='circle', marker=dict(color='blue', size=5)))

    # Show the candlestick chart using Plotly's iplot function
    #fig_candlestick.show()

    # Display the subplots
    #plt.tight_layout()
    #plt.show()
    return fig, fig_candlestick


Transition = namedtuple("Transition", ["States", "Actions", "Rewards", "NextStates", "Dones"])


class ReplayMemory:
  """
  Implementation of Agent memory
  """
  def __init__(self, capacity=MEMORY_LEN):
    self.memory = deque(maxlen=capacity)

  def store(self, t):
    self.memory.append(t)

  def sample(self, n):
    a = random.sample(self.memory, n)
    return a

  def __len__(self):
    return len(self.memory)



class DuellingDQN(nn.Module):
  """
  Acrchitecture for Duelling Deep Q Network Agent
  """

  def __init__(self, input_dim=STATE_SPACE, output_dim=ACTION_SPACE):
    super(DuellingDQN, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.fc1 = nn.Linear(self.input_dim, 500)
    self.fc2 = nn.Linear(500, 500)
    self.fc3 = nn.Linear(500, 300)
    self.fc4 = nn.Linear(300, 200)
    self.fc5 = nn.Linear(200, 10)

    self.fcs = nn.Linear(10, 1)
    self.fcp = nn.Linear(10, self.output_dim)
    self.fco = nn.Linear(self.output_dim + 1, self.output_dim)

    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self.sig = nn.Sigmoid()
    self.sm = nn.Softmax(dim=1)

  def forward(self, state):
    x = self.relu(self.fc1(state))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.relu(self.fc4(x))
    x = self.relu(self.fc5(x))
    xs = self.relu(self.fcs(x))
    xp = self.relu(self.fcp(x))

    x = xs + xp - xp.mean()
    return x


class DQNAgent:
  """
  Implements the Agent components
  """

  def __init__(self, actor_net=DuellingDQN, memory=ReplayMemory()):
    
    self.actor_online = actor_net(STATE_SPACE, ACTION_SPACE).to(DEVICE)
    self.actor_target = actor_net(STATE_SPACE, ACTION_SPACE).to(DEVICE)
    self.actor_target.load_state_dict(self.actor_online.state_dict())
    self.actor_target.eval()

    self.memory = memory

    self.actor_criterion = nn.MSELoss()
    self.actor_op = optim.Adam(self.actor_online.parameters(), lr=LR_DQN)

    self.t_step = 0


  def act(self, state, eps=0.):
    self.t_step += 1
    state = torch.from_numpy(state).float().to(DEVICE).view(1, -1)
    
    self.actor_online.eval()
    with torch.no_grad():
      actions = self.actor_online(state)
    self.actor_online.train()

    if random.random() > eps:
      act = np.argmax(actions.cpu().data.numpy())
    else:
      act = random.choice(np.arange(ACTION_SPACE))
    return int(act)


  def learn(self):
    if len(self.memory) <= MEMORY_THRESH:
      return 0

    if self.t_step > LEARN_AFTER and self.t_step % LEARN_EVERY==0:
    # Sample experiences from the Memory
      batch = self.memory.sample(BATCH_SIZE)

      states = np.vstack([t.States for t in batch])
      states = torch.from_numpy(states).float().to(DEVICE)

      actions = np.vstack([t.Actions for t in batch])
      actions = torch.from_numpy(actions).float().to(DEVICE)

      rewards = np.vstack([t.Rewards for t in batch])
      rewards = torch.from_numpy(rewards).float().to(DEVICE)

      next_states = np.vstack([t.NextStates for t in batch])
      next_states = torch.from_numpy(next_states).float().to(DEVICE)

      dones = np.vstack([t.Dones for t in batch]).astype(np.uint8)
      dones = torch.from_numpy(dones).float().to(DEVICE)

      # ACTOR UPDATE
      # Compute next state actions and state values
      next_state_values = self.actor_target(next_states).max(1)[0].unsqueeze(1)
      y = rewards + (1-dones) * GAMMA * next_state_values
      state_values = self.actor_online(states).gather(1, actions.type(torch.int64))
      # Compute Actor loss
      actor_loss = self.actor_criterion(y, state_values)
      # Minimize Actor loss
      self.actor_op.zero_grad()
      actor_loss.backward()
      self.actor_op.step()

      if self.t_step % UPDATE_EVERY == 0:
        self.soft_update(self.actor_online, self.actor_target)
      # return actor_loss.item()


  def soft_update(self, local_model, target_model, tau=TAU):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


