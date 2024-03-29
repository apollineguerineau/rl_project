import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from modules.data_loader import DataLoader
from modules.single_asset_env import SingleAssetEnv
from modules.q_network import Q_network
from modules.memory import Memory
from modules.trading_agent import TradingAgent
import func
import torch
import numpy as np

def test_agent(assets, test_envs, agent):
    total_profit = 0
    final_running_balance_dict = {}
    for asset, env in zip(assets, test_envs):

        state = env.reset()

        test_actions = []
        test_rewards = []

        for _ in range(len(env.data)-1):
            
            action = agent.qnets[agent.map_assets[asset]](torch.from_numpy(np.array(state, dtype=np.float32).reshape(1, -1)))
            action = np.argmax(action.data)

            test_actions.append(action.item())
                    
            next_state, reward, done = env.step(action.numpy())
            test_rewards.append(reward)

            state = next_state
    
        final_running_balance = (env.data.iloc[env.t]['Close']*env.positions) + env.balance
        final_running_balance_dict[asset] = final_running_balance
        total_profit += (env.data.iloc[env.t]['Close']*env.positions) + env.balance - env.initial_balance
        print(f"Balance : {asset}: {final_running_balance}")

    print("-"*27)
    print(f"Total profit made: {total_profit}")

    return final_running_balance_dict

def train_agent(assets, train_envs, agent, NUM_EPOCHS):
    scores = {key: [] for key in assets}
    for epoch in range(NUM_EPOCHS):

        # intialise score for the epoch
        score = {key: 0 for key in assets}
        step_count = 1

        for asset, env in zip(assets, train_envs):

            # reset the environment before each epoch + get initial state
            state = env.reset()

            while True:

                # find epsilon greedy action from state
                action = agent.act(asset, state, 1/step_count) # epsilon = 1/t

                # perform step in the environment and get completing info
                next_state, reward, done = env.step(action)

                agent.step(asset, state, action, reward, next_state, done)

                # prepare for next iteration
                step_count += 1
                state = next_state

                score[asset] += reward

                if done:
                    break

        # compute info about the epoch
        for key in scores.keys():
            scores[key].append(score[key])

        print(f"Epoch {epoch:2} | Scores = {score}")

    print("Training done!")

    # save Q_network model weights
    agent.save_models("weights")

