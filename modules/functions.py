"""
Train and test functions
"""


import numpy as np
import torch

def train_agent(assets, agent, train_envs, num_epochs=10):

    scores = {key: [] for key in assets}

    for epoch in range(num_epochs):

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

def test_agent(assets, agent, test_envs):

    total_profit = 0
    final_running_balance_dict = {}
    profits = {}

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
    
        # compute balance and profit
        final_running_balance = (env.data.iloc[env.t]['Close']*env.positions) + env.balance
        final_running_balance_dict[asset] = final_running_balance
        total_profit += (env.data.iloc[env.t]['Close']*env.positions) + env.balance - env.initial_balance
        profits[asset] = (env.data.iloc[env.t]['Close']*env.positions) + env.balance - env.initial_balance
        print(f"Balance : {asset}: {round(final_running_balance,2)}")

    print("-"*28)
    print(f"Total profit made: {round(total_profit,2)}")

    return final_running_balance_dict, profits

def test_agent_one_asset(asset, agent, test_env):

    final_running_balance = 0
    profits = 0

    state = test_env.reset()

    test_actions = []
    test_rewards = []

    for _ in range(len(test_env.data)-1):
            
        action = agent.qnets[agent.map_assets[asset]](torch.from_numpy(np.array(state, dtype=np.float32).reshape(1, -1)))
        action = np.argmax(action.data)

        test_actions.append(action.item())
                
        next_state, reward, done = test_env.step(action.numpy())
        test_rewards.append(reward)

        state = next_state
    
        # compute balance and profit
    final_running_balance += (test_env.data.iloc[test_env.t]['Close']*test_env.positions) + test_env.balance
        
    profits += (test_env.data.iloc[test_env.t]['Close']*test_env.positions) + test_env.balance - test_env.initial_balance

    print(f"Balance : {asset}: {round(final_running_balance,2)}")

    print("-"*28)
    print(f"Total profit made: {round(profits,2)}")

    return final_running_balance, profits