
import os
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
from model import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns


st.set_page_config(page_title="Gestion de portefeuille", page_icon="🌿", layout = 'wide')

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# create a Streamlit selectbox to choose the variable
asset = st.sidebar.selectbox(
    'Choisir les actifs:',
    options=["ETH-USD", "BNB-USD"],
    index=0
)

asset_codes = ["ETH-USD", "BNB-USD"]

## Training and Testing Environments
assets = [DataGetter(a, start_date="2020-01-01", end_date="2023-01-01") for a in asset_codes]
validation_assets = [DataGetter(a, start_date="2023-01-01", end_date="2024-01-01", freq="1d") for a in asset_codes]
test_assets = [DataGetter(a, start_date="2024-01-01", end_date="2024-03-01", freq="1d") for a in asset_codes]

envs = [SingleAssetTradingEnvironment(a) for a in assets]
test_envs = [SingleAssetTradingEnvironment(a) for a in test_assets]
val_envs = [SingleAssetTradingEnvironment(a) for a in validation_assets]

# Parameters
N_EPISODES = 10
act_dict = {0: -1, 1: 1, 2: 0}


def test_agent(test_envs, agent, N_EPISODES):
    # Load the trained model
    online_state_dict = torch.load("online.pt")
    target_state_dict = torch.load("target.pt")

    st.sidebar.success("Trained model loaded successfully!")

    agent.actor_online.load_state_dict(online_state_dict)
    agent.actor_target.load_state_dict(target_state_dict)

    te_score_min = -np.Inf
    test_score = 0
    test_score2 = 0
    N_EPISODES = 10  # No of episodes/epochs
    for episode in range(1, 1 + N_EPISODES):
        for i, test_env in enumerate(test_envs):
            state = test_env.reset()
            done = False
            score_te = 0
            scores_te = [score_te]

            while True:

                actions = agent.act(state)
                action = act_dict[actions]
                next_state, reward, done, _ = test_env.step(action)
                next_state = next_state.reshape(-1, STATE_SPACE)
                state= next_state
                score_te += reward
                scores_te.append(score_te)
                if done:
                    break

            test_score += score_te
            test_score2 += (test_env.store['running_capital'][-1] - test_env.store['running_capital'][0])
        if test_score > te_score_min:
            te_score_min = test_score
            torch.save(agent.actor_online.state_dict(), "online.pt")
            torch.save(agent.actor_target.state_dict(), "target.pt")
        print(f"Episode: {episode}, Validation Score: {test_score:.5f}")
        print(f"Episode: {episode}, Validation Value: ${test_score2:.5f}", "\n")

def train_agent(envs, val_envs, agent, N_EPISODES):
    scores = []
    eps = EPS_START
    te_score_min = -np.Inf
    for episode in range(1, 1 + N_EPISODES):
        counter = 0
        episode_score = 0
        episode_score2 = 0
        test_score = 0
        test_score2 = 0

        for env in envs:
            score = 0
            state = env.reset()
            state = state.reshape(-1, STATE_SPACE)
            while True:
                actions = agent.act(state, eps)
                action = act_dict[actions]
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape(-1, STATE_SPACE)

                t = Transition(state, actions, reward, next_state, done)
                agent.memory.store(t)
                agent.learn()

                state = next_state
                score += reward
                counter += 1
                if done:
                    break

            episode_score += score
            episode_score2 += (env.store['running_capital'][-1] - env.store['running_capital'][0])

        scores.append(episode_score)
        eps = max(EPS_END, EPS_DECAY * eps)

        for i, test_env in enumerate(val_envs):
            state = test_env.reset()
            done = False
            score_te = 0
            scores_te = [score_te]

            while True:
                actions = agent.act(state)
                action = act_dict[actions]
                next_state, reward, done, _ = test_env.step(action)
                next_state = next_state.reshape(-1, STATE_SPACE)
                state = next_state
                score_te += reward
                scores_te.append(score_te)
                if done:
                    break

            test_score += score_te
            test_score2 += (test_env.store['running_capital'][-1] - test_env.store['running_capital'][0])
        if test_score > te_score_min:
            te_score_min = test_score
            torch.save(agent.actor_online.state_dict(), "online.pt")
            torch.save(agent.actor_target.state_dict(), "target.pt")

        print(f"Episode: {episode}, Train Score: {episode_score:.5f}, Validation Score: {test_score:.5f}")
        print(f"Episode: {episode}, Train Value: ${episode_score2:.5f}, Validation Value: ${test_score2:.5f}", "\n")



# Check if the trained model exists
# Delete the files online.pt and target.pt if you want to retrain the model
if os.path.exists("online.pt") and os.path.exists("target.pt"):
    ## Agent
    memory = ReplayMemory()
    agent = DQNAgent(actor_net=DuellingDQN, memory=memory)

    test_agent(val_envs, agent, N_EPISODES)

else:  # TRAINING MODEL IN CASE IT IS NOT TRAINED YET

    ## Agent
    memory = ReplayMemory()
    agent = DQNAgent(actor_net=DuellingDQN, memory=memory)

    train_agent(envs, val_envs, agent, N_EPISODES)
    test_agent(test_envs, agent, N_EPISODES)
    st.sidebar.success("Model trained successfully!")


def get_graph():
    st.title("Reinforcement Learning with Duelling Deep Q Networks Algorithm")
    st.write('\n\n')
    st.subheader("Résultat de l'aglgorithme  ")
    st.write('\n')
 
    
    st.subheader('Validation set')
    # Plotting val set
    if asset == "ETH-USD":
        val_env = val_envs[0]
    elif asset == "BNB-USD":
        val_env = val_envs[1]
    
    st.subheader('Statistics')
    st.write('Initial Capital: ',CAPITAL)
    st.write('Agent Net-worth for this Asset: ',val_env.store['running_capital'][-1])
    st.write('\n')

    fig_test, candle_fig_test = val_env.render()
    st.plotly_chart(candle_fig_test, use_container_width=True)
    st.write('\n\n')
    st.pyplot(fig_test)
    

if __name__ == '__main__':
    get_graph()