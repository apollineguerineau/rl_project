
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

import streamlit as st
import numpy as np
import torch

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from modules.data_loader import DataLoader
from modules.single_asset_env import SingleAssetEnv
from modules.q_network import Q_network
from modules.memory import Memory
from modules.trading_agent import TradingAgent
from modules.functions import train_agent, test_agent


st.set_page_config(page_title="Gestion d'actifs", page_icon="🌿", layout = 'wide')

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# create a Streamlit selectbox to choose the variable
asset_name = st.sidebar.selectbox(
    'Choisir les actifs:',
    options=["BTC-USD", "ETH-USD", "BNB-USD"],
    index=0
)
assets = ["BTC-USD", "ETH-USD", "BNB-USD"]

## Load Training and Testing Dataset
start_date = '2020-01-01'
end_date =  '2024-03-30'
date_split = '2023-05-01'

dataloader = DataLoader(start_date, end_date, '1d', date_split)


trains = []
tests = []

for asset in assets:
    train, test = dataloader.load(asset)

    trains.append(train)
    tests.append(test)

# Environment and Agent Initiation
train_envs = [SingleAssetEnv(train) for train in trains]
test_envs = [SingleAssetEnv(test) for test in tests]

# Parameters and variables
num_actions = 3
states_dim = 30

num_epochs = 40
batch_size = 64
memory_size = 128

learning_rate = 1e-3
learning_freq = 5

tau = 3e-2
gamma = 0.95

device = 'cpu'
seed = 123


# Check if the trained model exists
# Delete the files online.pt and target.pt if you want to retrain the model
model_path = "weights/trained_agent_model_"+asset+".pth"

if os.path.exists(model_path) :
    ## Agent
    agent = TradingAgent(states_dim, num_actions, assets, 
                         batch_size=batch_size,
                         memory_size=memory_size,
                         learning_rate=learning_rate,
                         tau=tau,
                         gamma=gamma,
                         learning_freq=learning_freq,
                         device=device,
                         seed=seed)

    model_path = "weights/trained_agent_model_"+asset+".pth"
    # Load the state dict
    state_dict = torch.load(model_path)

    # Set the loaded state dict to the Q network of the corresponding asset
    agent.qnets[agent.map_assets[asset]].load_state_dict(state_dict)

    final_running_balance_dict = test_agent(assets, agent, test_envs)

else:  # TRAINING MODEL IN CASE IT IS NOT TRAINED YET

    ## Agent
    # j
    agent = TradingAgent(states_dim, num_actions, assets, 
                         batch_size=batch_size,
                         memory_size=memory_size,
                         learning_rate=learning_rate,
                         tau=tau,
                         gamma=gamma,
                         learning_freq=learning_freq,
                         device=device,
                         seed=seed)
    
    train_agent(assets, agent, train_envs, num_epochs)
    final_running_balance_dict = test_agent(assets, agent, test_envs)
    st.sidebar.success("Model trained successfully!")


def get_graph():
    st.title("Reinforcement Learning with Deep Q Networks Algorithm")
    st.write('\n\n')
    st.subheader("Visualisation de dataset ")
    st.write('\n')
 
    
    st.write('Prix de ', asset_name)
    # Plotting data
    if asset_name == "BTC-USD":
        fig_dataset = DataLoader(start_date, end_date, '1d', date_split).plot_train_test("BTC-USD")
    if asset_name == "ETH-USD":
        fig_dataset = DataLoader(start_date, end_date, '1d', date_split).plot_train_test("ETH-USD")
    if asset_name == "BNB-USD":
        fig_dataset = DataLoader(start_date, end_date, '1d', date_split).plot_train_test("BNB-USD")
    st.write('\n\n')
    st.plotly_chart(fig_dataset, use_container_width=True)

    

if __name__ == '__main__':
    get_graph()