
import os
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
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

st.set_page_config(page_title="Gestion des actifs", page_icon="🌿", layout = 'wide')

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
date_split = '2023-01-01'
start_date = '2020-01-01'
end_date =  '2024-01-01'
dataloader = DataLoader(start_date, '1d', date_split, end_date)


trains = []
tests = []

for asset in assets:
    train, test = dataloader.load(asset)

    trains.append(train)
    tests.append(test)

# Environment and Agent Initiation
train_envs = [SingleAssetEnv(train) for train in trains]
test_envs = [SingleAssetEnv(test) for test in tests]


# Parameters
NUM_ACTIONS = 3 # Buy, Sell, Hold
LEN_HISTORY = 30 # consider last week to predict next value
STATES_DIM = LEN_HISTORY # history + predicted value

# Q network params
INPUT_DIM = STATES_DIM
HIDDEN_DIM = 64
OUTPUT_DIM = NUM_ACTIONS

LEARNING_RATE = 1e-3

NUM_EPOCHS = 20
BATCH_SIZE = 64

MEMORY_SIZE = 200

GAMMA = 0.97

EPSILON = 1.0
EPSILON_DECREASE = 1e-3
EPSILON_MIN = 0.1
START_REDUCE_EPSILON = 200

TRAIN_FREQ = 10
UPDATE_Q_FREQ = 20
SHOW_LOG_FREQ = 5

DEVICE = 'cpu'
SEED = 123

# Check if the trained model exists
# Delete the files online.pt and target.pt if you want to retrain the model
model_path = "weights/trained_agent_model_"+asset+".pth"
if os.path.exists(model_path) :
    ## Agent
    agent = TradingAgent(STATES_DIM, NUM_ACTIONS, assets, seed=SEED)
    model_path = "weights/trained_agent_model_"+asset+".pth"
    # Load the state dict
    state_dict = torch.load(model_path)

    # Set the loaded state dict to the Q network of the corresponding asset
    agent.qnets[agent.map_assets[asset]].load_state_dict(state_dict)

    final_running_balance_dict = func.test_agent(assets, test_envs, agent)

else:  # TRAINING MODEL IN CASE IT IS NOT TRAINED YET

    ## Agent
    # j
    agent = TradingAgent(STATES_DIM, NUM_ACTIONS, assets, seed=SEED)
    
    func.train_agent(assets, train_envs, agent, NUM_EPOCHS)
    final_running_balance_dict = func.test_agent(assets, test_envs, agent)
    st.sidebar.success("Model trained successfully!")


def get_graph():
    st.title("Reinforcement Learning with Deep Q Networks Algorithm")
    st.write('\n\n')
    st.subheader("Visualisation de dataset ")
    st.write('\n')
 
    
    st.write('Prix de ', asset_name)
    # Plotting data
    if asset_name == "BTC-USD":
        fig_dataset = DataLoader(start_date, '1d', date_split, end_date).plot_train_test("BTC-USD")
    if asset_name == "ETH-USD":
        fig_dataset = DataLoader(start_date, '1d', date_split, end_date).plot_train_test("ETH-USD")
    if asset_name == "BNB-USD":
        fig_dataset = DataLoader(start_date, '1d', date_split, end_date).plot_train_test("BNB-USD")
    st.write('\n\n')
    st.plotly_chart(fig_dataset, use_container_width=True)

    

if __name__ == '__main__':
    get_graph()