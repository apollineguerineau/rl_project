import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import torch

#from app import *

from modules.data_loader import DataLoader
from modules.single_asset_env import SingleAssetEnv
from modules.q_network import Q_network
from modules.memory import Memory
from modules.trading_agent import TradingAgent
from modules.functions import train_agent, test_agent

st.title(f"Test with test dataset (2023-05-01 to 2024-03-30)")

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# create a Streamlit selectbox to choose the variable
asset_name_test = st.sidebar.selectbox(
    'Choisir les actifs:',
    options=["BTC-USD", "ETH-USD", "BNB-USD"],
    index=0,
    key="asset_selectbox"  

)
assets = ["BTC-USD", "ETH-USD", "BNB-USD"]

## Load Training and Testing Dataset
start_date = '2020-01-01'
end_date = '2024-03-30'
date_split = '2023-05-01'

dataloader = DataLoader(start_date, end_date, '1d', date_split)


trains = []
tests = []

for asset in assets:
    train, test = dataloader.load(asset)

    trains.append(train)
    tests.append(test)

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

# Environment and Agent Initiation
train_envs = [SingleAssetEnv(train) for train in trains]
test_envs = [SingleAssetEnv(test) for test in tests]

agent = TradingAgent(states_dim, num_actions, assets, 
                     batch_size=batch_size,
                     memory_size=memory_size,
                     learning_rate=learning_rate,
                     tau=tau,
                     gamma=gamma,
                     learning_freq=learning_freq,
                     device=device,
                     seed=seed)
# Load model
# Check if the trained model exists
# Delete the files online.pt and target.pt if you want to retrain the model
model_path = "weights/trained_agent_model_"+asset_name_test+".pth"


if os.path.exists(model_path) :
    ## Agent
    
    model_path = "weights/trained_agent_model_"+asset_name_test+".pth"
    # Load the state dict
    state_dict = torch.load(model_path)

    # Set the loaded state dict to the Q network of the corresponding asset
    agent.qnets[agent.map_assets[asset]].load_state_dict(state_dict)

    final_running_balance_dict = test_agent(assets, agent, test_envs)

else:  # TRAINING MODEL IN CASE IT IS NOT TRAINED YET

    train_agent(assets, agent, train_envs, num_epochs)
    final_running_balance_dict = test_agent(assets, agent, test_envs)
    st.sidebar.success("Model trained successfully!")



st.title("Graph")
st.write('\n\n')
st.subheader("Résultat de l'aglgorithme : ")
st.write('\n')
# Plotting test set
# Plot testset
asset_dict = {name: element for name, element in zip(assets, range(len(assets)))}
test_env = test_envs[asset_dict[asset_name_test]]


st.subheader('Statistics')
st.write('Initial Capital: ',100000)
st.write('Agent Net-worth for Asset ',asset_name_test, ": ",final_running_balance_dict[asset_name_test])
st.write('Profit: ',final_running_balance_dict[asset_name_test] - 100000)
st.write('\n')

fig_test, candle_fig_test = test_env.render(asset_name_test)
st.plotly_chart(candle_fig_test, use_container_width=True)
st.write('\n\n')
st.pyplot(fig_test)