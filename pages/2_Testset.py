import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
from app import *
import func 
from modules.data_loader import DataLoader
from modules.single_asset_env import SingleAssetEnv
from modules.q_network import Q_network
from modules.memory import Memory
from modules.trading_agent import TradingAgent
import func
import torch
import numpy as np

st.title("Test with test dataset (2023-05-01 to now")
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
agent = TradingAgent(STATES_DIM, NUM_ACTIONS, assets, seed=SEED)

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

    final_running_balance_dict = func.test_agent(assets, test_envs, agent)

else:  # TRAINING MODEL IN CASE IT IS NOT TRAINED YET

    func.train_agent(assets, train_envs, agent, NUM_EPOCHS)
    final_running_balance_dict = func.test_agent(assets, test_envs, agent)
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