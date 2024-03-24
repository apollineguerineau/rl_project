import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
from model import *
from app import *


st.title("Test with new data (2024-01-01 to 2024-03-01")
if "theme" not in st.session_state:
    st.session_state.theme = "light"
# create a Streamlit selectbox to choose the variable
asset = st.sidebar.selectbox(
    'Choisir les actifs:',
    options=["ETH-USD", "BNB-USD"],
    index=0
)

# Environment and Agent Initiation

## Cryptocurrency Tickers
asset_codes = ["ETH-USD", "BNB-USD" ]
# Load trained model state dictionaries
online_state_dict = torch.load("online.pt")
target_state_dict = torch.load("target.pt")

# Create an instance of DQNAgent and initialize it with the loaded state dictionaries
memory = ReplayMemory()
agent = DQNAgent(actor_net=DuellingDQN, memory=memory)
agent.actor_online.load_state_dict(online_state_dict)
agent.actor_target.load_state_dict(target_state_dict)


test_assets = [DataGetter(a, start_date="2024-01-01", end_date="2024-03-01", freq="1d") for a in asset_codes]
test_envs = [SingleAssetTradingEnvironment(a) for a in test_assets]


test_agent(test_envs, agent, N_EPISODES)

st.title("Graph")
st.write('\n\n')
st.subheader("Résultat de l'aglgorithme : ")
st.write('\n')
# Plotting test set
if asset == "ETH-USD":
    test_env = test_envs[0]
elif asset == "BNB-USD":
    test_env = test_envs[1]

st.subheader('Statistics')
st.write('Initial Capital: ',CAPITAL)
st.write('Agent Net-worth for this Asset: ',test_env.store['running_capital'][-1])
st.write('\n')

fig_test, candle_fig_test = test_env.render()
st.plotly_chart(candle_fig_test, use_container_width=True)
st.write('\n\n')
st.pyplot(fig_test)