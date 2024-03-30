
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