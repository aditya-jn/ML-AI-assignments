import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load rewards data
def load_rewards(mode):
    try:
        rewards = np.load(f'rl_trader_rewards/{mode}.npy')
        return rewards
    except FileNotFoundError:
        st.error(f"No rewards data found for mode: {mode}. Make sure to run the training first.")
        return None

# Main Streamlit app
def main():
    st.title("Reinforcement Learning Trader Visualization")

    # Select mode
    mode = st.selectbox("Select mode:", ("train", "test"))

    # Load the rewards data
    rewards = load_rewards(mode)
    
    if rewards is not None:
        st.subheader("Rewards Statistics")
        st.write(f"Average Reward: {rewards.mean():.2f}")
        st.write(f"Minimum Reward: {rewards.min():.2f}")
        st.write(f"Maximum Reward: {rewards.max():.2f}")

        # Plotting rewards
        if mode == "train":
            st.subheader("Training Progress")
            plt.figure(figsize=(10, 5))
            plt.plot(rewards)
            plt.title("Training Rewards Over Episodes")
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            st.pyplot(plt)
        else:
            st.subheader("Test Rewards Distribution")
            plt.figure(figsize=(10, 5))
            plt.hist(rewards, bins=20, edgecolor='black')
            plt.title("Distribution of Test Rewards")
            plt.xlabel("Rewards")
            plt.ylabel("Frequency")
            st.pyplot(plt)

if __name__ == "__main__":
    main()
