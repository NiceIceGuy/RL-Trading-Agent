import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

# Use a non-interactive backend so plots can be saved to files
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import random
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# Global configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Market data settings
TICKER = "TSLA"
PERIOD = "60d"
INTERVAL = "5m"

# Training setup
EPISODE_DAYS = 5
TOTAL_TIMESTEPS = 100_000


# Download historical price data from Yahoo Finance
df = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=True
).dropna().reset_index()

# Store calendar date separately so episodes can be sampled by day
df["Date"] = df["Datetime"].dt.date


# Technical indicators

# Short-term and medium-term exponential moving averages
df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

# MACD momentum indicator
df["MACD"] = df["EMA_12"] - df["EMA_26"]

# Relative Strength Index (RSI)
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-10)
df["RSI"] = 100 - (100 / (1 + rs))

# Remove rows created by rolling window calculations
df.dropna(inplace=True)


class MarginTradingEnv(gym.Env):
    """
    Custom margin trading environment.

    Actions:
      0 = hold
      1 = buy as much as possible
      2 = sell all held shares

    Observation:
      OHLCV data plus RSI, EMA, and MACD

    Reward:
      Change in net worth between steps
    """

    metadata = {"render_modes": []}

    def __init__(self, df, interest_rate=0.08, leverage=2.0, step_minutes=5, episode_days=5, max_buy_shares=1000):
        super().__init__()

        # Full dataset
        self.df = df.reset_index(drop=True)

        # Financial parameters
        self.interest_rate = interest_rate
        self.leverage = leverage
        self.step_minutes = step_minutes
        self.episode_days = episode_days
        self.max_buy_shares = max_buy_shares

        # Three discrete actions: hold, buy, sell
        self.action_space = gym.spaces.Discrete(3)

        # Observation vector (8 numerical features)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )

        # These are initialized when an episode starts
        self.episode_df = None
        self.current_step = 0

    def _get_row(self, idx: int) -> pd.Series:
        # Clamp the index
        idx = int(np.clip(idx, 0, len(self.episode_df) - 1))
        return self.episode_df.iloc[idx]

    def _next_observation(self) -> np.ndarray:
        # Build the observation vector for the current timestep
        row = self._get_row(self.current_step)
        return np.array([
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"]),
            float(row["RSI"]),
            float(row["EMA_9"]),
            float(row["MACD"]),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Start a new episode by selecting a random multi-day window
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        unique_dates = self.df["Date"].unique()
        if len(unique_dates) < self.episode_days:
            raise ValueError("Not enough unique dates for the chosen episode_days.")

        # Pick a random starting day
        start_day_idx = random.randint(0, len(unique_dates) - self.episode_days)
        start_date = unique_dates[start_day_idx]
        end_date = unique_dates[start_day_idx + self.episode_days - 1]

        # Slice the data to just the episode window
        mask = (self.df["Date"] >= start_date) & (self.df["Date"] <= end_date)
        self.episode_df = self.df[mask].reset_index(drop=True)

        # Reset portfolio state
        self.current_step = 0
        self.initial_cash = 100_000.0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.borrowed = 0.0
        self.net_worth = self.cash
        self.trades = []

        return self._next_observation(), {}

    def step(self, action):
        # Get the current market price
        row = self._get_row(self.current_step)
        price = max(float(row["Close"]), 0.01)

        # Interest charged on borrowed capital at each timestep
        minutes_per_year = 252 * 390
        step_fraction = self.step_minutes / minutes_per_year
        interest_cost = self.borrowed * self.interest_rate * step_fraction

        prev_worth = self.net_worth

        # Buy action: deploy maximum allowed capital using leverage
        if action == 1:
            buying_power = self.cash * self.leverage
            max_buy = int(min(buying_power // price, self.max_buy_shares))
            if max_buy > 0:
                required_funds = max_buy * price
                margin_needed = max(required_funds - self.cash, 0.0)

                # Borrow only what leverage allows
                max_borrow_allowed = self.cash * (self.leverage - 1)
                borrow = min(margin_needed, max_borrow_allowed)

                self.borrowed += borrow
                self.cash = max(self.cash - (required_funds - borrow), 0.0)
                self.shares_held += max_buy
                self.trades.append((self.current_step, price, "buy"))

        # Sell action: close all open positions
        elif action == 2:
            if self.shares_held > 0:
                sale_amount = self.shares_held * price
                self.cash += sale_amount
                self.shares_held = 0
                self.trades.append((self.current_step, price, "sell"))

        # Apply interest cost to remaining cash
        self.cash = max(self.cash - interest_cost, 0.0)

        # Move to the next timestep
        self.current_step += 1

        # Check whether the episode has ended
        done = self.current_step >= len(self.episode_df)

        # Update net worth using the last valid market price
        last_row = self._get_row(self.current_step)
        last_price = max(float(last_row["Close"]), 0.01)
        self.net_worth = self.cash + self.shares_held * last_price - self.borrowed

        # Reward is the change in portfolio value since the previous step
        reward = self.net_worth - prev_worth

        # Observation is clamped, so this is safe even at the end of the episode
        obs = self._next_observation()
        return obs, reward, done, False, {}


class RewardLoggerCallback(BaseCallback):
    # Placeholder callback for potential future logging
    def _on_step(self) -> bool:
        return True


# Factory function required by DummyVecEnv
def make_env():
    return MarginTradingEnv(df, episode_days=EPISODE_DAYS)


# Vectorized environment wrapper for Stable-Baselines
vec_env = DummyVecEnv([make_env])

# DQN agent configuration
model = DQN(
    "MlpPolicy",
    vec_env,
    verbose=1,
    seed=SEED,
    learning_starts=1000,
    buffer_size=50_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
)

# Train the agent
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=RewardLoggerCallback())


def run_episode(policy_fn, seed=None):
    # Run a single evaluation episode using a given policy
    env = MarginTradingEnv(df, episode_days=EPISODE_DAYS)
    obs, _ = env.reset(seed=seed)

    portfolio = [env.net_worth]

    for _ in range(len(env.episode_df) + 5):
        action = policy_fn(env, obs)
        obs, reward, done, truncated, _ = env.step(action)
        portfolio.append(env.net_worth)
        if done or truncated:
            break

    return env, portfolio


def policy_dqn(env, obs):
    # Deterministic policy from the trained DQN
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def policy_random(env, obs):
    # Random baseline
    return int(env.action_space.sample())


def policy_buy_and_hold(env, obs):
    # Buy once at the beginning and then do nothing
    return 1 if env.current_step == 0 else 0


def summarize(portfolios):
    # Compute basic statistics over final portfolio values
    finals = [p[-1] for p in portfolios]
    return {
        "episodes": len(finals),
        "mean_final": float(np.mean(finals)),
        "std_final": float(np.std(finals)),
        "mean_profit": float(np.mean([f - 100_000.0 for f in finals])),
    }


# Evaluate against baselines over multiple random episodes
N_EVAL = 20
dqn_ports, rnd_ports, bh_ports = [], [], []

for i in range(N_EVAL):
    seed_i = SEED + i
    _, p1 = run_episode(policy_dqn, seed=seed_i)
    _, p2 = run_episode(policy_random, seed=seed_i)
    _, p3 = run_episode(policy_buy_and_hold, seed=seed_i)

    dqn_ports.append(p1)
    rnd_ports.append(p2)
    bh_ports.append(p3)

print("\n=== Evaluation over multiple random 5-day episodes ===")
print("DQN:", summarize(dqn_ports))
print("Random:", summarize(rnd_ports))
print("Buy&Hold:", summarize(bh_ports))


# Plot one example DQN episode for visualization
example_env, example_portfolio = run_episode(policy_dqn, seed=SEED)
prices = example_env.episode_df["Close"].values.tolist()

plt.figure(figsize=(14, 6))
plt.plot(prices, label=f"{TICKER} Price", alpha=0.6)
plt.plot(example_portfolio[:len(prices)], label="Portfolio Value", linestyle="--")

for step, price, act in example_env.trades:
    if act == "buy":
        plt.plot(step, price, marker="^", color="green", label="Buy")
    elif act == "sell":
        plt.plot(step, price, marker="v", color="red", label="Sell")

plt.title(f"DQN Trading Example ({TICKER}, {INTERVAL} candles, {EPISODE_DAYS}-day episode)")
plt.xlabel("Time Step")
plt.ylabel("Price / Portfolio Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trading_results.png")

