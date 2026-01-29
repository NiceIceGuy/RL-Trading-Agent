# RL Margin Trading Agent

A reinforcement learning trading agent trained using DQN in a custom Gymnasium environment.
The environment models leverage, borrowing interest, and multi-day trading episodes using real TSLA 5-minute data.

The project focuses on environment design, evaluation, and reproducible experiments rather than profitability.

## Features
- Custom margin trading environment with leverage and interest costs
- Technical indicators (RSI, EMA, MACD) as part of the observation space
- Evaluation against buy-and-hold and random baselines
- Reproducible training using fixed random seeds

## Notes
Some strategy-specific tuning and experimental configs are intentionally omitted.
