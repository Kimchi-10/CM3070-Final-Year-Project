"""
FYP Dashboard — Reinforcement Learning Adaptive Veto Agent
============================================================
A Q-Learning agent that learns the optimal veto threshold by training
on historical signal + sentiment + price outcome data.

Instead of hardcoding 0.6 sentiment threshold, the RL agent learns
WHEN to veto based on:
  - State: (signal_direction, sentiment_score_bucket, rsi_bucket, volatility_bucket)
  - Action: VETO (0) or ALLOW (1)
  - Reward: based on simulated trade outcome (next-bar return)

The agent is trained offline on historical data and the learned Q-table
is saved to disk. During live dashboard use, the agent loads the Q-table
and makes veto decisions based on learned policy.

Student: Ng Chang Yan (23036719)
Module: CM3070 — Final Year Project
"""

import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------
class VetoQLearningAgent:
    """
    Tabular Q-Learning agent for adaptive signal veto decisions.

    State space (discretized):
        - signal_direction: 0=BUY, 1=SELL
        - sentiment_score: 5 buckets [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
        - rsi_bucket: 5 buckets [0-20, 20-40, 40-60, 60-80, 80-100]
        - volatility_bucket: 3 buckets [low, medium, high] based on ATR percentile

    Action space:
        - 0: VETO the signal (don't trade)
        - 1: ALLOW the signal (trade)

    Reward:
        - If ALLOW + trade was profitable: +1.0
        - If ALLOW + trade was unprofitable: -1.0
        - If VETO + trade would have been profitable: -0.5 (missed opportunity)
        - If VETO + trade would have been unprofitable: +0.5 (avoided loss)
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: np.zeros(2))  # {state: [Q_veto, Q_allow]}
        self.training_history = []

    def _discretize_sentiment(self, score: float) -> int:
        """Bucket sentiment score into 5 levels."""
        if score < 0.2:
            return 0
        elif score < 0.4:
            return 1
        elif score < 0.6:
            return 2
        elif score < 0.8:
            return 3
        return 4

    def _discretize_rsi(self, rsi: float) -> int:
        """Bucket RSI into 5 levels."""
        if rsi < 20:
            return 0
        elif rsi < 40:
            return 1
        elif rsi < 60:
            return 2
        elif rsi < 80:
            return 3
        return 4

    def _discretize_volatility(self, atr: float, atr_median: float) -> int:
        """Bucket volatility into 3 levels based on ATR relative to median."""
        if atr_median == 0:
            return 1
        ratio = atr / atr_median
        if ratio < 0.7:
            return 0  # Low volatility
        elif ratio < 1.3:
            return 1  # Medium
        return 2      # High

    def get_state(self, signal_dir: str, sentiment_score: float,
                  rsi: float, atr: float, atr_median: float) -> tuple:
        """Convert raw features into discretized state tuple."""
        dir_code = 0 if signal_dir == "BUY" else 1
        sent_bucket = self._discretize_sentiment(sentiment_score)
        rsi_bucket = self._discretize_rsi(rsi)
        vol_bucket = self._discretize_volatility(atr, atr_median)
        return (dir_code, sent_bucket, rsi_bucket, vol_bucket)

    def choose_action(self, state: tuple, training: bool = False) -> int:
        """Choose action: 0=VETO, 1=ALLOW. Epsilon-greedy during training."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(2)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        """Q-Learning update rule."""
        best_next = np.max(self.q_table[next_state])
        current = self.q_table[state][action]
        self.q_table[state][action] = current + self.alpha * (
            reward + self.gamma * best_next - current
        )

    def should_veto(self, signal: str, sentiment_score: float,
                    sentiment_label: str, rsi: float,
                    atr: float, atr_median: float) -> bool:
        """
        RL-based veto decision (replaces hardcoded threshold).

        Returns True if the agent recommends vetoing the signal.
        """
        if signal == "HOLD":
            return False

        # Check if sentiment contradicts signal direction
        if signal == "BUY" and sentiment_label != "negative":
            return False  # No contradiction → no veto consideration
        if signal == "SELL" and sentiment_label != "positive":
            return False

        state = self.get_state(signal, sentiment_score, rsi, atr, atr_median)
        action = self.choose_action(state, training=False)
        return action == 0  # 0 = VETO

    def save(self, path: str):
        """Save Q-table to JSON file."""
        # Convert defaultdict to regular dict with string keys
        q_dict = {}
        for state, values in self.q_table.items():
            key = str(state)
            q_dict[key] = values.tolist()

        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q_table": q_dict,
            "training_episodes": len(self.training_history),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> bool:
        """Load Q-table from JSON file. Returns True if loaded successfully."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self.alpha = data.get("alpha", self.alpha)
            self.gamma = data.get("gamma", self.gamma)
            self.epsilon = data.get("epsilon", self.epsilon)
            for key_str, values in data.get("q_table", {}).items():
                state = ast.literal_eval(key_str) 
                self.q_table[state] = np.array(values)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Training on Historical Data
# ---------------------------------------------------------------------------
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR indicator."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def simulate_sentiment(close_prices: pd.Series, idx: int) -> tuple:
    """
    Simulate sentiment score and label based on recent price momentum.

    Uses 5-day momentum as a proxy for news sentiment:
    - Strong recent gains → positive sentiment
    - Strong recent losses → negative sentiment
    - Flat → neutral

    This is a reasonable proxy because news sentiment tends to follow
    recent price action (momentum-driven headlines).
    """
    if idx < 5:
        return 0.5, "neutral"

    # 5-day return as sentiment proxy
    ret_5d = (close_prices.iloc[idx] - close_prices.iloc[idx - 5]) / close_prices.iloc[idx - 5]

    if ret_5d > 0.02:
        score = min(0.5 + ret_5d * 10, 0.95)
        return score, "positive"
    elif ret_5d < -0.02:
        score = min(0.5 + abs(ret_5d) * 10, 0.95)
        return score, "negative"
    else:
        return 0.5, "neutral"


def train_agent(ticker: str = "SPY", period: str = "5y",
                episodes: int = 50, verbose: bool = True) -> VetoQLearningAgent:
    """
    Train the RL veto agent on historical data.

    For each bar in the historical data:
    1. Compute RSI → determine if signal would be BUY or SELL
    2. Simulate sentiment score from recent momentum
    3. Agent decides: VETO or ALLOW
    4. Check next-bar return to compute reward
    5. Update Q-table

    Args:
        ticker: Asset to train on (e.g., "SPY", "QQQ")
        period: Historical data period (e.g., "2y" for 2 years)
        episodes: Number of training passes over the data
        verbose: Print training progress

    Returns:
        Trained VetoQLearningAgent
    """
    import yfinance as yf

    if verbose:
        print(f"Fetching {period} of daily data for {ticker}...")

    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if len(df) < 50:
        print(f"Insufficient data for {ticker} ({len(df)} bars)")
        return VetoQLearningAgent()

    close = df["Close"]
    rsi = _compute_rsi(close, 14)
    atr = _compute_atr(df, 14)
    atr_median = float(atr.median())

    # Forward returns
    df["fwd_return"] = close.shift(-1) / close - 1 

    agent = VetoQLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.3)

    total_rewards = []

    for episode in range(episodes):
        episode_reward = 0
        decisions = {"veto_correct": 0, "veto_wrong": 0, "allow_correct": 0, "allow_wrong": 0}

        # Decay epsilon over episodes (explore less as we learn)
        agent.epsilon = max(0.05, 0.3 * (1 - episode / episodes))

        for i in range(20, len(df) - 1):  # Need 20 bars warmup for indicators
            rsi_val = rsi.iloc[i]
            atr_val = atr.iloc[i]
            fwd_ret = df["fwd_return"].iloc[i]

            if pd.isna(rsi_val) or pd.isna(atr_val) or pd.isna(fwd_ret):
                continue

            # Determine signal direction from RSI
            if rsi_val < 35:
                signal_dir = "BUY"
            elif rsi_val > 65:
                signal_dir = "SELL"
            else:
                continue  # No signal when RSI is neutral

            # Simulate sentiment
            sent_score, sent_label = simulate_sentiment(close, i)

            # Check if there's even a contradiction (if not, skip — no veto possible)
            if signal_dir == "BUY" and sent_label != "negative":
                continue
            if signal_dir == "SELL" and sent_label != "positive":
                continue

            # This is a veto candidate — agent must decide
            state = agent.get_state(signal_dir, sent_score, rsi_val, atr_val, atr_median)

            # Get next state (for Q-learning update)
            next_rsi = rsi.iloc[min(i + 1, len(rsi) - 1)]
            next_atr = atr.iloc[min(i + 1, len(atr) - 1)]
            next_sent_score, next_sent_label = simulate_sentiment(close, min(i + 1, len(close) - 1))
            next_state = agent.get_state(signal_dir, next_sent_score,
                                         next_rsi if not pd.isna(next_rsi) else 50,
                                         next_atr if not pd.isna(next_atr) else atr_median,
                                         atr_median)

            action = agent.choose_action(state, training=True)

            # Compute reward based on trade outcome
            trade_profitable = (signal_dir == "BUY" and fwd_ret > 0) or \
                             (signal_dir == "SELL" and fwd_ret < 0)

            if action == 1:  # ALLOW (don't veto)
                reward = 1.0 if trade_profitable else -1.0
                if trade_profitable:
                    decisions["allow_correct"] += 1
                else:
                    decisions["allow_wrong"] += 1
            else:  # VETO
                reward = 0.5 if not trade_profitable else -0.5
                if not trade_profitable:
                    decisions["veto_correct"] += 1
                else:
                    decisions["veto_wrong"] += 1

            agent.update(state, action, reward, next_state)
            episode_reward += reward

        total_rewards.append(episode_reward)
        agent.training_history.append({
            "episode": episode + 1,
            "reward": episode_reward,
            "epsilon": agent.epsilon,
            "decisions": decisions,
        })

        if verbose:
            correct = decisions["veto_correct"] + decisions["allow_correct"]
            total = sum(decisions.values())
            accuracy = correct / total * 100 if total > 0 else 0
            print(f"  Episode {episode + 1}/{episodes} | "
                  f"Reward: {episode_reward:+.1f} | "
                  f"Accuracy: {accuracy:.0f}% | "
                  f"Veto correct: {decisions['veto_correct']} | "
                  f"Allow correct: {decisions['allow_correct']} | "
                  f"Epsilon: {agent.epsilon:.2f}")

    if verbose and total_rewards:
        print(f"\n  Training complete. Final avg reward: {np.mean(total_rewards[-3:]):+.1f}")
        print(f"  Q-table states learned: {len(agent.q_table)}")

    # Set epsilon to 0 for inference (no exploration)
    agent.epsilon = 0.0
    return agent


# ---------------------------------------------------------------------------
# Convenience: Train on multiple assets and save
# ---------------------------------------------------------------------------
QTABLE_PATH = Path(__file__).resolve().parent.parent / "strategies" / "rl_qtable.json"


def train_and_save(tickers: list = None, episodes: int = 50, verbose: bool = True):
    """Train RL agent across multiple assets and save Q-table."""
    if tickers is None:
        tickers = ["SPY", "QQQ", "AAPL", "TSLA"]

    combined_agent = VetoQLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.3)

    for ticker in tickers:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training on {ticker}")
            print(f"{'='*50}")

        agent = train_agent(ticker, period="5y", episodes=episodes, verbose=verbose)

        # Merge Q-tables 
        for state, values in agent.q_table.items():
            if state in combined_agent.q_table:
                combined_agent.q_table[state] = (
                    combined_agent.q_table[state] + values
                ) / 2
            else:
                combined_agent.q_table[state] = values.copy()

    combined_agent.epsilon = 0.0
    combined_agent.save(str(QTABLE_PATH))

    if verbose:
        print(f"\n{'='*50}")
        print(f"Combined Q-table saved to {QTABLE_PATH}")
        print(f"Total states learned: {len(combined_agent.q_table)}")
        print(f"{'='*50}")

    return combined_agent


def load_trained_agent() -> VetoQLearningAgent | None:
    """Load a pre-trained RL agent. Returns None if no trained model exists."""
    agent = VetoQLearningAgent()
    if agent.load(str(QTABLE_PATH)):
        return agent
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL Veto Agent")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "TSLA"],
                        help="Tickers to train on")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Training episodes per ticker")
    args = parser.parse_args()

    train_and_save(tickers=args.tickers, episodes=args.episodes)
