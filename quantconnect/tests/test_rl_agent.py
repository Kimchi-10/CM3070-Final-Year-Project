"""
Unit tests for the RL Adaptive Veto Agent.
Tests Q-learning mechanics, state discretization, training, and save/load.
"""

import pytest
import numpy as np
import tempfile
import os

from dashboard.components.rl_agent import (
    VetoQLearningAgent,
    simulate_sentiment,
    load_trained_agent,
    QTABLE_PATH,
)


# ---------------------------------------------------------------------------
# Agent initialization and state discretization
# ---------------------------------------------------------------------------
class TestAgentBasics:
    @pytest.fixture
    def agent(self):
        return VetoQLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.0)

    def test_init_defaults(self, agent):
        assert agent.alpha == 0.1
        assert agent.gamma == 0.95
        assert agent.epsilon == 0.0

    def test_state_space_buy(self, agent):
        """BUY signal should produce direction code 0."""
        state = agent.get_state("BUY", 0.8, 30, 1.0, 1.0)
        assert state[0] == 0  # BUY = 0

    def test_state_space_sell(self, agent):
        """SELL signal should produce direction code 1."""
        state = agent.get_state("SELL", 0.8, 70, 1.0, 1.0)
        assert state[0] == 1  # SELL = 1

    def test_sentiment_discretization(self, agent):
        """Sentiment scores should map to 5 buckets."""
        assert agent._discretize_sentiment(0.1) == 0
        assert agent._discretize_sentiment(0.3) == 1
        assert agent._discretize_sentiment(0.5) == 2
        assert agent._discretize_sentiment(0.7) == 3
        assert agent._discretize_sentiment(0.9) == 4

    def test_rsi_discretization(self, agent):
        """RSI values should map to 5 buckets."""
        assert agent._discretize_rsi(10) == 0
        assert agent._discretize_rsi(30) == 1
        assert agent._discretize_rsi(50) == 2
        assert agent._discretize_rsi(70) == 3
        assert agent._discretize_rsi(90) == 4

    def test_volatility_discretization(self, agent):
        """Volatility relative to median maps to 3 buckets."""
        assert agent._discretize_volatility(0.5, 1.0) == 0  # Low
        assert agent._discretize_volatility(1.0, 1.0) == 1  # Medium
        assert agent._discretize_volatility(2.0, 1.0) == 2  # High

    def test_state_is_tuple(self, agent):
        state = agent.get_state("BUY", 0.7, 25, 1.5, 1.0)
        assert isinstance(state, tuple)
        assert len(state) == 4


# ---------------------------------------------------------------------------
# Q-Learning mechanics
# ---------------------------------------------------------------------------
class TestQLearning:
    def test_q_update(self):
        """Q-value should update after training step."""
        agent = VetoQLearningAgent(alpha=0.5, gamma=0.0, epsilon=0.0)
        state = (0, 3, 1, 1)
        next_state = (0, 3, 2, 1)

        assert agent.q_table[state][0] == 0.0
        assert agent.q_table[state][1] == 0.0

        agent.update(state, 1, 1.0, next_state)

        assert agent.q_table[state][1] > 0

    def test_greedy_action_selection(self):
        """Agent should select action with highest Q-value when not exploring."""
        agent = VetoQLearningAgent(epsilon=0.0)
        state = (0, 3, 1, 1)

        agent.q_table[state] = np.array([-1.0, 2.0])
        assert agent.choose_action(state, training=False) == 1

        agent.q_table[state] = np.array([3.0, 1.0])
        assert agent.choose_action(state, training=False) == 0

    def test_exploration(self):
        """With epsilon=1.0, actions should be random."""
        agent = VetoQLearningAgent(epsilon=1.0)
        state = (0, 3, 1, 1)
        agent.q_table[state] = np.array([10.0, 0.0])

        actions = [agent.choose_action(state, training=True) for _ in range(100)]
        assert 1 in actions  

    def test_learning_improves(self):
        """Agent should learn to prefer correct actions after training."""
        agent = VetoQLearningAgent(alpha=0.3, gamma=0.0, epsilon=0.1)
        state = (0, 3, 1, 1)
        next_state = state

        # Train: ALLOW is always rewarded, VETO is always punished
        for _ in range(50):
            agent.update(state, 1, 1.0, next_state)   # ALLOW → +1
            agent.update(state, 0, -1.0, next_state)   # VETO → -1

        # Agent should prefer ALLOW
        assert agent.q_table[state][1] > agent.q_table[state][0]
        assert agent.choose_action(state, training=False) == 1


# ---------------------------------------------------------------------------
# Veto decision logic
# ---------------------------------------------------------------------------
class TestVetoDecisions:
    @pytest.fixture
    def trained_agent(self):
        """Create an agent with manually set Q-values for predictable behavior."""
        agent = VetoQLearningAgent(epsilon=0.0)
        agent.q_table[(0, 4, 0, 1)] = np.array([2.0, -1.0]) 
        agent.q_table[(0, 2, 2, 1)] = np.array([-1.0, 2.0]) 
        return agent

    def test_hold_never_vetoed(self, trained_agent):
        assert trained_agent.should_veto("HOLD", 0.9, "negative", 20, 1.0, 1.0) is False

    def test_no_contradiction_no_veto(self, trained_agent):
        """BUY + positive sentiment → no contradiction → no veto."""
        assert trained_agent.should_veto("BUY", 0.9, "positive", 20, 1.0, 1.0) is False

    def test_veto_on_strong_contradiction(self, trained_agent):
        """BUY + strong negative + oversold RSI → agent learned to VETO."""
        result = trained_agent.should_veto("BUY", 0.9, "negative", 10, 1.0, 1.0)
        assert result is True

    def test_allow_on_moderate_contradiction(self, trained_agent):
        """BUY + moderate negative → agent learned to ALLOW."""
        result = trained_agent.should_veto("BUY", 0.5, "negative", 50, 1.0, 1.0)
        assert result is False


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------
class TestSaveLoad:
    def test_save_and_load(self):
        """Q-table should survive save/load cycle."""
        agent = VetoQLearningAgent(alpha=0.2, gamma=0.9, epsilon=0.0)
        state = (0, 3, 1, 1)
        agent.q_table[state] = np.array([1.5, -0.5])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            agent.save(path)

            loaded = VetoQLearningAgent()
            assert loaded.load(path) is True
            assert loaded.alpha == 0.2
            assert loaded.gamma == 0.9
            np.testing.assert_array_almost_equal(
                loaded.q_table[state], np.array([1.5, -0.5])
            )
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        """Loading from nonexistent file should return False."""
        agent = VetoQLearningAgent()
        assert agent.load("/nonexistent/path.json") is False

    def test_load_pretrained_qtable(self):
        """Should be able to load the pre-trained Q-table if it exists."""
        agent = load_trained_agent()
        if QTABLE_PATH.exists():
            assert agent is not None
            assert len(agent.q_table) > 0
        else:
            assert agent is None


# ---------------------------------------------------------------------------
# Sentiment simulation
# ---------------------------------------------------------------------------
class TestSentimentSimulation:
    def test_neutral_on_flat(self):
        """Flat prices should produce neutral sentiment."""
        prices = pd.Series([100.0] * 20)
        score, label = simulate_sentiment(prices, 15)
        assert label == "neutral"

    def test_positive_on_rally(self):
        """Strong upward move should produce positive sentiment."""
        prices = pd.Series([100 + i * 2 for i in range(20)])
        score, label = simulate_sentiment(prices, 15)
        assert label == "positive"
        assert score > 0.5

    def test_negative_on_crash(self):
        """Strong downward move should produce negative sentiment."""
        prices = pd.Series([200 - i * 2 for i in range(20)])
        score, label = simulate_sentiment(prices, 15)
        assert label == "negative"
        assert score > 0.5

    def test_early_index_neutral(self):
        """Index < 5 should return neutral (insufficient history)."""
        prices = pd.Series([100.0] * 10)
        score, label = simulate_sentiment(prices, 3)
        assert label == "neutral"


import pandas as pd
