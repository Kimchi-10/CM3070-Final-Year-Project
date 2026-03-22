from AlgorithmImports import *
from datetime import datetime, timedelta
from collections import deque


class ForexStrategyEngine(QCAlgorithm):
    """
    FYP Forex Trading Engine — 5 Strategy Families (Mean Reversion + Trend)
    ======================================================================
    Family 1: RSI Mean Reversion
    Family 2: Bollinger Band Bounce
    Family 3: EMA Crossover (trend)
    Family 4: Stochastic Mean Reversion
    Family 5: EMA Trend + ADX (trend)

    All parameters are received via self.get_parameter() from the GA runner.
    Handles synchronous fills correctly (LEAN backtest vs live).
    """

    # ──────────────────────────────────────────────
    #  INITIALIZATION
    # ──────────────────────────────────────────────

    def initialize(self):

        # --- Helper functions for reading GA parameters ---
        def param_str(name, default):
            val = self.get_parameter(name)
            return val if val not in (None, "") else default

        def param_int(name, default):
            return int(param_str(name, str(default)))

        def param_float(name, default):
            return float(param_str(name, str(default)))

        # --- Mode & Date Ranges ---
        self.mode = param_str("mode", "test").lower()
        date_ranges = {
            "train":      (param_str("train_start", "2020-01-01"), param_str("train_end", "2022-12-31")),
            "validation": (param_str("valid_start", "2023-01-01"), param_str("valid_end", "2023-12-31")),
            "test":       (param_str("test_start", "2024-01-01"),  param_str("test_end", "2025-12-31")),
        }
        start_str, end_str = date_ranges.get(self.mode, date_ranges["test"])
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
        self.set_start_date(start_dt.year, start_dt.month, start_dt.day)
        self.set_end_date(end_dt.year, end_dt.month, end_dt.day)
        self.set_time_zone("UTC")

        # --- Capital & Risk ---
        self.starting_cash = param_float("starting_cash", 10000.0)
        self.set_cash(self.starting_cash)
        self.risk_per_trade_pct = param_float("risk_per_trade_pct", 0.5)
        self.max_trades_per_day = max(param_int("max_trades_per_day", 12), 1)

        # --- Asset Setup ---
        pairs_str = param_str("pairs", "EURUSD")
        pairs = [p.strip().upper() for p in pairs_str.split(",") if p.strip()]
        if not pairs:
            pairs = ["EURUSD"]

        # --- Strategy Family ---
        self.strategy_family = param_int("strategy_family", 1)
        self.FAMILY_NAMES = {
            1: "RSI Mean Reversion",
            2: "Bollinger Band Bounce",
            3: "EMA Crossover",
            4: "Stochastic Mean Reversion",
            5: "EMA Trend + ADX",
        }

        # --- Bar Timeframe ---
        self.bar_minutes = max(param_int("bar_minutes", 5), 1)

        # --- Session Filter (Day Trade) ---
        self.use_session_filter = param_int("use_session_filter", 1) == 1
        self.session_start_hour = param_int("session_start_hour", 7)
        self.session_end_hour = param_int("session_end_hour", 16)
        self.last_entry_hour = param_int("last_entry_hour", 15)
        self.last_entry_minute = param_int("last_entry_minute", 0)
        self.use_force_flat = param_int("use_force_flat", 1) == 1
        self.force_flat_hour = param_int("force_flat_hour", 15)
        self.force_flat_minute = param_int("force_flat_minute", 55)

        # --- Higher-Timeframe (HTF) Trend + Volatility Filter ---
        self.htf_minutes = max(param_int("htf_minutes", 60), self.bar_minutes + 1)
        self.htf_ema_period = param_int("htf_ema_period", 50)
        self.vol_filter_enabled = param_int("vol_filter_enabled", 1) == 1
        self.vol_bb_period = param_int("vol_bb_period", 20)

        # --- Shared Indicator / Trade Parameters ---
        self.atr_period = param_int("atr_period", 15)
        self.stop_atr_mult = param_float("stop_atr_mult", 2.0)
        self.reward_ratio = param_float("reward_ratio", 1.5)
        self.max_hold_bars = max(param_int("max_hold_bars", 20), 1)
        self.cooldown_bars = max(param_int("cooldown_bars", 2), 0)

        # --- Family 1: RSI Mean Reversion ---
        self.rsi_period = param_int("rsi_period", 10)
        self.rsi_oversold = param_float("rsi_oversold", 30.0)
        self.rsi_overbought = param_float("rsi_overbought", 70.0)
        self.trend_ema_period = param_int("trend_ema_period", 50)

        # --- Family 2: Bollinger Band Bounce ---
        self.bb_period = param_int("bb_period", 20)
        self.bb_std_dev = param_float("bb_std_dev", 2.0)
        self.bb_rsi_long = param_float("bb_rsi_long", 35.0)
        self.bb_rsi_short = param_float("bb_rsi_short", 65.0)

        # --- Family 4: Stochastic Mean Reversion ---
        self.stoch_k = param_int("stoch_k", 14)
        self.stoch_d = param_int("stoch_d", 3)
        self.stoch_oversold = param_float("stoch_oversold", 20.0)
        self.stoch_overbought = param_float("stoch_overbought", 80.0)

        # --- Family 5: EMA Trend + ADX ---
        self.ema_fast_period = param_int("ema_fast_period", 15)
        self.ema_slow_period = param_int("ema_slow_period", 50)
        self.adx_period = param_int("adx_period", 14)
        self.adx_min = param_float("adx_min", 25.0)

        # Ensure EMA fast < slow for Family 5
        if self.ema_fast_period >= self.ema_slow_period:
            self.ema_slow_period = self.ema_fast_period + 10

        # --- Brokerage ---
        self.set_brokerage_model(BrokerageName.OANDA_BROKERAGE, AccountType.MARGIN)

        # --- Per-Symbol State ---
        self.symbols = []
        self.latest_bar = {}
        self.prev_close = {}
        self.new_bar_ready = {}
        self.trade_day = {}
        self.trade_count_today = {}

        # Indicators
        self.rsi_ind = {}
        self.atr_ind = {}
        self.adx_ind = {}
        self.ema_fast = {}
        self.ema_slow = {}

        # Family indicators
        self.trend_ema = {}       # Family 1: trend direction EMA
        self.prev_rsi_val = {}    # Family 1: previous RSI for mean reversion
        self.bb_ind = {}          # Family 2: Bollinger Bands
        # Family 3: EMA Crossover — reuses ema_fast, ema_slow, trend_ema (no extra state)
        self.stoch_ind = {}       # Family 4: Stochastic indicator
        self.prev_stoch_k = {}    # Family 4: previous %K for crossover
        self.prev_stoch_d = {}    # Family 4: previous %D for crossover

        # Higher-timeframe indicators
        self.htf_ema_ind = {}         # HTF EMA for trend direction
        self.htf_bb_ind = {}          # HTF Bollinger for volatility regime
        self.htf_trend_dir = {}       # +1 uptrend, -1 downtrend, 0 neutral
        self.htf_vol_regime = {}      # "normal", "low", "high"
        self.htf_bb_width_history = {}  # Rolling BB width values for percentile

        # Bar history
        self.fast_bars = {}

        # Previous indicator values (for crossover / rising detection)
        self.prev_ema_fast_val = {}
        self.prev_ema_slow_val = {}
        self.prev_adx_val = {}

        # Position tracking
        self.entry_ticket = {}
        self.stop_ticket = {}
        self.tp_ticket = {}
        self.entry_price = {}
        self.entry_atr = {}
        self.entry_dir = {}
        self.active_stop_price = {}
        self.high_watermark = {}
        self.low_watermark = {}
        self.hold_bars = {}
        self.last_exit_bar_index = {}
        self.exits_placed = {}
        self.pending_family = {}
        self.bar_index = 0

        # Statistics
        self.exit_reason_counts = {}
        self.entry_fill_count = 0

        # --- Compute history window ---
        lower_window = max(
            self.ema_slow_period, self.bb_period, self.rsi_period,
            self.adx_period, self.atr_period, self.trend_ema_period,
            self.stoch_k, 50
        ) + 10

        # --- Add Forex Pairs ---
        for pair in pairs:
            security = self.add_forex(pair, Resolution.MINUTE, Market.OANDA)
            security.set_leverage(20)
            sym = security.symbol
            self.symbols.append(sym)

            # Initialize indicators
            self.rsi_ind[sym] = RelativeStrengthIndex(self.rsi_period, MovingAverageType.WILDERS)
            self.atr_ind[sym] = AverageTrueRange(self.atr_period, MovingAverageType.WILDERS)
            self.trend_ema[sym] = ExponentialMovingAverage(self.trend_ema_period)
            self.prev_rsi_val[sym] = None

            # Family 2: Bollinger Bands
            self.bb_ind[sym] = BollingerBands(self.bb_period, self.bb_std_dev, MovingAverageType.SIMPLE)

            # Family 4: Stochastic
            self.stoch_ind[sym] = Stochastic(self.stoch_k, self.stoch_d, 3)
            self.prev_stoch_k[sym] = None
            self.prev_stoch_d[sym] = None

            # Family 5: EMA Trend + ADX
            self.ema_fast[sym] = ExponentialMovingAverage(self.ema_fast_period)
            self.ema_slow[sym] = ExponentialMovingAverage(self.ema_slow_period)
            self.adx_ind[sym] = AverageDirectionalIndex(self.adx_period)

            # Bar history deque
            self.fast_bars[sym] = deque(maxlen=lower_window)

            # Consolidator — single fast-bar consolidator per symbol
            fast_consolidator = QuoteBarConsolidator(timedelta(minutes=self.bar_minutes))
            fast_consolidator.data_consolidated += self._make_fast_handler(sym)
            self.subscription_manager.add_consolidator(sym, fast_consolidator)

            # HTF consolidator for trend direction + volatility filter
            htf_consolidator = QuoteBarConsolidator(timedelta(minutes=self.htf_minutes))
            htf_consolidator.data_consolidated += self._make_htf_handler(sym)
            self.subscription_manager.add_consolidator(sym, htf_consolidator)

            self.htf_ema_ind[sym] = ExponentialMovingAverage(self.htf_ema_period)
            self.htf_bb_ind[sym] = BollingerBands(self.vol_bb_period, 2.0, MovingAverageType.SIMPLE)
            self.htf_trend_dir[sym] = 0
            self.htf_vol_regime[sym] = "normal"
            self.htf_bb_width_history[sym] = deque(maxlen=100)

            # Initialize state
            self.latest_bar[sym] = None
            self.prev_close[sym] = None
            self.new_bar_ready[sym] = False
            self.trade_day[sym] = None
            self.trade_count_today[sym] = 0
            self.entry_ticket[sym] = None
            self.stop_ticket[sym] = None
            self.tp_ticket[sym] = None
            self.entry_price[sym] = None
            self.entry_atr[sym] = None
            self.entry_dir[sym] = 0
            self.active_stop_price[sym] = None
            self.high_watermark[sym] = None
            self.low_watermark[sym] = None
            self.hold_bars[sym] = 0
            self.last_exit_bar_index[sym] = -999999
            self.exits_placed[sym] = False
            self.pending_family[sym] = None
            self.prev_ema_fast_val[sym] = None
            self.prev_ema_slow_val[sym] = None
            self.prev_adx_val[sym] = None

        # Warmup — must cover the longest indicator period on BOTH timeframes
        fast_warmup = max(
            self.ema_slow_period, self.bb_period, self.atr_period,
            self.rsi_period, self.adx_period, self.trend_ema_period,
            self.stoch_k
        ) + 50
        htf_warmup = max(self.htf_ema_period, self.vol_bb_period) + 20
        # Use whichever requires more wall-clock minutes
        warmup_minutes = max(
            fast_warmup * self.bar_minutes,
            htf_warmup * self.htf_minutes,
        )
        self.set_warm_up(warmup_minutes, Resolution.MINUTE)

        family_name = self.FAMILY_NAMES.get(self.strategy_family, "Unknown")
        self.debug(
            f"[INIT] Family={family_name} | Mode={self.mode} | "
            f"Pairs={pairs} | Bar={self.bar_minutes}m | HTF={self.htf_minutes}m | "
            f"Cash=${self.starting_cash:,.0f} | Risk={self.risk_per_trade_pct}% | "
            f"VolFilter={'ON' if self.vol_filter_enabled else 'OFF'}"
        )

    # ──────────────────────────────────────────────
    #  DATA HANDLERS
    # ──────────────────────────────────────────────

    def _make_fast_handler(self, sym):
        def handler(sender, bar):
            close_val = float(bar.close)
            if self.latest_bar[sym] is not None:
                self.prev_close[sym] = float(self.latest_bar[sym].close)

            # Store previous indicator values BEFORE updating
            if self.rsi_ind[sym].is_ready:
                self.prev_rsi_val[sym] = float(self.rsi_ind[sym].current.value)
            if self.ema_fast[sym].is_ready:
                self.prev_ema_fast_val[sym] = float(self.ema_fast[sym].current.value)
            if self.ema_slow[sym].is_ready:
                self.prev_ema_slow_val[sym] = float(self.ema_slow[sym].current.value)
            if self.adx_ind[sym].is_ready:
                self.prev_adx_val[sym] = float(self.adx_ind[sym].current.value)
            if self.stoch_ind[sym].is_ready:
                self.prev_stoch_k[sym] = float(self.stoch_ind[sym].fast_stoch.current.value)
                self.prev_stoch_d[sym] = float(self.stoch_ind[sym].stoch_d.current.value)

            # Update price-only indicators
            self.rsi_ind[sym].update(bar.end_time, close_val)
            self.trend_ema[sym].update(bar.end_time, close_val)
            self.ema_fast[sym].update(bar.end_time, close_val)
            self.ema_slow[sym].update(bar.end_time, close_val)
            self.bb_ind[sym].update(bar.end_time, close_val)

            # Update bar-based indicators (need high/low/close)
            self.atr_ind[sym].update(bar)
            self.adx_ind[sym].update(bar)
            self.stoch_ind[sym].update(bar)

            self.fast_bars[sym].append(bar)
            self.latest_bar[sym] = bar
            self.new_bar_ready[sym] = True
        return handler

    def _make_htf_handler(self, sym):
        """Handler for higher-timeframe (HTF) bars — updates trend direction + volatility regime."""
        def handler(sender, bar):
            close_val = float(bar.close)

            # Update HTF EMA (trend direction)
            self.htf_ema_ind[sym].update(bar.end_time, close_val)

            # Update HTF Bollinger Bands (volatility regime)
            self.htf_bb_ind[sym].update(bar.end_time, close_val)

            # Determine trend direction from HTF EMA
            if self.htf_ema_ind[sym].is_ready:
                htf_ema = float(self.htf_ema_ind[sym].current.value)
                margin = htf_ema * 0.001  # 0.1% buffer to avoid whipsaw
                if close_val > htf_ema + margin:
                    self.htf_trend_dir[sym] = 1   # Uptrend
                elif close_val < htf_ema - margin:
                    self.htf_trend_dir[sym] = -1  # Downtrend
                else:
                    self.htf_trend_dir[sym] = 0   # Neutral / choppy

            # Determine volatility regime 
            if self.htf_bb_ind[sym].is_ready:
                upper = float(self.htf_bb_ind[sym].upper_band.current.value)
                lower = float(self.htf_bb_ind[sym].lower_band.current.value)
                middle = float(self.htf_bb_ind[sym].middle_band.current.value)
                if middle > 0:
                    bb_width = (upper - lower) / middle
                    self.htf_bb_width_history[sym].append(bb_width)

                    # Use percentile of recent BB width to classify regime
                    if len(self.htf_bb_width_history[sym]) >= 20:
                        sorted_widths = sorted(self.htf_bb_width_history[sym])
                        pct_rank = sorted_widths.index(bb_width) / len(sorted_widths)
                        if pct_rank < 0.25:
                            self.htf_vol_regime[sym] = "low"     # Flat, choppy
                        elif pct_rank > 0.85:
                            self.htf_vol_regime[sym] = "high"    # Extreme volatility
                        else:
                            self.htf_vol_regime[sym] = "normal"  # Good for trading
                    else:
                        self.htf_vol_regime[sym] = "normal"

        return handler

    def _htf_allows_entry(self, sym, signal):
        """Check if HTF trend direction and volatility regime allow this entry."""
        # If HTF EMA not ready yet, allow entry
        if not self.htf_ema_ind[sym].is_ready:
            return True

        # Volatility filter: skip entries in low or extreme volatility
        if self.vol_filter_enabled:
            regime = self.htf_vol_regime[sym]
            if regime == "low":
                return False  
            if regime == "high":
                return False  

        # Trend direction filter: block counter-trend, allow neutral
        trend = self.htf_trend_dir[sym]
        if trend == 0:
            return True   # Neutral — let signal-level filters decide

        # Only block when trading AGAINST a clear HTF trend
        if signal > 0 and trend < 0:
            return False  # Don't buy in a downtrend
        if signal < 0 and trend > 0:
            return False  # Don't short in an uptrend

        return True

    # ──────────────────────────────────────────────
    #  MAIN EVENT LOOP
    # ──────────────────────────────────────────────

    def on_data(self, data):
        self._reset_daily_counters()
        if self.use_force_flat:
            self._force_flat_if_needed()

        for sym in self.symbols:
            self._manage_open_position(sym, data)

        if self.is_warming_up:
            return
        if not any(self.new_bar_ready.values()):
            return

        for sym in self.symbols:
            if not self.new_bar_ready.get(sym, False):
                continue
            self.new_bar_ready[sym] = False
            self.bar_index += 1

            if not self._indicators_ready(sym):
                continue
            if self.use_session_filter and not self._in_session():
                continue
            if self.portfolio[sym].invested:
                continue

            signal = self._generate_signal(sym)
            if signal == 0:
                continue
            if not self._htf_allows_entry(sym, signal):
                continue
            if self._cooldown_active(sym):
                continue
            if self._has_active_orders(sym):
                continue
            if not self._can_trade_today(sym):
                continue

            self._enter_position(sym, signal)

    # ──────────────────────────────────────────────
    #  SESSION & TIME MANAGEMENT
    # ──────────────────────────────────────────────

    def _reset_daily_counters(self):
        today = self.time.date()
        for sym in self.symbols:
            if self.trade_day[sym] != today:
                self.trade_day[sym] = today
                self.trade_count_today[sym] = 0

    def _in_session(self):
        hour = self.time.hour
        if self.session_start_hour <= self.session_end_hour:
            return self.session_start_hour <= hour <= self.session_end_hour
        return hour >= self.session_start_hour or hour <= self.session_end_hour

    def _past_last_entry_time(self):
        now_min = self.time.hour * 60 + self.time.minute
        cutoff = self.last_entry_hour * 60 + self.last_entry_minute
        return now_min > cutoff

    def _past_force_flat_time(self):
        now_min = self.time.hour * 60 + self.time.minute
        flat_min = self.force_flat_hour * 60 + self.force_flat_minute
        return now_min >= flat_min

    def _force_flat_if_needed(self):
        if not self._past_force_flat_time():
            return
        for sym in self.symbols:
            if self.portfolio[sym].invested or self._has_active_orders(sym):
                self._close_position(sym, reason="EndOfSession")

    def _can_trade_today(self, sym):
        if self.trade_count_today[sym] >= self.max_trades_per_day:
            return False
        if self.use_session_filter and self._past_last_entry_time():
            return False
        return True

    def _cooldown_active(self, sym):
        return (self.bar_index - self.last_exit_bar_index[sym]) <= self.cooldown_bars

    # ──────────────────────────────────────────────
    #  INDICATOR READINESS
    # ──────────────────────────────────────────────

    def _indicators_ready(self, sym):
        if self.latest_bar[sym] is None:
            return False
        if not self.atr_ind[sym].is_ready:
            return False

        family = self.strategy_family

        if family == 1:  # RSI Mean Reversion
            return (self.rsi_ind[sym].is_ready
                    and self.trend_ema[sym].is_ready)

        if family == 2:  # Bollinger Band Bounce
            return (self.bb_ind[sym].is_ready
                    and self.rsi_ind[sym].is_ready)

        if family == 3:  # EMA Crossover
            return (self.ema_fast[sym].is_ready
                    and self.ema_slow[sym].is_ready
                    and self.trend_ema[sym].is_ready
                    and self.prev_ema_fast_val[sym] is not None
                    and self.prev_ema_slow_val[sym] is not None)

        if family == 4:  # Stochastic Mean Reversion
            return (self.stoch_ind[sym].is_ready
                    and self.trend_ema[sym].is_ready)

        if family == 5:  # EMA Trend + ADX
            return (self.ema_fast[sym].is_ready
                    and self.ema_slow[sym].is_ready
                    and self.adx_ind[sym].is_ready
                    and self.prev_ema_fast_val[sym] is not None
                    and self.prev_ema_slow_val[sym] is not None
                    and self.prev_adx_val[sym] is not None)

        return False

    # ──────────────────────────────────────────────
    #  SIGNAL GENERATION (5 FAMILIES)
    # ──────────────────────────────────────────────

    def _generate_signal(self, sym):
        family = self.strategy_family

        if family == 1:
            return self._signal_rsi_mean_reversion(sym)
        if family == 2:
            return self._signal_bollinger_bounce(sym)
        if family == 3:
            return self._signal_ema_crossover(sym)
        if family == 4:
            return self._signal_stochastic_mean_reversion(sym)
        if family == 5:
            return self._signal_ema_trend_adx(sym)
        return 0

    def _signal_rsi_mean_reversion(self, sym):
        """
        Family 1: RSI Mean Reversion (fade oversold/overbought extremes)
        Long:  RSI < oversold AND price > trend EMA (buy dip in uptrend)
        Short: RSI > overbought AND price < trend EMA (sell rally in downtrend)
        """
        rsi = float(self.rsi_ind[sym].current.value)
        price = float(self.latest_bar[sym].close)
        trend = float(self.trend_ema[sym].current.value)

        # Long: RSI oversold → fade the drop, buy in uptrend
        if rsi < self.rsi_oversold and price > trend:
            return 1

        # Short: RSI overbought → fade the rally, sell in downtrend
        if rsi > self.rsi_overbought and price < trend:
            return -1

        return 0

    def _signal_bollinger_bounce(self, sym):
        """
        Family 2: Bollinger Band Bounce (mean reversion from band extremes)
        Long:  Price < lower band AND RSI < bb_rsi_long (oversold confirmation)
        Short: Price > upper band AND RSI > bb_rsi_short (overbought confirmation)
        """
        price = float(self.latest_bar[sym].close)
        rsi = float(self.rsi_ind[sym].current.value)
        upper = float(self.bb_ind[sym].upper_band.current.value)
        lower = float(self.bb_ind[sym].lower_band.current.value)

        # Long: price below lower band + RSI confirms oversold
        if price < lower and rsi < self.bb_rsi_long:
            return 1

        # Short: price above upper band + RSI confirms overbought
        if price > upper and rsi > self.bb_rsi_short:
            return -1

        return 0

    def _signal_ema_crossover(self, sym):
        """
        Family 3: EMA Crossover + Trend Filter
        Long:  Fast EMA crosses above Slow EMA AND price > Trend EMA
        Short: Fast EMA crosses below Slow EMA AND price < Trend EMA
        """
        fast = float(self.ema_fast[sym].current.value)
        slow = float(self.ema_slow[sym].current.value)
        prev_fast = self.prev_ema_fast_val[sym]
        prev_slow = self.prev_ema_slow_val[sym]
        price = float(self.latest_bar[sym].close)
        trend = float(self.trend_ema[sym].current.value)

        if prev_fast is None or prev_slow is None:
            return 0

        # Bullish crossover: fast crosses above slow, uptrend confirmed
        if prev_fast <= prev_slow and fast > slow and price > trend:
            return 1

        # Bearish crossover: fast crosses below slow, downtrend confirmed
        if prev_fast >= prev_slow and fast < slow and price < trend:
            return -1

        return 0

    def _signal_stochastic_mean_reversion(self, sym):
        """
        Family 4: Stochastic Mean Reversion (fade oversold/overbought zones)
        Long:  %K is in oversold zone + price > trend EMA (buy dips in uptrend)
        Short: %K is in overbought zone + price < trend EMA (sell rallies in downtrend)
        """
        k = float(self.stoch_ind[sym].fast_stoch.current.value)
        price = float(self.latest_bar[sym].close)
        trend = float(self.trend_ema[sym].current.value)

        # Long: %K in oversold zone, uptrend
        if k < self.stoch_oversold and price > trend:
            return 1

        # Short: %K in overbought zone, downtrend
        if k > self.stoch_overbought and price < trend:
            return -1

        return 0

    def _signal_ema_trend_adx(self, sym):
        """
        Family 5: EMA Trend Following + ADX (was Family 4, USDJPY winner)
        Long:  Fast EMA crosses above Slow EMA AND ADX > adx_min AND ADX rising
        Short: Fast EMA crosses below Slow EMA AND ADX > adx_min AND ADX rising
        """
        fast = float(self.ema_fast[sym].current.value)
        slow = float(self.ema_slow[sym].current.value)
        adx = float(self.adx_ind[sym].current.value)

        prev_fast = self.prev_ema_fast_val[sym]
        prev_slow = self.prev_ema_slow_val[sym]
        prev_adx = self.prev_adx_val[sym]

        if prev_fast is None or prev_slow is None or prev_adx is None:
            return 0

        # ADX must show sufficient trend strength and be rising
        if adx < self.adx_min:
            return 0
        adx_rising = adx > prev_adx

        if not adx_rising:
            return 0

        # Long: fast EMA crosses ABOVE slow EMA
        if fast > slow and prev_fast <= prev_slow:
            return 1

        # Short: fast EMA crosses BELOW slow EMA
        if fast < slow and prev_fast >= prev_slow:
            return -1

        return 0

    # ──────────────────────────────────────────────
    #  ORDER MANAGEMENT (SYNC FILL FIX)
    # ──────────────────────────────────────────────

    def _enter_position(self, sym, signal):
        atr_value = float(self.atr_ind[sym].current.value)
        if atr_value <= 0:
            return

        # Min ATR filter — skip if no volatility (flat market)
        if atr_value < 0.0001:
            return

        stop_dist = self.stop_atr_mult * atr_value

        # Spread filter — skip if spread is too wide relative to stop
        ask = float(self.securities[sym].ask_price)
        bid = float(self.securities[sym].bid_price)
        spread = ask - bid
        if spread > 0.20 * stop_dist:
            return

        qty = self._calculate_quantity(sym, stop_dist, signal)
        if qty == 0:
            return

        family_name = self.FAMILY_NAMES.get(self.strategy_family, "Unknown")

        # Store direction and ATR BEFORE placing the order
        self.entry_dir[sym] = 1 if signal > 0 else -1
        self.entry_atr[sym] = atr_value
        self.exits_placed[sym] = False
        self.pending_family[sym] = family_name

        # Place market order
        ticket = self.market_order(sym, qty, tag=f"ENTRY {family_name}")
        self.entry_ticket[sym] = ticket

        if ticket.status == OrderStatus.FILLED and not self.exits_placed[sym]:
            self._setup_exit_orders(
                sym,
                float(ticket.average_fill_price),
                atr_value,
                int(ticket.quantity_filled),
            )

    def _setup_exit_orders(self, sym, fill_price, atr_value, filled_qty):
        """Place stop loss and take profit orders after entry fill."""
        if self.exits_placed[sym]:
            return  

        self.exits_placed[sym] = True
        direction = 1 if filled_qty > 0 else -1
        stop_dist = self.stop_atr_mult * atr_value
        tp_dist = self.reward_ratio * stop_dist

        if direction > 0:
            stop_price = fill_price - stop_dist
            tp_price = fill_price + tp_dist
            exit_qty = -abs(filled_qty)
        else:
            stop_price = fill_price + stop_dist
            tp_price = fill_price - tp_dist
            exit_qty = abs(filled_qty)

        self.entry_price[sym] = fill_price
        self.entry_dir[sym] = direction
        self.active_stop_price[sym] = stop_price
        self.high_watermark[sym] = fill_price
        self.low_watermark[sym] = fill_price
        self.hold_bars[sym] = 0
        self.trade_count_today[sym] += 1
        self.entry_fill_count += 1

        self.stop_ticket[sym] = self.stop_market_order(sym, exit_qty, stop_price, tag="STOP LOSS")
        self.tp_ticket[sym] = self.limit_order(sym, exit_qty, tp_price, tag="TAKE PROFIT")

        family_name = self.pending_family[sym] or "Unknown"
        self.debug(
            f"[ENTRY] {family_name} | {sym} | "
            f"{'BUY' if direction > 0 else 'SELL'} @ {fill_price:.5f} | "
            f"Stop: {stop_price:.5f} | TP: {tp_price:.5f} | "
            f"Qty: {abs(filled_qty)}"
        )

    def _close_position(self, sym, reason="Liquidate"):
        had_position = self.portfolio[sym].invested

        # Cancel existing stop/TP
        for ticket in [self.stop_ticket.get(sym), self.tp_ticket.get(sym)]:
            if ticket is not None and self._ticket_active(ticket):
                ticket.cancel(reason)

        self.stop_ticket[sym] = None
        self.tp_ticket[sym] = None
        self.entry_ticket[sym] = None
        self.active_stop_price[sym] = None

        if had_position:
            self._count_exit(reason)
            self.liquidate(sym, tag=reason)

        self._clear_state(sym)

    def _manage_open_position(self, sym, data):
        if not self.portfolio[sym].invested:
            return
        if not self._indicators_ready(sym):
            return

        bar = data.quote_bars.get(sym)
        if bar is None:
            return

        direction = 1 if self.portfolio[sym].quantity > 0 else -1
        entry_price = self.entry_price[sym]
        entry_atr = self.entry_atr[sym]

        if entry_price is None or entry_atr is None or entry_atr <= 0:
            return

        # Count hold bars
        if self.new_bar_ready.get(sym, False):
            self.hold_bars[sym] += 1
            if self.hold_bars[sym] >= self.max_hold_bars:
                self._close_position(sym, reason="MaxHoldReached")
                return

        # Update watermarks
        if direction > 0:
            high = float(bar.high)
            if self.high_watermark[sym] is None or high > self.high_watermark[sym]:
                self.high_watermark[sym] = high
        else:
            low = float(bar.low)
            if self.low_watermark[sym] is None or low < self.low_watermark[sym]:
                self.low_watermark[sym] = low

    # ──────────────────────────────────────────────
    #  ORDER EVENTS
    # ──────────────────────────────────────────────

    def on_order_event(self, order_event):
        if order_event.status != OrderStatus.FILLED:
            return

        sym = order_event.symbol
        if sym not in self.symbols:
            return

        entry_t = self.entry_ticket.get(sym)
        stop_t = self.stop_ticket.get(sym)
        tp_t = self.tp_ticket.get(sym)

        # Entry fill
        if entry_t is not None and order_event.order_id == entry_t.order_id:
            if not self.exits_placed[sym]:
                filled_qty = int(order_event.fill_quantity)
                if filled_qty == 0:
                    return
                atr_value = self.entry_atr[sym] or float(self.atr_ind[sym].current.value)
                self._setup_exit_orders(
                    sym,
                    float(order_event.fill_price),
                    atr_value,
                    filled_qty,
                )
            return

        # Stop loss fill
        if stop_t is not None and order_event.order_id == stop_t.order_id:
            if tp_t is not None and self._ticket_active(tp_t):
                tp_t.cancel("Stop loss filled")
            self._count_exit("StopLoss")
            self.debug(
                f"[EXIT] Stop Loss | {sym} | "
                f"Entry: {self.entry_price[sym]:.5f} | "
                f"Exit: {order_event.fill_price:.5f}"
            )
            self._clear_state(sym)
            return

        # Take profit fill
        if tp_t is not None and order_event.order_id == tp_t.order_id:
            if stop_t is not None and self._ticket_active(stop_t):
                stop_t.cancel("Take profit filled")
            self._count_exit("TakeProfit")
            self.debug(
                f"[EXIT] Take Profit | {sym} | "
                f"Entry: {self.entry_price[sym]:.5f} | "
                f"Exit: {order_event.fill_price:.5f}"
            )
            self._clear_state(sym)
            return

    # ──────────────────────────────────────────────
    #  POSITION SIZING
    # ──────────────────────────────────────────────

    def _calculate_quantity(self, sym, stop_dist, signal):
        if signal not in (1, -1) or stop_dist <= 0:
            return 0

        equity = float(self.portfolio.total_portfolio_value)
        dollar_risk = equity * (self.risk_per_trade_pct / 100.0)
        if dollar_risk <= 0:
            return 0

        loss_per_unit = self._loss_per_unit_usd(sym, stop_dist)
        if loss_per_unit <= 0:
            return 0

        units = int(dollar_risk / loss_per_unit)
        if units <= 0:
            return 0

        return units if signal > 0 else -units

    def _loss_per_unit_usd(self, sym, stop_dist):
        price = float(self.latest_bar[sym].close) if self.latest_bar[sym] else float(self.securities[sym].price)
        if price <= 0 or stop_dist <= 0:
            return 0.0

        quote_currency = str(self.securities[sym].quote_currency.symbol).upper()
        if quote_currency == "USD":
            return stop_dist
        if quote_currency == "JPY":
            return stop_dist / price
        return stop_dist

    # ──────────────────────────────────────────────
    #  UTILITY HELPERS
    # ──────────────────────────────────────────────

    def _ticket_active(self, ticket):
        if ticket is None:
            return False
        return ticket.status in (
            OrderStatus.NEW, OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED, OrderStatus.UPDATE_SUBMITTED,
        )

    def _has_active_orders(self, sym):
        return (
            self._ticket_active(self.entry_ticket.get(sym))
            or self._ticket_active(self.stop_ticket.get(sym))
            or self._ticket_active(self.tp_ticket.get(sym))
        )

    def _count_exit(self, reason):
        self.exit_reason_counts[reason] = self.exit_reason_counts.get(reason, 0) + 1

    def _clear_state(self, sym):
        self.entry_ticket[sym] = None
        self.stop_ticket[sym] = None
        self.tp_ticket[sym] = None
        self.pending_family[sym] = None
        self.entry_price[sym] = None
        self.entry_atr[sym] = None
        self.entry_dir[sym] = 0
        self.active_stop_price[sym] = None
        self.high_watermark[sym] = None
        self.low_watermark[sym] = None
        self.hold_bars[sym] = 0
        self.exits_placed[sym] = False
        self.last_exit_bar_index[sym] = self.bar_index

    # ──────────────────────────────────────────────
    #  END OF ALGORITHM — REPORT STATISTICS
    # ──────────────────────────────────────────────

    def on_end_of_algorithm(self):
        family_name = self.FAMILY_NAMES.get(self.strategy_family, "Unknown")

        self.set_runtime_statistic("Family", family_name)
        self.set_runtime_statistic("Entries", str(self.entry_fill_count))

        for reason, count in self.exit_reason_counts.items():
            self.set_runtime_statistic(f"Exit_{reason}", str(count))

        self.debug(
            f"[FINAL] Family={family_name} | "
            f"Entries={self.entry_fill_count} | "
            f"Exits={dict(self.exit_reason_counts)}"
        )
