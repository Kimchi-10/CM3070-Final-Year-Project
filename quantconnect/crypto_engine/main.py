from AlgorithmImports import *
from datetime import datetime, timedelta
from collections import deque


class CryptoStrategyEngine(QCAlgorithm):
    """
    FYP Crypto Trading Engine — 5 Mean-Reversion Strategy Families
    ===============================================================
    Family 1: RSI Mean Reversion
    Family 2: Bollinger Band Mean Reversion
    Family 3: Connors RSI (2-Period) Mean Reversion
    Family 4: EMA Trend Following + ADX
    Family 5: Williams %R + RSI Confluence

    Supports 3 trading styles:
      - day_trade:  15-min bars, 24/7 trading
      - swing:      4-hour bars, wider hold window
      - long_term:  daily bars, longest hold window

    All parameters are received via self.get_parameter() from the GA runner.
    Handles synchronous fills correctly (LEAN backtest vs live).
    Uses TradeBarConsolidator (crypto uses trade bars, not quote bars).
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
        # Binance uses USDT pairs (BTCUSDT, SOLUSDT)
        pairs_str = param_str("pairs", "BTCUSDT")
        pairs = [p.strip().upper() for p in pairs_str.split(",") if p.strip()]
        if not pairs:
            pairs = ["BTCUSDT"]

        # --- Strategy Family ---
        self.strategy_family = param_int("strategy_family", 1)
        self.FAMILY_NAMES = {
            1: "RSI Mean Reversion",
            2: "Bollinger Band Mean Reversion",
            3: "Connors RSI (2-Period)",
            4: "EMA Trend + ADX",
            5: "Williams %R + RSI",
        }

        # --- Bar Timeframe ---
        self.bar_minutes = max(param_int("bar_minutes", 15), 1)

        self.use_session_filter = param_int("use_session_filter", 0) == 1
        self.use_force_flat = param_int("use_force_flat", 0) == 1
        self.session_start_hour = param_int("session_start_hour", 0)
        self.session_end_hour = param_int("session_end_hour", 23)
        self.last_entry_hour = param_int("last_entry_hour", 23)
        self.last_entry_minute = param_int("last_entry_minute", 59)
        self.force_flat_hour = param_int("force_flat_hour", 23)
        self.force_flat_minute = param_int("force_flat_minute", 55)

        # --- Shared Indicator / Trade Parameters ---
        self.atr_period = param_int("atr_period", 14)
        self.stop_atr_mult = param_float("stop_atr_mult", 2.0)
        self.reward_ratio = param_float("reward_ratio", 1.5)
        self.max_hold_bars = max(param_int("max_hold_bars", 20), 1)
        self.cooldown_bars = max(param_int("cooldown_bars", 2), 0)

        # --- Family 1: RSI Mean Reversion ---
        self.rsi_period = param_int("rsi_period", 14)
        self.rsi_oversold = param_float("rsi_oversold", 30.0)
        self.rsi_overbought = param_float("rsi_overbought", 70.0)
        self.trend_ema_period = param_int("trend_ema_period", 200)

        # --- Family 2: Bollinger Band Mean Reversion ---
        self.bb_period = param_int("bb_period", 20)
        self.bb_std_dev = param_float("bb_std_dev", 2.0)
        self.bb_rsi_long = param_float("bb_rsi_long", 35.0)
        self.bb_rsi_short = param_float("bb_rsi_short", 65.0)

        # --- Family 3: Connors RSI (2-Period) ---
        self.crsi_period = param_int("crsi_period", 2)
        self.crsi_entry = param_float("crsi_entry", 10.0)
        self.crsi_entry_short = param_float("crsi_entry_short", 90.0)
        self.exit_sma_period = param_int("exit_sma_period", 5)

        # --- Family 4: EMA Trend + ADX ---
        self.ema_fast_period = param_int("ema_fast_period", 15)
        self.ema_slow_period = param_int("ema_slow_period", 50)
        self.adx_period = param_int("adx_period", 14)
        self.adx_min = param_float("adx_min", 25.0)

        # --- Family 5: Williams %R + RSI ---
        self.wr_period = param_int("wr_period", 14)
        self.wr_oversold = param_float("wr_oversold", -80.0)
        self.wr_overbought = param_float("wr_overbought", -20.0)
        self.wr_rsi_long = param_float("wr_rsi_long", 35.0)
        self.wr_rsi_short = param_float("wr_rsi_short", 65.0)

        # Ensure EMA fast < slow for Family 4
        if self.ema_fast_period >= self.ema_slow_period:
            self.ema_slow_period = self.ema_fast_period + 10

        self.is_spot = True  
        self.set_brokerage_model(BrokerageName.BINANCE, AccountType.CASH)

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
        self.bb_ind = {}
        self.adx_ind = {}
        self.ema_fast = {}
        self.ema_slow = {}
        self.trend_ema = {}
        self.exit_sma = {}
        self.crsi_ind = {}
        self.wilr_ind = {}

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
            self.exit_sma_period, self.crsi_period, self.wr_period, 50
        ) + 10

        for pair in pairs:
            security = self.add_crypto(pair, Resolution.MINUTE, Market.BINANCE)
            security.set_leverage(1)  # Spot crypto: no leverage
            sym = security.symbol
            self.symbols.append(sym)

            # Initialize indicators
            self.ema_fast[sym] = ExponentialMovingAverage(self.ema_fast_period)
            self.ema_slow[sym] = ExponentialMovingAverage(self.ema_slow_period)
            self.rsi_ind[sym] = RelativeStrengthIndex(self.rsi_period, MovingAverageType.WILDERS)
            self.atr_ind[sym] = AverageTrueRange(self.atr_period, MovingAverageType.WILDERS)
            self.bb_ind[sym] = BollingerBands(self.bb_period, self.bb_std_dev, MovingAverageType.SIMPLE)
            self.adx_ind[sym] = AverageDirectionalIndex(self.adx_period)
            self.trend_ema[sym] = ExponentialMovingAverage(self.trend_ema_period)
            self.exit_sma[sym] = SimpleMovingAverage(self.exit_sma_period)
            self.crsi_ind[sym] = RelativeStrengthIndex(self.crsi_period, MovingAverageType.WILDERS)
            self.wilr_ind[sym] = WilliamsPercentR(self.wr_period)

            # Bar history deque
            self.fast_bars[sym] = deque(maxlen=lower_window)

            fast_consolidator = TradeBarConsolidator(timedelta(minutes=self.bar_minutes))
            fast_consolidator.data_consolidated += self._make_fast_handler(sym)
            self.subscription_manager.add_consolidator(sym, fast_consolidator)

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

        warmup_bars = max(
            self.ema_slow_period, self.bb_period, self.atr_period,
            self.rsi_period, self.adx_period, self.trend_ema_period,
            self.exit_sma_period, self.crsi_period, self.wr_period
        ) + 50
        warmup_minutes = warmup_bars * self.bar_minutes
        self.set_warm_up(warmup_minutes, Resolution.MINUTE)

        family_name = self.FAMILY_NAMES.get(self.strategy_family, "Unknown")
        self.debug(
            f"[INIT] Family={family_name} | Mode={self.mode} | "
            f"Pairs={pairs} | Bar={self.bar_minutes}m | "
            f"Cash=${self.starting_cash:,.0f} | Risk={self.risk_per_trade_pct}%"
        )

    # ──────────────────────────────────────────────
    #  DATA HANDLERS
    # ──────────────────────────────────────────────

    def _make_fast_handler(self, sym):
        def handler(sender, bar):
            close_val = float(bar.close)
            if self.latest_bar[sym] is not None:
                self.prev_close[sym] = float(self.latest_bar[sym].close)

            # Store previous indicator values BEFORE updating (for crossover / rising detection)
            if self.ema_fast[sym].is_ready:
                self.prev_ema_fast_val[sym] = float(self.ema_fast[sym].current.value)
            if self.ema_slow[sym].is_ready:
                self.prev_ema_slow_val[sym] = float(self.ema_slow[sym].current.value)
            if self.adx_ind[sym].is_ready:
                self.prev_adx_val[sym] = float(self.adx_ind[sym].current.value)

            # Update price-only indicators
            self.ema_fast[sym].update(bar.end_time, close_val)
            self.ema_slow[sym].update(bar.end_time, close_val)
            self.rsi_ind[sym].update(bar.end_time, close_val)
            self.bb_ind[sym].update(bar.end_time, close_val)
            self.trend_ema[sym].update(bar.end_time, close_val)
            self.exit_sma[sym].update(bar.end_time, close_val)
            self.crsi_ind[sym].update(bar.end_time, close_val)

            # Update bar-based indicators (need high/low/close)
            self.atr_ind[sym].update(bar)
            self.adx_ind[sym].update(bar)
            self.wilr_ind[sym].update(bar)

            self.fast_bars[sym].append(bar)
            self.latest_bar[sym] = bar
            self.new_bar_ready[sym] = True
        return handler

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
            if self.is_spot and signal == -1:
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

        if family == 1:
            return (self.rsi_ind[sym].is_ready
                    and self.trend_ema[sym].is_ready)

        if family == 2:
            return (self.bb_ind[sym].is_ready
                    and self.rsi_ind[sym].is_ready)

        if family == 3:
            return (self.crsi_ind[sym].is_ready
                    and self.trend_ema[sym].is_ready
                    and self.exit_sma[sym].is_ready)

        if family == 4:
            return (self.ema_fast[sym].is_ready
                    and self.ema_slow[sym].is_ready
                    and self.adx_ind[sym].is_ready
                    and self.prev_ema_fast_val[sym] is not None
                    and self.prev_ema_slow_val[sym] is not None
                    and self.prev_adx_val[sym] is not None)

        if family == 5:
            return (self.wilr_ind[sym].is_ready
                    and self.rsi_ind[sym].is_ready)

        return False

    # ──────────────────────────────────────────────
    #  SIGNAL GENERATION (5 FAMILIES)
    # ──────────────────────────────────────────────

    def _generate_signal(self, sym):
        family = self.strategy_family

        if family == 1:
            return self._signal_rsi_mean_reversion(sym)
        if family == 2:
            return self._signal_bollinger_mean_reversion(sym)
        if family == 3:
            return self._signal_connors_rsi(sym)
        if family == 4:
            return self._signal_ema_trend_adx(sym)
        if family == 5:
            return self._signal_williams_rsi(sym)
        return 0

    def _signal_rsi_mean_reversion(self, sym):
        """
        Family 1: RSI Mean Reversion (Expected 60-65% WR)
        Long:  RSI < oversold AND price above trend EMA (uptrend filter)
        Short: RSI > overbought AND price below trend EMA
        """
        rsi = float(self.rsi_ind[sym].current.value)
        price = float(self.latest_bar[sym].close)
        trend = float(self.trend_ema[sym].current.value)

        if rsi < self.rsi_oversold and price > trend:
            return 1
        if rsi > self.rsi_overbought and price < trend:
            return -1
        return 0

    def _signal_bollinger_mean_reversion(self, sym):
        """
        Family 2: Bollinger Band Mean Reversion (Expected 65-70% WR)
        Long:  Price closes below lower BB AND RSI < bb_rsi_long
        Short: Price closes above upper BB AND RSI > bb_rsi_short
        """
        upper = float(self.bb_ind[sym].upper_band.current.value)
        lower = float(self.bb_ind[sym].lower_band.current.value)
        price = float(self.latest_bar[sym].close)
        rsi = float(self.rsi_ind[sym].current.value)

        if price < lower and rsi < self.bb_rsi_long:
            return 1
        if price > upper and rsi > self.bb_rsi_short:
            return -1
        return 0

    def _signal_connors_rsi(self, sym):
        """
        Family 3: Connors RSI (2-Period) Mean Reversion (Expected 65-75% WR)
        Long:  RSI(2) < crsi_entry AND price above trend EMA AND price below exit SMA
        Short: RSI(2) > crsi_entry_short AND price below trend EMA AND price above exit SMA
        """
        crsi = float(self.crsi_ind[sym].current.value)
        price = float(self.latest_bar[sym].close)
        trend = float(self.trend_ema[sym].current.value)
        exit_sma = float(self.exit_sma[sym].current.value)

        if crsi < self.crsi_entry and price > trend and price < exit_sma:
            return 1
        if crsi > self.crsi_entry_short and price < trend and price > exit_sma:
            return -1
        return 0

    def _signal_ema_trend_adx(self, sym):
        """
        Family 4: EMA Trend Following + ADX (Expected 40-50% WR, larger wins)
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

        if adx < self.adx_min:
            return 0
        if adx <= prev_adx:
            return 0

        if fast > slow and prev_fast <= prev_slow:
            return 1
        if fast < slow and prev_fast >= prev_slow:
            return -1
        return 0

    def _signal_williams_rsi(self, sym):
        """
        Family 5: Williams %R + RSI Confluence (Expected 60-70% WR)
        Long:  Williams %R < wr_oversold AND RSI < wr_rsi_long
        Short: Williams %R > wr_overbought AND RSI > wr_rsi_short
        """
        wr = float(self.wilr_ind[sym].current.value)
        rsi = float(self.rsi_ind[sym].current.value)

        if wr < self.wr_oversold and rsi < self.wr_rsi_long:
            return 1
        if wr > self.wr_overbought and rsi > self.wr_rsi_short:
            return -1
        return 0

    # ──────────────────────────────────────────────
    #  ORDER MANAGEMENT
    # ──────────────────────────────────────────────

    def _enter_position(self, sym, signal):
        atr_value = float(self.atr_ind[sym].current.value)
        if atr_value <= 0:
            return

        stop_dist = self.stop_atr_mult * atr_value

        # Spread filter — skip if spread is too wide relative to stop
        ask = float(self.securities[sym].ask_price)
        bid = float(self.securities[sym].bid_price)
        if ask > 0 and bid > 0:
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

        # Handle synchronous fill (backtest mode)
        if ticket.status == OrderStatus.FILLED and not self.exits_placed[sym]:
            self._setup_exit_orders(
                sym,
                float(ticket.average_fill_price),
                atr_value,
                float(ticket.quantity_filled),
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
            f"{'BUY' if direction > 0 else 'SELL'} @ {fill_price:.2f} | "
            f"Stop: {stop_price:.2f} | TP: {tp_price:.2f} | "
            f"Qty: {abs(filled_qty):.6f}"
        )

    def _close_position(self, sym, reason="Liquidate"):
        had_position = self.portfolio[sym].invested

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

        bar = data.bars.get(sym)
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
                filled_qty = float(order_event.fill_quantity)
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
                f"Entry: {self.entry_price[sym]:.2f} | "
                f"Exit: {order_event.fill_price:.2f}"
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
                f"Entry: {self.entry_price[sym]:.2f} | "
                f"Exit: {order_event.fill_price:.2f}"
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

        # Crypto supports fractional quantities (e.g. 0.001 BTC)
        units = round(dollar_risk / loss_per_unit, 6)
        if units <= 0:
            return 0

        return units if signal > 0 else -units

    def _loss_per_unit_usd(self, sym, stop_dist):
        """
        For crypto pairs quoted in USDT (e.g. BTCUSDT, SOLUSDT),
        the loss per unit ≈ stop distance in USD terms.
        USDT is pegged ~1:1 with USD so this is effectively the same.
        """
        if stop_dist <= 0:
            return 0.0
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
