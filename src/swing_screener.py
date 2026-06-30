"""
Swing Trading Screener — Weekly & Monthly Timeframes
=====================================================
Designed for medium-term swing trades (5-30 day holding).
Resamples daily data into Weekly/Monthly OHLC candles.
Uses ATR-based stop-losses, trend confirmation, pullback entries.

SCORING FORMULA (100 pts):
    Trend Confirmation (30): Price above 50/200 SMA on resampled timeframe
    Pullback Quality (25): Distance from 20-SMA + bullish reversal candle
    Volatility / ATR (20): ATR-based risk/reward ratio favorability
    Volume Confirmation (15): Volume > previous period average
    Sector Rotation (10): Sector outperformance vs NIFTY-50 over 1 month
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Optional
import logging
import time
import math
import json
import os

from src.config import (
    NSE_UNIVERSE, get_yfinance_ticker,
    MIN_PRICE, MIN_VOLUME, MIN_MARKET_CAP
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── SAFETY UTILITIES ───────────────────────────────────────

def _safe_float(value, default=0.0):
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_round(value, ndigits=2, default=0.0):
    return round(_safe_float(value, default), ndigits)


# ─── NIFTY-50 TICKER FOR SECTOR ROTATION ────────────────────

NIFTY_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo Finance


# ─── RESAMPLING FUNCTIONS ───────────────────────────────────

def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily OHLC data to Weekly candles (Mon-Fri).
    Handles Indian market holidays by using W-FRI anchor.
    """
    if df.empty or len(df) < 5:
        return pd.DataFrame()

    # Ensure timezone-naive for reliable resampling
    if df.index.tz is not None:
        df = df.tz_localize(None)

    weekly = df.resample("W-FRI", closed="right", label="right").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })

    weekly = weekly.dropna(subset=["Close"])
    return weekly


def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily OHLC data to Monthly candles (last trading day of month).
    """
    if df.empty or len(df) < 20:
        return pd.DataFrame()

    if df.index.tz is not None:
        df = df.tz_localize(None)

    monthly = df.resample("ME", closed="right", label="right").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })

    monthly = monthly.dropna(subset=["Close"])
    return monthly


# ─── INDICATORS FOR RESAMPLED DATA ──────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range on already-resampled data."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    val = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    return _safe_float(val)


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_bullish_reversal_candle(row: pd.Series) -> bool:
    """
    Detect bullish reversal candle patterns:
    - Hammer: small body at top of range, long lower wick
    - Bullish Engulfing: current close > previous open AND current open < previous close
    - Piercing Line: opens below prev close, closes above 50% of prev body
    """
    open_p = _safe_float(row.get("Open", 0))
    high_p = _safe_float(row.get("High", 0))
    low_p = _safe_float(row.get("Low", 0))
    close_p = _safe_float(row.get("Close", 0))

    body = abs(close_p - open_p)
    total_range = high_p - low_p
    if total_range <= 0:
        return False

    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p

    # Hammer pattern: small real body, long lower wick (2x body), small upper wick
    is_hammer = (
        body > 0 and
        lower_wick >= 2 * body and
        upper_wick <= 0.3 * body and
        close_p > open_p
    )
    return is_hammer


def compute_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        "0.0": low,
        "23.6": low + 0.236 * diff,
        "38.2": low + 0.382 * diff,
        "50.0": low + 0.500 * diff,
        "61.8": low + 0.618 * diff,
        "78.6": low + 0.786 * diff,
        "100.0": high,
    }


# ─── SECTOR ROTATION ANALYZER ───────────────────────────────

class SectorRotationAnalyzer:
    """
    Tracks which sectors are outperforming NIFTY-50 over the last month.
    """

    @staticmethod
    def get_sector_performance() -> Dict[str, float]:
        """
        Returns sector -> 1-month performance relative to NIFTY.
        Positive = outperforming, Negative = underperforming.
        """
        try:
            nifty_ticker = yf.Ticker(NIFTY_TICKER)
            end = datetime.now()
            start = end - timedelta(days=45)
            nifty_hist = nifty_ticker.history(start=start, end=end)

            if len(nifty_hist) < 20:
                return {}

            nifty_close = nifty_hist["Close"]
            nifty_1m = ((nifty_close.iloc[-1] / nifty_close.iloc[-20]) - 1) * 100

            # Get a sample of sector ETFs/stocks to gauge sector rotation
            # Using prominent sector representatives on NSE
            sector_map = {
                "IT": ["TCS", "INFY", "WIPRO", "HCLTECH"],
                "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK"],
                "Automobile": ["TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO"],
                "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
                "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR"],
                "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "JINDALSTEL"],
                "Energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID"],
                "Financial Services": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE"],
            }

            sector_perf = {}
            for sector, tickers in sector_map.items():
                perf_values = []
                for t in tickers[:2]:  # Sample 2 per sector to save API calls
                    try:
                        stock = yf.Ticker(t + ".NS")
                        hist = stock.history(start=start, end=end)
                        if len(hist) >= 20:
                            perf = ((hist["Close"].iloc[-1] / hist["Close"].iloc[-20]) - 1) * 100
                            perf_values.append(perf)
                        time.sleep(0.3)
                    except:
                        pass
                if perf_values:
                    sector_perf[sector] = (np.mean(perf_values) - nifty_1m)

            return sector_perf

        except Exception as e:
            logger.warning(f"Sector rotation analysis failed: {e}")
            return {}

# ─── SWING SCORER ────────────────────────────────────────────

class SwingScorer:
    """
    Scores a single stock for swing trade suitability on resampled data.
    """

    def __init__(self):
        self.sector_rotator = SectorRotationAnalyzer()
        self._sector_perf_cache = None

    def _get_sector_perf(self) -> Dict[str, float]:
        if self._sector_perf_cache is None:
            self._sector_perf_cache = self.sector_rotator.get_sector_performance()
        return self._sector_perf_cache

    def score_swing(self, weekly_df: pd.DataFrame, info: Dict, ticker: str) -> Dict:
        """
        Complete swing scoring using resampled weekly data.
        Also runs monthly check if enough data exists.

        Scoring breakdown:
        - Trend Confirmation (30 pts): Above 20/50/200 SMA on resampled timeframe
        - Pullback Quality (25 pts): Near 20-SMA or Fib level + bullish reversal candle
        - ATR Risk/Reward (20 pts): Favorable R:R using 2xATR stop, 3xATR target
        - Volume Confirmation (15 pts): Volume expansion
        - Sector Rotation (10 pts): Sector leadership
        """
        if weekly_df.empty or len(weekly_df) < 20:
            return {
                "score": 0, "error": "Insufficient weekly data for swing analysis",
                "atr": 0, "smas": {}, "reasons": []
            }

        close = weekly_df["Close"]
        high = weekly_df["High"]
        low = weekly_df["Low"]
        volume = weekly_df["Volume"]

        current_price = _safe_float(close.iloc[-1])

        # ── 1. TREND CONFIRMATION (30 pts) ──
        sma_20 = compute_sma(close, 20)
        sma_50 = compute_sma(close, 50)
        sma_200 = None
        if len(close) >= 200:
            sma_200 = compute_sma(close, 200)

        trend_score = 0
        sma_20_val = _safe_float(sma_20.iloc[-1])
        sma_50_val = _safe_float(sma_50.iloc[-1])
        sma_200_val = _safe_float(sma_200.iloc[-1]) if sma_200 is not None and len(sma_200) > 0 else current_price

        reasons = []

        # Above 200 SMA = strong long-term uptrend (12 pts)
        if sma_200_val > 0 and current_price > sma_200_val:
            trend_score += 12
            reasons.append(f"Price above 200-SMA (long-term uptrend)")
        elif sma_200_val > 0:
            trend_score += 0  # Below 200 SMA = avoid
            reasons.append("Below 200-SMA, long-term trend uncertain")

        # Above 50 SMA = medium-term trend (10 pts)
        if current_price > sma_50_val:
            trend_score += 10
            reasons.append("Above 50-SMA (medium-term bullish)")
        else:
            trend_score += 3

        # 50 SMA > 200 SMA = golden cross / uptrend (8 pts)
        if sma_50_val > sma_200_val and sma_200_val > 0:
            trend_score += 8
            reasons.append("50-SMA > 200-SMA (uptrend confirmed)")

        # ── 2. PULLBACK QUALITY (25 pts) ──
        pullback_score = 0
        dist_from_20 = ((current_price - sma_20_val) / sma_20_val) * 100 if sma_20_val > 0 else 0

        # Near 20-SMA (within -2% to +2%) = ideal pullback entry
        if -2 <= dist_from_20 <= 2:
            pullback_score += 15
            reasons.append(f"Price near 20-SMA (pullback zone, {dist_from_20:.1f}%)")

        # Check Fibonacci retracement from recent swing high/low
        recent_high = _safe_float(high.tail(20).max())
        recent_low = _safe_float(low.tail(20).min())
        fib_levels = compute_fibonacci_levels(recent_high, recent_low)

        near_fib_382 = abs(current_price - fib_levels["38.2"]) / current_price < 0.02
        near_fib_50 = abs(current_price - fib_levels["50.0"]) / current_price < 0.02

        if near_fib_382:
            pullback_score += 5
            reasons.append("At 38.2% Fibonacci retracement level")
        elif near_fib_50:
            pullback_score += 5
            reasons.append("At 50% Fibonacci retracement level")

        # Bullish reversal candle on latest bar
        latest_candle = weekly_df.iloc[-1].to_dict()
        if compute_bullish_reversal_candle(latest_candle):
            pullback_score += 5
            reasons.append("Bullish reversal candle detected on weekly close")

        # ── 3. ATR RISK/REWARD (20 pts) ──
        atr_val = compute_atr(weekly_df)
        atr_score = 0

        if atr_val > 0 and current_price > 0:
            # Stop loss = 2 * ATR below
            sl_distance = 2 * atr_val
            sl_pct = (sl_distance / current_price) * 100

            # Target = 3 * ATR above
            tgt_distance = 3 * atr_val
            tgt_pct = (tgt_distance / current_price) * 100

            rr_ratio = tgt_distance / sl_distance if sl_distance > 0 else 0

            if rr_ratio >= 1.5:
                atr_score = 20
                reasons.append(f"ATR R:R = {rr_ratio:.2f} (favorable, stop={sl_pct:.1f}%, target={tgt_pct:.1f}%)")
            elif rr_ratio >= 1.2:
                atr_score = 15
            elif rr_ratio >= 1.0:
                atr_score = 10
            else:
                atr_score = 3
        else:
            atr_val = 0
            atr_score = 5

        # ── 4. VOLUME CONFIRMATION (15 pts) ──
        vol_score = 0
        if len(volume) >= 10:
            avg_vol = volume.tail(10).mean()
            current_vol = volume.iloc[-1]
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

            if vol_ratio >= 2.0:
                vol_score = 15
                reasons.append(f"Weekly volume {vol_ratio:.1f}x average (strong institutional interest)")
            elif vol_ratio >= 1.5:
                vol_score = 12
                reasons.append(f"Weekly volume {vol_ratio:.1f}x average (above normal)")
            elif vol_ratio >= 1.0:
                vol_score = 8
            else:
                vol_score = 3
        else:
            vol_ratio = 1.0
            vol_score = 5

        # ── 5. SECTOR ROTATION (10 pts) ──
        sector = info.get("sector", "N/A")
        sector_perf = self._get_sector_perf()
        sector_score = 5  # Neutral base

        if sector and sector in sector_perf:
            perf = sector_perf[sector]
            if perf > 3:
                sector_score = 10
                reasons.append(f"{sector} sector outperforming NIFTY by {perf:.1f}%")
            elif perf > 1:
                sector_score = 8
            elif perf < -2:
                sector_score = 2

        # ── COMPUTE ATR-BASED STOP & TARGETS ──
        if atr_val > 0 and current_price > 0:
            stop_loss = _safe_round(current_price - (2 * atr_val), 2)
            target_1 = _safe_round(current_price + (3 * atr_val), 2)  # ~5-10% move
            target_2 = _safe_round(current_price + (4 * atr_val), 2)  # ~10-15% move
        else:
            stop_loss = _safe_round(current_price * 0.95, 2)
            target_1 = _safe_round(current_price * 1.05, 2)
            target_2 = _safe_round(current_price * 1.10, 2)

        # Suggested holding period based on timeframe
        holding_period = "1-3 Weeks"  # Default for weekly

        total_score_raw = trend_score + pullback_score + atr_score + vol_score + sector_score

        return {
            "score": _safe_round(total_score_raw, 1),
            "trend_score": trend_score,
            "pullback_score": pullback_score,
            "atr_score": atr_score,
            "volume_score": vol_score,
            "sector_score": sector_score,
            "atr": _safe_round(atr_val, 2),
            "current_price": current_price,
            "smas": {
                "sma_20": _safe_round(sma_20_val, 2),
                "sma_50": _safe_round(sma_50_val, 2),
                "sma_200": _safe_round(sma_200_val, 2),
            },
            "dist_from_20sma_pct": _safe_round(dist_from_20, 1),
            "volume_ratio": _safe_round(vol_ratio if 'vol_ratio' in dir() else 1.0, 2),
            "fib_levels": fib_levels,
            "reasons": reasons[:5],
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "holding_period": holding_period,
            "risk_level": "Moderate" if pullback_score >= 10 else "Higher Risk",
        }


# ─── SWING SCREENER ─────────────────────────────────────────

class SwingScreener:
    """
    Full swing trading screener that scans NSE universe on Weekly/Monthly timeframes.
    """

    def __init__(self):
        self.scorer = SwingScorer()
        self.results_weekly = []
        self.results_monthly = []

    def _fetch_stock_data(self, ticker: str, lookback_days: int = 400) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Fetch daily NSE stock data for resampling.
        Need longer history (400+ days) for 200-SMA on weekly.
        """
        try:
            yf_ticker = get_yfinance_ticker(ticker)
            stock = yf.Ticker(yf_ticker)
            info = stock.info

            ltp = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            if not ltp or _safe_float(ltp) <= 0:
                stock = yf.Ticker(ticker)
                info = stock.info
                ltp = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                if not ltp or _safe_float(ltp) <= 0:
                    return None

            current_price = _safe_float(ltp)
            previous_close = _safe_float(info.get("previousClose", 0))
            if current_price <= 0:
                current_price = previous_close if previous_close > 0 else 0

            if current_price < MIN_PRICE:
                return None

            market_cap = _safe_float(info.get("marketCap", 0))
            if market_cap < MIN_MARKET_CAP:
                return None

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty or len(hist) < 200:
                # Need at least 200 daily bars for meaningful resampling
                return None

            return hist, info

        except Exception as e:
            logger.debug(f"Error fetching NSE {ticker}: {e}")
            return None

    def screen_swing_weekly(self, max_stocks: int = 200) -> List[Dict]:
        """
        Run swing screening on Weekly candles.
        """
        all_scored = []
        universe = NSE_UNIVERSE[:max_stocks]  # Limit to conserve API calls

        logger.info(f"Starting Swing Weekly screening of {len(universe)} NSE stocks...")

        for i, ticker in enumerate(universe):
            if i > 0 and i % 50 == 0:
                logger.info(f"Processed {i}/{len(universe)}...")
                time.sleep(2)

            result = self._fetch_stock_data(ticker, lookback_days=400)
            if result is None:
                continue

            hist, info = result
            weekly_df = resample_to_weekly(hist)

            if weekly_df.empty or len(weekly_df) < 30:
                logger.debug(f"{ticker}: Insufficient weekly bars ({len(weekly_df)})")
                continue

            swing_data = self.scorer.score_swing(weekly_df, info, ticker)

            if swing_data.get("score", 0) < 30:
                continue  # Filter weak scores

            entry_price = _safe_float(info.get("currentPrice") or info.get("previousClose", 0))
            if entry_price <= 0:
                entry_price = _safe_float(weekly_df["Close"].iloc[-1])

            pick = {
                "ticker": ticker,
                "company_name": info.get("longName", info.get("shortName", ticker)),
                "current_price": _safe_round(entry_price, 2),
                "previous_close": _safe_round(info.get("previousClose", weekly_df["Close"].iloc[-2] if len(weekly_df) > 1 else weekly_df["Close"].iloc[-1]), 2),
                "market_cap": _safe_float(info.get("marketCap", 0)),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "pe_ratio": _safe_float(info.get("trailingPE") or info.get("forwardPE") or 0),
                "swing_score": swing_data["score"],
                "trend_score": swing_data.get("trend_score", 0),
                "pullback_score": swing_data.get("pullback_score", 0),
                "atr_score": swing_data.get("atr_score", 0),
                "volume_score": swing_data.get("volume_score", 0),
                "sector_rot_score": swing_data.get("sector_score", 0),
                "atr": swing_data.get("atr", 0),
                "sma_20": swing_data["smas"]["sma_20"],
                "sma_50": swing_data["smas"]["sma_50"],
                "sma_200": swing_data["smas"]["sma_200"],
                "dist_from_20sma": swing_data.get("dist_from_20sma_pct", 0),
                "volume_ratio": swing_data.get("volume_ratio", 1.0),
                "reasons": swing_data.get("reasons", []),
                "entry_zone": _safe_round(swing_data["smas"]["sma_20"], 2),
                "stop_loss": swing_data.get("stop_loss", _safe_round(entry_price * 0.95, 2)),
                "target_1": swing_data.get("target_1", _safe_round(entry_price * 1.05, 2)),
                "target_2": swing_data.get("target_2", _safe_round(entry_price * 1.10, 2)),
                "holding_period": swing_data.get("holding_period", "1-3 Weeks"),
                "risk_level": swing_data.get("risk_level", "Moderate"),
                "timeframe": "Weekly",
            }

            all_scored.append(pick)

            if i % 10 == 0:
                time.sleep(0.5)

        # Sort by swing score
        all_scored.sort(key=lambda x: x["swing_score"], reverse=True)
        self.results_weekly = all_scored[:10]

        logger.info(f"Swing Weekly screening complete. Top {len(self.results_weekly)} picks.")
        return self.results_weekly

    def screen_swing_monthly(self, max_stocks: int = 200) -> List[Dict]:
        """
        Run swing screening on Monthly candles.
        """
        all_scored = []
        universe = NSE_UNIVERSE[:max_stocks]

        logger.info(f"Starting Swing Monthly screening of {len(universe)} NSE stocks...")

        for i, ticker in enumerate(universe):
            if i > 0 and i % 50 == 0:
                logger.info(f"Processed {i}/{len(universe)}...")
                time.sleep(2)

            result = self._fetch_stock_data(ticker, lookback_days=800)
            if result is None:
                continue

            hist, info = result
            monthly_df = resample_to_monthly(hist)

            if monthly_df.empty or len(monthly_df) < 12:
                logger.debug(f"{ticker}: Insufficient monthly bars ({len(monthly_df)})")
                continue

            swing_data = self.scorer.score_swing(monthly_df, info, ticker)

            if swing_data.get("score", 0) < 25:
                continue

            entry_price = _safe_float(info.get("currentPrice") or info.get("previousClose", 0))
            if entry_price <= 0:
                entry_price = _safe_float(monthly_df["Close"].iloc[-1])

            pick = {
                "ticker": ticker,
                "company_name": info.get("longName", info.get("shortName", ticker)),
                "current_price": _safe_round(entry_price, 2),
                "previous_close": _safe_round(info.get("previousClose", monthly_df["Close"].iloc[-2] if len(monthly_df) > 1 else monthly_df["Close"].iloc[-1]), 2),
                "market_cap": _safe_float(info.get("marketCap", 0)),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "pe_ratio": _safe_float(info.get("trailingPE") or info.get("forwardPE") or 0),
                "swing_score": swing_data["score"],
                "trend_score": swing_data.get("trend_score", 0),
                "pullback_score": swing_data.get("pullback_score", 0),
                "atr_score": swing_data.get("atr_score", 0),
                "volume_score": swing_data.get("volume_score", 0),
                "sector_rot_score": swing_data.get("sector_score", 0),
                "atr": swing_data.get("atr", 0),
                "sma_20": swing_data["smas"]["sma_20"],
                "sma_50": swing_data["smas"]["sma_50"],
                "sma_200": swing_data["smas"]["sma_200"],
                "dist_from_20sma": swing_data.get("dist_from_20sma_pct", 0),
                "volume_ratio": swing_data.get("volume_ratio", 1.0),
                "reasons": swing_data.get("reasons", []),
                "entry_zone": _safe_round(swing_data["smas"]["sma_20"], 2),
                "stop_loss": swing_data.get("stop_loss", _safe_round(entry_price * 0.92, 2)),
                "target_1": swing_data.get("target_1", _safe_round(entry_price * 1.10, 2)),
                "target_2": swing_data.get("target_2", _safe_round(entry_price * 1.20, 2)),
                "holding_period": "1-2 Months",
                "risk_level": swing_data.get("risk_level", "Moderate"),
                "timeframe": "Monthly",
            }

            all_scored.append(pick)

            if i % 10 == 0:
                time.sleep(0.5)

        all_scored.sort(key=lambda x: x["swing_score"], reverse=True)
        self.results_monthly = all_scored[:7]

        logger.info(f"Swing Monthly screening complete. Top {len(self.results_monthly)} picks.")
        return self.results_monthly

    def run_full_swing_screen(self) -> Dict[str, List[Dict]]:
        """Run both weekly and monthly screens."""
        weekly = self.screen_swing_weekly()
        monthly = self.screen_swing_monthly()
        return {"weekly": weekly, "monthly": monthly}


# ─── PERSISTENCE UTILITIES ──────────────────────────────────

def save_swing_picks(picks: List[Dict], timeframe: str):
    """Save swing picks to JSON file for offline loading."""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    filepath = f"{reports_dir}/swing_{timeframe.lower()}_picks.json"

    output = {
        "date": date.today().isoformat(),
        "generated_at": datetime.now().isoformat(),
        "timeframe": timeframe,
        "total_analyzed": "200 NSE stocks",
        "top_picks": picks,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved {len(picks)} {timeframe} swing picks to {filepath}")


def load_swing_picks(timeframe: str) -> List[Dict]:
    """Load swing picks from JSON file."""
    filepath = f"reports/swing_{timeframe.lower()}_picks.json"
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            return data.get("top_picks", [])
    except:
        return []


# ─── MAIN RUNNER ────────────────────────────────────────────

def run_swing_pipeline(timeframe: str = "both"):
    """
    Run the swing screening pipeline.
    Called by GitHub Actions on schedule (weekly/monthly).
    """
    screener = SwingScreener()

    if timeframe in ("weekly", "both"):
        try:
            weekly_picks = screener.screen_swing_weekly()
            save_swing_picks(weekly_picks, "weekly")
            logger.info(f"Saved {len(weekly_picks)} weekly swing picks")
        except Exception as e:
            logger.error(f"Weekly swing pipeline failed: {e}")

    if timeframe in ("monthly", "both"):
        try:
            monthly_picks = screener.screen_swing_monthly()
            save_swing_picks(monthly_picks, "monthly")
            logger.info(f"Saved {len(monthly_picks)} monthly swing picks")
        except Exception as e:
            logger.error(f"Monthly swing pipeline failed: {e}")

    return True