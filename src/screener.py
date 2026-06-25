"""
NSE Stock Screening & Scoring Engine
======================================
Daily technical + fundamental screener for National Stock Exchange of India.
Scans 500+ NSE stocks, scores them using a multi-factor algorithm,
and returns the Top 10 daily picks with entry/exit suggestions.

SCORING FORMULA:
    Total Score = Technical (65) + Fundamental (25) + Sentiment (10) = 100
    
    Technical: RSI (15) + MACD (15) + Volume Spike (15) + MA Cross (10) + Breakout (10)
    Fundamental: P/E (10) + Market Cap (5) + Volume Ratio (10)
    Sentiment: News Sentiment (5) + Relative Strength (5)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
import time

from src.config import (
    NSE_UNIVERSE, get_yfinance_ticker, WEIGHTS,
    MIN_PRICE, MIN_VOLUME, MIN_MARKET_CAP,
    LOOKBACK_DAYS, SCREENING_PERIOD
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Computes all technical indicators used in scoring.
    """
    
    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> float:
        """Relative Strength Index with Wilder's smoothing."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Wilder's smoothing
        for i in range(period, len(avg_gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    @staticmethod
    def compute_macd(series: pd.Series) -> Dict:
        """MACD: Moving Average Convergence Divergence."""
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd_line": macd_line.iloc[-1],
            "signal_line": signal_line.iloc[-1],
            "histogram": histogram.iloc[-1],
            "histogram_prev": histogram.iloc[-2] if len(histogram) > 1 else 0,
            "is_bullish": macd_line.iloc[-1] > signal_line.iloc[-1],
            "crossover_up": (histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0) if len(histogram) > 1 else False
        }

    @staticmethod
    def compute_bollinger_bands(series: pd.Series, period: int = 20) -> Dict:
        """Bollinger Bands for volatility/breakout detection."""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        current_price = series.iloc[-1]
        bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if upper.iloc[-1] != lower.iloc[-1] else 0.5
        
        return {
            "upper": upper.iloc[-1],
            "lower": lower.iloc[-1],
            "middle": sma.iloc[-1],
            "position": bb_position,  # 0=lower, 1=upper
            "squeeze": (upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1]
        }

    @staticmethod
    def compute_volume_spike(volume_series: pd.Series, period: int = 20) -> Dict:
        """Detect abnormal volume activity."""
        avg_volume = volume_series.tail(period).mean()
        current_volume = volume_series.iloc[-1]
        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return {
            "current_volume": current_volume,
            "avg_volume_20d": avg_volume,
            "ratio": ratio,
            "is_spike": ratio > 1.5
        }

    @staticmethod
    def compute_moving_averages(df: pd.DataFrame) -> Dict:
        """SMA crossovers and price relative to MAs."""
        close = df["Close"]
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean() if len(close) >= 200 else pd.Series(index=close.index)
        
        current_price = close.iloc[-1]
        
        return {
            "sma_20": sma_20.iloc[-1],
            "sma_50": sma_50.iloc[-1],
            "sma_200": sma_200.iloc[-1] if len(sma_200) > 0 else current_price,
            "price_above_sma_20": current_price > sma_20.iloc[-1],
            "price_above_sma_50": current_price > sma_50.iloc[-1],
            "price_above_sma_200": current_price > sma_200.iloc[-1] if len(sma_200) > 0 else True,
            "sma_50_above_200": sma_50.iloc[-1] > sma_200.iloc[-1] if len(sma_200) > 0 else True,
            "golden_cross_detected": (
                sma_50.iloc[-1] > sma_50.iloc[-2] and 
                sma_200.iloc[-1] > sma_200.iloc[-2] and
                sma_50.iloc[-1] > sma_200.iloc[-1]
            ) if len(sma_200) > 0 else False
        }

    @staticmethod
    def compute_support_resistance(df: pd.DataFrame, lookback: int = 60) -> Dict:
        """Identify key support/resistance levels using pivot points."""
        close = df["Close"].tail(lookback)
        high = df["High"].tail(lookback)
        low = df["Low"].tail(lookback)
        
        current_price = close.iloc[-1]
        
        # Find pivot highs and lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(2, len(close) - 2):
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and \
               high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
                pivot_highs.append(high.iloc[i])
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and \
               low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
                pivot_lows.append(low.iloc[i])
        
        # Nearest resistance (above current price)
        resistances = [p for p in pivot_highs if p > current_price]
        nearest_resistance = min(resistances) if resistances else current_price * 1.05
        
        # Nearest support (below current price)
        supports = [p for p in pivot_lows if p < current_price]
        nearest_support = max(supports) if supports else current_price * 0.95
        
        # Distance to resistance/support as %
        dist_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
        dist_to_support = ((current_price - nearest_support) / current_price) * 100
        
        return {
            "nearest_resistance": nearest_resistance,
            "nearest_support": nearest_support,
            "dist_to_resistance_pct": dist_to_resistance,
            "dist_to_support_pct": dist_to_support,
            "has_near_resistance": dist_to_resistance < 3.0,
            "has_near_support": dist_to_support < 3.0
        }


class StockScorer:
    """
    Multi-factor scoring system that computes a composite score
    for each stock. Higher score = better daily pick.
    """
    
    def __init__(self):
        self.weights = WEIGHTS
        self.analyzer = TechnicalAnalyzer()
    
    def score_technical(self, df: pd.DataFrame) -> Dict:
        """
        Technical Scoring (50 pts max)
        """
        close = df["Close"]
        volume = df["Volume"]
        
        # --- RSI Momentum Score (15 pts) ---
        rsi = self.analyzer.compute_rsi(close)
        if 50 <= rsi <= 70:
            rsi_score = 15  # Sweet spot: bullish momentum, not overbought
        elif 40 <= rsi < 50:
            rsi_score = 10  # Neutral to slightly oversold
        elif 70 < rsi <= 80:
            rsi_score = 8   # Still bullish but caution
        elif rsi > 80:
            rsi_score = 3   # Overbought - risk of reversal
        elif 30 <= rsi < 40:
            rsi_score = 5   # Oversold - potential bounce
        else:
            rsi_score = 1   # Extremes
        
        # --- MACD Score (15 pts) ---
        macd_data = self.analyzer.compute_macd(close)
        macd_score = 0
        if macd_data["is_bullish"]:
            macd_score += 8
            if macd_data["crossover_up"]:
                macd_score += 7  # Bonus for fresh crossover
            else:
                macd_score += 4  # Already in bullish territory
        elif macd_data["histogram"] > macd_data["histogram_prev"]:
            macd_score += 5  # Improving momentum
        else:
            macd_score += 2
        
        # --- Volume Spike Score (15 pts) ---
        vol_data = self.analyzer.compute_volume_spike(volume)
        if vol_data["ratio"] >= 2.0:
            vol_score = 15  # Major volume surge
        elif vol_data["ratio"] >= 1.5:
            vol_score = 12  # Significant volume
        elif vol_data["ratio"] >= 1.2:
            vol_score = 8   # Above average
        elif vol_data["ratio"] >= 0.8:
            vol_score = 5   # Normal
        else:
            vol_score = 2   # Low volume
        
        # --- MA Crossover Score (10 pts) ---
        ma_data = self.analyzer.compute_moving_averages(df)
        ma_score = 0
        if ma_data["golden_cross_detected"]:
            ma_score += 10  # Golden cross = strong bullish
        elif ma_data["sma_50_above_200"] and ma_data["price_above_sma_50"]:
            ma_score += 8   # Confirmed uptrend
        elif ma_data["price_above_sma_20"]:
            ma_score += 5   # Short-term bullish
        elif ma_data["price_above_sma_50"]:
            ma_score += 3
        else:
            ma_score += 1
        
        # --- Gap-up Potential Score (10 pts) ---
        # Looking for tight consolidation near resistance breakout
        bb_data = self.analyzer.compute_bollinger_bands(close)
        sr_data = self.analyzer.compute_support_resistance(df)
        
        gap_score = 5  # Base score
        if 0.7 <= bb_data["position"] <= 0.9:
            gap_score += 3  # Near upper band but not overextended
        if sr_data["has_near_resistance"]:
            gap_score += 2  # Near resistance = breakout potential
        
        technical_scores = {
            "rsi_value": rsi,
            "rsi_score": rsi_score,
            "macd_score": macd_score,
            "volume_score": vol_score,
            "volume_ratio": vol_data["ratio"],
            "ma_score": ma_score,
            "gap_score": gap_score,
            "total_technical": rsi_score + macd_score + vol_score + ma_score + gap_score,
            "max_technical": 65
        }
        
        return technical_scores
    
    def score_fundamental(self, info: Dict) -> Dict:
        """
        Fundamental Scoring (25 pts max)
        Uses yfinance stock info data.
        """
        # --- P/E Relative to Sector (10 pts) ---
        pe = info.get("trailingPE") or info.get("forwardPE") or 0
        sector_pe = info.get("sector")  # Placeholder - we'd compare to sector median
        market_cap = info.get("marketCap", 0)
        
        if pe > 0:
            if pe < 15:
                pe_score = 10  # Value
            elif pe < 25:
                pe_score = 7   # Moderate
            elif pe < 40:
                pe_score = 4   # Growth
            elif pe < 60:
                pe_score = 2
            else:
                pe_score = 0   # Extremely high
        else:
            pe_score = 3  # No data / negative earnings
        
        # --- Market Cap Score (5 pts) ---
        if market_cap >= 100_000_000_000:  # Mega cap
            cap_score = 5
        elif market_cap >= 10_000_000_000:  # Large cap
            cap_score = 4
        elif market_cap >= 2_000_000_000:  # Mid cap
            cap_score = 3
        elif market_cap >= MIN_MARKET_CAP:
            cap_score = 2
        else:
            cap_score = 0  # Below threshold, should be filtered
        
        # --- Volume Ratio Score (10 pts) ---
        avg_volume_90d = info.get("averageVolume", 0)
        volume = info.get("volume", 0)
        if market_cap > 0 and avg_volume_90d > 0:
            # How many times does the entire float turn over?
            # Higher = more liquidity and attention
            vol_ratio = volume / market_cap
            if vol_ratio > 0.005:  # 0.5% of market cap trading hands
                vol_ratio_score = 10
            elif vol_ratio > 0.002:
                vol_ratio_score = 7
            elif vol_ratio > 0.001:
                vol_ratio_score = 5
            else:
                vol_ratio_score = 2
        else:
            vol_ratio_score = 2
        
        fundamental_scores = {
            "pe_ratio": pe,
            "pe_score": pe_score,
            "market_cap": market_cap,
            "cap_score": cap_score,
            "volume_ratio_score": vol_ratio_score,
            "total_fundamental": pe_score + cap_score + vol_ratio_score,
            "max_fundamental": 25
        }
        
        return fundamental_scores
    
    def compute_full_score(self, df: pd.DataFrame, info: Dict) -> Tuple[float, Dict]:
        """
        Compute the complete composite score.
        
        Returns:
            Tuple of (total_score, breakdown_dict)
        """
        tech_scores = self.score_technical(df)
        fund_scores = self.score_fundamental(info)
        
        total_score = (
            tech_scores["total_technical"] +
            fund_scores["total_fundamental"]
            # Sentiment will be added via Finnhub later
        )
        
        # Normalize to 0-100 scale
        # Raw max = 65 (technical) + 25 (fundamental) = 90
        # We leave room for sentiment (10 pts) = 100 total
        normalized_score = min(100, (total_score / 90) * 100)
        
        breakdown = {
            "total_score": round(normalized_score, 1),
            "raw_score": total_score,
            "technical": tech_scores,
            "fundamental": fund_scores,
            "rsi": round(tech_scores.get("rsi_value", 0), 1),
            "volume_ratio": round(tech_scores.get("volume_ratio", 0), 2),
        }
        
        return normalized_score, breakdown


class MarketScreener:
    """
    Main screener that scans the market universe and returns
    the Top 10 picks.
    """
    
    def __init__(self):
        self.scorer = StockScorer()
        self.results = []
    
    def _fetch_stock_data(self, ticker: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Fetch NSE stock data for a single ticker.
        Uses .NS suffix for yfinance (Yahoo Finance NSE format).
        Returns (historical_data, info) or None if invalid.
        """
        try:
            yf_ticker = get_yfinance_ticker(ticker)
            stock = yf.Ticker(yf_ticker)
            
            # Get info
            info = stock.info
            
            # Fast fail for invalid tickers
            if not info.get("regularMarketPrice") and not info.get("currentPrice"):
                # Try without .NS suffix as fallback (some stocks work both ways)
                stock = yf.Ticker(ticker)
                info = stock.info
                if not info.get("regularMarketPrice") and not info.get("currentPrice"):
                    return None
            
            # Get price
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
            
            # FILTER: Penny stocks (threshold in INR)
            if current_price < MIN_PRICE:
                return None
            
            # FILTER: Market cap (in INR)
            market_cap = info.get("marketCap", 0)
            if market_cap < MIN_MARKET_CAP:
                return None
            
            # FILTER: Volume
            volume = info.get("volume", 0)
            if volume < MIN_VOLUME:
                return None
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=LOOKBACK_DAYS + 60)  # Extra buffer
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty or len(hist) < 20:
                return None
            
            return hist, info
            
        except Exception as e:
            logger.debug(f"Error fetching NSE {ticker}: {e}")
            return None
    
    def screen_market(self) -> List[Dict]:
        """
        Run the full screening pipeline.
        Returns list of scored stocks, sorted by score descending.
        """
        all_scored = []
        errors = []
        
        universe = list(set(NSE_UNIVERSE))  # Deduplicate NSE stocks
        logger.info(f"Starting NSE screening of {len(universe)} stocks...")
        
        for i, ticker in enumerate(universe):
            if i > 0 and i % 50 == 0:
                logger.info(f"Processed {i}/{len(universe)} stocks...")
                time.sleep(2)  # Rate limit buffer
            
            result = self._fetch_stock_data(ticker)
            if result is None:
                continue
            
            hist, info = result
            score, breakdown = self.scorer.compute_full_score(hist, info)
            
            stock_data = {
                "ticker": ticker,
                "company_name": info.get("longName", info.get("shortName", ticker)),
                "current_price": round(info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1], 2),
                "previous_close": round(info.get("previousClose", hist["Close"].iloc[-2] if len(hist) > 1 else hist["Close"].iloc[-1]), 2),
                "change_pct": round(info.get("regularMarketChangePercent", 0) or 0, 2),
                "volume": info.get("volume", 0),
                "avg_volume_90d": info.get("averageVolume", 0),
                "market_cap": info.get("marketCap", 0),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE") or 0,
                "score": score,
                "breakdown": breakdown,
                "rsi": breakdown.get("rsi", 50),
                "volume_ratio_20d": breakdown.get("volume_ratio", 1.0),
                "technicals": self._get_technical_summary(hist),
            }
            
            all_scored.append(stock_data)
            
            if i % 10 == 0:
                time.sleep(0.5)  # Prevent rate limiting
        
        # Sort by score descending
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top 20 (we'll refine to 10 with sentiment)
        self.results = all_scored[:20]
        
        logger.info(f"Screening complete. Found {len(all_scored)} qualifying stocks.")
        return self.results
    
    def _get_technical_summary(self, df: pd.DataFrame) -> Dict:
        """Generate a human-readable technical summary."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        
        return {
            "high_20d": round(high.tail(20).max(), 2),
            "low_20d": round(low.tail(20).min(), 2),
            "close_20d_ago": round(close.iloc[-20], 2) if len(close) >= 20 else round(close.iloc[0], 2),
            "price_pct_5d": round(((close.iloc[-1] / close.iloc[-5]) - 1) * 100, 2) if len(close) >= 5 else 0,
            "price_pct_20d": round(((close.iloc[-1] / close.iloc[-20]) - 1) * 100, 2) if len(close) >= 20 else 0,
        }
    
    def get_top_10_picks(self, news_sentiment: Dict[str, float] = None) -> List[Dict]:
        """
        Refine top 20 to top 10 using sentiment analysis.
        
        Args:
            news_sentiment: Dict mapping ticker -> sentiment score (0-1)
        """
        if not self.results:
            self.screen_market()
        
        top_picks = self.results[:20].copy()
        
        # Apply sentiment boost if available
        if news_sentiment:
            for pick in top_picks:
                sentiment = news_sentiment.get(pick["ticker"], 0.5)
                # Sentiment adjustment: +/- 5 points
                sentiment_boost = (sentiment - 0.5) * 10
                pick["score"] += sentiment_boost
                pick["sentiment"] = sentiment
        
        # Re-sort
        top_picks.sort(key=lambda x: x["score"], reverse=True)
        
        # Return exactly 10
        final_picks = top_picks[:10]
        
        # Generate rationales
        for pick in final_picks:
            pick["rationale"] = self._generate_rationale(pick)
        
        return final_picks
    
    def _generate_rationale(self, stock: Dict) -> Dict:
        """
        Generate a professional buy/sell rationale for each pick.
        """
        bd = stock.get("breakdown", {})
        tech = bd.get("technical", {})
        fund = bd.get("fundamental", {})
        ts = stock.get("technicals", {})
        
        # Build rationale components
        reasons = []
        
        # Technical reasons
        if tech.get("rsi_score", 0) >= 12:
            reasons.append(f"RSI at {stock.get('rsi', 50):.1f} (bullish momentum zone)")
        elif tech.get("rsi_score", 0) >= 8:
            reasons.append("RSI showing improving momentum")
        
        if tech.get("macd_score", 0) >= 12:
            reasons.append("MACD bullish crossover confirmed")
        elif tech.get("macd_score", 0) >= 8:
            reasons.append("MACD in positive territory")
        
        if tech.get("volume_score", 0) >= 12:
            reasons.append(f"Volume spike at {stock.get('volume_ratio_20d', 1):.1f}x average")
        
        if tech.get("ma_score", 0) >= 8:
            reasons.append("Price above key moving averages (uptrend confirmed)")
        elif tech.get("ma_score", 0) >= 5:
            reasons.append("Price above 20-day SMA (short-term bullish)")
        
        # Fundamental reasons
        if fund.get("pe_score", 0) >= 7:
            reasons.append(f"P/E ratio of {stock.get('pe_ratio', 'N/A')} is attractive")
        
        if fund.get("volume_ratio_score", 0) >= 7:
            reasons.append("High relative volume to market cap (strong interest)")
        
        # Price action reasons
        if ts.get("price_pct_5d", 0) > 0:
            reasons.append(f"Up {ts['price_pct_5d']:.1f}% in last 5 days")
        if ts.get("price_pct_20d", 0) > 5:
            reasons.append(f"Strong {ts['price_pct_20d']:.1f}% gain over 20 days")
        
        # Entry/exit suggestions
        entry_suggestion = "Market open" 
        if stock.get("rsi", 50) < 40:
            entry_suggestion = "Look for dip buy near support"
        elif stock.get("rsi", 50) > 70:
            entry_suggestion = "Wait for pullback to 20-day SMA"
        
        stop_loss = round(stock["current_price"] * 0.97, 2)
        target_price = round(stock["current_price"] * 1.03, 2)
        
        return {
            "bullish_reasons": reasons[:4],  # Top 4 reasons
            "entry_suggestion": entry_suggestion,
            "suggested_stop_loss": stop_loss,
            "suggested_target": target_price,
            "risk_level": "Moderate" if stock.get("rsi", 50) < 70 else "Higher Risk",
            "sector": stock.get("sector", "N/A"),
            "market_cap_formatted": self._format_market_cap(stock.get("market_cap", 0)),
            "score_breakdown": f"Tech: {tech.get('total_technical', 0):.0f}/65 + Fund: {fund.get('total_fundamental', 0):.0f}/25"
        }
    
    @staticmethod
    def _format_market_cap(cap: float) -> str:
        """Format market cap in Indian numbering (Crore/Cr)."""
        crore = cap / 10_000_000
        if crore >= 100_000:
            return f"₹{crore/100_000:.2f} Lakh Cr"
        elif crore >= 1_000:
            return f"₹{crore/1_000:.2f}K Cr"
        elif crore >= 1:
            return f"₹{crore:.2f} Cr"
        else:
            lakh = cap / 100_000
            return f"₹{lakh:.2f} Lakh"


if __name__ == "__main__":
    # Quick test with NSE stocks
    screener = MarketScreener()
    print("Running NSE market screen...")
    top10 = screener.get_top_10_picks()
    print(f"\n=== TOP 10 NSE PICKS ===")
    for i, pick in enumerate(top10, 1):
        print(f"\n{'-'*50}")
        print(f"#{i}: {pick['ticker']} - Rs.{pick['current_price']} | Score: {pick['score']:.1f}/100")
        print(f"   {pick['company_name']}")
        print(f"   RSI: {pick['rsi']:.1f} | Vol Ratio: {pick['volume_ratio_20d']:.2f}x")
        if 'rationale' in pick:
            print(f"   Entry: {pick['rationale']['entry_suggestion']}")
            print(f"   Stop: Rs.{pick['rationale']['suggested_stop_loss']} | Target: Rs.{pick['rationale']['suggested_target']}")
            for reason in pick['rationale']['bullish_reasons']:
                print(f"   [OK] {reason}")
