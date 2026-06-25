"""
Streamlit Dashboard — Daily Top 10 NSE Stock Picks
====================================================
Interactive dashboard for viewing, analyzing, and downloading
daily NSE stock picks with charts, rationales, and PDF reports.

Supports NSE (National Stock Exchange of India) stocks.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date, timedelta
import os
import sys
import io
import json
from fpdf import FPDF

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.screener import MarketScreener
from src.database import PickDatabase
from src.pipeline import run_daily_pipeline

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Daily Top 10 NSE Stock Picks",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .score-high {
        color: #00cc66;
        font-weight: bold;
    }
    .score-mid {
        color: #ffaa00;
        font-weight: bold;
    }
    .score-low {
        color: #ff4444;
        font-weight: bold;
    }
    .card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #00cc66;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .card-moderate {
        border-left-color: #ffaa00;
    }
    .card-risk {
        border-left-color: #ff4444;
    }
    .stButton button {
        background-color: #1a1a2e;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_score_color(score: float) -> str:
    if score >= 70:
        return "score-high"
    elif score >= 50:
        return "score-mid"
    else:
        return "score-low"

def format_market_cap_nse(cap: float) -> str:
    crore = cap / 10_000_000
    if crore >= 100_000:
        return f"\u20B9{crore/100_000:.2f} Lakh Cr"
    elif crore >= 1_000:
        return f"\u20B9{crore/1_000:.2f}K Cr"
    elif crore >= 1:
        return f"\u20B9{crore:.2f} Cr"
    else:
        lakh = cap / 100_000
        return f"\u20B9{lakh:.2f} Lakh"

def format_volume_nse(vol: float) -> str:
    if vol >= 100_000_000:
        return f"{vol/100_000_000:.2f} Cr"
    elif vol >= 1_000_000:
        return f"{vol/1_000_000:.2f} L"
    elif vol >= 1_000:
        return f"{vol/1_000:.1f}K"
    else:
        return str(int(vol))

@st.cache_data(ttl=300)
def get_cached_picks() -> list:
    db = PickDatabase()
    today_picks = db.get_daily_picks(date.today())
    if today_picks:
        return today_picks
    yesterday = date.today() - timedelta(days=1)
    yesterday_picks = db.get_daily_picks(yesterday)
    if yesterday_picks:
        return yesterday_picks
    reports_dir = "reports"
    if os.path.exists(reports_dir):
        files = sorted(os.listdir(reports_dir))
        json_files = [f for f in files if f.endswith(".json") and f.startswith("picks_")]
        if json_files:
            latest = json_files[-1]
            with open(os.path.join(reports_dir, latest), "r") as f:
                data = json.load(f)
                return data.get("top_picks", [])
    return []


def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">:chart_with_upwards_trend: Daily Top 10 NSE Picks</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="sub-header">'
            f'NSE India stock screening &bull; {date.today().strftime("%A, %B %d, %Y")}'
            f'</p>',
            unsafe_allow_html=True
        )
    with col2:
        if st.button(":arrows_counterclockwise: Run Fresh Screen", use_container_width=True):
            with st.spinner("Screening 300+ NSE stocks... Takes ~60 sec..."):
                success = run_daily_pipeline()
                if success:
                    st.success("Pipeline complete! Data updated.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Pipeline failed. Check logs.")

def render_pick_card(pick: dict, rank: int):
    score = pick.get("score", 0)
    score_color = get_score_color(score)
    rationale = pick.get("rationale", {})
    if not rationale and "entry_suggestion" in pick:
        rationale = pick
    risk = rationale.get("risk_level", "Moderate")
    card_class = "card"
    if risk == "Higher Risk":
        card_class = "card card-risk"
    elif risk == "Moderate":
        card_class = "card card-moderate"

    with st.container():
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        cols = st.columns([0.5, 1.5, 1.5, 1, 1.5, 1.5])
        with cols[0]:
            st.markdown(f"**#{rank}**")
        with cols[1]:
            st.markdown(f"**{pick.get('ticker', 'N/A')}**  \n{pick.get('company_name', '')[:35]}")
        with cols[2]:
            change = pick.get("change_pct", 0)
            change_str = f"+{change}%" if change > 0 else f"{change}%"
            st.markdown(f"**\u20B9{pick.get('current_price', 0):.2f}**  \n{change_str}")
        with cols[3]:
            st.markdown(f'<span class="{score_color}">{score:.1f}</span>  \nScore', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f"**{pick.get('rsi', 50):.1f}**  \nRSI")
        with cols[5]:
            st.markdown(f"**{rationale.get('entry_suggestion', 'Market')}**  \nEntry")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander(f":bar_chart: Full Analysis - {pick.get('ticker', 'N/A')}"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(":dart: Action Plan")
            reasons = rationale.get("bullish_reasons", [])
            for reason in reasons:
                st.markdown(f":white_check_mark: {reason}")
            st.markdown("---")
            entry_cols = st.columns(3)
            with entry_cols[0]:
                st.metric("Entry Strategy", rationale.get("entry_suggestion", "Market Open"))
            with entry_cols[1]:
                sl_val = rationale.get("suggested_stop_loss", 0)
                pct_down = ((pick.get('current_price', 0) - sl_val) / pick.get('current_price', 1) * 100) if pick.get('current_price', 0) > 0 else 0
                st.metric("Suggested Stop Loss", f"\u20B9{sl_val:.2f}", delta=f"-{pct_down:.1f}%")
            with entry_cols[2]:
                tgt_val = rationale.get("suggested_target", 0)
                pct_up = ((tgt_val - pick.get('current_price', 0)) / pick.get('current_price', 1) * 100) if pick.get('current_price', 0) > 0 else 0
                st.metric("Target Price", f"\u20B9{tgt_val:.2f}", delta=f"+{pct_up:.1f}%")
            st.markdown("**Score Components:**")
            st.markdown(rationale.get("score_breakdown", ""))
            try:
                tech = pick.get("breakdown", {}).get("technical", {})
                fund = pick.get("breakdown", {}).get("fundamental", {})
                categories = ["RSI", "MACD", "Volume", "MA Cross", "Breakout", "P/E", "Mkt Cap", "Vol Ratio"]
                values = [
                    tech.get("rsi_score", 0) / 15 * 100,
                    tech.get("macd_score", 0) / 15 * 100,
                    tech.get("volume_score", 0) / 15 * 100,
                    tech.get("ma_score", 0) / 10 * 100,
                    tech.get("gap_score", 0) / 10 * 100,
                    fund.get("pe_score", 0) / 10 * 100,
                    fund.get("cap_score", 0) / 5 * 100,
                    fund.get("volume_ratio_score", 0) / 10 * 100,
                ]
                fig = go.Figure(data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name=pick.get("ticker", ""), line_color='#00cc66'
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, margin=dict(l=80, r=80, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

        with col2:
            st.subheader(":clipboard: Key Metrics")
            metrics = {
                "Sector": rationale.get("sector", pick.get("sector", "N/A")),
                "Market Cap": rationale.get("market_cap_formatted", format_market_cap_nse(pick.get("market_cap", 0))),
                "P/E Ratio": f"{pick.get('pe_ratio', 'N/A'):.2f}" if isinstance(pick.get('pe_ratio', 'N/A'), (int, float)) else pick.get('pe_ratio', 'N/A'),
                "Volume": format_volume_nse(pick.get("volume", 0)),
                "Avg Vol (90d)": format_volume_nse(pick.get("avg_volume_90d", 0)),
                "Vol Ratio (20d)": f"{pick.get('volume_ratio_20d', 1):.2f}x",
                "Risk Level": risk,
            }
            for label, value in metrics.items():
                st.markdown(f"**{label}:**  \n{value}")
                st.markdown("---")
            st.subheader(":chart_with_upwards_trend: Price Context")
            tech_summary = pick.get("technicals", {})
            if tech_summary:
                price_metrics = {
                    "20-Day High": f"\u20B9{tech_summary.get('high_20d', 0):.2f}",
                    "20-Day Low": f"\u20B9{tech_summary.get('low_20d', 0):.2f}",
                    "5-Day Change": f"{tech_summary.get('price_pct_5d', 0):.1f}%",
                    "20-Day Change": f"{tech_summary.get('price_pct_20d', 0):.1f}%",
                }
                for label, value in price_metrics.items():
                    st.markdown(f"**{label}:**  \n{value}")
                    st.markdown("---")


def render_top_picks():
    st.markdown("### :trophy: Today's Top 10 NSE Picks")
    st.markdown("*Sorted by composite score. Click arrow_downwards to expand.*")
    picks = get_cached_picks()
    if not picks:
        st.info("No picks available. Click 'Run Fresh Screen' to analyze NSE.")
        return
    for i, pick in enumerate(picks[:10], 1):
        render_pick_card(pick, i)


def render_download_section():
    st.markdown("---")
    st.markdown("### :inbox_tray: Download Reports")
    picks = get_cached_picks()
    if not picks:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        df_data = []
        for pick in picks:
            r = pick.get("rationale", {})
            df_data.append({
                "Ticker": pick.get("ticker", ""),
                "Company": pick.get("company_name", ""),
                "Price (Rs.)": pick.get("current_price", 0),
                "Score": round(pick.get("score", 0), 1),
                "RSI": round(pick.get("rsi", 0), 1),
                "Volume Ratio": round(pick.get("volume_ratio_20d", 0), 2),
                "Entry": r.get("entry_suggestion", ""),
                "Stop Loss": r.get("suggested_stop_loss", 0),
                "Target": r.get("suggested_target", 0),
                "Risk": r.get("risk_level", ""),
                "P/E": pick.get("pe_ratio", "N/A"),
            })
        df = pd.DataFrame(df_data)
        csv = df.to_csv(index=False)
        st.download_button(label=":page_facing_up: Download CSV", data=csv, file_name=f"nse_top10_{date.today()}.csv", mime="text/csv", use_container_width=True)

    with col2:
        if st.button(":green_book: Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 20)
                pdf.cell(0, 15, "NSE Daily Top 10 Picks", ln=True, align="C")
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 10, f"Date: {date.today().strftime('%B %d, %Y')}", ln=True, align="C")
                pdf.ln(10)
                for i, pick in enumerate(picks[:10], 1):
                    r = pick.get("rationale", {})
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, f"#{i} {pick.get('ticker', 'N/A')} - Rs.{pick.get('current_price', 0):.2f}", ln=True)
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 6, f"Score: {pick.get('score', 0):.1f}/100 | RSI: {pick.get('rsi', 50):.1f} | Entry: {r.get('entry_suggestion', 'N/A')}", ln=True)
                    pdf.cell(0, 6, f"Stop: Rs.{r.get('suggested_stop_loss', 0):.2f} | Target: Rs.{r.get('suggested_target', 0):.2f}", ln=True)
                    for reason in r.get("bullish_reasons", []):
                        pdf.cell(0, 6, f"  + {reason}", ln=True)
                    pdf.ln(3)
                pdf.set_y(-30)
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 10, "Disclaimer: For educational purposes only. Not financial advice.", ln=True, align="C")
                pdf_output = pdf.output(dest="S").encode("latin-1")
                st.download_button(label=":green_book: Download PDF", data=pdf_output, file_name=f"nse_top10_{date.today()}.pdf", mime="application/pdf", use_container_width=True)

    with col3:
        from src.database import ReportGenerator
        json_report = ReportGenerator.generate_json_report(picks)
        st.download_button(label=":card_index: Download JSON", data=json_report, file_name=f"nse_picks_{date.today()}.json", mime="application/json", use_container_width=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("## :bar_chart: Dashboard Info")
        st.markdown("---")
        st.markdown("### About This Tool")
        st.markdown("""Screens **300+ NSE stocks** daily using a multi-factor quant model:

**Technical (65 pts):** RSI, MACD, Volume, MA Trends, Breakout

**Fundamental (25 pts):** P/E, Market Cap, Volume Ratio

**Sentiment (10 pts):** News & Relative Strength""")
        st.markdown("---")
        st.markdown("### :bar_chart: Market Context")
        picks = get_cached_picks()
        if picks:
            top10 = picks[:10]
            avg_score = np.mean([p.get("score", 0) for p in top10]) if top10 else 0
            avg_rsi = np.mean([p.get("rsi", 50) for p in top10]) if top10 else 50
            st.metric("Average Score", f"{avg_score:.1f}/100")
            st.metric("Average RSI", f"{avg_rsi:.1f}")
            st.metric("Total Picks Available", len(picks))
        st.markdown("---")
        st.markdown("### :rocket: Quick Stats")
        st.markdown("- **Universe:** 300+ NSE stocks")
        st.markdown("- **Index:** NIFTY 50 + NIFTY Next 50 + F&O")
        st.markdown("- **Min Price:** Rs. 50 (no penny stocks)")
        st.markdown("- **Min Market Cap:** Rs. 500 Cr")
        st.markdown("- **Min Volume:** 1L shares/day")
        st.markdown("---")
        st.markdown(
            '<p class="disclaimer" style="font-size: 0.8rem; color: #999; font-style: italic; border-top: 1px solid #eee; padding-top: 1rem;">'
            ":warning: <strong>Disclaimer:</strong> Educational/research tool only. "
            "Past results do not guarantee future performance. "
            "Always do your own due diligence."
            "</p>",
            unsafe_allow_html=True
        )


def render_performance_tracker():
    st.markdown("---")
    st.markdown("### :chart_with_upwards_trend: Historical Performance")
    st.markdown("*Requires database connection*")
    db = PickDatabase()
    stats = db.get_performance_stats(days=30)
    if "error" in stats:
        st.info("Performance tracking requires active database connection.")
    else:
        st.metric("Tracked Days", stats.get("total_days", 0))


def render_visual_dashboard():
    picks = get_cached_picks()
    if not picks:
        return
    st.markdown("---")
    st.markdown("### :bar_chart: Visual Overview")
    top10 = picks[:10]
    fig1 = px.bar(
        x=[p.get("ticker", "") for p in top10],
        y=[p.get("score", 0) for p in top10],
        title="Top 10 Scores", labels={"x": "Ticker", "y": "Score"},
        color=[p.get("score", 0) for p in top10], color_continuous_scale="Greens"
    )
    fig1.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(
            x=[p.get("ticker", "") for p in top10],
            y=[p.get("rsi", 50) for p in top10],
            title="RSI Values", labels={"x": "Ticker", "y": "RSI"},
            color=[p.get("rsi", 50) for p in top10], color_continuous_scale="RdYlGn", range_color=[30, 80]
        )
        fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig2.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Bullish")
        fig2.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Oversold")
        fig2.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.bar(
            x=[p.get("ticker", "") for p in top10],
            y=[p.get("volume_ratio_20d", 1) for p in top10],
            title="Volume Ratio (20d Avg)", labels={"x": "Ticker", "y": "x Avg Volume"},
            color=[p.get("volume_ratio_20d", 1) for p in top10], color_continuous_scale="Blues"
        )
        fig3.add_hline(y=1.5, line_dash="dash", line_color="green", annotation_text="Spike Threshold")
        fig3.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig3, use_container_width=True)


def main():
    render_sidebar()
    render_header()
    render_top_picks()
    render_visual_dashboard()
    render_download_section()
    render_performance_tracker()
    st.markdown(
        """<div style="text-align:center;margin-top:3rem;padding:1rem;font-size:0.8rem;color:#999;border-top:1px solid #eee;">
        :warning: <strong>Disclaimer:</strong> For educational purposes only. Not financial advice.
        Trading involves risk. Consult a qualified advisor before investing.
        <p style="margin-top:0.5rem;">Built with :heart: using yfinance, Streamlit &amp; GitHub Actions</p></div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
