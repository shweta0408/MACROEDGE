"""
Macro-Economic Driven Portfolio Recommendation System
Built for BlackRock Application Demo
Author: [Your Name]
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="MacroEdge | Portfolio Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #050a0e;
    --bg-secondary: #0a1628;
    --bg-card: #0d1f35;
    --bg-card-hover: #112540;
    --accent-green: #00ff88;
    --accent-blue: #0066ff;
    --accent-gold: #ffd700;
    --accent-red: #ff4444;
    --accent-orange: #ff8c00;
    --text-primary: #e8f0fe;
    --text-secondary: #7a9cc8;
    --text-muted: #3d5a80;
    --border: #1a2f4a;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* Hide Streamlit default elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-secondary);
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
}

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: all 0.2s ease;
}
.metric-card:hover { border-color: var(--accent-blue); background: var(--bg-card-hover); }

.regime-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.regime-bull { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid #00ff88; }
.regime-bear { background: rgba(255,68,68,0.15); color: #ff4444; border: 1px solid #ff4444; }
.regime-stag { background: rgba(255,165,0,0.15); color: #ffa500; border: 1px solid #ffa500; }
.regime-recov { background: rgba(0,102,255,0.15); color: #4d9fff; border: 1px solid #4d9fff; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent-green);
}
.kpi-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.allocation-bar-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.alloc-label { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.3rem; }
.alloc-bar-bg { background: var(--border); border-radius: 4px; height: 8px; overflow: hidden; }
.alloc-bar-fill { height: 100%; border-radius: 4px; transition: width 1s ease; }

.info-box {
    background: rgba(0,102,255,0.08);
    border: 1px solid rgba(0,102,255,0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    line-height: 1.6;
    color: var(--text-primary);
}

.warn-box {
    background: rgba(255,165,0,0.08);
    border: 1px solid rgba(255,165,0,0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    line-height: 1.6;
}

.header-brand {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 2px;
}
.header-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
}

/* Streamlit widgets dark override */
.stSlider > div > div { background: var(--bg-card) !important; }
.stSelectbox > div > div { background: var(--bg-card) !important; border-color: var(--border) !important; }
.stNumberInput > div > div > input { background: var(--bg-card) !important; border-color: var(--border) !important; color: var(--text-primary) !important; }
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
}
div[data-testid="metric-container"] label { color: var(--text-secondary) !important; font-size: 0.75rem !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: var(--accent-green) !important; font-family: 'Space Mono', monospace; }

.stButton > button {
    background: var(--accent-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-secondary); border-radius: 8px; padding: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--text-secondary) !important; font-family: 'Space Mono', monospace; font-size: 0.75rem; border-radius: 6px; }
.stTabs [aria-selected="true"] { background: var(--bg-card) !important; color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)


FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# We'll use FRED's public CSV endpoint — no API key needed for these series
FRED_SERIES = {
    "CPI Inflation (YoY %)": "CPIAUCSL",
    "Federal Funds Rate (%)": "FEDFUNDS",
    "10Y Treasury Yield (%)": "DGS10",
    "2Y Treasury Yield (%)": "DGS2",
    "Unemployment Rate (%)": "UNRATE",
    "GDP Growth (QoQ %)": "A191RL1Q225SBEA",
    "ISM Manufacturing PMI": "MANEMP",      # proxy
    "Consumer Sentiment": "UMCSENT",
}


@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str, periods: int = 24) -> pd.Series:
    """Fetch latest N observations for a FRED series via public CSV."""
    try:
        url = f"{FRED_BASE}?id={series_id}"
        df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
        df = df[df[series_id] != "."]
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        df = df.dropna()
        series = df[series_id].tail(periods)
        return series
    except Exception:
        # Return synthetic fallback data if FRED unavailable
        idx = pd.date_range(end=datetime.today(), periods=periods, freq='MS')
        np.random.seed(42)
        return pd.Series(np.random.randn(periods) * 0.5 + 3.0, index=idx, name=series_id)


@st.cache_data(ttl=3600)
def load_all_macro_data():
    """Load all macro indicators and compute latest values."""
    data = {}
    series_map = {
        "inflation": "CPIAUCSL",
        "fed_rate": "FEDFUNDS",
        "yield_10y": "DGS10",
        "yield_2y": "DGS2",
        "unemployment": "UNRATE",
        "gdp_growth": "A191RL1Q225SBEA",
        "consumer_sentiment": "UMCSENT",
    }
    for key, sid in series_map.items():
        s = fetch_fred_series(sid, 36)
        data[key] = s

    # Compute YoY inflation from CPI level
    cpi = data["inflation"]
    if len(cpi) >= 13:
        yoy = ((cpi.iloc[-1] / cpi.iloc[-13]) - 1) * 100
    else:
        yoy = 3.2
    data["inflation_yoy"] = round(yoy, 2)
    data["fed_rate_current"] = round(float(data["fed_rate"].iloc[-1]), 2)
    data["yield_10y_current"] = round(float(data["yield_10y"].iloc[-1]), 2)
    data["yield_2y_current"] = round(float(data["yield_2y"].iloc[-1]), 2)
    data["yield_spread"] = round(
        data["yield_10y_current"] - data["yield_2y_current"], 2)
    data["unemployment_current"] = round(
        float(data["unemployment"].iloc[-1]), 2)
    gdp = data["gdp_growth"]
    data["gdp_current"] = round(float(gdp.iloc[-1]), 2)
    sentiment = data["consumer_sentiment"]
    data["sentiment_current"] = round(float(sentiment.iloc[-1]), 1)

    return data


def detect_regime(inflation, fed_rate, gdp_growth, yield_spread, unemployment):
    """
    Rule-based macro regime classifier.
    Returns: (regime_name, regime_color_class, score_dict, explanation)
    """
    scores = {"Bull": 0, "Bear": 0, "Stagflation": 0, "Recovery": 0}

    # GDP signal
    if gdp_growth > 2.5:
        scores["Bull"] += 3
        scores["Recovery"] += 1
    elif gdp_growth > 0:
        scores["Recovery"] += 2
        scores["Bull"] += 1
    else:
        scores["Bear"] += 3
        scores["Stagflation"] += 1

    # Inflation signal
    if inflation > 5.0:
        scores["Stagflation"] += 3
        scores["Bear"] += 1
    elif inflation > 3.0:
        scores["Stagflation"] += 1
        scores["Bear"] += 1
    elif inflation < 1.5:
        scores["Recovery"] += 1
        scores["Bear"] += 1
    else:
        scores["Bull"] += 2

    # Fed rate signal
    if fed_rate > 4.5:
        scores["Bear"] += 2
        scores["Stagflation"] += 1
    elif fed_rate > 2.5:
        scores["Bear"] += 1
    else:
        scores["Bull"] += 2
        scores["Recovery"] += 2

    # Yield curve (spread = 10Y - 2Y)
    if yield_spread < 0:  # Inverted — recession signal
        scores["Bear"] += 3
    elif yield_spread < 0.5:
        scores["Bear"] += 1
    else:
        scores["Bull"] += 2
        scores["Recovery"] += 1

    # Unemployment
    if unemployment < 4.5:
        scores["Bull"] += 2
    elif unemployment < 6.0:
        scores["Recovery"] += 1
    else:
        scores["Bear"] += 2
        scores["Stagflation"] += 1

    regime = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = round((scores[regime] / total) * 100, 1)

    explanations = {
        "Bull": "Strong GDP growth, controlled inflation, and low unemployment signal an expansionary environment. Risk assets historically outperform.",
        "Bear": "Elevated rates, inverted yield curve, or weak GDP suggest a contractionary cycle. Capital preservation becomes priority.",
        "Stagflation": "High inflation persisting alongside slowing growth — the most challenging regime for traditional portfolios.",
        "Recovery": "Economy emerging from contraction. Early-cycle opportunities in equities and credit with improving fundamentals."
    }
    colors = {"Bull": "regime-bull", "Bear": "regime-bear",
              "Stagflation": "regime-stag", "Recovery": "regime-recov"}

    return regime, colors[regime], scores, confidence, explanations[regime]


REGIME_ALLOCATIONS = {
    "Bull": {
        "US Equities": 40, "International Equities": 20,
        "Bonds (IG)": 15, "High Yield Credit": 10,
        "Gold": 5, "Commodities": 5, "Cash": 5
    },
    "Bear": {
        "US Equities": 15, "International Equities": 5,
        "Bonds (IG)": 35, "High Yield Credit": 5,
        "Gold": 20, "Commodities": 5, "Cash": 15
    },
    "Stagflation": {
        "US Equities": 15, "International Equities": 10,
        "Bonds (IG)": 10, "High Yield Credit": 5,
        "Gold": 30, "Commodities": 20, "Cash": 10
    },
    "Recovery": {
        "US Equities": 30, "International Equities": 15,
        "Bonds (IG)": 20, "High Yield Credit": 15,
        "Gold": 10, "Commodities": 5, "Cash": 5
    }
}

SECTOR_ROTATION = {
    "Bull": {
        "Overweight": ["Technology", "Consumer Discretionary", "Financials", "Industrials"],
        "Underweight": ["Utilities", "Consumer Staples", "Healthcare"],
        "Rationale": "Growth sectors lead in expansionary cycles. Cyclicals outperform defensives."
    },
    "Bear": {
        "Overweight": ["Utilities", "Consumer Staples", "Healthcare", "REITs (selective)"],
        "Underweight": ["Technology", "Consumer Discretionary", "Financials"],
        "Rationale": "Defensive sectors with stable cash flows preserve capital. Reduce cyclical exposure."
    },
    "Stagflation": {
        "Overweight": ["Energy", "Materials", "Agriculture", "Infrastructure"],
        "Underweight": ["Technology", "Bonds", "Consumer Discretionary"],
        "Rationale": "Real assets and commodity producers hedge purchasing power erosion."
    },
    "Recovery": {
        "Overweight": ["Financials", "Industrials", "Materials", "Small-Cap Equities"],
        "Underweight": ["Gold", "Long-Duration Bonds", "Defensive Sectors"],
        "Rationale": "Early-cycle sectors — financials and industrials — benefit most from inflection."
    }
}

RISK_PROFILES = {
    "Conservative": {"equity_scale": 0.6, "bond_scale": 1.4, "label": "Capital Preservation"},
    "Moderate": {"equity_scale": 1.0, "bond_scale": 1.0, "label": "Balanced Growth"},
    "Aggressive": {"equity_scale": 1.4, "bond_scale": 0.6, "label": "Maximum Growth"},
}


def compute_allocation(regime, risk_profile):
    base = REGIME_ALLOCATIONS[regime].copy()
    scales = RISK_PROFILES[risk_profile]
    equity_keys = ["US Equities",
                   "International Equities", "High Yield Credit"]
    bond_keys = ["Bonds (IG)", "Gold", "Cash"]

    adjusted = {}
    for k, v in base.items():
        if k in equity_keys:
            adjusted[k] = v * scales["equity_scale"]
        elif k in bond_keys:
            adjusted[k] = v * scales["bond_scale"]
        else:
            adjusted[k] = v

    # Normalize to 100
    total = sum(adjusted.values())
    return {k: round((v / total) * 100, 1) for k, v in adjusted.items()}


def compute_risk_score(inflation, fed_rate, gdp_growth, yield_spread, unemployment, vix_approx=20):
    """
    Composite risk score 0–100 (higher = more risk in market).
    """
    score = 0
    score += min(inflation * 4, 25)          # Inflation: max 25pts
    score += min(fed_rate * 3, 20)            # Fed rate: max 20pts
    # Negative GDP: max 15pts (if negative)
    score += max(-gdp_growth * 3, 0)
    # Inversion: up to 15pts
    score += (15 if yield_spread < 0 else max(5 - yield_spread * 5, 0))
    score += min(unemployment * 2, 15)       # Unemployment: max 15pts
    score += min(vix_approx * 0.5, 10)       # VIX proxy: max 10pts
    return round(min(score, 100), 1)


CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,31,53,0.5)',
    font=dict(family='DM Sans', color='#7a9cc8', size=11),
    xaxis=dict(gridcolor='#1a2f4a', linecolor='#1a2f4a',
               tickfont=dict(color='#7a9cc8')),
    yaxis=dict(gridcolor='#1a2f4a', linecolor='#1a2f4a',
               tickfont=dict(color='#7a9cc8')),
    margin=dict(l=40, r=20, t=40, b=40),
)

ASSET_COLORS = {
    "US Equities": "#0066ff",
    "International Equities": "#4d9fff",
    "Bonds (IG)": "#00ff88",
    "High Yield Credit": "#00cc6a",
    "Gold": "#ffd700",
    "Commodities": "#ff8c00",
    "Cash": "#3d5a80"
}


def make_donut(allocation: dict, title: str):
    labels = list(allocation.keys())
    values = list(allocation.values())
    colors = [ASSET_COLORS.get(l, "#7a9cc8") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color='#050a0e', width=2)),
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{label}</b><br>%{value}%<extra></extra>'
    ))
    fig.update_layout(
        **CHART_THEME,
        title=dict(text=title, font=dict(size=13, color='#e8f0fe'), x=0.5),
        showlegend=True,
        legend=dict(font=dict(color='#7a9cc8', size=10),
                    bgcolor='rgba(0,0,0,0)', x=1.0),
        annotations=[dict(text='<b>Allocation</b>', x=0.5, y=0.5, font_size=13,
                          font_color='#e8f0fe', showarrow=False)],
        height=320,
    )
    return fig


def make_macro_sparkline(series: pd.Series, color: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1)',
        hovertemplate='%{x|%b %Y}<br><b>%{y:.2f}</b><extra></extra>'
    ))
    fig.update_layout(
        **CHART_THEME,
        title=dict(text=title, font=dict(size=11, color='#7a9cc8')),
        height=160,
        showlegend=False,
    )
    return fig


def make_rebalancing_chart(current: dict, recommended: dict):
    assets = list(recommended.keys())
    curr_vals = [current.get(a, 0) for a in assets]
    rec_vals = [recommended[a] for a in assets]
    diff = [r - c for r, c in zip(rec_vals, curr_vals)]

    colors = ['#00ff88' if d >= 0 else '#ff4444' for d in diff]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Current', x=assets, y=curr_vals,
        marker_color='#3d5a80',
        hovertemplate='%{x}<br>Current: <b>%{y}%</b><extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Recommended', x=assets, y=rec_vals,
        marker_color='#0066ff',
        hovertemplate='%{x}<br>Recommended: <b>%{y}%</b><extra></extra>'
    ))
    fig.update_layout(
        **{k: v for k, v in CHART_THEME.items() if k != 'xaxis'},
        barmode='group',
        title=dict(text='Current vs Recommended Allocation',
                   font=dict(size=13, color='#e8f0fe')),
        height=340,
        legend=dict(font=dict(color='#7a9cc8'), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(**CHART_THEME['xaxis'], tickangle=-20)
    )
    return fig


def make_gauge(score, title):
    color = "#00ff88" if score < 35 else (
        "#ffd700" if score < 65 else "#ff4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'color': '#7a9cc8', 'size': 13}},
        number={'font': {'color': color, 'size': 30, 'family': 'Space Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#3d5a80', 'tickfont': {'color': '#3d5a80'}},
            'bar': {'color': color},
            'bgcolor': '#0d1f35',
            'bordercolor': '#1a2f4a',
            'steps': [
                {'range': [0, 35], 'color': 'rgba(0,255,136,0.1)'},
                {'range': [35, 65], 'color': 'rgba(255,215,0,0.1)'},
                {'range': [65, 100], 'color': 'rgba(255,68,68,0.1)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 2},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def make_yield_curve(y2, y10):
    maturities = ['2Y', '5Y', '7Y', '10Y']
    y5_approx = y2 + (y10 - y2) * 0.4
    y7_approx = y2 + (y10 - y2) * 0.7
    yields = [y2, y5_approx, y7_approx, y10]
    color = '#ff4444' if y2 > y10 else '#00ff88'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities, y=yields,
        mode='lines+markers',
        line=dict(color=color, width=2.5),
        marker=dict(color=color, size=8),
        fill='tozeroy',
        fillcolor=f'rgba({255 if color == "#ff4444" else 0},{136 if color == "#00ff88" else 68},{68 if color == "#ff4444" else 0},0.12)',
        hovertemplate='%{x}: <b>%{y:.2f}%</b><extra></extra>'
    ))
    fig.update_layout(
        **CHART_THEME,
        title=dict(text='US Treasury Yield Curve (Approximate)',
                   font=dict(size=13, color='#e8f0fe')),
        height=220,
        showlegend=False,
    )
    return fig


with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem'>
        <div style='font-family: Space Mono; font-size:1.1rem; color:#e8f0fe; font-weight:700; letter-spacing:2px'>
            MACRO<span style='color:#0066ff'>EDGE</span>
        </div>
        <div style='font-size:0.65rem; color:#3d5a80; font-family:Space Mono; letter-spacing:1px'>
            PORTFOLIO INTELLIGENCE SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">RISK PROFILE</div>',
                unsafe_allow_html=True)
    risk_profile = st.selectbox(
        "Investor Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        help="Adjusts allocation weights relative to your risk appetite"
    )
    st.caption(f"→ {RISK_PROFILES[risk_profile]['label']}")

    st.markdown('<div class="section-title" style="margin-top:1.5rem">CURRENT PORTFOLIO</div>',
                unsafe_allow_html=True)
    st.caption("Enter your current allocation (%) — leave blank if starting fresh")

    portfolio_inputs = {}
    asset_classes = ["US Equities", "International Equities", "Bonds (IG)",
                     "High Yield Credit", "Gold", "Commodities", "Cash"]
    for asset in asset_classes:
        portfolio_inputs[asset] = st.number_input(
            asset, min_value=0.0, max_value=100.0, value=0.0, step=1.0,
            key=f"port_{asset}"
        )

    total_input = sum(portfolio_inputs.values())
    if total_input > 0:
        if abs(total_input - 100) > 5:
            st.warning(
                f"⚠ Portfolio sums to {total_input:.0f}% — should be ~100%")
        else:
            st.success(f"✓ Portfolio: {total_input:.0f}%")

    st.markdown('<div class="section-title" style="margin-top:1.5rem">DATA SOURCE</div>',
                unsafe_allow_html=True)
    st.caption("Live macro data via FRED (Federal Reserve Bank of St. Louis)")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


with st.spinner("Fetching live macro data from FRED..."):
    macro = load_all_macro_data()

inflation = macro["inflation_yoy"]
fed_rate = macro["fed_rate_current"]
yield_10y = macro["yield_10y_current"]
yield_2y = macro["yield_2y_current"]
yield_spread = macro["yield_spread"]
unemployment = macro["unemployment_current"]
gdp_growth = macro["gdp_current"]
sentiment = macro["sentiment_current"]

regime, regime_class, regime_scores, confidence, regime_explanation = detect_regime(
    inflation, fed_rate, gdp_growth, yield_spread, unemployment
)
recommended_allocation = compute_allocation(regime, risk_profile)
risk_score = compute_risk_score(
    inflation, fed_rate, gdp_growth, yield_spread, unemployment)
sector_info = SECTOR_ROTATION[regime]


col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
with col_h1:
    st.markdown("""
    <div style='margin-bottom:0.5rem'>
        <span class='header-brand'>MACROEDGE</span>
        <span style='font-family:Space Mono; color:#3d5a80; font-size:0.75rem; margin-left:1rem'>
            PORTFOLIO INTELLIGENCE
        </span>
    </div>
    <div class='header-sub'>Macro-Economic Driven Portfolio Recommendation System</div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='text-align:right; padding-top:0.5rem'>
        <div style='font-size:0.7rem; color:#3d5a80; font-family:Space Mono'>LAST UPDATED</div>
        <div style='font-family:Space Mono; color:#7a9cc8; font-size:0.85rem'>
            {datetime.now().strftime("%b %d, %Y")}
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_h3:
    st.markdown(f"""
    <div style='text-align:right; padding-top:0.5rem'>
        <div style='font-size:0.7rem; color:#3d5a80; font-family:Space Mono'>DATA SOURCE</div>
        <div style='font-family:Space Mono; color:#0066ff; font-size:0.85rem'>FRED / Live</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1a2f4a; margin: 0.5rem 0 1rem'>",
            unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "📡  MACRO DASHBOARD",
    "🧠  REGIME & RISK",
    "💼  ALLOCATION",
    "⚖️  REBALANCING"
])


with tab1:
    st.markdown('<div class="section-title">LIVE MACRO INDICATORS</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "CPI Inflation", f"{inflation}%", "YoY", inflation > 3.5),
        (c2, "Fed Funds Rate", f"{fed_rate}%", "Current", fed_rate > 4.0),
        (c3, "10Y Yield", f"{yield_10y}%", "Treasury", None),
        (c4, "Yield Spread", f"{yield_spread:+.2f}%",
         "10Y - 2Y", yield_spread < 0),
        (c5, "Unemployment", f"{unemployment}%",
         "U-3 Rate", unemployment > 5.5),
    ]
    for col, label, val, sub, warn in metrics:
        with col:
            color = "#ff4444" if warn else (
                "#ffd700" if warn is None else "#00ff88")
            st.markdown(f"""
            <div class="metric-card" style="text-align:center">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{val}</div>
                <div style="font-size:0.7rem; color:#3d5a80; font-family:Space Mono">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1rem">MACRO TRENDS (36M)</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        cpi_series = macro["inflation"].pct_change(12).dropna() * 100
        st.plotly_chart(make_macro_sparkline(
            cpi_series.tail(24), "#ff4444", "CPI Inflation YoY %"
        ), use_container_width=True, config={'displayModeBar': False})

        st.plotly_chart(make_macro_sparkline(
            macro["fed_rate"].tail(24), "#ffd700", "Federal Funds Rate %"
        ), use_container_width=True, config={'displayModeBar': False})

    with col_b:
        st.plotly_chart(make_macro_sparkline(
            macro["yield_10y"].tail(24), "#0066ff", "10-Year Treasury Yield %"
        ), use_container_width=True, config={'displayModeBar': False})

        st.plotly_chart(make_macro_sparkline(
            macro["unemployment"].tail(24), "#00ff88", "Unemployment Rate %"
        ), use_container_width=True, config={'displayModeBar': False})

    col_yc, col_sent = st.columns(2)
    with col_yc:
        st.plotly_chart(make_yield_curve(yield_2y, yield_10y),
                        use_container_width=True, config={'displayModeBar': False})
        if yield_spread < 0:
            st.markdown('<div class="warn-box">⚠️ <b>Yield Curve Inverted</b> — Historically a leading indicator of recession (avg 12–18 month lead time). 2Y yields exceeding 10Y yields signal tight monetary conditions.</div>', unsafe_allow_html=True)

    with col_sent:
        st.plotly_chart(make_macro_sparkline(
            macro["consumer_sentiment"].tail(
                24), "#ff8c00", "Consumer Sentiment Index"
        ), use_container_width=True, config={'displayModeBar': False})
        st.plotly_chart(make_macro_sparkline(
            macro["gdp_growth"].tail(
                12), "#4d9fff", "GDP Growth QoQ % (Annualized)"
        ), use_container_width=True, config={'displayModeBar': False})


with tab2:
    col_reg, col_gauge = st.columns([3, 2])

    with col_reg:
        st.markdown(
            '<div class="section-title">DETECTED MARKET REGIME</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="padding: 1.5rem">
            <div style="margin-bottom:1rem">
                <span class="regime-badge {regime_class}">{regime}</span>
                <span style="font-family:Space Mono; color:#3d5a80; font-size:0.75rem; margin-left:1rem">
                    {confidence}% confidence
                </span>
            </div>
            <div style="color:#e8f0fe; font-size:0.95rem; line-height:1.7; margin-bottom:1rem">
                {regime_explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title" style="margin-top:1rem">REGIME SCORE BREAKDOWN</div>', unsafe_allow_html=True)
        total_scores = sum(regime_scores.values())
        score_colors = {"Bull": "#00ff88", "Bear": "#ff4444",
                        "Stagflation": "#ffa500", "Recovery": "#4d9fff"}
        for r, s in sorted(regime_scores.items(), key=lambda x: -x[1]):
            pct = round((s / total_scores) * 100, 1)
            color = score_colors[r]
            st.markdown(f"""
            <div class="allocation-bar-wrap">
                <div style="display:flex; justify-content:space-between">
                    <span class="alloc-label">{r}</span>
                    <span style="font-family:Space Mono; font-size:0.8rem; color:{color}">{pct}%</span>
                </div>
                <div class="alloc-bar-bg">
                    <div class="alloc-bar-fill" style="width:{pct}%; background:{color}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown(
            '<div class="section-title">MARKET RISK SCORE</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(risk_score, "Composite Risk Score"), use_container_width=True,
                        config={'displayModeBar': False})
        risk_label = "LOW RISK" if risk_score < 35 else (
            "MODERATE RISK" if risk_score < 65 else "HIGH RISK")
        risk_color = "#00ff88" if risk_score < 35 else (
            "#ffd700" if risk_score < 65 else "#ff4444")
        st.markdown(f"""
        <div style="text-align:center; margin-top:-0.5rem">
            <span style="font-family:Space Mono; color:{risk_color}; font-size:0.85rem; letter-spacing:2px">
                {risk_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:1rem">
            <b>How it's computed:</b><br>
            Weighted composite of: inflation level, fed rate, GDP contraction, yield curve inversion, and unemployment. Higher = more macro stress.
        </div>
        """, unsafe_allow_html=True)

    # Indicator signal table
    st.markdown('<div class="section-title" style="margin-top:1rem">INDICATOR SIGNAL MATRIX</div>',
                unsafe_allow_html=True)
    signals = {
        "CPI Inflation": (inflation, "> 5% = Bearish / 2–4% = Neutral / < 2% = Bullish",
                          "🔴" if inflation > 5 else "🟡" if inflation > 3 else "🟢"),
        "Fed Funds Rate": (f"{fed_rate}%", "> 4.5% = Restrictive / 2–4% = Neutral / < 2% = Accommodative",
                           "🔴" if fed_rate > 4.5 else "🟡" if fed_rate > 2 else "🟢"),
        "GDP Growth": (f"{gdp_growth}%", "> 2.5% = Strong / 0–2.5% = Moderate / < 0% = Contraction",
                       "🟢" if gdp_growth > 2.5 else "🟡" if gdp_growth > 0 else "🔴"),
        "Yield Spread (10Y–2Y)": (f"{yield_spread:+.2f}%", "< 0% = Inverted (Bearish) / > 0.5% = Normal",
                                  "🔴" if yield_spread < 0 else "🟡" if yield_spread < 0.5 else "🟢"),
        "Unemployment": (f"{unemployment}%", "< 4.5% = Strong / 4.5–6% = Moderate / > 6% = Weak",
                         "🟢" if unemployment < 4.5 else "🟡" if unemployment < 6 else "🔴"),
    }
    sig_df = pd.DataFrame([
        {"Indicator": k, "Current Value": str(
            v[0]), "Interpretation": v[1], "Signal": v[2]}
        for k, v in signals.items()
    ])
    st.dataframe(sig_df, use_container_width=True, hide_index=True,
                 column_config={"Signal": st.column_config.TextColumn("Signal", width="small")})


with tab3:
    col_donut, col_bars = st.columns([1, 1])

    with col_donut:
        st.markdown(
            '<div class="section-title">RECOMMENDED ALLOCATION</div>', unsafe_allow_html=True)
        st.plotly_chart(make_donut(recommended_allocation, f"{regime} Regime · {risk_profile} Profile"),
                        use_container_width=True, config={'displayModeBar': False})

    with col_bars:
        st.markdown(
            '<div class="section-title">ALLOCATION BREAKDOWN</div>', unsafe_allow_html=True)
        for asset, pct in sorted(recommended_allocation.items(), key=lambda x: -x[1]):
            color = ASSET_COLORS.get(asset, "#7a9cc8")
            st.markdown(f"""
            <div class="allocation-bar-wrap">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <span class="alloc-label">{asset}</span>
                    <span style="font-family:Space Mono; font-size:0.9rem; color:{color}; font-weight:700">{pct}%</span>
                </div>
                <div class="alloc-bar-bg">
                    <div class="alloc-bar-fill" style="width:{pct}%; background:{color}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Sector rotation
    st.markdown('<div class="section-title" style="margin-top:1.5rem">SECTOR ROTATION SIGNALS</div>',
                unsafe_allow_html=True)
    col_ow, col_uw = st.columns(2)
    with col_ow:
        st.markdown("**OVERWEIGHT →**")
        for s in sector_info["Overweight"]:
            st.markdown(f"""
            <div style="background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.2);
                         border-radius:6px; padding:0.5rem 0.8rem; margin:0.3rem 0;
                         font-size:0.88rem; color:#00ff88">
                ↑ {s}
            </div>""", unsafe_allow_html=True)
    with col_uw:
        st.markdown("**UNDERWEIGHT →**")
        for s in sector_info["Underweight"]:
            st.markdown(f"""
            <div style="background:rgba(255,68,68,0.08); border:1px solid rgba(255,68,68,0.2);
                         border-radius:6px; padding:0.5rem 0.8rem; margin:0.3rem 0;
                         font-size:0.88rem; color:#ff4444">
                ↓ {s}
            </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div class="info-box" style="margin-top:1rem">💡 <b>Rationale:</b> {sector_info["Rationale"]}</div>', unsafe_allow_html=True)


with tab4:
    has_portfolio = total_input > 1
    if not has_portfolio:
        st.markdown("""
        <div class="warn-box" style="margin-bottom:1rem">
            👈 Enter your current portfolio allocation in the left sidebar to see rebalancing recommendations.
            Use 0% for assets you don't hold.
        </div>
        """, unsafe_allow_html=True)
        current_alloc = {a: round(100 / len(asset_classes), 1)
                         for a in asset_classes}
        st.caption("_Showing example 60/40-style portfolio for demonstration_")
        demo = {"US Equities": 50, "International Equities": 10, "Bonds (IG)": 30,
                "High Yield Credit": 5, "Gold": 3, "Commodities": 0, "Cash": 2}
        current_alloc = demo
    else:
        total_pct = sum(portfolio_inputs.values())
        current_alloc = {k: round((v / total_pct) * 100, 1)
                         for k, v in portfolio_inputs.items()}

    st.plotly_chart(make_rebalancing_chart(current_alloc, recommended_allocation),
                    use_container_width=True, config={'displayModeBar': False})

    # Delta table
    st.markdown('<div class="section-title" style="margin-top:1rem">REBALANCING ACTIONS</div>',
                unsafe_allow_html=True)
    rebal_rows = []
    for asset in asset_classes:
        curr = current_alloc.get(asset, 0)
        rec = recommended_allocation.get(asset, 0)
        delta = rec - curr
        action = "INCREASE ↑" if delta > 1 else (
            "DECREASE ↓" if delta < -1 else "HOLD ─")
        rebal_rows.append({
            "Asset Class": asset,
            "Current (%)": f"{curr:.1f}",
            "Recommended (%)": f"{rec:.1f}",
            "Δ Change": f"{delta:+.1f}%",
            "Action": action
        })

    rebal_df = pd.DataFrame(rebal_rows)
    st.dataframe(
        rebal_df, use_container_width=True, hide_index=True,
        column_config={
            "Δ Change": st.column_config.TextColumn("Δ Change"),
            "Action": st.column_config.TextColumn("Action"),
        }
    )

    st.markdown("""
    <div class="info-box" style="margin-top:1rem">
        ⚠️ <b>Disclaimer:</b> This tool is for educational and demonstration purposes only.
        It does not constitute financial advice. All recommendations are model-driven and based on
        macro indicators. Consult a qualified financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
