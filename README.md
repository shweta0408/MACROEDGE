# 📊 MacroEdge — Portfolio Intelligence System
### Macro-Economic Driven Portfolio Recommendation System

---

## 🚀 Live Demo
`https://macroedge-by-shweta.streamlit.app/`

---

## 🧠 What It Does

| Feature | Description |
|---|---|
| **Live Macro Data** | Auto-fetches real-time data from FRED (Federal Reserve Economic Data) |
| **Regime Detection** | Classifies market as Bull / Bear / Stagflation / Recovery |
| **Risk Scoring** | Composite 0–100 macro stress score |
| **Asset Allocation** | Recommends Stocks/Bonds/Gold/Commodities/Cash split |
| **Sector Rotation** | Overweight/Underweight sector signals per regime |
| **Rebalancing Engine** | Compare your current portfolio vs. recommended and get action items |

---

## 📦 Tech Stack

- **Python 3.10+**
- **Streamlit** — Web UI framework
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing
- **FRED Public API** — Live macro data (no API key required)

---

## 🏃 Run Locally

```bash
# 1. Clone or download this folder
cd macro-portfolio-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```
App opens at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (Free, 5 minutes)

1. Push this folder to a **GitHub repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: MacroEdge Portfolio System"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/macroedge.git
   git push -u origin main
   ```

2. Go to **[share.streamlit.io](https://share.streamlit.io)**

3. Click **"New App"** → Connect GitHub → Select your repo

4. Set:
   - **Repository:** your repo name
   - **Branch:** main
   - **Main file path:** `app.py`

5. Click **Deploy** — done! 🎉

---

## 📐 How the Models Work

### Regime Detection Engine
Uses a **rule-based scoring system** across 5 macro indicators:

| Indicator | Signal Logic |

| GDP Growth | >2.5% → Bull (+3), <0% → Bear (+3) |
| CPI Inflation | >5% → Stagflation (+3), 2–4% → Bull (+2) |
| Fed Funds Rate | >4.5% → Bear (+2), <2% → Bull (+2) |
| Yield Spread | Negative → Bear (+3) (inverted yield curve) |
| Unemployment | <4.5% → Bull (+2), >6% → Bear (+2) |

The regime with the **highest total score** is declared the active regime.

### Risk Score (0–100)
Composite of: Inflation (max 25pts) + Fed Rate (20pts) + GDP contraction (15pts) + Yield inversion (15pts) + Unemployment (15pts) + VaR proxy (10pts)

### Allocation Engine
Base regime allocations are **scaled by risk profile**:
- Conservative: Reduces equity exposure by 40%, increases bonds
- Moderate: Base regime weights
- Aggressive: Increases equity exposure by 40%, reduces bonds

---

## 📝 Disclaimer
This tool is for educational and demonstration purposes only.
It does not constitute financial advice.

---

## 🏦 Why This Matters 
This project demonstrates:
- Understanding of macro economic cycles and their impact on asset classes
- Factor-based investment logic (Fama-French, risk regimes)
- Real-time data engineering (FRED API)
- Full-stack data product development
- Clear communication of quantitative outputs

## Connect Me!
[Linkedin](https://www.linkedin.com/in/shweta-mishra-4777681a4)
[Github](https://github.com/shweta0408)
