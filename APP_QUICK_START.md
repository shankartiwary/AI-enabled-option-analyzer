# 🚀 Quick Start - Options Strategy Analyzer

## Installation (30 seconds)

```bash
pip install streamlit numpy pandas scipy plotly
streamlit run options_strategy_analyzer.py
```

## Features At-A-Glance

| Feature | Description |
|---------|-------------|
| **What-If Analysis** | Test unlimited scenarios (Spot × IV × Time) |
| **AI Hedging** | Get top 5 hedge recommendations automatically |
| **P&L Curves** | Interactive charts with breakevens |
| **Greeks Dashboard** | Real-time Delta, Gamma, Theta, Vega, Rho |
| **Risk Scoring** | Automated Low/Medium/High assessment |
| **Presets** | Iron Condor, Strangle, Straddle ready to use |
| **Position Manager** | Add/remove/analyze legs individually |

## Common Tasks

### Load Preset
```
Sidebar → Click "Iron Condor"
```

### Add Leg
```
Sidebar → Fill form → "➕ Add Leg"
```

### Run Scenario
```
Tab 2 → Adjust sliders → View heatmap
```

### Get Hedge
```
Tab 3 → Set cost → "Implement Hedge"
```

**Ready?** Run: `streamlit run options_strategy_analyzer.py`
