# Options Strategy Analyzer - Plotly Dash Version

## Overview
This is a complete conversion of your Streamlit options strategy analyzer to Plotly Dash. The app provides comprehensive what-if analysis, hedging recommendations, and P&L visualizations for options trading strategies.

## Key Differences: Streamlit vs Dash

### Architecture
- **Streamlit**: Sequential execution model - code runs top to bottom on every interaction
- **Dash**: Callback-based architecture - specific functions respond to specific inputs

### UI Components

| Feature | Streamlit | Dash |
|---------|-----------|------|
| Sidebar | `st.sidebar` | `dbc.Col` with custom styling |
| Tabs | `st.tabs()` | `dcc.Tabs` |
| Slider | `st.slider()` | `dcc.Slider` or `dbc.Input` |
| Dropdown | `st.selectbox()` | `dcc.Dropdown` |
| Button | `st.button()` | `dbc.Button` |
| Charts | `st.plotly_chart()` | `dcc.Graph` |
| Tables | `st.dataframe()` | `dash_table.DataTable` |
| State | `st.session_state` | `dcc.Store` + callbacks |

### Major Changes Made

1. **State Management**: 
   - Replaced `st.session_state` with `dcc.Store` components
   - Legs stored in `legs-store`
   - Spot price stored in `spot-store`

2. **Rerun Pattern**:
   - Removed all `st.rerun()` calls
   - Dash automatically updates UI through callbacks

3. **Layout**:
   - Converted to Bootstrap grid system using `dash-bootstrap-components`
   - Sidebar is now a fixed column layout

4. **Callbacks**:
   - `manage_legs()`: Handles adding legs, clearing, and quick strategies
   - `render_tab_content()`: Renders the appropriate tab based on selection
   - Uses `ctx.triggered` to detect which button was clicked

5. **Styling**:
   - Custom CSS moved to `app.index_string`
   - Bootstrap classes used for cards and layout

## Installation

```bash
pip install dash dash-bootstrap-components plotly scipy numpy pandas --break-system-packages
```

## Running the App

```bash
python options_strategy_analyzer_dash.py
```

Then open your browser to: `http://127.0.0.1:8050`

## Features

### 1. Position Setup (Sidebar)
- Set underlying spot price
- Add option legs manually with:
  - Strike price
  - Expiry days
  - Option type (Call/Put)
  - Side (Buy/Sell)
  - Quantity & lot size
  - Entry price & IV
- Quick strategy buttons:
  - Bull Call Spread
  - Iron Condor
  - Straddle

### 2. Overview Tab
- Current position value and P&L
- Maximum profit/loss
- Risk level assessment
- Complete Greeks analysis (Delta, Gamma, Theta, Vega, Rho)
- Breakeven points
- Position legs table

### 3. Scenarios Tab
- Scenario matrix heatmap
- Best/worst case scenarios
- Multi-dimensional analysis (spot, IV, time)

### 4. Recommendations Tab
- AI-powered hedge recommendations
- Delta hedging
- Gamma hedging
- Theta protection
- EV scoring for each recommendation

### 5. P&L Curves Tab
- Interactive P&L curve
- Time decay comparison (Today, 3 days, 7 days, Expiry)
- Multiple curve overlays

### 6. Position Manager Tab
- View all current legs
- Detailed leg information

## Code Structure

```
options_strategy_analyzer_dash.py
├── Pricing Engine
│   ├── black_scholes()
│   ├── calculate_greeks()
│   └── implied_volatility()
│
├── Core Classes
│   ├── OptionLeg (dataclass)
│   ├── PositionAnalyzer
│   └── HedgeRecommender
│
├── Plotting Functions
│   ├── plot_pnl_curve()
│   └── plot_scenario_heatmap()
│
├── Dash App
│   ├── Layout
│   │   ├── create_sidebar()
│   │   └── create_main_content()
│   │
│   ├── Callbacks
│   │   ├── manage_legs()
│   │   └── render_tab_content()
│   │
│   └── Render Functions
│       ├── render_overview()
│       ├── render_scenarios()
│       ├── render_recommendations()
│       ├── render_pnl()
│       └── render_manager()
```

## What Wasn't Converted

The following features from the original Streamlit app were simplified or removed:

1. **Interactive hedge implementation**: The "Implement Hedge" buttons don't add legs in this version (would require additional callbacks)
2. **Solo leg analysis**: The "Analyze Solo" feature was removed (could be added with more callbacks)
3. **Remove individual legs**: Simplified to just "Clear All" (individual removal would need index-based callbacks)
4. **Custom scenario parameters**: Uses fixed parameters instead of sliders (can be added with additional inputs)

## Future Enhancements

To make this even more powerful, you could add:

1. **Persistent storage**: Use a database or file storage for positions
2. **Real-time data**: Integrate with options data API
3. **Export functionality**: Download reports as PDF/Excel
4. **User authentication**: Multi-user support
5. **Advanced scenarios**: More customizable scenario analysis
6. **Live Greeks**: Real-time updates with market data

## Troubleshooting

**Port already in use:**
```bash
python options_strategy_analyzer_dash.py
# Change port in code: app.run_server(debug=True, port=8051)
```

**Import errors:**
Make sure all dependencies are installed with `--break-system-packages` flag.

**Callbacks not firing:**
Check browser console for errors. Dash is more strict about data types than Streamlit.

## Performance Notes

- Dash is generally faster than Streamlit for complex apps
- Callbacks are cached and only rerun when inputs change
- Large scenario analyses may still take a few seconds
- Consider using `dcc.Loading` for long operations

## License

Same as original Streamlit version.
