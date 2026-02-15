# Options Strategy Analyzer - Complete Setup Guide

## 🎯 What You're Getting

A **production-ready Streamlit application** that provides:

✅ **What-If Analysis** - Test unlimited market scenarios  
✅ **AI-Powered Hedging** - Get optimal hedge recommendations  
✅ **P&L Visualization** - Interactive charts and heatmaps  
✅ **Greeks Dashboard** - Real-time risk metrics  
✅ **Strategy Presets** - Iron Condor, Strangle, Straddle, and more  
✅ **Scenario Heatmaps** - Spot × IV × Time analysis  
✅ **Risk Assessment** - Automated risk scoring  

---

## 📦 Installation

### Step 1: Install Dependencies

```bash
pip install streamlit numpy pandas scipy plotly
```

### Step 2: Download the App

Save the file `options_strategy_analyzer.py` to your computer.

### Step 3: Run the App

```bash
streamlit run options_strategy_analyzer.py
```

The app will open in your web browser at `http://localhost:8501`

---

## 🚀 Quick Start

### Method 1: Use Preset Strategies

1. Click a preset button in the sidebar:
   - **Iron Condor** - Classic defined-risk strategy
   - **Short Strangle** - High-IV premium collection
   - **Long Straddle** - Volatility play

2. View instant analysis across all tabs

### Method 2: Build Custom Strategy

1. Set **Spot Price** (e.g., 24000)
2. Click **Add Leg** for each option:
   - Strike: 24000
   - Type: CE or PE
   - Side: BUY or SELL
   - Days to Expiry: 14
   - Quantity: 50 lots
   - Lot Size: 50
   - Entry Price: 150
   - IV: 0.20 (20%)

3. Repeat for all legs
4. Analyze!

---

## 📊 Features Breakdown

### Tab 1: Overview

**What You See:**
- Current P&L
- Maximum profit/loss
- Risk percentage used
- Complete Greeks (Delta, Gamma, Theta, Vega, Rho)
- Risk assessment (Low/Medium/High)
- Greeks gauge dashboard
- Position legs table
- Breakeven points

**How to Use:**
- Check current position health
- Identify risk exposure
- Find breakeven levels

### Tab 2: Scenario Analysis

**What You See:**
- Scenario matrix (Spot × IV)
- Interactive heatmap
- Worst 5 scenarios
- Best 5 scenarios
- P&L across different conditions

**Controls:**
- Spot Range: 5% to 30%
- IV Changes: -10% to +10%
- Time Points: 0, 3, 7, 14 days

**How to Use:**
- Adjust sliders to test scenarios
- Identify danger zones (red areas)
- Plan for different market conditions

### Tab 3: AI Recommendations

**What You Get:**
- Up to 5 hedge recommendations
- Three types of hedges:
  1. **Delta Hedge** - Neutralize directional bias
  2. **Gamma Hedge** - Reduce sensitivity
  3. **Protective Hedge** - Cap downside

**For Each Recommendation:**
- Specific action (e.g., "BUY 25 lots 24200 CE")
- Cost estimate
- Impact on Greeks
- Expected value score
- One-click implementation

**How to Use:**
- Set max hedge cost with slider
- Review recommendations
- Click "Implement Hedge" to add leg
- System automatically updates analysis

### Tab 4: P&L Curves

**What You See:**
- Interactive P&L curve
- Breakeven markers
- Current position indicator
- Multiple curves (time comparison)
- Time decay visualization

**Controls:**
- IV Scenario: -10% to +10%
- Time Scenario: Today to Expiry

**How to Use:**
- Select scenarios to visualize
- See P&L at different spot prices
- Compare today vs expiry curves
- Understand time decay impact

### Tab 5: Position Manager

**What You Can Do:**
- View all legs in detail
- Remove individual legs
- Analyze legs in isolation
- Modify position structure

**How to Use:**
- Expand any leg to see details
- Click "Remove" to delete
- Click "Analyze Solo" for single-leg view
- Build "what-if" scenarios

---

## 💡 Example Workflows

### Workflow 1: Entering a New Trade

**Goal:** Evaluate an Iron Condor before entry

1. **Load Preset**
   - Click "Iron Condor" in sidebar
   
2. **Adjust Parameters**
   - Modify strikes to match market
   - Update entry prices
   - Set current IV

3. **Check Overview**
   - Review max profit/loss
   - Check Greeks
   - Verify risk level

4. **Run Scenarios**
   - Go to "Scenario Analysis"
   - Test worst cases
   - Identify danger zones

5. **Make Decision**
   - If acceptable: Enter trade
   - If risky: Adjust strikes or pass

### Workflow 2: Managing Active Position

**Goal:** Monitor and adjust existing trade

1. **Input Current Position**
   - Add all legs manually
   - Use current market prices

2. **Check Risk**
   - View "Overview" tab
   - Check Greeks for imbalances
   - Note risk level

3. **Test Scenarios**
   - "Scenario Analysis" tab
   - Look at worst cases
   - Check P&L heatmap

4. **Get Hedge Recommendations**
   - "Recommendations" tab
   - Set max cost willing to spend
   - Review AI suggestions

5. **Implement Hedge**
   - Click "Implement Hedge"
   - System adds leg
   - Re-check metrics

### Workflow 3: What-If Testing

**Goal:** Test different adjustments

1. **Load Base Strategy**
   - Use preset or build custom

2. **Test Addition #1**
   - Go to "Position Manager"
   - Add a protective call
   - Check new P&L curve

3. **Compare Results**
   - Note changes in:
     - Max loss
     - Greeks
     - Breakevens
     - Cost

4. **Test Addition #2**
   - Remove first addition
   - Try different strike/type
   - Compare again

5. **Choose Best Option**
   - Select adjustment with:
     - Best cost/benefit
     - Acceptable Greeks
     - Manageable risk

---

## 🎨 Understanding the Visuals

### P&L Curve

```
     *                    ← Profit Zone (green fill)
   *   *
 *       *
─────────────── Zero Line (red dashed)
         *   *
           *              ← Loss Zone
         
|       ↑
Current  Spot Price (green dashed)
```

**How to Read:**
- **Green fill above zero** = Profit
- **Below zero** = Loss
- **Orange diamonds** = Breakevens
- **Green dot** = Current position

### Scenario Heatmap

```
IV +10%  | 🟥 | 🟨 | 🟩 | 🟨 | 🟥 |
IV  +5%  | 🟨 | 🟩 | 🟩 | 🟩 | 🟨 |
IV   0%  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
IV  -5%  | 🟨 | 🟩 | 🟩 | 🟩 | 🟨 |
IV -10%  | 🟥 | 🟨 | 🟩 | 🟨 | 🟥 |
         -200 -100   0  +100 +200
              Spot Change (₹)
```

**Color Guide:**
- 🟩 **Green** = Profit
- 🟨 **Yellow** = Small loss/profit
- 🟥 **Red** = Large loss

### Greeks Gauges

```
Delta:    [-500 ←→ +500]
           ↑
        Current: +150
        
Gamma:    [-2 ←→ +2]
           ↑
        Current: -0.8
```

**Ideal Zones:**
- **Delta near 0** = Directionally neutral
- **Positive Theta** = Earning from time decay
- **Negative Vega** = Benefiting from IV drop

---

## 🧮 Key Formulas Used

### Black-Scholes Pricing
```
Call = S×N(d1) - K×e^(-rT)×N(d2)
Put  = K×e^(-rT)×N(-d2) - S×N(-d1)

where:
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

### Greeks
```
Delta (Δ) = ∂V/∂S    (Price sensitivity)
Gamma (Γ) = ∂²V/∂S²  (Delta change rate)
Theta (Θ) = ∂V/∂t    (Time decay)
Vega (ν)  = ∂V/∂σ    (IV sensitivity)
Rho (ρ)   = ∂V/∂r    (Rate sensitivity)
```

### P&L Calculation
```
P&L = (Current Value - Initial Value)

For each leg:
Current Value = BS_Price(current_params) × Qty × Lot
Initial Value = Entry_Price × Qty × Lot

Total P&L = Σ(Current - Initial) for all legs
```

---

## 🎓 Understanding Greeks

### Delta (Δ)
**What it means:** Change in option value per ₹1 move in underlying

**Position Delta:**
- **+100**: Long 100 shares equivalent (bullish)
- **-100**: Short 100 shares equivalent (bearish)
- **0**: Market neutral

**Action:**
- High delta → Consider hedging
- Delta moves against you → Add opposite positions

### Gamma (Γ)
**What it means:** How fast delta changes

**High Gamma:**
- **Positive**: Profits accelerate with moves (good!)
- **Negative**: Losses accelerate with moves (danger!)

**Action:**
- Negative gamma + large moves → High risk
- Near expiry → Gamma explodes
- Consider delta hedging

### Theta (Θ)
**What it means:** Daily time decay

**Theta Per Day:**
- **+1000**: Earning ₹1000/day from decay (seller)
- **-1000**: Losing ₹1000/day to decay (buyer)

**Action:**
- Positive theta → Time is your friend
- Negative theta → Need quick moves
- Accelerates near expiry

### Vega (ν)
**What it means:** Change per 1% IV move

**Position Vega:**
- **+10000**: Gain ₹10,000 if IV ↑ 1%
- **-10000**: Lose ₹10,000 if IV ↑ 1%

**Action:**
- Negative vega → Benefit from IV drop
- IV expansion → Hedges get expensive
- Check IV percentile before trading

---

## 🔧 Advanced Features

### Custom Scenarios

**Test Specific Events:**

1. **Earnings Announcement**
   - IV +10%
   - Spot ±5%
   - Next day (1 day passed)

2. **Market Crash**
   - Spot -15%
   - IV +20%
   - Immediate

3. **Slow Grind**
   - Spot +2%
   - IV -5%
   - 7 days passed

### Position Adjustments

**Common Adjustments:**

1. **Roll Up (Bullish)**
   - Remove lower strike
   - Add higher strike
   - Test in app

2. **Roll Down (Bearish)**
   - Remove higher strike
   - Add lower strike
   - Compare results

3. **Add Width**
   - Move long legs further
   - Increases max profit
   - Check new risk

4. **Reduce Width**
   - Move long legs closer
   - Decreases max loss
   - Check new cost

### Strategy Builder Tips

**Iron Condor:**
- Width: 500 points typical
- Premium target: ₹150-250/leg
- Delta: Keep near zero

**Short Strangle:**
- 16-30 delta options
- Weekly expiry preferred
- Monitor gamma closely

**Long Straddle:**
- ATM strikes
- High IV → Expensive
- Need large moves

---

## 📱 Keyboard Shortcuts

When running the app:

- `R` - Rerun / Refresh
- `C` - Clear cache
- `Ctrl+S` - Screenshot
- `Ctrl+F` - Search

---

## 💾 Saving Your Work

### Export Scenarios

```python
# From "Scenario Analysis" tab
# Copy the scenario matrix
# Paste into Excel for records
```

### Save P&L Curves

```python
# Right-click on any chart
# "Save image as..."
# PNG format recommended
```

### Track Adjustments

```python
# Keep a spreadsheet:
Date | Action | Greeks Before | Greeks After | Cost | Result
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install --upgrade streamlit numpy pandas scipy plotly
```

### Issue: App won't start
**Solution:**
```bash
# Kill existing streamlit processes
pkill -f streamlit
# Restart
streamlit run options_strategy_analyzer.py
```

### Issue: Slow performance
**Solution:**
- Reduce scenario range
- Close other apps
- Restart Streamlit

### Issue: Charts not showing
**Solution:**
```bash
# Update Plotly
pip install --upgrade plotly
```

### Issue: Wrong calculations
**Solution:**
- Check entry prices
- Verify IV is decimal (0.20, not 20)
- Confirm lot sizes
- Review side (BUY/SELL)

---

## 🚀 Next Steps

### Phase 1: Master the Basics (Week 1)
- [ ] Load all presets
- [ ] Understand each tab
- [ ] Test simple scenarios
- [ ] Compare P&L curves

### Phase 2: Build Strategies (Week 2)
- [ ] Create custom strategies
- [ ] Test with real market data
- [ ] Practice adjustments
- [ ] Use hedge recommendations

### Phase 3: Advanced Usage (Week 3)
- [ ] Multi-scenario testing
- [ ] Greek management
- [ ] Strategy optimization
- [ ] Risk budgeting

### Phase 4: Live Trading Integration (Week 4)
- [ ] Paper trade alongside app
- [ ] Track actual vs predicted
- [ ] Refine entry rules
- [ ] Develop adjustment triggers

---

## 📚 Additional Resources

### Learn More About:

**Options Basics:**
- YouTube: "Options Trading for Beginners"
- Book: "Options as a Strategic Investment"

**Greeks:**
- Investopedia: Options Greeks
- TastyTrade: Greeks Explained

**Strategies:**
- OptionAlpha: Strategy Guides
- Zerodha Varsity: Options Module

**Technical:**
- Black-Scholes Model
- Volatility Trading
- Risk Management

---

## 🎯 Pro Tips

1. **Always Test First**
   - Never enter a trade without analysis
   - Use scenario analysis extensively
   - Check worst cases

2. **Monitor Greeks**
   - Delta: Keep manageable
   - Gamma: Watch near expiry
   - Theta: Know your decay
   - Vega: Track IV changes

3. **Use Hedges Wisely**
   - Don't over-hedge
   - Cost vs benefit
   - Keep it simple

4. **Adjust Proactively**
   - Don't wait for max loss
   - Set adjustment triggers
   - Have a plan

5. **Track Everything**
   - Record all trades
   - Note what worked
   - Learn from mistakes

---

## 🤝 Support & Enhancement

### Want to Add Features?

The app is built modularly. Easy to extend:

**New Strategy Presets:**
```python
if st.button("Butterfly"):
    st.session_state.legs = [
        # Add butterfly legs here
    ]
```

**Custom Greeks:**
```python
def calculate_custom_greek():
    # Your formula
    pass
```

**More Visualizations:**
```python
def plot_custom_chart():
    fig = go.Figure()
    # Your visualization
    return fig
```

### Need Help?

- Check troubleshooting section
- Review example workflows
- Test with preset strategies first
- Start simple, add complexity

---

## ✅ Launch Checklist

Before going live:

- [ ] Dependencies installed
- [ ] App runs successfully
- [ ] All tabs working
- [ ] Presets load correctly
- [ ] Can add custom legs
- [ ] Charts render properly
- [ ] Scenarios calculate
- [ ] Recommendations appear
- [ ] Can implement hedges
- [ ] Position manager works

---

## 🎉 You're Ready!

**Launch the app:**
```bash
streamlit run options_strategy_analyzer.py
```

**Start with a preset:**
1. Click "Iron Condor"
2. Explore all tabs
3. Test scenarios
4. Get comfortable

**Then build your own:**
1. Clear all
2. Add legs manually
3. Analyze
4. Optimize

**Happy Trading!** 📈

---

## 📄 License & Credits

Built with:
- Streamlit (UI framework)
- Plotly (Visualizations)
- SciPy (Black-Scholes)
- NumPy & Pandas (Calculations)

Based on:
- Black-Scholes option pricing model
- Standard options Greeks formulas
- Portfolio optimization theory

---

**Version:** 1.0  
**Last Updated:** February 2026  
**Status:** Production Ready  

Enjoy your Options Strategy Analyzer! 🚀
