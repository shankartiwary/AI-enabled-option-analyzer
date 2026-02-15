#!/usr/bin/env python3
"""
Options Strategy Analyzer - Streamlit App
Complete what-if analysis, hedging recommendations, and P&L visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Options Strategy Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    .recommendation-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== OPTIONS PRICING ENGINE ====================

def black_scholes(spot, strike, time, volatility, rate=0.06, option_type='CE'):
    """Black-Scholes option pricing"""
    if time <= 0:
        if option_type == 'CE':
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)
    
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time) / (volatility * np.sqrt(time))
    d2 = d1 - volatility * np.sqrt(time)
    
    if option_type == 'CE':
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2)
    else:
        price = strike * np.exp(-rate * time) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    return price

def calculate_greeks(spot, strike, time, volatility, rate=0.06, option_type='CE'):
    """Calculate all Greeks"""
    if time <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time) / (volatility * np.sqrt(time))
    d2 = d1 - volatility * np.sqrt(time)
    
    # Delta
    if option_type == 'CE':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time))
    
    # Theta
    term1 = -(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time))
    if option_type == 'CE':
        term2 = rate * strike * np.exp(-rate * time) * norm.cdf(d2)
        theta = (term1 - term2) / 365
    else:
        term2 = rate * strike * np.exp(-rate * time) * norm.cdf(-d2)
        theta = (term1 + term2) / 365
    
    # Vega (same for calls and puts)
    vega = spot * norm.pdf(d1) * np.sqrt(time) / 100
    
    # Rho
    if option_type == 'CE':
        rho = strike * time * np.exp(-rate * time) * norm.cdf(d2) / 100
    else:
        rho = -strike * time * np.exp(-rate * time) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def implied_volatility(market_price, spot, strike, time, rate=0.06, option_type='CE', max_iterations=100):
    """Calculate implied volatility using Newton-Raphson"""
    if time <= 0:
        return 0.20
    
    iv = 0.20  # Initial guess
    
    for i in range(max_iterations):
        price = black_scholes(spot, strike, time, iv, rate, option_type)
        vega = calculate_greeks(spot, strike, time, iv, rate, option_type)['vega'] * 100
        
        diff = market_price - price
        
        if abs(diff) < 0.01 or vega == 0:
            return iv
        
        iv = iv + diff / vega
        iv = max(0.01, min(2.0, iv))  # Constrain between 1% and 200%
    
    return iv

# ==================== DATA STRUCTURES ====================

@dataclass
class OptionLeg:
    """Single option leg"""
    strike: float
    expiry_days: int
    option_type: str  # 'CE' or 'PE'
    side: str  # 'BUY' or 'SELL'
    quantity: int
    lot_size: int
    entry_price: float
    current_iv: float = 0.20
    
    def __hash__(self):
        return hash((self.strike, self.expiry_days, self.option_type, self.side))

# ==================== POSITION ANALYZER ====================

class PositionAnalyzer:
    """Core analysis engine"""
    
    def __init__(self, spot_price: float, legs: List[OptionLeg], risk_free_rate: float = 0.06):
        self.spot_price = spot_price
        self.legs = legs
        self.rate = risk_free_rate
    
    def calculate_position_value(self, spot: float = None, iv_multiplier: float = 1.0, days_passed: int = 0):
        """Calculate current position value"""
        if spot is None:
            spot = self.spot_price
        
        total_value = 0
        
        for leg in self.legs:
            new_iv = leg.current_iv * iv_multiplier
            new_time = max(0, leg.expiry_days - days_passed)
            
            current_price = black_scholes(
                spot, leg.strike, new_time / 365, new_iv, self.rate, leg.option_type
            )
            
            leg_value = current_price * leg.quantity * leg.lot_size
            
            if leg.side == 'SELL':
                leg_value = -leg_value
            
            total_value += leg_value
        
        return total_value
    
    def calculate_initial_value(self):
        """Calculate initial investment/credit"""
        total = 0
        for leg in self.legs:
            value = leg.entry_price * leg.quantity * leg.lot_size
            if leg.side == 'BUY':
                total -= value
            else:
                total += value
        return total
    
    def calculate_pnl(self, spot: float = None, iv_multiplier: float = 1.0, days_passed: int = 0):
        """Calculate P&L"""
        current_value = self.calculate_position_value(spot, iv_multiplier, days_passed)
        initial_value = self.calculate_initial_value()
        return current_value + initial_value
    
    def calculate_greeks(self):
        """Calculate position Greeks"""
        net_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for leg in self.legs:
            greeks = calculate_greeks(
                self.spot_price, leg.strike, leg.expiry_days / 365,
                leg.current_iv, self.rate, leg.option_type
            )
            
            multiplier = leg.quantity * leg.lot_size
            if leg.side == 'SELL':
                multiplier *= -1
            
            for key in net_greeks:
                net_greeks[key] += greeks[key] * multiplier
        
        return net_greeks
    
    def scenario_analysis(self, spot_range: np.ndarray, iv_changes: List[float], time_points: List[int]):
        """Comprehensive scenario analysis"""
        scenarios = []
        
        for spot in spot_range:
            for iv_mult in iv_changes:
                for days in time_points:
                    pnl = self.calculate_pnl(spot, iv_mult, days)
                    scenarios.append({
                        'spot': spot,
                        'spot_change': spot - self.spot_price,
                        'spot_pct': (spot - self.spot_price) / self.spot_price * 100,
                        'iv_multiplier': iv_mult,
                        'iv_change_pct': (iv_mult - 1) * 100,
                        'days_passed': days,
                        'days_remaining': max(0, min([leg.expiry_days for leg in self.legs]) - days),
                        'pnl': pnl
                    })
        
        return pd.DataFrame(scenarios)
    
    def calculate_breakevens(self):
        """Calculate breakeven points"""
        spot_range = np.linspace(self.spot_price * 0.7, self.spot_price * 1.3, 1000)
        pnls = [self.calculate_pnl(spot, 1.0, 0) for spot in spot_range]
        
        breakevens = []
        for i in range(len(pnls) - 1):
            if (pnls[i] < 0 and pnls[i+1] > 0) or (pnls[i] > 0 and pnls[i+1] < 0):
                breakevens.append(spot_range[i])
        
        return breakevens
    
    def calculate_max_profit_loss(self):
        """Calculate theoretical max profit and loss"""
        spot_range = np.linspace(self.spot_price * 0.5, self.spot_price * 1.5, 1000)
        pnls = [self.calculate_pnl(spot, 1.0, min([leg.expiry_days for leg in self.legs])) for spot in spot_range]
        
        max_profit = max(pnls)
        max_loss = min(pnls)
        
        # Find spot prices where max profit/loss occur
        max_profit_spot = spot_range[np.argmax(pnls)]
        max_loss_spot = spot_range[np.argmin(pnls)]
        
        return {
            'max_profit': max_profit,
            'max_profit_spot': max_profit_spot,
            'max_loss': max_loss,
            'max_loss_spot': max_loss_spot
        }

# ==================== HEDGE RECOMMENDER ====================

class HedgeRecommender:
    """AI-powered hedge recommendation engine"""
    
    def __init__(self, analyzer: PositionAnalyzer):
        self.analyzer = analyzer
    
    def recommend_hedges(self, available_strikes: List[float], max_cost: float = 100000):
        """Recommend optimal hedges"""
        current_greeks = self.analyzer.calculate_greeks()
        current_pnl = self.analyzer.calculate_pnl()
        max_loss_info = self.analyzer.calculate_max_profit_loss()
        
        recommendations = []
        
        # Analyze delta exposure
        if abs(current_greeks['delta']) > 100:
            recommendations.extend(
                self._recommend_delta_hedge(available_strikes, current_greeks, max_cost)
            )
        
        # Analyze gamma exposure
        if abs(current_greeks['gamma']) > 0.5:
            recommendations.extend(
                self._recommend_gamma_hedge(available_strikes, current_greeks, max_cost)
            )
        
        # Protective hedges if approaching max loss
        if current_pnl < max_loss_info['max_loss'] * 0.5:
            recommendations.extend(
                self._recommend_protective_hedge(available_strikes, max_loss_info, max_cost)
            )
        
        # Rank by expected value
        for rec in recommendations:
            rec['ev_score'] = self._calculate_expected_value(rec)
        
        recommendations.sort(key=lambda x: x['ev_score'], reverse=True)
        
        return recommendations[:5]  # Top 5
    
    def _recommend_delta_hedge(self, strikes: List[float], current_greeks: dict, max_cost: float):
        """Recommend delta-neutral hedge"""
        recommendations = []
        current_delta = current_greeks['delta']
        
        for strike in strikes:
            # Determine option type
            if current_delta > 0:  # Long delta, buy puts
                option_type = 'PE'
                side = 'BUY'
            else:  # Short delta, buy calls
                option_type = 'CE'
                side = 'BUY'
            
            # Calculate hedge greeks
            hedge_greeks = calculate_greeks(
                self.analyzer.spot_price, strike, 30 / 365, 0.20, self.analyzer.rate, option_type
            )
            
            # Calculate quantity needed
            quantity_needed = abs(current_delta / (hedge_greeks['delta'] * 50))
            quantity_needed = int(np.ceil(quantity_needed))
            
            if quantity_needed > 0:
                price = black_scholes(
                    self.analyzer.spot_price, strike, 30 / 365, 0.20, self.analyzer.rate, option_type
                )
                
                cost = price * quantity_needed * 50
                
                if cost <= max_cost:
                    new_delta = current_delta + (hedge_greeks['delta'] * quantity_needed * 50 * (1 if side == 'BUY' else -1))
                    
                    recommendations.append({
                        'type': 'Delta Hedge',
                        'action': f"{side} {quantity_needed} lots {strike} {option_type}",
                        'strike': strike,
                        'option_type': option_type,
                        'side': side,
                        'quantity': quantity_needed,
                        'price': price,
                        'cost': cost,
                        'new_delta': new_delta,
                        'delta_reduction': abs(current_delta) - abs(new_delta),
                        'reason': f"Neutralize {current_delta:.0f} delta exposure"
                    })
        
        return recommendations
    
    def _recommend_gamma_hedge(self, strikes: List[float], current_greeks: dict, max_cost: float):
        """Recommend gamma hedge"""
        recommendations = []
        current_gamma = current_greeks['gamma']
        
        # High gamma means sensitivity to price changes
        # Add opposite gamma position
        
        atm_strike = min(strikes, key=lambda x: abs(x - self.analyzer.spot_price))
        
        for option_type in ['CE', 'PE']:
            hedge_greeks = calculate_greeks(
                self.analyzer.spot_price, atm_strike, 30 / 365, 0.20, self.analyzer.rate, option_type
            )
            
            # Buy options to add positive gamma (reduce negative gamma)
            quantity = int(abs(current_gamma / (hedge_greeks['gamma'] * 50))) + 1
            
            price = black_scholes(
                self.analyzer.spot_price, atm_strike, 30 / 365, 0.20, self.analyzer.rate, option_type
            )
            
            cost = price * quantity * 50
            
            if cost <= max_cost:
                new_gamma = current_gamma + (hedge_greeks['gamma'] * quantity * 50)
                
                recommendations.append({
                    'type': 'Gamma Hedge',
                    'action': f"BUY {quantity} lots {atm_strike} {option_type}",
                    'strike': atm_strike,
                    'option_type': option_type,
                    'side': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'cost': cost,
                    'new_gamma': new_gamma,
                    'gamma_reduction': abs(current_gamma) - abs(new_gamma),
                    'reason': f"Reduce {current_gamma:.2f} gamma risk"
                })
        
        return recommendations
    
    def _recommend_protective_hedge(self, strikes: List[float], max_loss_info: dict, max_cost: float):
        """Recommend protective hedge to cap losses"""
        recommendations = []
        
        current_spot = self.analyzer.spot_price
        max_loss_spot = max_loss_info['max_loss_spot']
        
        # Determine direction of risk
        if max_loss_spot > current_spot:
            # Risk on upside, buy calls
            option_type = 'CE'
            relevant_strikes = [s for s in strikes if s >= current_spot]
        else:
            # Risk on downside, buy puts
            option_type = 'PE'
            relevant_strikes = [s for s in strikes if s <= current_spot]
        
        for strike in relevant_strikes[:3]:  # Try 3 strikes
            price = black_scholes(
                current_spot, strike, 30 / 365, 0.20, self.analyzer.rate, option_type
            )
            
            # Calculate how many lots needed to significantly reduce max loss
            for quantity in [10, 25, 50]:
                cost = price * quantity * 50
                
                if cost <= max_cost:
                    # Estimate new max loss with hedge
                    estimated_reduction = cost * 1.5  # Conservative estimate
                    
                    recommendations.append({
                        'type': 'Protective Hedge',
                        'action': f"BUY {quantity} lots {strike} {option_type}",
                        'strike': strike,
                        'option_type': option_type,
                        'side': 'BUY',
                        'quantity': quantity,
                        'price': price,
                        'cost': cost,
                        'estimated_loss_reduction': estimated_reduction,
                        'reason': f"Cap downside risk (max loss: ₹{max_loss_info['max_loss']:,.0f})"
                    })
        
        return recommendations
    
    def _calculate_expected_value(self, recommendation: dict):
        """Calculate expected value score for recommendation"""
        # Simple scoring: benefit / cost ratio
        cost = recommendation['cost']
        
        if 'delta_reduction' in recommendation:
            benefit = recommendation['delta_reduction'] * 100
        elif 'gamma_reduction' in recommendation:
            benefit = recommendation['gamma_reduction'] * 10000
        elif 'estimated_loss_reduction' in recommendation:
            benefit = recommendation['estimated_loss_reduction']
        else:
            benefit = 0
        
        if cost > 0:
            return benefit / cost
        return 0

# ==================== VOLATILITY ANALYZER ====================

def analyze_volatility(current_iv: float, historical_iv: float = None):
    """Analyze volatility metrics"""
    analysis = {
        'current_iv': current_iv,
        'current_iv_pct': current_iv * 100,
    }
    
    if historical_iv:
        analysis['historical_iv'] = historical_iv
        analysis['iv_percentile'] = (current_iv / historical_iv) * 100
        
        if analysis['iv_percentile'] > 80:
            analysis['level'] = 'HIGH'
            analysis['interpretation'] = 'Options are expensive. Consider selling strategies.'
        elif analysis['iv_percentile'] < 20:
            analysis['level'] = 'LOW'
            analysis['interpretation'] = 'Options are cheap. Consider buying strategies.'
        else:
            analysis['level'] = 'NORMAL'
            analysis['interpretation'] = 'IV is in normal range.'
    else:
        # Simple classification
        if current_iv > 0.30:
            analysis['level'] = 'HIGH'
        elif current_iv < 0.15:
            analysis['level'] = 'LOW'
        else:
            analysis['level'] = 'NORMAL'
    
    return analysis

# ==================== VISUALIZATION ====================

def plot_pnl_curve(analyzer: PositionAnalyzer, iv_multiplier: float = 1.0, days_passed: int = 0):
    """Create interactive P&L curve"""
    spot_range = np.linspace(analyzer.spot_price * 0.85, analyzer.spot_price * 1.15, 200)
    pnls = [analyzer.calculate_pnl(spot, iv_multiplier, days_passed) for spot in spot_range]
    
    fig = go.Figure()
    
    # P&L curve
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=pnls,
        mode='lines',
        name='P&L',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    
    # Current spot
    current_pnl = analyzer.calculate_pnl(analyzer.spot_price, iv_multiplier, days_passed)
    fig.add_vline(x=analyzer.spot_price, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_trace(go.Scatter(
        x=[analyzer.spot_price],
        y=[current_pnl],
        mode='markers',
        name='Current',
        marker=dict(size=12, color='green')
    ))
    
    # Breakevens
    breakevens = analyzer.calculate_breakevens()
    if breakevens:
        fig.add_trace(go.Scatter(
            x=breakevens,
            y=[0] * len(breakevens),
            mode='markers',
            name='Breakeven',
            marker=dict(size=10, color='orange', symbol='diamond')
        ))
    
    fig.update_layout(
        title=f'P&L Curve (IV: {(iv_multiplier-1)*100:+.0f}%, Days: {days_passed})',
        xaxis_title='Spot Price (₹)',
        yaxis_title='P&L (₹)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_scenario_heatmap(scenarios_df: pd.DataFrame):
    """Create scenario heatmap"""
    # Pivot for heatmap
    pivot = scenarios_df.pivot_table(
        values='pnl',
        index='iv_change_pct',
        columns='spot_change',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        text=np.round(pivot.values / 1000, 1),
        texttemplate='₹%{text}k',
        textfont={"size": 10},
        colorbar=dict(title="P&L (₹)")
    ))
    
    fig.update_layout(
        title='Scenario Analysis Heatmap',
        xaxis_title='Spot Change (₹)',
        yaxis_title='IV Change (%)',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_greeks_gauge(greeks: dict):
    """Create Greeks dashboard"""
    fig = go.Figure()
    
    greek_names = ['Delta', 'Gamma', 'Theta', 'Vega']
    greek_keys = ['delta', 'gamma', 'theta', 'vega']
    greek_ranges = [(-500, 500), (-2, 2), (-50000, 50000), (-100000, 100000)]
    
    for i, (name, key, (vmin, vmax)) in enumerate(zip(greek_names, greek_keys, greek_ranges)):
        value = greeks[key]
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': name},
            domain={'row': i // 2, 'column': i % 2},
            gauge={
                'axis': {'range': [vmin, vmax]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [vmin, vmin + (vmax - vmin) * 0.33], 'color': "lightgray"},
                    {'range': [vmin + (vmax - vmin) * 0.33, vmin + (vmax - vmin) * 0.67], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
    
    fig.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=500,
        template='plotly_white'
    )
    
    return fig

# ==================== STREAMLIT APP ====================

def initialize_session_state():
    """Initialize session state"""
    if 'legs' not in st.session_state:
        st.session_state.legs = []
    if 'spot_price' not in st.session_state:
        st.session_state.spot_price = 24000.0
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📊 Options Strategy Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Input Section
    with st.sidebar:
        st.header("📋 Strategy Input")
        
        # Market params
        st.subheader("Market Parameters")
        spot_price = st.number_input("Spot Price (₹)", value=st.session_state.spot_price, step=50.0)
        st.session_state.spot_price = spot_price
        
        st.markdown("---")
        
        # Add leg
        st.subheader("Add Option Leg")
        
        col1, col2 = st.columns(2)
        with col1:
            strike = st.number_input("Strike", value=24000.0, step=50.0, key="new_strike")
            option_type = st.selectbox("Type", ["CE", "PE"], key="new_type")
            side = st.selectbox("Side", ["BUY", "SELL"], key="new_side")
        
        with col2:
            expiry_days = st.number_input("Days to Expiry", value=14, min_value=1, step=1, key="new_expiry")
            quantity = st.number_input("Quantity (lots)", value=50, min_value=1, step=1, key="new_qty")
            lot_size = st.number_input("Lot Size", value=50, min_value=1, step=1, key="new_lot")
        
        entry_price = st.number_input("Entry Price (₹)", value=100.0, step=5.0, key="new_price")
        current_iv = st.slider("Current IV", 0.10, 0.50, 0.20, 0.01, key="new_iv")
        
        if st.button("➕ Add Leg", use_container_width=True):
            leg = OptionLeg(
                strike=strike,
                expiry_days=expiry_days,
                option_type=option_type,
                side=side,
                quantity=quantity,
                lot_size=lot_size,
                entry_price=entry_price,
                current_iv=current_iv
            )
            st.session_state.legs.append(leg)
            st.success("✅ Leg added!")
            st.rerun()
        
        st.markdown("---")
        
        # Load presets
        st.subheader("📦 Load Preset Strategy")
        
        if st.button("Iron Condor", use_container_width=True):
            st.session_state.legs = [
                OptionLeg(24000, 14, 'CE', 'SELL', 50, 50, 150, 0.18),
                OptionLeg(24500, 14, 'CE', 'BUY', 50, 50, 50, 0.18),
                OptionLeg(23500, 14, 'PE', 'SELL', 50, 50, 145, 0.18),
                OptionLeg(23000, 14, 'PE', 'BUY', 50, 50, 45, 0.18),
            ]
            st.success("✅ Iron Condor loaded!")
            st.rerun()
        
        if st.button("Short Strangle", use_container_width=True):
            st.session_state.legs = [
                OptionLeg(24200, 14, 'CE', 'SELL', 100, 50, 180, 0.20),
                OptionLeg(23800, 14, 'PE', 'SELL', 100, 50, 175, 0.20),
            ]
            st.success("✅ Short Strangle loaded!")
            st.rerun()
        
        if st.button("Long Straddle", use_container_width=True):
            st.session_state.legs = [
                OptionLeg(24000, 14, 'CE', 'BUY', 50, 50, 200, 0.22),
                OptionLeg(24000, 14, 'PE', 'BUY', 50, 50, 195, 0.22),
            ]
            st.success("✅ Long Straddle loaded!")
            st.rerun()
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.legs = []
            st.rerun()
    
    # Main content
    if not st.session_state.legs:
        st.info("👈 Add option legs using the sidebar to begin analysis")
        st.markdown("""
        ### Quick Start
        1. Enter spot price
        2. Add individual legs or load a preset strategy
        3. View comprehensive analysis and recommendations
        
        ### Features
        - 📊 Real-time P&L and Greeks
        - 🔍 Multi-dimensional scenario analysis  
        - 💡 AI-powered hedge recommendations
        - 📈 Interactive visualizations
        - ⚡ What-if simulations
        """)
        return
    
    # Create analyzer
    st.session_state.analyzer = PositionAnalyzer(spot_price, st.session_state.legs)
    analyzer = st.session_state.analyzer
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎯 Scenario Analysis",
        "💡 Recommendations",
        "📈 P&L Curves",
        "🔧 Position Manager"
    ])
    
    with tab1:
        show_overview(analyzer)
    
    with tab2:
        show_scenario_analysis(analyzer)
    
    with tab3:
        show_recommendations(analyzer)
    
    with tab4:
        show_pnl_curves(analyzer)
    
    with tab5:
        show_position_manager()

def show_overview(analyzer: PositionAnalyzer):
    """Overview tab"""
    st.header("Position Overview")
    
    # Calculate metrics
    current_pnl = analyzer.calculate_pnl()
    greeks = analyzer.calculate_greeks()
    max_profit_loss = analyzer.calculate_max_profit_loss()
    breakevens = analyzer.calculate_breakevens()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current P&L", f"₹{current_pnl:,.0f}", 
                 delta=None if current_pnl == 0 else f"{current_pnl:+,.0f}")
    
    with col2:
        st.metric("Max Profit", f"₹{max_profit_loss['max_profit']:,.0f}",
                 delta=f"@ {max_profit_loss['max_profit_spot']:.0f}")
    
    with col3:
        st.metric("Max Loss", f"₹{max_profit_loss['max_loss']:,.0f}",
                 delta=f"@ {max_profit_loss['max_loss_spot']:.0f}")
    
    with col4:
        risk_pct = abs(current_pnl / max_profit_loss['max_loss'] * 100) if max_profit_loss['max_loss'] != 0 else 0
        st.metric("Risk Used", f"{risk_pct:.1f}%")
    
    st.markdown("---")
    
    # Greeks
    st.subheader("Position Greeks")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        greeks_df = pd.DataFrame([greeks]).T
        greeks_df.columns = ['Value']
        greeks_df.index = ['Delta', 'Gamma', 'Theta (₹/day)', 'Vega (₹/1%)', 'Rho']
        greeks_df['Value'] = greeks_df['Value'].map(lambda x: f"{x:,.2f}")
        st.dataframe(greeks_df, use_container_width=True)
        
        # Risk assessment
        st.markdown("#### Risk Assessment")
        
        risk_score = 0
        if abs(greeks['delta']) > 200:
            risk_score += 3
        elif abs(greeks['delta']) > 100:
            risk_score += 2
        
        if abs(greeks['gamma']) > 1:
            risk_score += 3
        elif abs(greeks['gamma']) > 0.5:
            risk_score += 2
        
        if risk_pct > 70:
            risk_score += 4
        elif risk_pct > 40:
            risk_score += 2
        
        if risk_score >= 7:
            st.markdown('<div class="risk-high">🔴 HIGH RISK</div>', unsafe_allow_html=True)
        elif risk_score >= 4:
            st.markdown('<div class="risk-medium">🟡 MEDIUM RISK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">🟢 LOW RISK</div>', unsafe_allow_html=True)
    
    with col2:
        fig = plot_greeks_gauge(greeks)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Position details
    st.subheader("Position Legs")
    
    legs_data = []
    for i, leg in enumerate(st.session_state.legs):
        legs_data.append({
            '#': i + 1,
            'Strike': leg.strike,
            'Type': leg.option_type,
            'Side': leg.side,
            'Qty': leg.quantity,
            'Lot': leg.lot_size,
            'Entry': f"₹{leg.entry_price:.2f}",
            'IV': f"{leg.current_iv*100:.1f}%",
            'Expiry': f"{leg.expiry_days}d"
        })
    
    st.dataframe(pd.DataFrame(legs_data), use_container_width=True, hide_index=True)
    
    # Breakevens
    if breakevens:
        st.markdown("#### Breakeven Points")
        for i, be in enumerate(breakevens, 1):
            st.write(f"**Breakeven {i}:** ₹{be:.2f}")

def show_scenario_analysis(analyzer: PositionAnalyzer):
    """Scenario analysis tab"""
    st.header("Scenario Analysis")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spot_range_pct = st.slider("Spot Range (%)", 5, 30, 15)
    with col2:
        iv_range = st.multiselect("IV Changes", ["-10%", "-5%", "0%", "+5%", "+10%"], 
                                  default=["-5%", "0%", "+5%"])
    with col3:
        time_points = st.multiselect("Days Passed", [0, 3, 7, 14], default=[0, 7])
    
    # Convert IV to multipliers
    iv_map = {"-10%": 0.90, "-5%": 0.95, "0%": 1.0, "+5%": 1.05, "+10%": 1.10}
    iv_multipliers = [iv_map[iv] for iv in iv_range]
    
    # Generate scenarios
    spot_min = analyzer.spot_price * (1 - spot_range_pct / 100)
    spot_max = analyzer.spot_price * (1 + spot_range_pct / 100)
    spot_range = np.linspace(spot_min, spot_max, 15)
    
    scenarios_df = analyzer.scenario_analysis(spot_range, iv_multipliers, time_points)
    
    # Show immediate scenario matrix (days=0)
    st.subheader("Immediate Scenarios (Today)")
    
    immediate = scenarios_df[scenarios_df['days_passed'] == 0]
    pivot = immediate.pivot_table(
        values='pnl',
        index='iv_change_pct',
        columns='spot_change',
        aggfunc='mean'
    )
    
    # Format pivot
    pivot_display = pivot.applymap(lambda x: f"₹{x/1000:.1f}k")
    st.dataframe(pivot_display, use_container_width=True)
    
    # Heatmap
    fig_heatmap = plot_scenario_heatmap(immediate)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Worst/Best cases
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔻 Worst 5 Scenarios")
        worst = scenarios_df.nsmallest(5, 'pnl')[['spot', 'spot_pct', 'iv_change_pct', 'days_passed', 'pnl']]
        worst['pnl'] = worst['pnl'].map(lambda x: f"₹{x:,.0f}")
        worst['spot_pct'] = worst['spot_pct'].map(lambda x: f"{x:+.1f}%")
        worst['iv_change_pct'] = worst['iv_change_pct'].map(lambda x: f"{x:+.1f}%")
        st.dataframe(worst, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("🔺 Best 5 Scenarios")
        best = scenarios_df.nlargest(5, 'pnl')[['spot', 'spot_pct', 'iv_change_pct', 'days_passed', 'pnl']]
        best['pnl'] = best['pnl'].map(lambda x: f"₹{x:,.0f}")
        best['spot_pct'] = best['spot_pct'].map(lambda x: f"{x:+.1f}%")
        best['iv_change_pct'] = best['iv_change_pct'].map(lambda x: f"{x:+.1f}%")
        st.dataframe(best, hide_index=True, use_container_width=True)

def show_recommendations(analyzer: PositionAnalyzer):
    """Recommendations tab"""
    st.header("AI-Powered Recommendations")
    
    # Generate available strikes
    current_spot = analyzer.spot_price
    available_strikes = [
        current_spot - 400, current_spot - 200, current_spot - 100,
        current_spot, 
        current_spot + 100, current_spot + 200, current_spot + 400
    ]
    
    # Get recommendations
    recommender = HedgeRecommender(analyzer)
    max_cost = st.slider("Maximum Hedge Cost (₹)", 10000, 500000, 100000, 10000)
    
    recommendations = recommender.recommend_hedges(available_strikes, max_cost)
    
    if not recommendations:
        st.success("✅ Position looks good! No urgent hedges needed.")
        return
    
    st.markdown(f"Found **{len(recommendations)}** hedge recommendations:")
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"💡 Recommendation #{i}: {rec['type']}", expanded=(i == 1)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="recommendation-box">
                <h4>{rec['action']}</h4>
                <p><strong>Reason:</strong> {rec['reason']}</p>
                <p><strong>Cost:</strong> ₹{rec['cost']:,.0f}</p>
                <p><strong>Expected Value Score:</strong> {rec['ev_score']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Details
                st.write("**Impact:**")
                if 'new_delta' in rec:
                    current_delta = analyzer.calculate_greeks()['delta']
                    st.write(f"- Delta: {current_delta:.0f} → {rec['new_delta']:.0f} ({rec['delta_reduction']:.0f} reduction)")
                if 'new_gamma' in rec:
                    current_gamma = analyzer.calculate_greeks()['gamma']
                    st.write(f"- Gamma: {current_gamma:.2f} → {rec['new_gamma']:.2f}")
                if 'estimated_loss_reduction' in rec:
                    st.write(f"- Estimated loss reduction: ₹{rec['estimated_loss_reduction']:,.0f}")
            
            with col2:
                if st.button(f"✅ Implement Hedge #{i}", key=f"hedge_{i}"):
                    # Add hedge leg
                    new_leg = OptionLeg(
                        strike=rec['strike'],
                        expiry_days=30,
                        option_type=rec['option_type'],
                        side=rec['side'],
                        quantity=rec['quantity'],
                        lot_size=50,
                        entry_price=rec['price'],
                        current_iv=0.20
                    )
                    st.session_state.legs.append(new_leg)
                    st.success(f"✅ Hedge added!")
                    st.rerun()

def show_pnl_curves(analyzer: PositionAnalyzer):
    """P&L curves tab"""
    st.header("P&L Visualizations")
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        iv_change = st.selectbox("IV Scenario", 
                                 ["-10%", "-5%", "Current", "+5%", "+10%"],
                                 index=2)
        iv_map = {"-10%": 0.90, "-5%": 0.95, "Current": 1.0, "+5%": 1.05, "+10%": 1.10}
        iv_mult = iv_map[iv_change]
    
    with col2:
        days = st.selectbox("Time Scenario",
                           ["Today", "3 days", "7 days", "At Expiry"],
                           index=0)
        days_map = {"Today": 0, "3 days": 3, "7 days": 7, "At Expiry": min([leg.expiry_days for leg in st.session_state.legs])}
        days_passed = days_map[days]
    
    # Main P&L curve
    fig = plot_pnl_curve(analyzer, iv_mult, days_passed)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Multiple curves comparison
    st.subheader("Time Decay Comparison")
    
    fig_multi = go.Figure()
    
    colors = ['blue', 'green', 'orange', 'red']
    time_points = [0, 3, 7, min([leg.expiry_days for leg in st.session_state.legs])]
    labels = ['Today', '3 Days', '7 Days', 'Expiry']
    
    spot_range = np.linspace(analyzer.spot_price * 0.85, analyzer.spot_price * 1.15, 200)
    
    for time_point, label, color in zip(time_points, labels, colors):
        pnls = [analyzer.calculate_pnl(spot, 1.0, time_point) for spot in spot_range]
        fig_multi.add_trace(go.Scatter(
            x=spot_range,
            y=pnls,
            mode='lines',
            name=label,
            line=dict(color=color, width=2)
        ))
    
    fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_multi.add_vline(x=analyzer.spot_price, line_dash="dash", line_color="green", opacity=0.5)
    
    fig_multi.update_layout(
        title='P&L Evolution Over Time',
        xaxis_title='Spot Price (₹)',
        yaxis_title='P&L (₹)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_multi, use_container_width=True)

def show_position_manager():
    """Position manager tab"""
    st.header("Position Manager")
    
    if not st.session_state.legs:
        st.info("No legs to manage")
        return
    
    st.subheader("Current Legs")
    
    for i, leg in enumerate(st.session_state.legs):
        with st.expander(f"Leg {i+1}: {leg.side} {leg.strike} {leg.option_type}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Strike:** {leg.strike}")
                st.write(f"**Type:** {leg.option_type}")
                st.write(f"**Side:** {leg.side}")
            
            with col2:
                st.write(f"**Quantity:** {leg.quantity}")
                st.write(f"**Lot Size:** {leg.lot_size}")
                st.write(f"**Entry Price:** ₹{leg.entry_price}")
            
            with col3:
                st.write(f"**IV:** {leg.current_iv*100:.1f}%")
                st.write(f"**Days to Expiry:** {leg.expiry_days}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🗑️ Remove Leg {i+1}", key=f"remove_{i}"):
                    st.session_state.legs.pop(i)
                    st.rerun()
            
            with col2:
                if st.button(f"📊 Analyze Solo", key=f"solo_{i}"):
                    st.session_state.legs = [leg]
                    st.rerun()

if __name__ == "__main__":
    main()
