#!/usr/bin/env python3
"""
Options Strategy Analyzer - Plotly Dash App
Complete what-if analysis, hedging recommendations, and P&L visualization
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import json

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
    
    def max_profit(self):
        """Calculate maximum profit"""
        spot_range = np.linspace(self.spot_price * 0.5, self.spot_price * 1.5, 1000)
        pnls = [self.calculate_pnl(spot, 1.0, 0) for spot in spot_range]
        max_pnl = max(pnls)
        return max_pnl if max_pnl < 1e6 else float('inf')
    
    def max_loss(self):
        """Calculate maximum loss"""
        spot_range = np.linspace(self.spot_price * 0.5, self.spot_price * 1.5, 1000)
        pnls = [self.calculate_pnl(spot, 1.0, 0) for spot in spot_range]
        min_pnl = min(pnls)
        return min_pnl if min_pnl > -1e6 else float('-inf')
    
    def breakeven_points(self):
        """Find breakeven points"""
        spot_range = np.linspace(self.spot_price * 0.5, self.spot_price * 1.5, 1000)
        pnls = [self.calculate_pnl(spot, 1.0, 0) for spot in spot_range]
        
        breakevens = []
        for i in range(len(pnls) - 1):
            if pnls[i] * pnls[i + 1] < 0:
                breakevens.append(spot_range[i])
        
        return breakevens
    
    def scenario_analysis(self, spot_range, iv_multipliers, time_points):
        """Run comprehensive scenario analysis"""
        scenarios = []
        
        for spot in spot_range:
            spot_pct = ((spot - self.spot_price) / self.spot_price) * 100
            spot_change = f"{spot_pct:+.1f}%"
            
            for iv_mult in iv_multipliers:
                iv_change_pct = (iv_mult - 1.0) * 100
                
                for days in time_points:
                    pnl = self.calculate_pnl(spot, iv_mult, days)
                    
                    scenarios.append({
                        'spot': spot,
                        'spot_pct': spot_pct,
                        'spot_change': spot_change,
                        'iv_multiplier': iv_mult,
                        'iv_change_pct': iv_change_pct,
                        'days_passed': days,
                        'pnl': pnl
                    })
        
        return pd.DataFrame(scenarios)

# ==================== HEDGE RECOMMENDER ====================

class HedgeRecommender:
    """Smart hedging recommendations"""
    
    def __init__(self, analyzer: PositionAnalyzer):
        self.analyzer = analyzer
        self.greeks = analyzer.calculate_greeks()
    
    def recommend_hedges(self, available_strikes: List[float], max_cost: float):
        """Generate hedge recommendations"""
        recommendations = []
        
        # Delta hedge
        if abs(self.greeks['delta']) > 500:
            rec = self._delta_hedge(available_strikes, max_cost)
            if rec:
                recommendations.append(rec)
        
        # Gamma hedge
        if abs(self.greeks['gamma']) > 0.5:
            rec = self._gamma_hedge(available_strikes, max_cost)
            if rec:
                recommendations.append(rec)
        
        # Theta protection
        if self.greeks['theta'] < -500:
            rec = self._theta_protection(available_strikes, max_cost)
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    def _delta_hedge(self, strikes, max_cost):
        """Recommend delta hedge"""
        delta = self.greeks['delta']
        
        if abs(delta) < 500:
            return None
        
        # Find ATM strike
        atm_strike = min(strikes, key=lambda x: abs(x - self.analyzer.spot_price))
        
        # Determine hedge type
        if delta > 0:
            option_type = 'PE'
            side = 'BUY'
            reason = f"Position is long delta ({delta:.0f}). Buying puts reduces directional risk."
        else:
            option_type = 'CE'
            side = 'BUY'
            reason = f"Position is short delta ({delta:.0f}). Buying calls reduces directional risk."
        
        # Calculate quantity needed
        price = black_scholes(self.analyzer.spot_price, atm_strike, 30/365, 0.20, 0.06, option_type)
        quantity = min(abs(delta) // 50, max_cost // (price * 50))
        
        if quantity == 0:
            return None
        
        cost = price * quantity * 50
        
        return {
            'type': 'Delta Hedge',
            'action': f"{side} {int(quantity)} lots of {atm_strike} {option_type}",
            'reason': reason,
            'strike': atm_strike,
            'option_type': option_type,
            'side': side,
            'quantity': int(quantity),
            'price': price,
            'cost': cost,
            'ev_score': abs(delta) / (cost + 1),
            'new_delta': delta - (quantity * 50 * (0.5 if option_type == 'CE' else -0.5)),
            'delta_reduction': abs(quantity * 50 * 0.5)
        }
    
    def _gamma_hedge(self, strikes, max_cost):
        """Recommend gamma hedge"""
        gamma = self.greeks['gamma']
        
        if abs(gamma) < 0.5:
            return None
        
        atm_strike = min(strikes, key=lambda x: abs(x - self.analyzer.spot_price))
        
        if gamma > 0:
            side = 'SELL'
            option_type = 'CE'
            reason = f"Position has high positive gamma ({gamma:.2f}). Selling options reduces gamma exposure."
        else:
            side = 'BUY'
            option_type = 'CE'
            reason = f"Position has high negative gamma ({gamma:.2f}). Buying options increases gamma."
        
        price = black_scholes(self.analyzer.spot_price, atm_strike, 30/365, 0.20, 0.06, option_type)
        quantity = min(abs(gamma) * 100, max_cost // (price * 50))
        
        if quantity == 0:
            return None
        
        cost = price * quantity * 50
        
        return {
            'type': 'Gamma Hedge',
            'action': f"{side} {int(quantity)} lots of {atm_strike} {option_type}",
            'reason': reason,
            'strike': atm_strike,
            'option_type': option_type,
            'side': side,
            'quantity': int(quantity),
            'price': price,
            'cost': cost,
            'ev_score': abs(gamma) / (cost / 10000 + 1),
            'new_gamma': gamma - (quantity * 50 * 0.01) * (-1 if side == 'SELL' else 1)
        }
    
    def _theta_protection(self, strikes, max_cost):
        """Recommend theta protection"""
        theta = self.greeks['theta']
        
        if theta > -500:
            return None
        
        atm_strike = min(strikes, key=lambda x: abs(x - self.analyzer.spot_price))
        
        reason = f"Position loses ₹{abs(theta):.0f} per day to time decay. Selling options generates premium."
        
        price = black_scholes(self.analyzer.spot_price, atm_strike, 7/365, 0.20, 0.06, 'CE')
        quantity = min(abs(theta) // 20, max_cost // (price * 50))
        
        if quantity == 0:
            return None
        
        cost = -price * quantity * 50  # Negative because selling
        
        return {
            'type': 'Theta Protection',
            'action': f"SELL {int(quantity)} lots of {atm_strike} CE (weekly)",
            'reason': reason,
            'strike': atm_strike,
            'option_type': 'CE',
            'side': 'SELL',
            'quantity': int(quantity),
            'price': price,
            'cost': cost,
            'ev_score': abs(theta) / (abs(cost) + 1),
            'estimated_loss_reduction': quantity * price * 50 * 7
        }

# ==================== PLOTTING FUNCTIONS ====================

def plot_pnl_curve(analyzer: PositionAnalyzer, iv_multiplier: float = 1.0, days_passed: int = 0):
    """Plot P&L curve"""
    spot_range = np.linspace(analyzer.spot_price * 0.85, analyzer.spot_price * 1.15, 200)
    pnls = [analyzer.calculate_pnl(spot, iv_multiplier, days_passed) for spot in spot_range]
    
    fig = go.Figure()
    
    # Main P&L curve
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=pnls,
        mode='lines',
        name='P&L',
        line=dict(color='blue', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 123, 255, 0.1)'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    
    # Current spot
    fig.add_vline(x=analyzer.spot_price, line_dash="dash", line_color="green", 
                  opacity=0.5, annotation_text="Current Spot")
    
    # Breakevens
    breakevens = analyzer.breakeven_points()
    for be in breakevens:
        fig.add_vline(x=be, line_dash="dot", line_color="orange", opacity=0.5)
    
    fig.update_layout(
        title=f'P&L Curve (IV: {iv_multiplier*100:.0f}%, Days: {days_passed})',
        xaxis_title='Spot Price (₹)',
        yaxis_title='P&L (₹)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_scenario_heatmap(scenarios_df):
    """Create scenario heatmap"""
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
        zmid=0,
        text=pivot.values.round(0),
        texttemplate='₹%{text:,.0f}',
        textfont={"size": 10},
        colorbar=dict(title="P&L (₹)")
    ))
    
    fig.update_layout(
        title='Scenario Matrix: Immediate P&L',
        xaxis_title='Spot Change',
        yaxis_title='IV Change (%)',
        height=400,
        template='plotly_white'
    )
    
    return fig

# ==================== DASH APP ====================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Options Strategy Analyzer"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                border: 1px solid #dee2e6;
            }
            .risk-high { color: #d32f2f; font-weight: bold; }
            .risk-medium { color: #f57c00; font-weight: bold; }
            .risk-low { color: #388e3c; font-weight: bold; }
            .recommendation-box {
                background-color: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0.25rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ==================== LAYOUT ====================

def create_sidebar():
    """Create sidebar with position setup"""
    return dbc.Col([
        html.H3("📊 Position Setup", className="text-primary mb-3"),
        
        # Spot Price
        dbc.Card([
            dbc.CardBody([
                html.Label("Underlying Spot Price"),
                dbc.Input(id='spot-price', type='number', value=22000, step=50),
            ])
        ], className="mb-3"),
        
        # Add Leg Section
        dbc.Card([
            dbc.CardHeader(html.H5("Add Option Leg")),
            dbc.CardBody([
                html.Label("Strike"),
                dbc.Input(id='strike', type='number', value=22000, step=50, className="mb-2"),
                
                html.Label("Expiry (Days)"),
                dbc.Input(id='expiry', type='number', value=30, min=1, className="mb-2"),
                
                html.Label("Option Type"),
                dcc.Dropdown(
                    id='option-type',
                    options=[
                        {'label': 'Call (CE)', 'value': 'CE'},
                        {'label': 'Put (PE)', 'value': 'PE'}
                    ],
                    value='CE',
                    className="mb-2"
                ),
                
                html.Label("Side"),
                dcc.Dropdown(
                    id='side',
                    options=[
                        {'label': 'Buy', 'value': 'BUY'},
                        {'label': 'Sell', 'value': 'SELL'}
                    ],
                    value='BUY',
                    className="mb-2"
                ),
                
                html.Label("Quantity (Lots)"),
                dbc.Input(id='quantity', type='number', value=1, min=1, className="mb-2"),
                
                html.Label("Lot Size"),
                dbc.Input(id='lot-size', type='number', value=50, className="mb-2"),
                
                html.Label("Entry Price"),
                dbc.Input(id='entry-price', type='number', value=100, step=0.5, className="mb-2"),
                
                html.Label("Current IV (%)"),
                dbc.Input(id='current-iv', type='number', value=20, step=1, className="mb-3"),
                
                dbc.Button("➕ Add Leg", id='add-leg-btn', color='success', className="w-100 mb-2"),
                dbc.Button("🗑️ Clear All", id='clear-all-btn', color='danger', className="w-100"),
            ])
        ], className="mb-3"),
        
        # Quick Strategies
        dbc.Card([
            dbc.CardHeader(html.H5("Quick Strategies")),
            dbc.CardBody([
                dbc.Button("Bull Call Spread", id='bull-call-btn', color='info', size='sm', className="w-100 mb-1"),
                dbc.Button("Iron Condor", id='iron-condor-btn', color='info', size='sm', className="w-100 mb-1"),
                dbc.Button("Straddle", id='straddle-btn', color='info', size='sm', className="w-100"),
            ])
        ])
        
    ], width=3, className="bg-light p-3")

def create_main_content():
    """Create main content area with tabs"""
    return dbc.Col([
        html.H1("Options Strategy Analyzer", className="text-center text-primary mb-4"),
        
        dcc.Tabs(id='tabs', value='overview', children=[
            dcc.Tab(label='📈 Overview', value='overview'),
            dcc.Tab(label='🎯 Scenarios', value='scenarios'),
            dcc.Tab(label='💡 Recommendations', value='recommendations'),
            dcc.Tab(label='📊 P&L Curves', value='pnl'),
            dcc.Tab(label='⚙️ Position Manager', value='manager'),
        ]),
        
        html.Div(id='tab-content', className="mt-3")
        
    ], width=9, className="p-3")

app.layout = dbc.Container([
    dcc.Store(id='legs-store', data=[]),
    dcc.Store(id='spot-store', data=22000),
    
    dbc.Row([
        create_sidebar(),
        create_main_content()
    ])
], fluid=True)

# ==================== CALLBACKS ====================

@app.callback(
    [Output('legs-store', 'data'),
     Output('spot-store', 'data')],
    [Input('add-leg-btn', 'n_clicks'),
     Input('clear-all-btn', 'n_clicks'),
     Input('bull-call-btn', 'n_clicks'),
     Input('iron-condor-btn', 'n_clicks'),
     Input('straddle-btn', 'n_clicks'),
     Input('spot-price', 'value')],
    [State('legs-store', 'data'),
     State('strike', 'value'),
     State('expiry', 'value'),
     State('option-type', 'value'),
     State('side', 'value'),
     State('quantity', 'value'),
     State('lot-size', 'value'),
     State('entry-price', 'value'),
     State('current-iv', 'value'),
     State('spot-store', 'data')]
)
def manage_legs(add_clicks, clear_clicks, bull_clicks, condor_clicks, straddle_clicks, spot_value,
                legs_data, strike, expiry, opt_type, side, qty, lot_size, entry_price, iv, spot_store):
    """Manage option legs"""
    if not ctx.triggered:
        return legs_data, spot_value or 22000
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'spot-price':
        return legs_data, spot_value
    
    if button_id == 'clear-all-btn':
        return [], spot_value
    
    if button_id == 'add-leg-btn' and add_clicks:
        new_leg = {
            'strike': strike,
            'expiry_days': expiry,
            'option_type': opt_type,
            'side': side,
            'quantity': qty,
            'lot_size': lot_size,
            'entry_price': entry_price,
            'current_iv': iv / 100
        }
        legs_data.append(new_leg)
        return legs_data, spot_value
    
    # Quick strategies
    spot = spot_value or 22000
    
    if button_id == 'bull-call-btn' and bull_clicks:
        legs_data = [
            {'strike': spot, 'expiry_days': 30, 'option_type': 'CE', 'side': 'BUY', 
             'quantity': 1, 'lot_size': 50, 'entry_price': 150, 'current_iv': 0.20},
            {'strike': spot + 200, 'expiry_days': 30, 'option_type': 'CE', 'side': 'SELL', 
             'quantity': 1, 'lot_size': 50, 'entry_price': 80, 'current_iv': 0.20}
        ]
        return legs_data, spot_value
    
    if button_id == 'iron-condor-btn' and condor_clicks:
        legs_data = [
            {'strike': spot - 200, 'expiry_days': 30, 'option_type': 'PE', 'side': 'BUY', 
             'quantity': 1, 'lot_size': 50, 'entry_price': 50, 'current_iv': 0.20},
            {'strike': spot - 100, 'expiry_days': 30, 'option_type': 'PE', 'side': 'SELL', 
             'quantity': 1, 'lot_size': 50, 'entry_price': 100, 'current_iv': 0.20},
            {'strike': spot + 100, 'expiry_days': 30, 'option_type': 'CE', 'side': 'SELL', 
             'quantity': 1, 'lot_size': 50, 'entry_price': 100, 'current_iv': 0.20},
            {'strike': spot + 200, 'expiry_days': 30, 'option_type': 'CE', 'side': 'BUY', 
             'quantity': 1, 'lot_size': 50, 'entry_price': 50, 'current_iv': 0.20}
        ]
        return legs_data, spot_value
    
    if button_id == 'straddle-btn' and straddle_clicks:
        legs_data = [
            {'strike': spot, 'expiry_days': 30, 'option_type': 'CE', 'side': 'BUY', 
             'quantity': 1, 'lot_size': 50, 'entry_price': 150, 'current_iv': 0.20},
            {'strike': spot, 'expiry_days': 30, 'option_type': 'PE', 'side': 'BUY', 
             'quantity': 1, 'lot_size': 50, 'entry_price': 150, 'current_iv': 0.20}
        ]
        return legs_data, spot_value
    
    return legs_data, spot_value

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('legs-store', 'data'),
     Input('spot-store', 'data')]
)
def render_tab_content(active_tab, legs_data, spot):
    """Render content based on active tab"""
    if not legs_data:
        return dbc.Alert("Please add at least one option leg to begin analysis.", color="info")
    
    # Create analyzer
    legs = [OptionLeg(**leg) for leg in legs_data]
    analyzer = PositionAnalyzer(spot, legs)
    
    if active_tab == 'overview':
        return render_overview(analyzer, legs_data)
    elif active_tab == 'scenarios':
        return render_scenarios(analyzer)
    elif active_tab == 'recommendations':
        return render_recommendations(analyzer)
    elif active_tab == 'pnl':
        return render_pnl(analyzer, legs)
    elif active_tab == 'manager':
        return render_manager(legs_data)
    
    return html.Div("Select a tab")

def render_overview(analyzer, legs_data):
    """Render overview tab"""
    greeks = analyzer.calculate_greeks()
    initial_value = analyzer.calculate_initial_value()
    current_value = analyzer.calculate_position_value()
    current_pnl = analyzer.calculate_pnl()
    max_profit = analyzer.max_profit()
    max_loss = analyzer.max_loss()
    breakevens = analyzer.breakeven_points()
    
    # Risk assessment
    risk_score = abs(greeks['delta']) + abs(greeks['gamma'] * 100) + abs(greeks['theta'])
    if risk_score > 2000:
        risk_level = "HIGH"
        risk_class = "risk-high"
    elif risk_score > 1000:
        risk_level = "MEDIUM"
        risk_class = "risk-medium"
    else:
        risk_level = "LOW"
        risk_class = "risk-low"
    
    return html.Div([
        # Position Summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Current Position", className="text-muted"),
                        html.H3(f"₹{current_value:,.0f}", className="text-primary"),
                        html.P(f"Initial: ₹{initial_value:,.0f}")
                    ])
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Current P&L", className="text-muted"),
                        html.H3(f"₹{current_pnl:,.0f}", 
                               className="text-success" if current_pnl >= 0 else "text-danger"),
                        html.P(f"{(current_pnl/abs(initial_value)*100):.1f}%" if initial_value != 0 else "N/A")
                    ])
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Profit", className="text-muted"),
                        html.H3("Unlimited" if max_profit == float('inf') else f"₹{max_profit:,.0f}",
                               className="text-success"),
                        html.P(f"Max Loss: ₹{max_loss:,.0f}" if max_loss != float('-inf') else "Unlimited Loss")
                    ])
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Risk Level", className="text-muted"),
                        html.H3(risk_level, className=risk_class),
                        html.P(f"Score: {risk_score:.0f}")
                    ])
                ], className="metric-card")
            ], width=3),
        ], className="mb-4"),
        
        # Greeks
        html.H4("Greeks Analysis", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Delta (Δ)"),
                        html.H4(f"{greeks['delta']:.2f}"),
                        html.Small("Price sensitivity")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Gamma (Γ)"),
                        html.H4(f"{greeks['gamma']:.4f}"),
                        html.Small("Delta change rate")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Theta (Θ)"),
                        html.H4(f"₹{greeks['theta']:.0f}"),
                        html.Small("Daily decay")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Vega (ν)"),
                        html.H4(f"₹{greeks['vega']:.0f}"),
                        html.Small("IV sensitivity")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Rho (ρ)"),
                        html.H4(f"₹{greeks['rho']:.0f}"),
                        html.Small("Rate sensitivity")
                    ])
                ])
            ], width=2),
        ], className="mb-4"),
        
        # Breakeven points
        html.H4("Breakeven Points", className="mb-3"),
        html.P([
            html.Span(f"₹{be:,.0f}", className="badge bg-warning text-dark me-2")
            for be in breakevens
        ] if breakevens else "No breakeven points found"),
        
        # Legs table
        html.H4("Position Legs", className="mt-4 mb-3"),
        dash_table.DataTable(
            data=[{
                'Strike': leg['strike'],
                'Expiry': f"{leg['expiry_days']}d",
                'Type': leg['option_type'],
                'Side': leg['side'],
                'Qty': leg['quantity'],
                'Lot': leg['lot_size'],
                'Entry': f"₹{leg['entry_price']}",
                'IV': f"{leg['current_iv']*100:.1f}%"
            } for leg in legs_data],
            columns=[{'name': k, 'id': k} for k in ['Strike', 'Expiry', 'Type', 'Side', 'Qty', 'Lot', 'Entry', 'IV']],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
        )
    ])

def render_scenarios(analyzer):
    """Render scenarios tab"""
    # Default parameters
    spot_range = np.linspace(analyzer.spot_price * 0.85, analyzer.spot_price * 1.15, 15)
    iv_multipliers = [0.95, 1.0, 1.05]
    time_points = [0, 7]
    
    scenarios_df = analyzer.scenario_analysis(spot_range, iv_multipliers, time_points)
    
    # Immediate scenarios
    immediate = scenarios_df[scenarios_df['days_passed'] == 0]
    
    # Heatmap
    fig_heatmap = plot_scenario_heatmap(immediate)
    
    # Worst and best scenarios
    worst = scenarios_df.nsmallest(5, 'pnl')
    best = scenarios_df.nlargest(5, 'pnl')
    
    return html.Div([
        html.H4("Scenario Analysis", className="mb-3"),
        
        dcc.Graph(figure=fig_heatmap),
        
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.H5("🔻 Worst 5 Scenarios"),
                dash_table.DataTable(
                    data=worst[['spot', 'spot_pct', 'iv_change_pct', 'days_passed', 'pnl']].to_dict('records'),
                    columns=[
                        {'name': 'Spot', 'id': 'spot', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Spot %', 'id': 'spot_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'IV %', 'id': 'iv_change_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Days', 'id': 'days_passed'},
                        {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                    ],
                    style_data_conditional=[
                        {'if': {'column_id': 'pnl', 'filter_query': '{pnl} < 0'}, 'color': 'red'}
                    ]
                )
            ], width=6),
            dbc.Col([
                html.H5("🔺 Best 5 Scenarios"),
                dash_table.DataTable(
                    data=best[['spot', 'spot_pct', 'iv_change_pct', 'days_passed', 'pnl']].to_dict('records'),
                    columns=[
                        {'name': 'Spot', 'id': 'spot', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Spot %', 'id': 'spot_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'IV %', 'id': 'iv_change_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Days', 'id': 'days_passed'},
                        {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                    ],
                    style_data_conditional=[
                        {'if': {'column_id': 'pnl', 'filter_query': '{pnl} > 0'}, 'color': 'green'}
                    ]
                )
            ], width=6),
        ])
    ])

def render_recommendations(analyzer):
    """Render recommendations tab"""
    available_strikes = [
        analyzer.spot_price - 400, analyzer.spot_price - 200, analyzer.spot_price - 100,
        analyzer.spot_price, 
        analyzer.spot_price + 100, analyzer.spot_price + 200, analyzer.spot_price + 400
    ]
    
    recommender = HedgeRecommender(analyzer)
    recommendations = recommender.recommend_hedges(available_strikes, 100000)
    
    if not recommendations:
        return dbc.Alert("✅ Position looks good! No urgent hedges needed.", color="success")
    
    cards = []
    for i, rec in enumerate(recommendations, 1):
        card = dbc.Card([
            dbc.CardHeader(html.H5(f"💡 Recommendation #{i}: {rec['type']}")),
            dbc.CardBody([
                html.H6(rec['action'], className="text-primary"),
                html.P(rec['reason']),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Strong("Cost: "),
                        html.Span(f"₹{rec['cost']:,.0f}")
                    ], width=6),
                    dbc.Col([
                        html.Strong("EV Score: "),
                        html.Span(f"{rec['ev_score']:.2f}")
                    ], width=6),
                ]),
                html.Hr(),
                html.H6("Impact:"),
                html.Ul([
                    html.Li(f"Delta: → {rec['new_delta']:.0f} ({rec['delta_reduction']:.0f} reduction)")
                    if 'new_delta' in rec else None,
                    html.Li(f"Gamma: → {rec['new_gamma']:.2f}") if 'new_gamma' in rec else None,
                    html.Li(f"Loss reduction: ₹{rec['estimated_loss_reduction']:,.0f}")
                    if 'estimated_loss_reduction' in rec else None
                ]),
            ])
        ], className="mb-3")
        cards.append(card)
    
    return html.Div([
        html.H4("AI-Powered Recommendations", className="mb-3"),
        html.P(f"Found {len(recommendations)} hedge recommendations:"),
        html.Div(cards)
    ])

def render_pnl(analyzer, legs):
    """Render P&L curves tab"""
    # Main P&L curve
    fig = plot_pnl_curve(analyzer, 1.0, 0)
    
    # Multi-curve comparison
    fig_multi = go.Figure()
    
    colors = ['blue', 'green', 'orange', 'red']
    time_points = [0, 3, 7, min([leg.expiry_days for leg in legs])]
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
    
    return html.Div([
        html.H4("P&L Visualizations", className="mb-3"),
        dcc.Graph(figure=fig),
        html.Hr(),
        html.H5("Time Decay Comparison"),
        dcc.Graph(figure=fig_multi)
    ])

def render_manager(legs_data):
    """Render position manager tab"""
    if not legs_data:
        return dbc.Alert("No legs to manage", color="info")
    
    cards = []
    for i, leg in enumerate(legs_data):
        card = dbc.Card([
            dbc.CardHeader(f"Leg {i+1}: {leg['side']} {leg['strike']} {leg['option_type']}"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P([html.Strong("Strike: "), f"₹{leg['strike']}"]),
                        html.P([html.Strong("Type: "), leg['option_type']]),
                        html.P([html.Strong("Side: "), leg['side']]),
                    ], width=4),
                    dbc.Col([
                        html.P([html.Strong("Quantity: "), str(leg['quantity'])]),
                        html.P([html.Strong("Lot Size: "), str(leg['lot_size'])]),
                        html.P([html.Strong("Entry Price: "), f"₹{leg['entry_price']}"]),
                    ], width=4),
                    dbc.Col([
                        html.P([html.Strong("IV: "), f"{leg['current_iv']*100:.1f}%"]),
                        html.P([html.Strong("Days to Expiry: "), str(leg['expiry_days'])]),
                    ], width=4),
                ])
            ])
        ], className="mb-2")
        cards.append(card)
    
    return html.Div([
        html.H4("Position Manager", className="mb-3"),
        html.Div(cards)
    ])

# ==================== RUN APP ====================

if __name__ == '__main__':
    app.run(debug=True, port=8050)
