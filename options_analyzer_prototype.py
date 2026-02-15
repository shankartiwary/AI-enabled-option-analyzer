#!/usr/bin/env python3
"""
Options Strategy What-If Analyzer - Basic Prototype
Demonstrates scenario analysis and P&L calculations
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt

# ==================== OPTIONS PRICING ====================

def black_scholes(spot, strike, time, volatility, rate=0.06, option_type='CE'):
    """Calculate option price using Black-Scholes model"""
    if time <= 0:
        # At expiry
        if option_type == 'CE':
            return max(0, spot - strike)
        else:  # PE
            return max(0, strike - spot)
    
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time) / (volatility * np.sqrt(time))
    d2 = d1 - volatility * np.sqrt(time)
    
    if option_type == 'CE':
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2)
    else:  # PE
        price = strike * np.exp(-rate * time) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    return price


def calculate_greeks(spot, strike, time, volatility, rate=0.06, option_type='CE'):
    """Calculate option Greeks"""
    if time <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
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
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }


# ==================== POSITION MANAGEMENT ====================

@dataclass
class OptionLeg:
    """Represents a single option leg"""
    strike: float
    expiry_days: int
    option_type: str  # 'CE' or 'PE'
    side: str  # 'BUY' or 'SELL'
    quantity: int
    lot_size: int
    entry_price: float
    current_iv: float = 0.20


class PositionAnalyzer:
    """Analyzes option positions and scenarios"""
    
    def __init__(self, spot_price: float, legs: List[OptionLeg]):
        self.spot_price = spot_price
        self.legs = legs
    
    def calculate_current_pnl(self):
        """Calculate current P&L of the position"""
        total_pnl = 0
        
        for leg in self.legs:
            current_price = black_scholes(
                self.spot_price,
                leg.strike,
                leg.expiry_days / 365,
                leg.current_iv,
                option_type=leg.option_type
            )
            
            if leg.side == 'BUY':
                pnl = (current_price - leg.entry_price) * leg.quantity * leg.lot_size
            else:  # SELL
                pnl = (leg.entry_price - current_price) * leg.quantity * leg.lot_size
            
            total_pnl += pnl
        
        return total_pnl
    
    def calculate_position_greeks(self):
        """Calculate net Greeks for the entire position"""
        net_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        for leg in self.legs:
            greeks = calculate_greeks(
                self.spot_price,
                leg.strike,
                leg.expiry_days / 365,
                leg.current_iv,
                option_type=leg.option_type
            )
            
            multiplier = leg.quantity * leg.lot_size
            if leg.side == 'SELL':
                multiplier *= -1
            
            for key in net_greeks:
                net_greeks[key] += greeks[key] * multiplier
        
        return net_greeks
    
    def scenario_analysis(self, spot_moves: List[float], iv_changes: List[float], days_passed: List[int]):
        """Perform comprehensive scenario analysis"""
        scenarios = []
        
        for spot_move in spot_moves:
            new_spot = self.spot_price + spot_move
            
            for iv_change in iv_changes:
                for days in days_passed:
                    pnl = self._calculate_scenario_pnl(new_spot, iv_change, days)
                    
                    scenarios.append({
                        'spot': new_spot,
                        'spot_move': spot_move,
                        'iv_change': iv_change,
                        'days_passed': days,
                        'pnl': pnl
                    })
        
        return pd.DataFrame(scenarios)
    
    def _calculate_scenario_pnl(self, new_spot: float, iv_change: float, days_passed: int):
        """Calculate P&L for a specific scenario"""
        total_pnl = 0
        
        for leg in self.legs:
            new_iv = leg.current_iv * (1 + iv_change)
            new_time = max(0, leg.expiry_days - days_passed)
            
            new_price = black_scholes(
                new_spot,
                leg.strike,
                new_time / 365,
                new_iv,
                option_type=leg.option_type
            )
            
            if leg.side == 'BUY':
                pnl = (new_price - leg.entry_price) * leg.quantity * leg.lot_size
            else:
                pnl = (leg.entry_price - new_price) * leg.quantity * leg.lot_size
            
            total_pnl += pnl
        
        return total_pnl
    
    def find_max_loss_scenarios(self, scenarios_df: pd.DataFrame, top_n: int = 5):
        """Find worst-case scenarios"""
        return scenarios_df.nsmallest(top_n, 'pnl')
    
    def find_max_profit_scenarios(self, scenarios_df: pd.DataFrame, top_n: int = 5):
        """Find best-case scenarios"""
        return scenarios_df.nlargest(top_n, 'pnl')
    
    def plot_pnl_curve(self, iv_change: float = 0, days_passed: int = 0):
        """Plot P&L curve across spot prices"""
        spot_range = np.linspace(self.spot_price * 0.85, self.spot_price * 1.15, 100)
        pnls = []
        
        for spot in spot_range:
            pnl = self._calculate_scenario_pnl(spot, iv_change, days_passed)
            pnls.append(pnl)
        
        plt.figure(figsize=(12, 6))
        plt.plot(spot_range, pnls, linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=self.spot_price, color='g', linestyle='--', alpha=0.5, label='Current Spot')
        plt.xlabel('Spot Price')
        plt.ylabel('P&L (₹)')
        plt.title(f'P&L Curve (IV change: {iv_change*100:.1f}%, Days passed: {days_passed})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        return plt


# ==================== HEDGE RECOMMENDER ====================

class HedgeRecommender:
    """Recommends optimal hedges based on risk analysis"""
    
    def __init__(self, position: PositionAnalyzer):
        self.position = position
    
    def recommend_delta_hedge(self, available_strikes: List[float]):
        """Recommend hedge to neutralize delta"""
        current_greeks = self.position.calculate_position_greeks()
        current_delta = current_greeks['delta']
        
        if abs(current_delta) < 100:
            return None  # Delta already manageable
        
        recommendations = []
        
        # Try hedging with options at different strikes
        for strike in available_strikes:
            # Calculate which option type would best offset delta
            if current_delta > 0:  # Long delta, need to sell calls or buy puts
                option_type = 'PE'
                side = 'BUY'
            else:  # Short delta, need to buy calls or sell puts
                option_type = 'CE'
                side = 'BUY'
            
            # Calculate hedge greeks
            hedge_greeks = calculate_greeks(
                self.position.spot_price,
                strike,
                30 / 365,  # 1 month
                0.20,
                option_type=option_type
            )
            
            # Calculate quantity needed
            hedge_delta = hedge_greeks['delta']
            if side == 'SELL':
                hedge_delta *= -1
            
            quantity_needed = int(abs(current_delta / (hedge_delta * 50)))  # 50 lot size
            
            if quantity_needed > 0:
                price = black_scholes(
                    self.position.spot_price,
                    strike,
                    30 / 365,
                    0.20,
                    option_type=option_type
                )
                
                cost = price * quantity_needed * 50
                
                recommendations.append({
                    'strike': strike,
                    'type': option_type,
                    'side': side,
                    'quantity': quantity_needed,
                    'price': price,
                    'cost': cost,
                    'new_delta': current_delta + (hedge_delta * quantity_needed * 50)
                })
        
        # Return best recommendation (closest to zero delta)
        if recommendations:
            return min(recommendations, key=lambda x: abs(x['new_delta']))
        return None


# ==================== DEMO EXAMPLE ====================

def demo_iron_condor():
    """Demo: Analyze an Iron Condor position"""
    
    print("="*70)
    print("OPTIONS STRATEGY ANALYZER - DEMO")
    print("Strategy: NIFTY Iron Condor")
    print("="*70)
    
    # Current market
    spot = 23750
    
    # Define the Iron Condor
    legs = [
        # Short Call spread
        OptionLeg(strike=24000, expiry_days=14, option_type='CE', side='SELL', 
                 quantity=50, lot_size=50, entry_price=150, current_iv=0.18),
        OptionLeg(strike=24500, expiry_days=14, option_type='CE', side='BUY', 
                 quantity=50, lot_size=50, entry_price=50, current_iv=0.18),
        # Short Put spread
        OptionLeg(strike=23500, expiry_days=14, option_type='PE', side='SELL', 
                 quantity=50, lot_size=50, entry_price=145, current_iv=0.18),
        OptionLeg(strike=23000, expiry_days=14, option_type='PE', side='BUY', 
                 quantity=50, lot_size=50, entry_price=45, current_iv=0.18),
    ]
    
    analyzer = PositionAnalyzer(spot, legs)
    
    # Current Position
    print(f"\nCurrent Spot: ₹{spot}")
    print(f"Days to Expiry: 14")
    
    current_pnl = analyzer.calculate_current_pnl()
    print(f"\nCurrent P&L: ₹{current_pnl:,.0f}")
    
    greeks = analyzer.calculate_position_greeks()
    print(f"\nPosition Greeks:")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    print(f"  Theta: ₹{greeks['theta']:.2f} per day")
    print(f"  Vega: ₹{greeks['vega']:.2f} per 1% IV")
    
    # Scenario Analysis
    print("\n" + "="*70)
    print("SCENARIO ANALYSIS")
    print("="*70)
    
    scenarios = analyzer.scenario_analysis(
        spot_moves=[-300, -200, -100, 0, 100, 200, 300],
        iv_changes=[-0.05, 0, 0.05],
        days_passed=[0, 7, 14]
    )
    
    # Create pivot table for display
    pivot = scenarios[scenarios['days_passed'] == 0].pivot_table(
        values='pnl',
        index='iv_change',
        columns='spot_move'
    )
    
    print("\nP&L Matrix (Immediate scenarios, days_passed=0):")
    print("\nSpot Move →")
    print(pivot.to_string())
    
    # Worst case scenarios
    print("\n" + "="*70)
    print("WORST CASE SCENARIOS")
    print("="*70)
    worst = analyzer.find_max_loss_scenarios(scenarios)
    print(worst[['spot', 'spot_move', 'iv_change', 'days_passed', 'pnl']].to_string(index=False))
    
    # Best case scenarios
    print("\n" + "="*70)
    print("BEST CASE SCENARIOS")
    print("="*70)
    best = analyzer.find_max_profit_scenarios(scenarios)
    print(best[['spot', 'spot_move', 'iv_change', 'days_passed', 'pnl']].to_string(index=False))
    
    # Risk Alert
    max_loss = worst['pnl'].min()
    if max_loss < -500000:
        print("\n" + "="*70)
        print("⚠️  RISK ALERT: Potential large loss detected!")
        print("="*70)
        print(f"Maximum potential loss: ₹{max_loss:,.0f}")
        
        # Recommend hedge
        recommender = HedgeRecommender(analyzer)
        hedge = recommender.recommend_delta_hedge([23800, 24000, 24200, 24400])
        
        if hedge:
            print(f"\n💡 RECOMMENDED HEDGE:")
            print(f"   Action: {hedge['side']} {hedge['quantity']} lots")
            print(f"   Strike: {hedge['strike']} {hedge['type']}")
            print(f"   Price: ₹{hedge['price']:.2f}")
            print(f"   Total Cost: ₹{hedge['cost']:,.0f}")
            print(f"   New Delta: {hedge['new_delta']:.2f} (from {greeks['delta']:.2f})")
    
    # Generate P&L curve
    print("\n" + "="*70)
    print("Generating P&L curve...")
    print("="*70)
    
    plt = analyzer.plot_pnl_curve(iv_change=0, days_passed=0)
    plt.savefig('/mnt/user-data/outputs/pnl_curve.png', dpi=150, bbox_inches='tight')
    print("✅ P&L curve saved to: pnl_curve.png")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == "__main__":
    demo_iron_condor()
