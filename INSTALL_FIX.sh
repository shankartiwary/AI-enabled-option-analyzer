#!/bin/bash
# Quick fix for NumPy 2.0 compatibility issue

echo "🔧 Fixing NumPy compatibility..."

# Downgrade NumPy to 1.x
pip install "numpy<2.0"

echo "✅ Fixed! Now run:"
echo "   streamlit run options_strategy_analyzer.py"
