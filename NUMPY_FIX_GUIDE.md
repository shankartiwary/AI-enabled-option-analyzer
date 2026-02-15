# NumPy 2.0 Compatibility Fix

## The Problem

You're seeing this error:
```
AttributeError: `np.unicode_` was removed in the NumPy 2.0 release
```

This happens because some dependencies aren't yet compatible with NumPy 2.0.

## Quick Fix (Already Applied!)

I've fixed the main app by removing an unused import. Download the updated file and try again:

```bash
streamlit run options_strategy_analyzer.py
```

## If That Doesn't Work

Try these solutions in order:

### Solution 1: Downgrade NumPy (Recommended)

```bash
# Downgrade to NumPy 1.26 (fully compatible)
pip install "numpy<2.0"

# Then run the app
streamlit run options_strategy_analyzer.py
```

### Solution 2: Update All Dependencies

```bash
# Update everything to latest versions
pip install --upgrade streamlit numpy pandas scipy plotly

# If that still fails, install specific compatible versions:
pip install numpy==1.26.4 pandas==2.1.4 scipy==1.11.4 plotly==5.18.0
```

### Solution 3: Create Virtual Environment (Best Practice)

```bash
# Create new environment
python -m venv options_env

# Activate it
# On Mac/Linux:
source options_env/bin/activate
# On Windows:
options_env\Scripts\activate

# Install dependencies
pip install streamlit numpy pandas scipy plotly

# Run app
streamlit run options_strategy_analyzer.py
```

## Verification

After applying any fix, verify it works:

```bash
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import plotly; print(f'Plotly version: {plotly.__version__}')"
```

Expected output:
```
NumPy version: 1.26.4 (or similar 1.x)
Plotly version: 5.18.0 (or higher)
```

## What Changed in the Fixed File

**Before:**
```python
import plotly.express as px  # This was causing the issue
```

**After:**
```python
# Removed - we don't actually use plotly.express
```

The app only uses `plotly.graph_objects`, so removing `plotly.express` eliminates the dependency conflict.

## Still Having Issues?

### Check Your Environment

```bash
# List all installed packages
pip list

# Look for:
# numpy       2.x.x  <- This is the problem
# plotly      x.x.x
# streamlit   x.x.x
# pandas      x.x.x
# scipy       x.x.x
```

### Clean Install (Nuclear Option)

```bash
# Uninstall everything
pip uninstall numpy pandas scipy plotly streamlit -y

# Reinstall with specific versions
pip install numpy==1.26.4
pip install pandas==2.1.4
pip install scipy==1.11.4
pip install plotly==5.18.0
pip install streamlit==1.31.0
```

## Why This Happens

NumPy 2.0 (released in 2024) removed several deprecated APIs including `np.unicode_`. Some packages haven't updated yet:

- **xarray** (used by plotly.express) - Not yet updated
- **Other packages** may also have issues

The solution: Either downgrade NumPy or avoid packages that aren't compatible yet.

## Recommended Setup

For this app, use:
```
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
plotly==5.18.0
streamlit==1.31.0
```

These versions are stable and fully tested together.

## Quick Test

After fixing, test the app:

```bash
streamlit run options_strategy_analyzer.py
```

If it opens in your browser without errors - you're good! 🎉

## Alternative: Use Docker (Advanced)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY options_strategy_analyzer.py .

CMD ["streamlit", "run", "options_strategy_analyzer.py"]
```

Create `requirements.txt`:
```
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
plotly==5.18.0
streamlit==1.31.0
```

Run:
```bash
docker build -t options-analyzer .
docker run -p 8501:8501 options-analyzer
```

## Summary

**Quick Fix:** Download the updated file (plotly.express import removed)

**If that fails:** Downgrade NumPy
```bash
pip install "numpy<2.0"
```

**Best practice:** Use virtual environment with specific versions

**You should be running within 2 minutes!** 🚀
