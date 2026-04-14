"""
XAUUSD (Gold) Price Direction Predictor
========================================
Project: Simple ML-based Gold Price Direction Predictor
Author: Arjun Singh Yadav
Description:
    Predicts whether Gold (XAUUSD) price will go UP or DOWN the next day
    using technical indicators and a Random Forest Classifier.

Skills demonstrated:
    - Data collection with yfinance
    - Feature engineering with technical indicators
    - ML model training and evaluation with scikit-learn
    - Proper train/test split to avoid lookahead bias
    - Model evaluation with classification metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: Download Historical Gold Data
# ─────────────────────────────────────────────
print("=" * 60)
print("XAUUSD Gold Price Direction Predictor")
print("=" * 60)

print("\n[1/5] Loading Gold price data...")

# To use your own data, export XAUUSD OHLCV from any broker/platform as CSV
# with columns: Date, Open, High, Low, Close, Volume

# ── We're using  Download live data ──────────────────────────────
gold = yf.download("GC=F", start="2018-01-01", progress=False)
if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.droplevel(1)
gold = gold[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"    Loaded {len(gold)} trading days of data")
print(f"    Date range: {gold.index[0].date()} to {gold.index[-1].date()}")


# ─────────────────────────────────────────────
# STEP 2: Feature Engineering
# Build technical indicators as ML features
# ─────────────────────────────────────────────
print("\n[2/5] Engineering technical indicator features...")

df = gold.copy()

# --- Trend Indicators ---
# Simple Moving Averages: capture short vs long term trend
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_30'] = df['Close'].rolling(window=30).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# EMA: gives more weight to recent prices
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

# MACD: difference between short and long EMA (momentum signal)
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# Price relative to moving averages (normalized)
df['Price_vs_SMA10'] = (df['Close'] - df['SMA_10']) / df['SMA_10']
df['Price_vs_SMA30'] = (df['Close'] - df['SMA_30']) / df['SMA_30']
df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

# --- Momentum Indicators ---
# RSI: measures speed and magnitude of price changes (0-100)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# Rate of change: % price change over N days
df['ROC_5']  = df['Close'].pct_change(5)
df['ROC_10'] = df['Close'].pct_change(10)
df['ROC_20'] = df['Close'].pct_change(20)

# --- Volatility Indicators ---
# Bollinger Bands: price relative to volatility bands
bb_mid  = df['Close'].rolling(window=20).mean()
bb_std  = df['Close'].rolling(window=20).std()
df['BB_Upper'] = bb_mid + (2 * bb_std)
df['BB_Lower'] = bb_mid - (2 * bb_std)
df['BB_Width']    = (df['BB_Upper'] - df['BB_Lower']) / bb_mid  # Band width = volatility measure
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])  # 0=lower, 1=upper

# Average True Range: measures daily volatility
high_low   = df['High'] - df['Low']
high_close = (df['High'] - df['Close'].shift()).abs()
low_close  = (df['Low']  - df['Close'].shift()).abs()
df['ATR_14'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
df['ATR_Pct'] = df['ATR_14'] / df['Close']  # Normalized ATR

# --- Candlestick Body Features ---
df['Body_Size']    = (df['Close'] - df['Open']).abs() / df['Open']  # Size of candle body
df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
df['Bullish_Candle'] = (df['Close'] > df['Open']).astype(int)  # 1 if green candle

# --- Volume Features ---
df['Volume_SMA20'] = df['Volume'].rolling(20).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']  # High ratio = unusual volume

# --- Lagged Returns (past N days returns as features) ---
for lag in [1, 2, 3, 5]:
    df[f'Return_Lag{lag}'] = df['Close'].pct_change(lag)


# ─────────────────────────────────────────────
# STEP 3: Create Target Variable
# 1 = price goes UP tomorrow, 0 = price goes DOWN
# ─────────────────────────────────────────────
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop rows with NaN values (from rolling calculations)
df.dropna(inplace=True)

# Define feature columns (everything except raw OHLCV and target)
FEATURE_COLS = [
    'SMA_10', 'SMA_30', 'SMA_50',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Price_vs_SMA10', 'Price_vs_SMA30', 'Price_vs_SMA50',
    'RSI_14',
    'ROC_5', 'ROC_10', 'ROC_20',
    'BB_Width', 'BB_Position',
    'ATR_Pct',
    'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Bullish_Candle',
    'Volume_Ratio',
    'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag5'
]

X = df[FEATURE_COLS]
y = df['Target']

print(f"    Built {len(FEATURE_COLS)} features across {len(df)} samples")


# ─────────────────────────────────────────────
# STEP 4: Train/Test Split + Model Training
# IMPORTANT: We use chronological split (not random!)
# Random split would cause data leakage (lookahead bias)
# ─────────────────────────────────────────────
print("\n[3/5] Training Random Forest model...")

SPLIT = int(len(df) * 0.80)  # 80% train, 20% test (chronological)
X_train, X_test = X.iloc[:SPLIT], X.iloc[SPLIT:]
y_train, y_test = y.iloc[:SPLIT], y.iloc[SPLIT:]

# Scale features (important for some models; good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Random Forest: ensemble of decision trees, robust to overfitting
model = RandomForestClassifier(
    n_estimators=200,    # Number of trees
    max_depth=6,         # Limit depth to reduce overfitting
    min_samples_leaf=20, # Each leaf needs 20+ samples
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print(f"    Trained on {len(X_train)} days | Testing on {len(X_test)} days")


# ─────────────────────────────────────────────
# STEP 5: Evaluation
# ─────────────────────────────────────────────
print("\n[4/5] Evaluating model performance...")

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"\n    Overall Accuracy: {acc:.2%}")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))


# ─────────────────────────────────────────────
# STEP 6: Visualizations
# ─────────────────────────────────────────────
print("\n[5/5] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0f0f0f')
for ax in axes.flat:
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#aaaaaa')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['top'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['right'].set_color('#333333')

GOLD  = '#FFD700'
GREEN = '#00ff88'
RED   = '#ff4455'
GRAY  = '#aaaaaa'

# --- Plot 1: Gold Price with Predictions ---
ax1 = axes[0, 0]
test_dates  = df.index[SPLIT:]
test_prices = df['Close'].iloc[SPLIT:].values

# Color background by prediction
for i in range(len(test_dates) - 1):
    color = GREEN if y_pred[i] == 1 else RED
    ax1.axvspan(test_dates[i], test_dates[i+1], alpha=0.15, color=color, linewidth=0)

ax1.plot(test_dates, test_prices, color=GOLD, linewidth=1.5, label='Gold Price')
ax1.set_title('Gold Price + Daily Prediction (Green=UP, Red=DOWN)', color='white', fontsize=11, pad=10)
ax1.set_ylabel('Price (USD)', color=GRAY)
ax1.legend(facecolor='#1a1a1a', labelcolor='white')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

# --- Plot 2: Feature Importance ---
ax2 = axes[0, 1]
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True).tail(12)
bars = ax2.barh(feat_imp.index, feat_imp.values, color=GOLD, alpha=0.8)
ax2.set_title('Top 12 Most Important Features', color='white', fontsize=11, pad=10)
ax2.set_xlabel('Importance Score', color=GRAY)
ax2.tick_params(axis='y', labelsize=8, colors=GRAY)

# --- Plot 3: Confusion Matrix ---
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
im = ax3.imshow(cm, cmap='YlOrRd', aspect='auto')
ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Pred DOWN', 'Pred UP'], color=GRAY)
ax3.set_yticklabels(['Actual DOWN', 'Actual UP'], color=GRAY)
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white', fontsize=16, fontweight='bold')
ax3.set_title('Confusion Matrix', color='white', fontsize=11, pad=10)

# --- Plot 4: Cumulative Strategy Returns ---
ax4 = axes[1, 1]
test_df = df.iloc[SPLIT:].copy()
test_df['Pred'] = y_pred
test_df['Daily_Return'] = test_df['Close'].pct_change()

# Strategy: go long if model predicts UP, flat if DOWN
test_df['Strategy_Return'] = test_df['Daily_Return'] * test_df['Pred'].shift(1)
test_df['Cumulative_BuyHold'] = (1 + test_df['Daily_Return']).cumprod()
test_df['Cumulative_Strategy'] = (1 + test_df['Strategy_Return']).cumprod()

ax4.plot(test_df.index, test_df['Cumulative_BuyHold'], color=GRAY,
         linewidth=1.5, label='Buy & Hold', linestyle='--')
ax4.plot(test_df.index, test_df['Cumulative_Strategy'], color=GREEN,
         linewidth=2, label='ML Strategy')
ax4.set_title('Cumulative Returns: ML Strategy vs Buy & Hold', color='white', fontsize=11, pad=10)
ax4.set_ylabel('Growth of $1', color=GRAY)
ax4.legend(facecolor='#1a1a1a', labelcolor='white')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.suptitle('XAUUSD Gold Price Direction Predictor — Model Analysis',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('model_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.close()
print("    Saved: model_analysis.png")

print("\n" + "=" * 60)
print("DONE! Files saved to: gold_predictor_simple/")
print("=" * 60)
