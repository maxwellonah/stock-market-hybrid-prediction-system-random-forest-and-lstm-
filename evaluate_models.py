import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

from rf_model import EnhancedRandomForestModel
from lstm_model import LSTMModel

# Create directory for models
os.makedirs('models', exist_ok=True)

# Generate synthetic stock data for evaluation
def generate_stock_data(days=1000, volatility=0.01, drift=0.0002):
    np.random.seed(42)  # For reproducibility
    price = 100  # Starting price
    prices = [price]
    
    for _ in range(days - 1):
        change = np.random.normal(drift, volatility)
        price *= (1 + change)
        prices.append(price)
    
    dates = pd.date_range(start='2020-01-01', periods=days)
    
    # Generate other columns
    high = np.array(prices) * (1 + np.random.uniform(0, 0.03, days))
    low = np.array(prices) * (1 - np.random.uniform(0, 0.03, days))
    open_prices = np.array(prices) * (1 + np.random.normal(0, 0.01, days))
    volume = np.random.randint(1000000, 10000000, days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': prices,
        'Volume': volume
    })
    
    return df

print('Generating synthetic stock data for evaluation...')
df = generate_stock_data(days=1000)

# Split data for training and testing
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f'\nTotal data points: {len(df)}')
print(f'Training data points: {len(train_df)}')
print(f'Testing data points: {len(test_df)}')

print('\n==== RANDOM FOREST MODEL EVALUATION ====')
# Train RF model
rf_model = EnhancedRandomForestModel()
print('Training Random Forest model...')
rf_model.train(train_df)

# Calculate RF metrics
print('Evaluating Random Forest model...')
y_true = test_df['Close'].values[1:]  # Next day's actual prices
y_pred = []

for i in range(len(test_df)-1):
    pred_df = test_df.iloc[i:i+1]
    try:
        pred = rf_model.predict_next_day(pred_df)
        y_pred.append(pred)
    except Exception as e:
        print(f'Error predicting day {i}: {str(e)}')
        continue

if len(y_pred) > 0:
    mae = np.mean(np.abs(np.array(y_pred) - y_true[:len(y_pred)]))
    mse = np.mean((np.array(y_pred) - y_true[:len(y_pred)])**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true[:len(y_pred)] - np.array(y_pred)) / y_true[:len(y_pred)])) * 100
    r2 = r2_score(y_true[:len(y_pred)], y_pred)
    
    # Directional accuracy
    actual_direction = np.sign(np.diff(y_true[:len(y_pred)+1]))
    pred_direction = np.sign(np.array(y_pred) - y_true[:len(y_pred)])
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print('\nRandom Forest Model Metrics:')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    print(f'R-squared (R²): {r2:.4f}')
    print(f'Directional Accuracy: {directional_accuracy:.2f}%')
    
    # Feature importance
    print('\nTop 10 Feature Importance:')
    feature_importance = rf_model.feature_importance()
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f'{i+1}. {feature}: {importance:.4f}')
else:
    print('Could not generate Random Forest predictions')

print('\n==== LSTM MODEL EVALUATION ====')
# Train and evaluate LSTM model
features = ['Close', 'Volume', 'High', 'Low', 'Open']
lstm_model = LSTMModel(time_steps=30, features=features, epochs=10)
print('Training LSTM model...')
try:
    lstm_model.train(train_df)
    
    # Evaluate LSTM model
    print('Evaluating LSTM model...')
    lstm_metrics = lstm_model.evaluate(test_df)
    
    print('\nLSTM Model Metrics:')
    for metric, value in lstm_metrics.items():
        if isinstance(value, float):
            print(f'{metric}: {value:.4f}')
        else:
            print(f'{metric}: {value}')
    
    # Test prediction with uncertainty
    print('\nLSTM Prediction Example with Uncertainty:')
    prediction = lstm_model.predict(test_df.iloc[-60:])
    
    if isinstance(prediction, dict):
        for key, value in prediction.items():
            if key == 'confidence_interval':
                lower, upper = value
                print(f'confidence_interval: ${lower:.2f} to ${upper:.2f}')
            elif key == 'predicted_price':
                print(f'predicted_price: ${value:.2f}')
            elif key == 'uncertainty':
                print(f'uncertainty: ±${value:.2f}')
            elif key != 'predictions':  # Skip the predictions array
                print(f'{key}: {value}')
    else:
        print(f'Predicted Price: ${prediction:.2f}')
    
except Exception as e:
    print(f'Error evaluating LSTM model: {str(e)}')

print('\n==== COMPARISON OF MODELS ====')
print('Random Forest Model (Daily Predictions):')
print(f'- MAE: {mae:.4f}')
print(f'- RMSE: {rmse:.4f}')
print(f'- R²: {r2:.4f}')
print(f'- Directional Accuracy: {directional_accuracy:.2f}%')

if 'lstm_metrics' in locals():
    print('\nLSTM Model (Monthly Predictions):')
    for metric, value in lstm_metrics.items():
        if isinstance(value, float):
            print(f'- {metric}: {value:.4f}')
        else:
            print(f'- {metric}: {value}')
    print(f'- Uncertainty Quantification: Yes')
    print(f'- Confidence Intervals: Yes')
else:
    print('\nLSTM Model: Evaluation failed')

print('\nStrengths of Each Model:')
print('Random Forest:')
print('- Better for short-term (daily) predictions')
print('- More interpretable (feature importance)')
print('- Less sensitive to noise')
print('- Faster training time')

print('\nLSTM:')
print('- Better for long-term (monthly) predictions')
print('- Captures temporal dependencies')
print('- Provides uncertainty estimates')
print('- Handles complex patterns')

# Save models
print('\nSaving models...')
rf_model.save('models/rf_model.pkl')
lstm_model.save('models/lstm_model.h5')
print('Models saved successfully!')
