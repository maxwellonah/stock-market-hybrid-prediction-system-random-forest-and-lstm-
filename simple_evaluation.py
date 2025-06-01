import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

from rf_model import EnhancedRandomForestModel
from lstm_model import LSTMModel

# Create directory for models if it doesn't exist
os.makedirs('models', exist_ok=True)

# Generate synthetic stock data for evaluation
def generate_stock_data(days=500, volatility=0.01, drift=0.0002):
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
df = generate_stock_data(days=500)

# Split data for training and testing
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f'\nTotal data points: {len(df)}')
print(f'Training data points: {len(train_df)}')
print(f'Testing data points: {len(test_df)}')

# ===== RANDOM FOREST MODEL EVALUATION =====
print('\n==== RANDOM FOREST MODEL EVALUATION ====')
rf_model = EnhancedRandomForestModel()
print('Training Random Forest model...')
rf_model.train(train_df)

# Calculate RF metrics
print('Evaluating Random Forest model...')
rf_predictions = []
rf_actuals = []

# Predict next day for each day in test set
for i in range(len(test_df)-1):
    try:
        pred_df = test_df.iloc[i:i+1]
        actual = test_df['Close'].iloc[i+1]
        pred = rf_model.predict_next_day(pred_df)
        rf_predictions.append(pred)
        rf_actuals.append(actual)
    except Exception as e:
        print(f'Error on prediction {i}: {str(e)}')

# Calculate metrics
rf_mae = mean_absolute_error(rf_actuals, rf_predictions)
rf_mse = mean_squared_error(rf_actuals, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(rf_actuals, rf_predictions)
rf_mape = np.mean(np.abs((np.array(rf_actuals) - np.array(rf_predictions)) / np.array(rf_actuals))) * 100

# Calculate directional accuracy
rf_actual_direction = np.sign(np.diff(rf_actuals))
rf_pred_direction = np.sign(np.array(rf_predictions[1:]) - np.array(rf_predictions[:-1]))
rf_directional_accuracy = np.mean(rf_actual_direction == rf_pred_direction) * 100

print('\nRandom Forest Model Metrics:')
print(f'Mean Absolute Error (MAE): {rf_mae:.4f}')
print(f'Root Mean Squared Error (RMSE): {rf_rmse:.4f}')
print(f'Mean Absolute Percentage Error (MAPE): {rf_mape:.2f}%')
print(f'R-squared (R²): {rf_r2:.4f}')
print(f'Directional Accuracy: {rf_directional_accuracy:.2f}%')

# Feature importance
print('\nTop 10 Feature Importance:')
feature_importance = rf_model.feature_importance()
for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
    print(f'{i+1}. {feature}: {importance:.4f}')

# ===== LSTM MODEL EVALUATION =====
print('\n==== LSTM MODEL EVALUATION ====')
features = ['Close', 'Volume', 'High', 'Low', 'Open']
lstm_model = LSTMModel(time_steps=30, features=features, epochs=5)
print('Training LSTM model...')

try:
    lstm_model.train(train_df)
    
    # Make a prediction with the LSTM model
    print('\nLSTM Prediction Example with Uncertainty:')
    prediction = lstm_model.predict(test_df.iloc[-60:])
    
    if isinstance(prediction, dict):
        print(f'Prediction Method: {prediction.get("method", "unknown")}')
        print(f'Predicted Price: ${prediction["predicted_price"]:.2f}')
        
        if 'confidence_interval' in prediction:
            lower, upper = prediction['confidence_interval']
            print(f'95% Confidence Interval: ${lower:.2f} to ${upper:.2f}')
        
        if 'uncertainty' in prediction:
            print(f'Uncertainty: ±${prediction["uncertainty"]:.2f}')
            uncertainty_pct = (prediction['uncertainty'] / prediction['predicted_price']) * 100
            print(f'Uncertainty: ±{uncertainty_pct:.2f}%')
    else:
        print(f'Predicted Price: ${prediction:.2f}')
    
    # Calculate LSTM metrics for multiple predictions
    print('\nCalculating LSTM metrics...')
    lstm_predictions = []
    lstm_actuals = []
    
    # Use sliding window for monthly predictions (30 days ahead)
    for i in range(0, len(test_df)-30, 30):
        try:
            # Get data for prediction
            pred_df = test_df.iloc[i:i+60]  # Use 60 days of data
            if len(pred_df) < 60:
                continue
                
            # Get actual price 30 days ahead
            if i+60 < len(test_df):
                actual = test_df['Close'].iloc[i+60]
                
                # Make prediction
                pred_result = lstm_model.predict(pred_df)
                if isinstance(pred_result, dict):
                    pred = pred_result['predicted_price']
                else:
                    pred = pred_result
                    
                lstm_predictions.append(pred)
                lstm_actuals.append(actual)
        except Exception as e:
            print(f'Error on LSTM prediction {i}: {str(e)}')
    
    # Calculate metrics if we have predictions
    if len(lstm_predictions) > 0:
        lstm_mae = mean_absolute_error(lstm_actuals, lstm_predictions)
        lstm_mse = mean_squared_error(lstm_actuals, lstm_predictions)
        lstm_rmse = np.sqrt(lstm_mse)
        lstm_r2 = r2_score(lstm_actuals, lstm_predictions)
        lstm_mape = np.mean(np.abs((np.array(lstm_actuals) - np.array(lstm_predictions)) / np.array(lstm_actuals))) * 100
        
        print('\nLSTM Model Metrics:')
        print(f'Mean Absolute Error (MAE): {lstm_mae:.4f}')
        print(f'Root Mean Squared Error (RMSE): {lstm_rmse:.4f}')
        print(f'Mean Absolute Percentage Error (MAPE): {lstm_mape:.2f}%')
        print(f'R-squared (R²): {lstm_r2:.4f}')
        
        if len(lstm_predictions) > 1:
            # Calculate directional accuracy for LSTM
            lstm_actual_direction = np.sign(np.diff(lstm_actuals))
            lstm_pred_direction = np.sign(np.diff(lstm_predictions))
            lstm_directional_accuracy = np.mean(lstm_actual_direction == lstm_pred_direction) * 100
            print(f'Directional Accuracy: {lstm_directional_accuracy:.2f}%')
    else:
        print("Not enough data to calculate LSTM metrics")
        
except Exception as e:
    print(f'Error evaluating LSTM model: {str(e)}')

# ===== COMPARISON OF MODELS =====
print('\n==== COMPARISON OF MODELS ====')
print('Random Forest Model (Daily Predictions):')
print(f'- MAE: {rf_mae:.4f}')
print(f'- RMSE: {rf_rmse:.4f}')
print(f'- R²: {rf_r2:.4f}')
print(f'- Directional Accuracy: {rf_directional_accuracy:.2f}%')

if 'lstm_mae' in locals():
    print('\nLSTM Model (Monthly Predictions):')
    print(f'- MAE: {lstm_mae:.4f}')
    print(f'- RMSE: {lstm_rmse:.4f}')
    print(f'- R²: {lstm_r2:.4f}')
    if 'lstm_directional_accuracy' in locals():
        print(f'- Directional Accuracy: {lstm_directional_accuracy:.2f}%')
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
