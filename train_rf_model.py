import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rf_model import EnhancedRandomForestModel
from sklearn.metrics import mean_absolute_error

def load_data(file_path):
    """Load and prepare stock data"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

def validate_data(df):
    """Ensure data meets minimum requirements"""
    if len(df) < 365:
        raise ValueError("Insufficient data: Need at least 1 year of history")
    if df.isnull().sum().any():
        raise ValueError("Data contains missing values")
    if 'Close' not in df.columns:
        raise ValueError("Missing required 'Close' column")

def train_and_evaluate_model(data_path, cv=5):
    """Train and evaluate the enhanced model"""
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    # Data validation
    try:
        print("\nValidating data...")
        validate_data(df)
    except ValueError as e:
        print(f"Data validation failed: {str(e)}")
        return None, None
    
    print(f"\nData Overview:")
    print(f"- Shape: {df.shape}")
    print(f"- Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"- Features Available: {len(df.columns)}")
    
    print("\nInitializing Enhanced Random Forest Model...")
    rf_model = EnhancedRandomForestModel(
        feature_selection_threshold=0.02,
        random_state=42
    )
    
    print("Training model with time-series cross-validation...")
    metrics = rf_model.train(df, cv=cv)
    
    print("\nTraining Results:")
    print(f"- Best Parameters: {metrics['best_params']}")
    print(f"- RMSE: {metrics['rmse']:.2f}")
    print(f"- MAE: {metrics['mae']:.2f}")
    print(f"- R² Score: {metrics['r2']:.4f}")
    
    print("\nSelected Features:")
    print(f"Total: {len(metrics['selected_features'])} features")
    print("Top 15:")
    for feat in metrics['selected_features'][:15]:
        print(f"  - {feat}")
    
    # Benchmark comparison
    def compare_to_baseline(model, df):
        """Compare model performance to naive forecast"""
        df_features = model.create_features(df)
        X, y = model.prepare_data(df_features)
        naive_pred = np.roll(y, 1)  # Yesterday's price
        
        # Filter X to only use selected features
        if hasattr(model, 'selected_features') and model.selected_features is not None:
            X = X[:, model.selected_features]
            
        model_pred = model.pipeline.predict(X)
        
        model_mae = mean_absolute_error(y[1:], model_pred[1:])
        naive_mae = mean_absolute_error(y[1:], naive_pred[1:])
        
        return {
            'model_mae': model_mae,
            'naive_mae': naive_mae,
            'improvement': 1 - (model_mae / naive_mae)
        }
    
    print("\nBenchmark Comparison:")
    benchmark = compare_to_baseline(rf_model, df)
    print(f"- Model MAE: {benchmark['model_mae']:.2f}")
    print(f"- Naive MAE: {benchmark['naive_mae']:.2f}")
    print(f"- Improvement: {benchmark['improvement']:.1%}")
    metrics['benchmark'] = benchmark
    
    # Save artifacts
    model_dir = os.path.join(os.path.dirname(data_path), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'enhanced_rf_model.joblib')
    rf_model.save_model(model_path)
    
    return rf_model, metrics

def plot_feature_importance(model, metrics, save_path=None):
    """Plot feature importance from trained model"""
    features = metrics['selected_features']
    importances = model.pipeline.named_steps['model'].feature_importances_
    
    feat_imp = sorted(zip(features, importances), 
                    key=lambda x: x[1], reverse=True)[:15]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[imp for _, imp in feat_imp], y=[feat for feat, _ in feat_imp])
    plt.title("Top 15 Feature Importances (Post-Selection)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\nSaved feature plot to {save_path}")
    plt.show()

def predict_next_day_price(model, data_path):
    """Make prediction with data validation"""
    df = load_data(data_path)
    
    try:
        prediction = model.predict_next_day(df)
        last_price = df['Close'].iloc[-1]
        change = prediction - last_price
        pct_change = (change/last_price) * 100
        
        print("\nNext Day Prediction:")
        print(f"- Last Closing Price: ${last_price:.2f}")
        print(f"- Predicted Price:   ${prediction:.2f}")
        print(f"- Change:            ${change:.2f} ({pct_change:.2f}%)")
        return prediction
        
    except ValueError as e:
        print(f"\nPrediction Error: {str(e)}")
        return None

def predict_next_30_days_prices(model, data_path, save_plot=True):
    """Make predictions for the next 30 days with visualization"""
    df = load_data(data_path)
    
    try:
        # Get predictions for next 30 days
        predictions_df = model.predict_next_30_days(df)
        
        # Get the last known price
        last_price = df['Close'].iloc[-1]
        last_date = df['Date'].iloc[-1]
        
        # Calculate overall change
        final_price = predictions_df['Predicted_Close'].iloc[-1]
        overall_change = final_price - last_price
        overall_pct_change = (overall_change/last_price) * 100
        
        print("\nNext 30 Days Prediction Summary:")
        print(f"- Start Date: {last_date.strftime('%Y-%m-%d')}")
        print(f"- End Date: {predictions_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"- Starting Price: ${last_price:.2f}")
        print(f"- Ending Price: ${final_price:.2f}")
        print(f"- Overall Change: ${overall_change:.2f} ({overall_pct_change:.2f}%)")
        
        # Visualize the predictions
        if save_plot:
            plt.figure(figsize=(14, 7))
            
            # Plot historical data (last 60 days)
            historical_days = 60
            if len(df) >= historical_days:
                hist_data = df.iloc[-historical_days:]
                plt.plot(hist_data['Date'], hist_data['Close'], 
                         label='Historical', color='blue')
            
            # Plot predictions
            plt.plot(predictions_df['Date'], predictions_df['Predicted_Close'], 
                     label='Predicted', color='red', linestyle='--')
            
            # Add markers for start and end points
            plt.scatter([last_date], [last_price], color='green', s=100, 
                        label='Last Known Price')
            plt.scatter([predictions_df['Date'].iloc[-1]], [final_price], 
                        color='purple', s=100, label='30-Day Prediction')
            
            plt.title(f'30-Day Stock Price Prediction\nOverall Change: {overall_pct_change:.2f}%')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            plot_dir = os.path.join(os.path.dirname(data_path), 'reports')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, '30day_prediction.png')
            plt.savefig(plot_path, dpi=300)
            print(f"\nSaved 30-day prediction plot to {plot_path}")
            plt.show()
            
        return predictions_df
        
    except ValueError as e:
        print(f"\nPrediction Error: {str(e)}")
        return None

if __name__ == "__main__":
    data_path = r"c:\Users\user\Desktop\school\FY project\ensemble web\cache_GOOGL_10y.csv"
    
    # Train model
    model, metrics = train_and_evaluate_model(data_path, cv=5)
    
    # Visualize results
    plot_dir = os.path.join(os.path.dirname(data_path), 'reports')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'feature_importance.png')
    plot_feature_importance(model, metrics, plot_path)
    
    # Generate predictions
    if model:
        predict_next_day_price(model, data_path)
        predict_next_30_days_prices(model, data_path)

        