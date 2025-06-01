import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from rf_model import RandomForestModel

def load_data(file_path):
    """
    Load stock data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with stock data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    return df

def visualize_predictions(model, data_path, days_to_visualize=60, save_path=None):
    """
    Visualize model predictions
    
    Parameters:
    -----------
    model : RandomForestModel
        Trained model
    data_path : str
        Path to the CSV file with stock data
    days_to_visualize : int
        Number of days to visualize
    save_path : str or None
        Path to save the plot
    """
    # Load data
    df = load_data(data_path)
    
    # Create features
    df_features = model.create_features(df)
    
    # Get the last n days
    last_days = df_features.iloc[-days_to_visualize:].copy()
    
    # Prepare data for prediction
    X, y = model.prepare_data(last_days)
    
    # Make predictions
    predictions = model.model.predict(X)
    
    # Create a dataframe with actual and predicted values
    results_df = pd.DataFrame({
        'Date': last_days['Date'],
        'Actual': last_days['Close'],
        'Predicted': predictions
    })
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))
    mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    r2 = r2_score(results_df['Actual'], results_df['Predicted'])
    
    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['Date'], results_df['Actual'], label='Actual', color='blue')
    plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', color='red')
    plt.title(f'GOOGL Stock Price: Actual vs Predicted\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Predictions plot saved to {save_path}")
    
    # Show plot
    plt.show()
    
    # Predict next day's price
    next_day_price = model.predict_next_day(df)
    last_price = df['Close'].iloc[-1]
    change = next_day_price - last_price
    change_pct = (change / last_price) * 100
    
    # Print prediction
    print("\nNext day price prediction:")
    print(f"Last known price (on {df['Date'].iloc[-1].strftime('%Y-%m-%d')}): ${last_price:.2f}")
    print(f"Predicted price for next trading day: ${next_day_price:.2f}")
    print(f"Change: ${change:.2f} ({change_pct:.2f}%)")
    
    return results_df

if __name__ == "__main__":
    # Path to data
    data_path = r"c:\Users\user\Desktop\school\FY project\ensemble web\cache_GOOGL_10y.csv"
    
    # Path to model
    model_dir = os.path.join(os.path.dirname(data_path), 'models')
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    
    # Check if model exists
    if os.path.exists(model_path):
        # Load model
        model = RandomForestModel.load_model(model_path)
    else:
        # Train model if it doesn't exist
        from train_rf_model import train_and_evaluate_model
        model, _ = train_and_evaluate_model(data_path)
    
    # Visualize predictions
    plot_dir = os.path.join(os.path.dirname(data_path), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'rf_predictions.png')
    visualize_predictions(model, data_path, days_to_visualize=60, save_path=plot_path)