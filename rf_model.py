import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
import os

class EnhancedRandomForestModel:
    def __init__(self, feature_selection_threshold=0.01, random_state=42):
        """
        Improved Random Forest model with feature selection and hyperparameter tuning
        
        Parameters:
        -----------
        feature_selection_threshold : float
            Threshold for feature selection based on importance
        random_state : int
            Random seed for reproducibility
        """
        self.feature_selection_threshold = feature_selection_threshold
        self.random_state = random_state
        self.pipeline = None
        self.feature_columns = None
        self.selected_features = None
        self.best_params_ = None

    def create_features(self, df):
        """
        Enhanced feature engineering with time-series characteristics
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with stock data
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with enhanced features
        """
        df = df.copy()
        
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        df = df.sort_values('Date')
        
        # Price Dynamics
        df['Return'] = df['Close'].pct_change()
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Volume Features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # Technical Indicators
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'MA{window}'] = df['Close'].rolling(window).mean()
            if window <= 50:  # Only calculate for shorter windows to avoid too many features
                df[f'Volatility{window}'] = df['Return'].rolling(window).std()
        
        # Price relative to moving averages
        df['Price_to_MA50'] = df['Close'] / df['MA50']
        df['Price_to_MA200'] = df['Close'] / df['MA200']
        
        # Moving Average Crossovers
        df['MA_Cross_5_20'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA_Cross_20_50'] = (df['MA20'] > df['MA50']).astype(int)
        df['MA_Cross_50_200'] = (df['MA50'] > df['MA200']).astype(int)
        
        # Momentum Indicators
        for period in [5, 10, 20, 30]:
            df[f'Momentum{period}'] = df['Close'].pct_change(period)
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 1e-10)
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Price Relationships
        df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Low']
        df['Close_Open_Spread'] = (df['Close'] - df['Open']) / df['Open']
        df['Close_to_High'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Date Features
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        df['Day_of_Month'] = df['Date'].dt.day
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week
        
        # Market Regime Features
        df['Trend_20_50'] = np.where(df['MA20'] > df['MA50'], 1, -1)
        df['Volatility_Regime'] = np.where(df['Volatility20'] > df['Volatility20'].rolling(50).mean(), 1, 0)
        
        # Target Variable
        df['Next_Day_Close'] = df['Close'].shift(-1)
        
        # Drop initial rows with missing values
        df = df.dropna().reset_index(drop=True)
        
        return df
        
    def use_pretrained_pipeline(self, best_params):
        """Reuse parameters from initial training"""
        self.pipeline = Pipeline([
            ('feature_selection', SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                threshold=self.feature_selection_threshold)),
            ('model', RandomForestRegressor(**best_params))
        ])

    def prepare_data(self, df):
        """Prepare feature matrix and target vector"""
        feature_df = df.drop(['Date', 'Next_Day_Close'], axis=1, errors='ignore')
        self.feature_columns = feature_df.columns.tolist()
        return feature_df.values, df['Next_Day_Close'].values
        
    def _select_features(self, X, y):
        """
        Select important features using a Random Forest model
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            list: Indices of selected features
        """
        # Train a simple RF model to get feature importances
        selector = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        selector.fit(X, y)
        
        # Get feature importances
        importances = selector.feature_importances_
        
        # Select features above threshold
        selected_indices = np.where(importances > self.feature_selection_threshold)[0]
        
        # If no features selected, take top 5
        if len(selected_indices) == 0:
            selected_indices = np.argsort(importances)[-5:]
            
        return selected_indices.tolist()
        
    def _tune_hyperparameters(self, X, y, cv):
        """
        Tune hyperparameters using grid search with time series cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            
        Returns:
            dict: Best hyperparameters
        """
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Grid search
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state),
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        self.best_params_ = grid_search.best_params_
        
        return grid_search.best_params_
        
    def _create_pipeline(self, best_params):
        """Create pipeline with feature selection and best model"""
        return Pipeline([
            ('model', RandomForestRegressor(**best_params, random_state=self.random_state))
        ])

    def train(self, df, cv=5):
        """
        Train the model with feature selection and hyperparameter tuning
        
        Args:
            df: DataFrame with historical data
            cv: Number of cross-validation folds
            
        Returns:
            dict: Training metrics and selected features
        """
        # Create features
        df_features = self.create_features(df)
        
        # Prepare data
        X, y = self.prepare_data(df_features)
        
        # Set feature columns - this is the critical fix
        self.feature_columns = df_features.drop(['Date', 'Next_Month_Close', 'Next_Day_Close'], axis=1, errors='ignore').columns.tolist()
        
        # Feature selection
        if self.feature_selection_threshold > 0:
            self.selected_features = self._select_features(X, y)
            X_selected = X[:, self.selected_features]
        else:
            self.selected_features = list(range(X.shape[1]))
            X_selected = X
        
        # Hyperparameter tuning
        best_params = self._tune_hyperparameters(X_selected, y, cv)
        
        # Train final model with best parameters
        self.pipeline = self._create_pipeline(best_params)
        self.pipeline.fit(X_selected, y)
        
        # Evaluate model
        y_pred = self.pipeline.predict(X_selected)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'best_params': best_params,
            'selected_features': [self.feature_columns[i] for i in self.selected_features]
        }
        
        return metrics
    
    def predict_next_day(self, df):
        """Predict next day's closing price"""
        if len(df) < 200:  # Ensure sufficient history
            raise ValueError("Insufficient historical data for prediction")
            
        df_engineered = self.create_features(df)
        last_data = df_engineered.iloc[-1:].drop(['Date', 'Next_Day_Close'], axis=1, errors='ignore')
        
        # Extract features in the correct format
        X = last_data.values
        
        # If we have selected features, use only those
        if hasattr(self, 'selected_features') and self.selected_features is not None:
            X = X[:, self.selected_features]
            
        return self.pipeline.predict(X)[0]
    
    def predict_next_30_days(self, df):
        """Predict closing prices for the next 30 trading days"""
        if len(df) < 200:  # Ensure sufficient history
            raise ValueError("Insufficient historical data for prediction")
            
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Get the last date in the dataframe
        last_date = df_copy['Date'].iloc[-1]
        
        # Create a dataframe to store predictions
        predictions = []
        dates = []
        
        # Predict for the next 30 days
        for i in range(1, 31):
            # Create features for the current state
            df_engineered = self.create_features(df_copy)
            
            # Get the last row for prediction
            last_data = df_engineered.iloc[-1:].drop(['Date', 'Next_Day_Close'], axis=1, errors='ignore')
            
            # Extract features in the correct format
            X = last_data.values
            
            # If we have selected features, use only those
            if hasattr(self, 'selected_features') and self.selected_features is not None:
                X = X[:, self.selected_features]
                
            # Make prediction for the next day
            next_day_price = self.pipeline.predict(X)[0]
            
            # Calculate the next date (assuming business days)
            next_date = last_date + pd.Timedelta(days=i)
            
            # Store the prediction
            predictions.append(next_day_price)
            dates.append(next_date)
            
            # Add the prediction to the dataframe for the next iteration
            new_row = df_copy.iloc[-1:].copy()
            new_row['Date'] = next_date
            new_row['Close'] = next_day_price
            new_row['Open'] = next_day_price  # Simplification
            new_row['High'] = next_day_price  # Simplification
            new_row['Low'] = next_day_price   # Simplification
            
            # Append the new row to the dataframe
            df_copy = pd.concat([df_copy, new_row], ignore_index=True)
            
        # Create a dataframe with the predictions
        predictions_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Close': predictions
        })
        
        return predictions_df
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load trained model from disk"""
        return joblib.load(filepath)