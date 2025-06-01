import os
import pandas as pd
import numpy as np
import json
import plotly
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from .models import StockData, TrainedModel, Prediction
from .forms import StockDataUploadForm, StockSymbolForm

# Import ML models
from rf_model import EnhancedRandomForestModel
from lstm_model import LSTMModel

# Create directory for models
os.makedirs(os.path.join(settings.MEDIA_ROOT, 'models'), exist_ok=True)

# Function to fetch latest market data from Polygon API
def fetch_latest_market_data(ticker, days=5):
    """Fetch the latest market data for a ticker from Polygon API"""
    api_key = settings.POLYGON_API_KEY
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'results' not in data:
            return None, f"Error fetching data: {data.get('error', 'Unknown error')}"
        
        # Convert to DataFrame
        df = pd.DataFrame(data['results'])
        
        # Rename columns to match our model expectations
        df = df.rename(columns={
            'v': 'Volume',
            'o': 'Open',
            'c': 'Close',
            'h': 'High',
            'l': 'Low',
            't': 'timestamp'
        })
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Select and reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df, None
    
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def index(request):
    """Home page view"""
    upload_form = StockDataUploadForm()
    symbol_form = StockSymbolForm()
    
    context = {
        'upload_form': upload_form,
        'symbol_form': symbol_form,
    }
    return render(request, 'stock_prediction/index.html', context)

def fetch_stock_data(request):
    """Fetch stock data from Polygon API"""
    if request.method == 'POST':
        form = StockSymbolForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['stock_symbol']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            
            # Set default dates if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            else:
                start_date = start_date.strftime('%Y-%m-%d')
                
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Fetch data from Polygon API
            api_key = settings.POLYGON_API_KEY
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"
            
            try:
                response = requests.get(url)
                data = response.json()
                
                if 'results' not in data:
                    messages.error(request, f"Error fetching data: {data.get('error', 'Unknown error')}")
                    return redirect('index')
                
                # Convert to DataFrame
                df = pd.DataFrame(data['results'])
                
                # Rename columns to match our model expectations
                df = df.rename(columns={
                    'v': 'Volume',
                    'o': 'Open',
                    'c': 'Close',
                    'h': 'High',
                    'l': 'Low',
                    't': 'timestamp'
                })
                
                # Convert timestamp to datetime
                df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Select and reorder columns
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Save to CSV
                file_path = os.path.join(settings.MEDIA_ROOT, 'stock_data', f"{ticker}_data.csv")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path, index=False)
                
                # Save to database
                stock_data = StockData(
                    name=f"{ticker} Data",
                    file=f"stock_data/{ticker}_data.csv"
                )
                stock_data.save()
                
                # Calculate technical indicators
                df_with_indicators = calculate_technical_indicators(df)
                
                # Create price chart
                price_chart = create_price_chart(df)
                
                # Create technical indicator charts
                ma_chart = create_moving_averages_chart(df_with_indicators)
                rsi_chart = create_rsi_chart(df_with_indicators)
                macd_chart = create_macd_chart(df_with_indicators)
                bb_chart = create_bollinger_chart(df_with_indicators)
                
                # Store data in session
                request.session['stock_data'] = {
                    'ticker': ticker,
                    'file_path': file_path,
                    'data_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                    'records': len(df)
                }
                
                context = {
                    'upload_form': StockDataUploadForm(),
                    'symbol_form': form,
                    'stock_data': request.session['stock_data'],
                    'price_chart': price_chart,
                    'ma_chart': ma_chart,
                    'rsi_chart': rsi_chart,
                    'macd_chart': macd_chart,
                    'bb_chart': bb_chart,
                }
                
                messages.success(request, f"Successfully fetched data for {ticker}")
                return render(request, 'stock_prediction/index.html', context)
                
            except Exception as e:
                messages.error(request, f"Error fetching stock data: {str(e)}")
                return redirect('index')
    
    return redirect('index')

def upload_stock_data(request):
    """Handle uploaded stock data files"""
    if request.method == 'POST':
        form = StockDataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            stock_data = form.save()
            
            # Read the uploaded file
            file_path = stock_data.file.path
            try:
                df = pd.read_csv(file_path)
                
                # Validate required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    stock_data.delete()
                    messages.error(request, f"Missing required columns: {', '.join(missing_columns)}")
                    return redirect('index')
                
                # Convert Date to datetime
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Calculate technical indicators
                df_with_indicators = calculate_technical_indicators(df)
                
                # Create price chart
                price_chart = create_price_chart(df)
                
                # Create technical indicator charts
                ma_chart = create_moving_averages_chart(df_with_indicators)
                rsi_chart = create_rsi_chart(df_with_indicators)
                macd_chart = create_macd_chart(df_with_indicators)
                bb_chart = create_bollinger_chart(df_with_indicators)
                
                # Store data in session
                request.session['stock_data'] = {
                    'ticker': stock_data.name,
                    'file_path': file_path,
                    'data_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                    'records': len(df)
                }
                
                context = {
                    'upload_form': StockDataUploadForm(),
                    'symbol_form': StockSymbolForm(),
                    'stock_data': request.session['stock_data'],
                    'price_chart': price_chart,
                    'ma_chart': ma_chart,
                    'rsi_chart': rsi_chart,
                    'macd_chart': macd_chart,
                    'bb_chart': bb_chart,
                }
                
                messages.success(request, f"Successfully uploaded {stock_data.name}")
                return render(request, 'stock_prediction/index.html', context)
                
            except Exception as e:
                stock_data.delete()
                messages.error(request, f"Error processing file: {str(e)}")
                return redirect('index')
    
    return redirect('index')

def train_models(request):
    """Train RF and LSTM models"""
    if request.method == 'POST':
        if 'stock_data' not in request.session:
            messages.error(request, "No stock data available. Please fetch or upload data first.")
            return redirect('index')
        
        file_path = request.session['stock_data']['file_path']
        ticker = request.session['stock_data']['ticker']
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Train RF model
            rf_model = EnhancedRandomForestModel(
                feature_selection_threshold=0.02,
                random_state=42
            )
            
            rf_metrics = rf_model.train(df, cv=3)
            
            # Save RF model
            model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
            os.makedirs(model_dir, exist_ok=True)
            rf_model_path = os.path.join(model_dir, f'rf_model_{ticker}.joblib')
            rf_model.save_model(rf_model_path)
            
            # Save RF model info to database
            rf_model_db = TrainedModel(
                name=f"RF Model for {ticker}",
                model_type='RF',
                stock_symbol=ticker,
                file_path=rf_model_path,
                accuracy=1.0 - rf_metrics.get('mae', 0)
            )
            rf_model_db.save()
            
            # Train LSTM model
            features = ['Close', 'Volume', 'MA7', 'MA21', 'RSI', 'MACD']
            lstm_model = LSTMModel(time_steps=60, features=features, epochs=50, batch_size=32)
            
            # Calculate indicators for LSTM
            df_with_indicators = calculate_technical_indicators(df)
            
            # Train LSTM model
            lstm_history = lstm_model.train(df_with_indicators)
            
            # Save LSTM model
            lstm_model_path = os.path.join(model_dir, f'lstm_model_{ticker}.h5')
            lstm_model.save(lstm_model_path)
            
            # Save LSTM model info to database
            lstm_model_db = TrainedModel(
                name=f"LSTM Model for {ticker}",
                model_type='LSTM',
                stock_symbol=ticker,
                file_path=lstm_model_path,
                accuracy=None  # LSTM doesn't provide simple accuracy metric
            )
            lstm_model_db.save()
            
            # Store model IDs in session
            request.session['rf_model_id'] = rf_model_db.id
            request.session['lstm_model_id'] = lstm_model_db.id
            
            messages.success(request, "Successfully trained Random Forest and LSTM models")
            
            # Redirect to the same page to show the models are trained
            return redirect('index')
            
        except Exception as e:
            messages.error(request, f"Error training models: {str(e)}")
            return redirect('index')
    
    return redirect('index')

def make_predictions(request):
    """Make predictions using trained models"""
    if request.method == 'POST':
        if 'stock_data' not in request.session:
            messages.error(request, "No stock data available. Please fetch or upload data first.")
            return redirect('index')
        
        if 'rf_model_id' not in request.session or 'lstm_model_id' not in request.session:
            messages.error(request, "Models not trained. Please train models first.")
            return redirect('index')
        
        file_path = request.session['stock_data']['file_path']
        ticker = request.session['stock_data']['ticker']
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Load RF model
            rf_model_db = TrainedModel.objects.get(id=request.session['rf_model_id'])
            rf_model = EnhancedRandomForestModel.load_model(rf_model_db.file_path)
            
            # Make RF predictions
            next_day_price = rf_model.predict_next_day(df)
            rf_predictions_df = rf_model.predict_next_30_days(df)
            
            # Create RF prediction chart
            rf_chart = create_rf_prediction_chart(df, rf_predictions_df)
            
            # Save RF prediction to database
            rf_prediction = Prediction(
                stock_symbol=ticker,
                prediction_type='daily',
                prediction_date=datetime.now().date() + timedelta(days=1),
                predicted_price=next_day_price,
                model=rf_model_db
            )
            rf_prediction.save()
            
            # Load LSTM model
            lstm_model_db = TrainedModel.objects.get(id=request.session['lstm_model_id'])
            lstm_model = LSTMModel.load(lstm_model_db.file_path)
            
            # Calculate indicators for LSTM
            df_with_indicators = calculate_technical_indicators(df)
            
            # Make LSTM prediction
            next_month_price = lstm_model.predict(df_with_indicators)
            
            # Create LSTM prediction chart
            lstm_chart = create_lstm_prediction_chart(df, next_month_price)
            
            # Save LSTM prediction to database
            lstm_prediction = Prediction(
                stock_symbol=ticker,
                prediction_type='monthly',
                prediction_date=datetime.now().date() + timedelta(days=30),
                predicted_price=next_month_price,
                model=lstm_model_db
            )
            lstm_prediction.save()
            
            # Current price
            last_price = df['Close'].iloc[-1]
            
            # Calculate changes
            rf_change = next_day_price - last_price
            rf_pct_change = (rf_change / last_price) * 100
            
            lstm_change = next_month_price - last_price
            lstm_pct_change = (lstm_change / last_price) * 100
            
            # Store predictions in session
            request.session['predictions'] = {
                'rf': {
                    'next_day_price': float(next_day_price),
                    'change': float(rf_change),
                    'pct_change': float(rf_pct_change)
                },
                'lstm': {
                    'next_month_price': float(next_month_price),
                    'change': float(lstm_change),
                    'pct_change': float(lstm_pct_change)
                },
                'last_price': float(last_price)
            }
            
            context = {
                'upload_form': StockDataUploadForm(),
                'symbol_form': StockSymbolForm(),
                'stock_data': request.session['stock_data'],
                'predictions': request.session['predictions'],
                'rf_chart': rf_chart,
                'lstm_chart': lstm_chart,
                'show_predictions': True
            }
            
            messages.success(request, "Successfully made predictions")
            return render(request, 'stock_prediction/predictions.html', context)
            
        except Exception as e:
            messages.error(request, f"Error making predictions: {str(e)}")
            return redirect('index')
    
    return redirect('index')

def live_prediction(request):
    """Make predictions using the latest market data from Polygon API"""
    if request.method == 'POST':
        form = StockSymbolForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['stock_symbol']
            
            # Check if we have trained models for this ticker
            rf_models = TrainedModel.objects.filter(stock_symbol=ticker, model_type='RF')
            lstm_models = TrainedModel.objects.filter(stock_symbol=ticker, model_type='LSTM')
            
            if not rf_models.exists() or not lstm_models.exists():
                messages.error(request, f"No trained models found for {ticker}. Please train models first.")
                return redirect('index')
            
            # Get the latest trained models
            rf_model_db = rf_models.order_by('-created_at').first()
            lstm_model_db = lstm_models.order_by('-created_at').first()
            
            try:
                # Fetch latest market data
                df, error = fetch_latest_market_data(ticker, days=100)  # Need enough history for features
                
                if error:
                    messages.error(request, error)
                    return redirect('index')
                
                if len(df) < 60:  # Minimum required for LSTM
                    messages.error(request, f"Insufficient market data for {ticker}. Need at least 60 days.")
                    return redirect('index')
                
                # Load models
                rf_model = EnhancedRandomForestModel.load_model(rf_model_db.file_path)
                lstm_model = LSTMModel.load(lstm_model_db.file_path)
                
                # Make RF predictions
                next_day_price = rf_model.predict_next_day(df)
                rf_predictions_df = rf_model.predict_next_30_days(df)
                
                # Create RF prediction chart
                rf_chart = create_rf_prediction_chart(df, rf_predictions_df)
                
                # Calculate indicators for LSTM
                df_with_indicators = calculate_technical_indicators(df)
                
                # Make LSTM prediction
                next_month_price = lstm_model.predict(df_with_indicators)
                
                # Create LSTM prediction chart
                lstm_chart = create_lstm_prediction_chart(df, next_month_price)
                
                # Current price
                last_price = df['Close'].iloc[-1]
                last_date = df['Date'].iloc[-1]
                
                # Calculate changes
                rf_change = next_day_price - last_price
                rf_pct_change = (rf_change / last_price) * 100
                
                lstm_change = next_month_price - last_price
                lstm_pct_change = (lstm_change / last_price) * 100
                
                # Save predictions to database
                rf_prediction = Prediction(
                    stock_symbol=ticker,
                    prediction_type='daily',
                    prediction_date=datetime.now().date() + timedelta(days=1),
                    predicted_price=next_day_price,
                    model=rf_model_db
                )
                rf_prediction.save()
                
                lstm_prediction = Prediction(
                    stock_symbol=ticker,
                    prediction_type='monthly',
                    prediction_date=datetime.now().date() + timedelta(days=30),
                    predicted_price=next_month_price,
                    model=lstm_model_db
                )
                lstm_prediction.save()
                
                # Store data in session for display
                request.session['stock_data'] = {
                    'ticker': ticker,
                    'data_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                    'records': len(df)
                }
                
                # Store predictions in session
                request.session['predictions'] = {
                    'rf': {
                        'next_day_price': float(next_day_price),
                        'change': float(rf_change),
                        'pct_change': float(rf_pct_change)
                    },
                    'lstm': {
                        'next_month_price': float(next_month_price),
                        'change': float(lstm_change),
                        'pct_change': float(lstm_pct_change)
                    },
                    'last_price': float(last_price),
                    'last_date': last_date.strftime('%Y-%m-%d'),
                    'is_live': True
                }
                
                # Create price chart
                price_chart = create_price_chart(df)
                
                # Create technical indicator charts
                ma_chart = create_moving_averages_chart(df_with_indicators)
                rsi_chart = create_rsi_chart(df_with_indicators)
                macd_chart = create_macd_chart(df_with_indicators)
                bb_chart = create_bollinger_chart(df_with_indicators)
                
                context = {
                    'upload_form': StockDataUploadForm(),
                    'symbol_form': form,
                    'stock_data': request.session['stock_data'],
                    'predictions': request.session['predictions'],
                    'rf_chart': rf_chart,
                    'lstm_chart': lstm_chart,
                    'price_chart': price_chart,
                    'ma_chart': ma_chart,
                    'rsi_chart': rsi_chart,
                    'macd_chart': macd_chart,
                    'bb_chart': bb_chart,
                    'show_predictions': True,
                    'is_live': True
                }
                
                messages.success(request, f"Successfully made live predictions for {ticker}")
                return render(request, 'stock_prediction/live_predictions.html', context)
                
            except Exception as e:
                messages.error(request, f"Error making live predictions: {str(e)}")
                return redirect('index')
    
    return redirect('index')

# Helper functions for technical indicators and charts

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataframe"""
    df = df.copy()
    
    # Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    return df

def create_price_chart(df):
    """Create price chart with Plotly"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name="Volume",
        yaxis="y2",
        opacity=0.3
    ))
    
    # Update layout
    fig.update_layout(
        title="Stock Price and Volume",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right"
        ),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False)

def create_moving_averages_chart(df):
    """Create moving averages chart with Plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA7'], name="MA7", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name="MA20", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name="MA50", line=dict(color='red')))
    fig.update_layout(title="Moving Averages", height=400, legend=dict(orientation="h"))
    
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False)

def create_rsi_chart(df):
    """Create RSI chart with Plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')))
    fig.add_shape(type="line", x0=df['Date'].min(), x1=df['Date'].max(), y0=70, y1=70,
                 line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=df['Date'].min(), x1=df['Date'].max(), y0=30, y1=30,
                 line=dict(color="green", width=2, dash="dash"))
    fig.update_layout(title="Relative Strength Index (RSI)", height=400, yaxis=dict(range=[0, 100]))
    
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False)

def create_macd_chart(df):
    """Create MACD chart with Plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name="Signal", line=dict(color='red')))
    fig.update_layout(title="MACD", height=400, legend=dict(orientation="h"))
    
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False)

def create_bollinger_chart(df):
    """Create Bollinger Bands chart with Plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_upper'], name="Upper Band", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_middle'], name="Middle Band", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_lower'], name="Lower Band", line=dict(color='green')))
    fig.update_layout(title="Bollinger Bands", height=400, legend=dict(orientation="h"))
    
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False)

def create_rf_prediction_chart(df, predictions_df):
    """Create Random Forest prediction chart with Plotly"""
    fig = go.Figure()
    
    # Add historical data (last 30 days)
    historical_df = df.iloc[-30:]
    fig.add_trace(go.Scatter(
        x=historical_df['Date'], 
        y=historical_df['Close'],
        name="Historical",
        line=dict(color='blue')
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=predictions_df['Date'],
        y=predictions_df['Predicted_Close'],
        name="Predicted",
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title="Random Forest 30-Day Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        legend=dict(orientation="h")
    )
    
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False)

def create_lstm_prediction_chart(df, next_month_price):
    """Create LSTM prediction chart with Plotly"""
    fig = go.Figure()
    
    # Add historical data (last 60 days)
    historical_df = df.iloc[-60:]
    fig.add_trace(go.Scatter(
        x=historical_df['Date'], 
        y=historical_df['Close'],
        name="Historical",
        line=dict(color='blue')
    ))
    
    # Create a simple projection line
    last_date = df['Date'].iloc[-1]
    last_price = df['Close'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(0, 31, 5)]
    
    # Linear interpolation between last price and predicted price
    change = next_month_price - last_price
    future_prices = [last_price + (change * i / 30) for i in range(0, 31, 5)]
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        name="Projected",
        line=dict(color='purple', dash='dash')
    ))
    
    # Add final prediction point
    fig.add_trace(go.Scatter(
        x=[future_dates[-1]],
        y=[next_month_price],
        name="Month Prediction",
        mode="markers",
        marker=dict(size=12, color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title="LSTM Monthly Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        legend=dict(orientation="h")
    )
    
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False)
