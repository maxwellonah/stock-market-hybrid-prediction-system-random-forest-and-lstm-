import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import requests
import io
from dotenv import load_dotenv

# Import models
from rf_model import EnhancedRandomForestModel
from lstm_model import LSTMModel

# Load environment variables
load_dotenv()
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', "mo0_G1UPGqllOOPmY37UvS9Ui6mpiPQL")

# Initialize the Dash app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

# Define stock options
STOCK_OPTIONS = [
    {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
    {'label': 'Amazon (AMZN)', 'value': 'AMZN'}
]

# Create directory for uploaded files
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Function to fetch stock data from Polygon API
def fetch_stock_data(ticker, timespan='day', multiplier=1, from_date=None, to_date=None):
    """Fetch stock data from Polygon API"""
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?apiKey={POLYGON_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'results' not in data:
            print(f"Error fetching data: {data.get('error', 'Unknown error')}")
            return None
        
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
        
        return df
    
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return None

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    """Calculate technical indicators for display"""
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

# Function to train RF model
def train_rf_model(df, status_div_id):
    """Train Random Forest model and return metrics"""
    try:
        # Create model
        rf_model = EnhancedRandomForestModel(
            feature_selection_threshold=0.02,
            random_state=42
        )
        
        # Train model
        metrics = rf_model.train(df, cv=3)
        
        # Save model
        model_path = os.path.join('models', 'rf_model.joblib')
        rf_model.save_model(model_path)
        
        return rf_model, metrics
    
    except Exception as e:
        print(f"Error training RF model: {str(e)}")
        return None, {"error": str(e)}

# Function to train LSTM model
def train_lstm_model(df, status_div_id):
    """Train LSTM model and return metrics"""
    try:
        # Define features
        features = ['Close', 'Volume', 'MA7', 'MA20', 'RSI', 'MACD']
        
        # Create model
        lstm_model = LSTMModel(time_steps=60, features=features, epochs=50, batch_size=32)
        
        # Train model
        history = lstm_model.train(df)
        
        # Save model
        model_path = os.path.join('models', 'lstm_model.h5')
        lstm_model.save(model_path)
        
        return lstm_model, history
    
    except Exception as e:
        print(f"Error training LSTM model: {str(e)}")
        return None, {"error": str(e)}

# App Layout
app.layout = dbc.Container([
    # Notifications container
    html.Div(
        [
            dbc.Toast(
                id="training-notification",
                header="Model Training",
                is_open=False,
                dismissable=True,
                duration=8000,  # Longer duration
                icon="primary",
                style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 1000},
            ),
            dbc.Toast(
                id="completion-notification",
                header="Training Complete",
                is_open=False,
                dismissable=True,
                duration=8000,  # Longer duration
                icon="success",
                style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 1000},
            ),
        ]
    ),
    
    # Loading spinners container
    html.Div(
        [
            dbc.Spinner(
                id="rf-spinner",
                color="primary",
                type="grow",
                fullscreen=False,
                children=html.Div(id="rf-spinner-output"),
                spinner_style={"width": "3rem", "height": "3rem"},
            ),
            dbc.Spinner(
                id="lstm-spinner",
                color="info",
                type="grow",
                fullscreen=False,
                children=html.Div(id="lstm-spinner-output"),
                spinner_style={"width": "3rem", "height": "3rem"},
            ),
        ],
        style={"textAlign": "center", "marginBottom": "20px"}
    ),
    dbc.Row([
        dbc.Col([
            html.H1("Hybrid Stock Prediction System", className="text-center my-4"),
            html.P("Combining Random Forest (Daily) and LSTM (Monthly) predictions", className="text-center text-muted mb-4")
        ])
    ]),
    
    # Status indicators
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("System Status", className="text-center"),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        html.Div(id="rf-status", className="text-center p-2", 
                                children="Random Forest: Ready")
                    ], color="light"), width=6),
                    dbc.Col(dbc.Card([
                        html.Div(id="lstm-status", className="text-center p-2", 
                                children="LSTM: Ready")
                    ], color="light"), width=6),
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Stock Selection & Data Source Tabs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Source"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.P("Select a stock to analyze:", className="mt-2"),
                            dcc.Dropdown(
                                id="stock-dropdown",
                                options=STOCK_OPTIONS,
                                value="GOOGL",
                                className="mb-3"
                            ),
                            dbc.Button("Fetch Data", id="fetch-data-btn", color="primary", className="mr-2"),
                            html.Div(id="api-data-info", className="mt-3")
                        ], label="Polygon API"),
                        
                        dbc.Tab([
                            html.P("Upload your own CSV file:", className="mt-2"),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Div(id="upload-data-info", className="mt-3")
                        ], label="Upload Data")
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Price Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Chart"),
                dbc.CardBody([
                    dcc.Graph(id="price-chart")
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Technical Indicators
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Technical Indicators"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id="moving-averages-chart")
                        ], label="Moving Averages"),
                        dbc.Tab([
                            dcc.Graph(id="rsi-chart")
                        ], label="RSI"),
                        dbc.Tab([
                            dcc.Graph(id="macd-chart")
                        ], label="MACD"),
                        dbc.Tab([
                            dcc.Graph(id="bollinger-chart")
                        ], label="Bollinger Bands")
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Predictions
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Predictions"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Train Models", id="train-models-btn", color="success", className="mb-3 w-100")
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Make Predictions", id="predict-btn", color="primary", className="mb-3 w-100")
                        ], width=6)
                    ]),
                    dbc.Tabs([
                        dbc.Tab([
                            html.Div(id="rf-prediction-results"),
                            dcc.Graph(id="rf-prediction-chart")
                        ], label="Random Forest (Daily)"),
                        dbc.Tab([
                            html.Div(id="lstm-prediction-results"),
                            dcc.Graph(id="lstm-prediction-chart")
                        ], label="LSTM (Monthly)")
                    ])
                ])
            ])
        ])
    ]),
    
    # Store components for data
    dcc.Store(id="stock-data-store"),
    dcc.Store(id="rf-model-store"),
    dcc.Store(id="lstm-model-store"),
    dcc.Store(id="rf-predictions-store"),
    dcc.Store(id="lstm-predictions-store"),
    
    # Interval for status updates
    dcc.Interval(id="status-interval", interval=1000, n_intervals=0),
    
    html.Footer([
        html.P("Hybrid Stock Prediction System © 2025", className="text-center text-muted mt-4")
    ])
], fluid=True)

# Callback for uploading data
@app.callback(
    [Output("upload-data-info", "children"),
     Output("stock-data-store", "data", allow_duplicate=True)],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
    prevent_initial_call=True
)
def update_output(contents, filename):
    if contents is None:
        return html.Div("No file uploaded yet."), None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = io.StringIO(content_string.decode('utf-8'))
        
        if 'csv' in filename:
            df = pd.read_csv(decoded)
        else:
            return html.Div(f"File type not supported: {filename}"), None
        
        # Validate data
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return html.Div(f"Missing required columns: {', '.join(missing_columns)}"), None
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Save file
        upload_path = os.path.join('uploads', filename)
        df.to_csv(upload_path, index=False)
        
        # Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        return html.Div([
            html.P(f"File uploaded: {filename}"),
            html.P(f"Data range: {df['Date'].min().date()} to {df['Date'].max().date()}"),
            html.P(f"Number of records: {len(df)}")
        ]), df_with_indicators.to_json(date_format='iso', orient='split')
    
    except Exception as e:
        return html.Div(f"Error processing file: {str(e)}"), None

# Callback for fetching data from API
@app.callback(
    [Output("api-data-info", "children"),
     Output("stock-data-store", "data")],
    [Input("fetch-data-btn", "n_clicks")],
    [State("stock-dropdown", "value")]
)
def fetch_data(n_clicks, ticker):
    if n_clicks is None:
        return html.Div("Click 'Fetch Data' to load stock information."), None
    
    # Fetch data
    df = fetch_stock_data(ticker)
    
    if df is None:
        return html.Div("Error fetching data. Please try again."), None
    
    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(df)
    
    # Save file
    file_path = os.path.join('uploads', f"{ticker}_data.csv")
    df.to_csv(file_path, index=False)
    
    return html.Div([
        html.P(f"Data fetched for {ticker}"),
        html.P(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}"),
        html.P(f"Number of records: {len(df)}")
    ]), df_with_indicators.to_json(date_format='iso', orient='split')

# Callback for price chart
@app.callback(
    Output("price-chart", "figure"),
    [Input("stock-data-store", "data")]
)
def update_price_chart(data):
    if data is None:
        return go.Figure().update_layout(title="No data available")
    
    df = pd.read_json(data, orient='split')
    
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
    
    return fig

# Callback for technical indicator charts
@app.callback(
    [Output("moving-averages-chart", "figure"),
     Output("rsi-chart", "figure"),
     Output("macd-chart", "figure"),
     Output("bollinger-chart", "figure")],
    [Input("stock-data-store", "data")]
)
def update_technical_charts(data):
    if data is None:
        empty_fig = go.Figure().update_layout(title="No data available")
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    df = pd.read_json(data, orient='split')
    
    # Moving Averages Chart
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color='black')))
    ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA7'], name="MA7", line=dict(color='blue')))
    ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name="MA20", line=dict(color='orange')))
    ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name="MA50", line=dict(color='red')))
    ma_fig.update_layout(title="Moving Averages", height=400, legend=dict(orientation="h"))
    
    # RSI Chart
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')))
    rsi_fig.add_shape(type="line", x0=df['Date'].min(), x1=df['Date'].max(), y0=70, y1=70,
                     line=dict(color="red", width=2, dash="dash"))
    rsi_fig.add_shape(type="line", x0=df['Date'].min(), x1=df['Date'].max(), y0=30, y1=30,
                     line=dict(color="green", width=2, dash="dash"))
    rsi_fig.update_layout(title="Relative Strength Index (RSI)", height=400, yaxis=dict(range=[0, 100]))
    
    # MACD Chart
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD", line=dict(color='blue')))
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name="Signal", line=dict(color='red')))
    macd_fig.update_layout(title="MACD", height=400, legend=dict(orientation="h"))
    
    # Bollinger Bands Chart
    bb_fig = go.Figure()
    bb_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color='black')))
    bb_fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_upper'], name="Upper Band", line=dict(color='red')))
    bb_fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_middle'], name="Middle Band", line=dict(color='blue')))
    bb_fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_lower'], name="Lower Band", line=dict(color='green')))
    bb_fig.update_layout(title="Bollinger Bands", height=400, legend=dict(orientation="h"))
    
    return ma_fig, rsi_fig, macd_fig, bb_fig

# Callback for training models
@app.callback(
    [Output("rf-status", "children"),
     Output("lstm-status", "children"),
     Output("rf-model-store", "data"),
     Output("lstm-model-store", "data"),
     Output("training-notification", "is_open", allow_duplicate=True),
     Output("training-notification", "children", allow_duplicate=True),
     Output("completion-notification", "is_open", allow_duplicate=True),
     Output("completion-notification", "children", allow_duplicate=True),
     Output("rf-spinner-output", "children"),
     Output("lstm-spinner-output", "children")],
    [Input("train-models-btn", "n_clicks")],
    [State("stock-data-store", "data")],
    prevent_initial_call=True
)
def train_models(n_clicks, data):
    if n_clicks is None or data is None:
        return "Random Forest: Ready", "LSTM: Ready", None, None, False, "", False, "", "", ""
    
    df = pd.read_json(data, orient='split')
    
    # Show training notification
    training_notification = True
    training_message = html.Div([
        html.H5("Training Models", className="mb-2"),
        html.P("Starting model training. This may take a few moments..."),
        html.P("Please wait while we train both models on your data.", className="mb-0")
    ])
    
    # Train RF model
    rf_status = "Random Forest: Training..."
    lstm_status = "LSTM: Waiting..."
    
    # Spinner content - show during training
    rf_spinner_content = "Training RF"
    lstm_spinner_content = "Waiting"
    
    # Train RF model
    rf_model, rf_metrics = train_rf_model(df, "rf-status")
    if rf_model is not None:
        rf_status = "Random Forest: Trained"
        rf_model_info = {"model_path": "models/rf_model.joblib", "metrics": rf_metrics}
    else:
        rf_status = f"Random Forest: Error - {rf_metrics.get('error', 'Unknown error')}"
        rf_model_info = None
    
    # Train LSTM model
    lstm_status = "LSTM: Training..."
    lstm_spinner_content = "Training LSTM"
    lstm_model, lstm_history = train_lstm_model(df, "lstm-status")
    if lstm_model is not None:
        lstm_status = "LSTM: Trained"
        lstm_model_info = {"model_path": "models/lstm_model.h5", "history": str(lstm_history)}
    else:
        lstm_status = f"LSTM: Error - {lstm_history.get('error', 'Unknown error')}"
        lstm_model_info = None
    
    # Show completion notification
    completion_notification = True
    completion_message = html.Div([
        html.H5("Training Complete!", className="mb-2"),
        html.P("Both models have been successfully trained."),
        html.P("You can now make predictions using the 'Predict' button.", className="mb-0")
    ])
    
    # Clear spinner content after training
    rf_spinner_content = ""
    lstm_spinner_content = ""
    
    return rf_status, lstm_status, rf_model_info, lstm_model_info, training_notification, training_message, completion_notification, completion_message, rf_spinner_content, lstm_spinner_content

# Callback for making predictions
@app.callback(
    [Output("rf-prediction-results", "children"),
     Output("lstm-prediction-results", "children"),
     Output("rf-prediction-chart", "figure"),
     Output("lstm-prediction-chart", "figure"),
     Output("rf-predictions-store", "data"),
     Output("lstm-predictions-store", "data"),
     Output("training-notification", "is_open", allow_duplicate=True),
     Output("training-notification", "children", allow_duplicate=True),
     Output("completion-notification", "is_open", allow_duplicate=True),
     Output("completion-notification", "children", allow_duplicate=True),
     Output("rf-spinner-output", "children", allow_duplicate=True),
     Output("lstm-spinner-output", "children", allow_duplicate=True)],
    [Input("predict-btn", "n_clicks")],
    [State("stock-data-store", "data"),
     State("rf-model-store", "data"),
     State("lstm-model-store", "data")],
    prevent_initial_call=True
)
def make_predictions(n_clicks, data, rf_model_info, lstm_model_info):
    if n_clicks is None or data is None:
        empty_fig = go.Figure().update_layout(title="No predictions available")
        return html.Div(), html.Div(), empty_fig, empty_fig, None, None, False, "", False, "", "", ""
    
    df = pd.read_json(data, orient='split')
    
    # Initialize results
    rf_results = html.Div("No Random Forest model trained.")
    lstm_results = html.Div("No LSTM model trained.")
    rf_fig = go.Figure().update_layout(title="No Random Forest predictions available")
    lstm_fig = go.Figure().update_layout(title="No LSTM predictions available")
    rf_predictions = None
    lstm_predictions = None
    
    # RF Predictions
    if rf_model_info is not None:
        try:
            # Load RF model
            rf_model = EnhancedRandomForestModel.load_model(rf_model_info["model_path"])
            
            # Make predictions
            next_day_price = rf_model.predict_next_day(df)
            
            # Current price
            last_price = df['Close'].iloc[-1]
            last_date = df['Date'].iloc[-1]
            
            # Calculate change
            change = next_day_price - last_price
            pct_change = (change / last_price) * 100
            
            # Create results display
            rf_results = html.Div([
                html.H5("Random Forest Daily Prediction"),
                html.P(f"Last Close: ${last_price:.2f}"),
                html.P(f"Predicted Next Day: ${next_day_price:.2f}"),
                html.P([
                    f"Change: ${change:.2f} (",
                    html.Span(f"{pct_change:.2f}%", 
                              style={"color": "green" if pct_change >= 0 else "red"}),
                    ")"
                ])
            ])
            
            # Create prediction chart
            rf_fig = go.Figure()
            
            # Add historical data (last 30 days)
            historical_df = df.iloc[-30:]
            rf_fig.add_trace(go.Scatter(
                x=historical_df['Date'], 
                y=historical_df['Close'],
                name="Historical",
                line=dict(color='blue')
            ))
            
            # Add next day prediction point
            next_day_date = last_date + timedelta(days=1)
            rf_fig.add_trace(go.Scatter(
                x=[next_day_date],
                y=[next_day_price],
                name="Next Day Prediction",
                mode="markers",
                marker=dict(size=12, color='red')
            ))
            
            # Update layout
            rf_fig.update_layout(
                title="Random Forest Daily Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                legend=dict(orientation="h")
            )
            
            # Store predictions
            rf_predictions = pd.DataFrame({
                'Date': [next_day_date],
                'Predicted_Close': [next_day_price]
            }).to_json(date_format='iso', orient='split')
            
        except Exception as e:
            rf_results = html.Div(f"Error making RF predictions: {str(e)}")
    
    # LSTM Predictions
    if lstm_model_info is not None:
        try:
            # Check if LSTM model file exists
            model_path = "models/lstm_model.h5"
            if os.path.exists(model_path) or os.path.exists(model_path.replace('.h5', '.keras')):
                # Load LSTM model
                print("Loading LSTM model...")
                lstm_model = LSTMModel.load(model_path)
                
                # Force using the main prediction code
                print("Making LSTM prediction...")
                # Make predictions - now returns a dictionary with uncertainty
                prediction_info = lstm_model.predict(df)
                print(f"LSTM prediction info: {prediction_info.keys() if isinstance(prediction_info, dict) else 'not a dict'}")
                
                # Extract prediction details
                if isinstance(prediction_info, dict):
                    # New format with uncertainty information
                    next_month_price = prediction_info['predicted_price']
                    lower_bound, upper_bound = prediction_info['confidence_interval']
                    uncertainty = prediction_info['uncertainty']
                    prediction_method = prediction_info.get('method', 'lstm_model')
                    print(f"Using prediction method: {prediction_method}")
                else:
                    # Handle old format for backward compatibility
                    next_month_price = prediction_info
                    # Estimate uncertainty as 10% of the prediction
                    uncertainty = next_month_price * 0.1
                    lower_bound = next_month_price * 0.9
                    upper_bound = next_month_price * 1.1
                    prediction_method = 'lstm_model_legacy'
                    print("Using legacy prediction format")
            else:
                # Create a simple prediction based on the last price with a small increase
                # This is a fallback when the model file doesn't exist
                print(f"LSTM model file not found at {model_path} or {model_path.replace('.h5', '.keras')}")
                print("Using simple fallback prediction")
                
                # Train a minimal model on the fly
                try:
                    print("Attempting to train a minimal LSTM model on the fly...")
                    features = ['Close', 'Volume', 'High', 'Low', 'Open']
                    minimal_model = LSTMModel(time_steps=30, features=features, epochs=3)
                    minimal_model.train(df)
                    prediction_info = minimal_model.predict(df)
                    
                    if isinstance(prediction_info, dict):
                        next_month_price = prediction_info['predicted_price']
                        lower_bound, upper_bound = prediction_info['confidence_interval']
                        uncertainty = prediction_info['uncertainty']
                        prediction_method = prediction_info.get('method', 'lstm_minimal')
                        print(f"Successfully used minimal model: {prediction_method}")
                    else:
                        raise ValueError("Minimal model did not return dictionary format")
                except Exception as e:
                    print(f"Error training minimal model: {str(e)}")
                    # Ultimate fallback
                    last_price = df['Close'].iloc[-1]
                    next_month_price = last_price * 1.05  # 5% increase as a placeholder
                    uncertainty = next_month_price * 0.15  # Higher uncertainty for fallback
                    lower_bound = next_month_price * 0.85
                    upper_bound = next_month_price * 1.15
                    prediction_method = 'simple_fallback'
                    print("Using ultimate simple fallback")
            
            # Save the model after successful prediction
            if 'lstm_model' in locals() and lstm_model.model is not None:
                print("Saving LSTM model after successful prediction")
                lstm_model.save(model_path)
            
            # Current price
            last_price = df['Close'].iloc[-1]
            last_date = df['Date'].iloc[-1]
            
            # Calculate change
            change = next_month_price - last_price
            pct_change = (change / last_price) * 100
            
            # Format confidence interval
            confidence_range = f"${lower_bound:.2f} to ${upper_bound:.2f}"
            confidence_pct = (upper_bound - lower_bound) / next_month_price * 100
            
            # Create results display with uncertainty information
            lstm_results = html.Div([
                html.H5("LSTM Monthly Prediction"),
                html.P(f"Last Close: ${last_price:.2f}"),
                html.P(f"Predicted Next Month: ${next_month_price:.2f}"),
                html.P([
                    f"Change: ${change:.2f} (",
                    html.Span(f"{pct_change:.2f}%", 
                             style={"color": "green" if pct_change >= 0 else "red"}),
                    ")"
                ]),
                html.P(f"95% Confidence Interval: {confidence_range}"),
                html.P(f"Uncertainty: ±{uncertainty:.2f} (±{confidence_pct:.1f}%)"),
                html.P(f"Prediction Method: {prediction_method}", className="text-muted small")
            ])
            
            # Create prediction chart
            lstm_fig = go.Figure()
            
            # Add historical data (last 60 days)
            historical_df = df.iloc[-60:]
            lstm_fig.add_trace(go.Scatter(
                x=historical_df['Date'], 
                y=historical_df['Close'],
                name="Historical",
                line=dict(color='blue')
            ))
            
            # Create a projection line with confidence intervals
            future_dates = [last_date + timedelta(days=i) for i in range(0, 31)]  # Daily points for smoother curve
            
            # Linear interpolation between current price and predicted price
            future_prices = [last_price + (change * i / 30) for i in range(0, 31)]
            
            # Calculate upper and lower bounds for each point
            # Uncertainty increases as we move further into the future
            upper_bounds = []
            lower_bounds = []
            for i in range(0, 31):
                # Increasing uncertainty with time - starts at 0 and grows to full confidence interval
                uncertainty_factor = i / 30
                point_uncertainty = uncertainty * uncertainty_factor
                upper_bounds.append(future_prices[i] + point_uncertainty)
                lower_bounds.append(future_prices[i] - point_uncertainty)
            
            # Add prediction line
            lstm_fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_prices,
                name="Projected",
                line=dict(color='purple')
            ))
            
            # Add confidence interval as a shaded area
            lstm_fig.add_trace(go.Scatter(
                x=future_dates + future_dates[::-1],  # x, then x reversed
                y=upper_bounds + lower_bounds[::-1],  # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(128, 0, 128, 0.2)',  # Light purple
                line=dict(color='rgba(128, 0, 128, 0)'),  # Transparent line
                name="95% Confidence Interval",
                showlegend=True
            ))
            
            # Add final prediction point with error bars
            lstm_fig.add_trace(go.Scatter(
                x=[future_dates[-1]],
                y=[next_month_price],
                name="Month Prediction",
                mode="markers",
                marker=dict(size=12, color='red'),
                error_y=dict(
                    type='data',
                    array=[uncertainty],
                    visible=True,
                    color='red'
                )
            ))
            
            # Update layout
            lstm_fig.update_layout(
                title="LSTM Monthly Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                legend=dict(orientation="h")
            )
            
            # Store predictions
            lstm_predictions = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Close': future_prices
            }).to_json(date_format='iso', orient='split')
            
        except Exception as e:
            lstm_results = html.Div(f"Error making LSTM predictions: {str(e)}")
    
    # Show prediction notification
    training_notification = True
    training_message = html.Div([
        html.H5("Generating Predictions", className="mb-2"),
        html.P("Processing data and generating predictions..."),
        html.P("This will only take a moment.", className="mb-0")
    ])
    
    # Show completion notification
    completion_notification = True
    completion_message = html.Div([
        html.H5("Predictions Complete!", className="mb-2"),
        html.P("All predictions have been generated."),
        html.P("Scroll down to view the results and charts.", className="mb-0")
    ])
    
    # Spinner content during prediction
    rf_spinner_content = "Predicting"
    lstm_spinner_content = "Predicting"
    
    return rf_results, lstm_results, rf_fig, lstm_fig, rf_predictions, lstm_predictions, training_notification, training_message, completion_notification, completion_message, rf_spinner_content, lstm_spinner_content

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
