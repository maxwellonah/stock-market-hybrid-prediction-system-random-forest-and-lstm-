# Hybrid Stock Prediction System

This Django web application implements a hybrid predictive system for stock market analysis, combining two powerful machine learning models:

1. **Random Forest Model**: For daily stock price predictions
2. **LSTM (Long Short-Term Memory) Model**: For monthly stock price predictions

## Features

- **Data Sources**:
  - Fetch stock data from Polygon API (GOOGL, AAPL, MSFT, AMZN)
  - Upload your own CSV dataset for custom analysis

- **Interactive Visualizations**:
  - Price charts with candlestick patterns using Plotly
  - Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
  - Prediction visualizations for both models

- **Prediction Capabilities**:
  - Daily predictions using Random Forest
  - Monthly predictions using LSTM
  - Visual comparison of predictions vs. historical data

## Technology Stack

- **Django 5.0.1**: Web framework
- **NumPy 1.26.3**: Numerical computing
- **Pandas 2.2.0**: Data manipulation
- **scikit-learn 1.4.0**: Machine learning for Random Forest
- **TensorFlow 2.16.1**: Deep learning for LSTM
- **Plotly 5.0.0**: Interactive visualizations
- **Bootstrap 5**: Frontend styling

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Polygon API key in the `.env` file (already included)

### Running the Application

```
python app.py
```

## Usage

1. **Select Data Source**:
   - Choose a stock from the dropdown and click "Fetch Data", or
   - Upload your own CSV file with the required columns (Date, Open, High, Low, Close, Volume)

2. **Analyze Technical Indicators**:
   - View the various technical indicators in the dedicated tabs

3. **Generate Predictions**:
   - Click "Train Models" to train both the Random Forest and LSTM models
   - Click "Make Predictions" to generate and visualize predictions

## Model Information

### Random Forest Model
- Uses enhanced feature engineering with time-series characteristics
- Implements feature selection to focus on the most important factors
- Provides daily predictions with visualization

### LSTM Model
- Bidirectional LSTM architecture for capturing long-term dependencies
- Designed specifically for time-series forecasting
- Provides monthly predictions with visualization

## Project Structure

```
ensemble_web/                  # Django project directory
├── settings.py                # Project settings
├── urls.py                    # URL configuration
└── wsgi.py                    # WSGI configuration

stock_prediction/              # Django app directory
├── models.py                  # Database models
├── views.py                   # View functions
├── forms.py                   # Form definitions
└── urls.py                    # App URL configuration

templates/                     # HTML templates
├── base.html                  # Base template
└── stock_prediction/          # App templates
    ├── index.html             # Home page
    └── predictions.html       # Predictions page

static/                        # Static files
├── css/                       # CSS files
└── js/                        # JavaScript files

media/                         # User-uploaded files
├── stock_data/                # Stock data CSV files
└── models/                    # Saved ML models

rf_model.py                    # Random Forest model implementation
lstm_model.py                  # LSTM model implementation
manage.py                      # Django management script
requirements.txt               # Project dependencies
```

## License

This project is for educational purposes only.
