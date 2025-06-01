import pandas as pd
from lstm_model import LSTMModel

# Load your stock data
df = pd.read_csv('cache_GOOGL_10y.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Define features to use
features = ['Close', 'Volume', 'MA7', 'MA21', 'RSI', 'MACD']

# Create and train the model
model = LSTMModel(time_steps=60, features=features, epochs=100, batch_size=64)
history = model.train(df)

# Predict next month's closing price
next_month_price = model.predict_next_month(df)
print(f"Predicted closing price for next month: ${next_month_price:.2f}")

# Optionally save the model
model.save('models/lstm_model.h5')