import requests
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, LeakyReLU, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas_ta as ta  # For technical indicators
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    Image as RLImage, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle
)
from tensorflow.keras.regularizers import l1_l2
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime

# Step 1: Fetch Data from Alpha Vantage API using TIME_SERIES_DAILY
API_KEY_ALPHA = "YOUR_API_KEY"  # Replace with your Alpha Vantage API key
SYMBOL = "AAPL"            # Replace with the stock symbol you're interested in
URL_ALPHA = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&outputsize=full&apikey={API_KEY_ALPHA}"

# Fetch stock data
response_alpha = requests.get(URL_ALPHA)
data_alpha = response_alpha.json()

# Extract "Time Series (Daily)" data
time_series = data_alpha.get("Time Series (Daily)", {})
if not time_series:
    print("Error fetching stock data. Please check the API key and symbol.")
    exit()

# Convert to DataFrame
df_stock = pd.DataFrame.from_dict(time_series, orient="index")

# Rename columns
df_stock.columns = ["open", "high", "low", "close", "volume"]

# Convert data types
df_stock.index = pd.to_datetime(df_stock.index)  # Convert index to datetime
df_stock = df_stock.astype(float)     # Convert all columns to float

# Sort the DataFrame by date
df_stock.sort_index(inplace=True)

# Keep only necessary columns
df_stock = df_stock[["close", "volume"]]

# Step 2: Fetch Economic Indicators from FRED API
API_KEY_FRED = "YOUR_API_KEY"  # Replace with your FRED API key

# List of economic indicators to fetch
indicators = {
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers: All Items',
    'CPILFESL': 'CPI excluding food and energy',
    'CPIENGSL': 'CPI for Energy',
    'CUSR0000SAF11': 'CPI for Food at Home',
    'FPCPITOTLZGUSA': 'Inflation, consumer prices for the United States',
    'CUUR0000SEHA': 'Rent of primary residence',
    'CUUS0000SEHC': 'Owners’ equivalent rent of residences',
    'CUSR0000SETB01': 'Gasoline prices',
    'CPIENGSL': 'Energy CPI',
    'UNRATE': 'Unemployment rate',
    'FEDFUNDS': 'Federal funds effective rate',
    'CUSR0000SETC': 'Motor vehicle parts and equipment prices',
    'CUSR0000SAF112': 'Meat, poultry, fish, and egg CPI'
}

# Function to fetch data from FRED
def fetch_fred_data(series_id, api_key, start_date=None, end_date=None):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json'
    }
    if start_date:
        params['observation_start'] = start_date
    if end_date:
        params['observation_end'] = end_date
    response = requests.get(base_url, params=params)
    data = response.json()
    observations = data.get('observations', [])
    df = pd.DataFrame(observations)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.rename(columns={'value': series_id}, inplace=True)
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
    return df[[series_id]]

# Fetch all indicators and merge them into a single DataFrame
start_date = df_stock.index.min().strftime('%Y-%m-%d')
end_date = df_stock.index.max().strftime('%Y-%m-%d')

dfs = []
for series_id, description in indicators.items():
    df_indicator = fetch_fred_data(series_id, API_KEY_FRED, start_date, end_date)
    dfs.append(df_indicator)

# Merge all indicators on date
df_economic = pd.concat(dfs, axis=1)

# Forward fill missing values
df_economic = df_economic.ffill()

# Step 3: Merge Economic Indicators with Stock Data
df = df_stock.merge(df_economic, left_index=True, right_index=True, how='left')

# Fill any remaining missing values
df = df.ffill()
df = df.bfill()

# Step 4: Continue with Existing Feature Engineering and Model Training
# Add date-based features
df["month"] = df.index.month
df["quarter"] = df.index.quarter

# Calculate technical indicators
df['rsi'] = ta.rsi(df['close'], length=14)
macd = ta.macd(df['close'])
df['macd'] = macd['MACD_12_26_9']
bbands = ta.bbands(df['close'], length=20)
df['bb_upper'] = bbands['BBU_20_2.0']
df['bb_middle'] = bbands['BBM_20_2.0']
df['bb_lower'] = bbands['BBL_20_2.0']

# Calculate volatility measures
df['daily_return'] = df['close'].pct_change()
df['volatility'] = df['daily_return'].rolling(window=14).std() * np.sqrt(14)

# Drop rows with NaN values (from calculations)
df.dropna(inplace=True)

# Step 5: Create Lag Features (using the updated function)
def create_lag_features(dataframe, lags, target_columns):
    lagged_data = {}
    for col in target_columns:
        for lag in range(1, lags + 1):
            lagged_col_name = f"{col}_lag_{lag}"
            lagged_data[lagged_col_name] = dataframe[col].shift(lag)
    lagged_df = pd.DataFrame(lagged_data, index=dataframe.index)
    dataframe = pd.concat([dataframe, lagged_df], axis=1)
    dataframe.dropna(inplace=True)
    return dataframe

# Define target columns for lag features (including economic indicators)
target_columns = ['close'] + list(indicators.keys())

# Add lagged features
df = create_lag_features(df, lags=60, target_columns=target_columns)

# Step 6: Normalize Features and Target
# Separate features and target
features = df.drop(columns=["close"])
target = df["close"]

# Initialize scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit scalers and transform data
scaled_features = feature_scaler.fit_transform(features)
scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))

# Combine scaled features and target into a DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)
df_scaled["close"] = scaled_target

# Step 7: Prepare Data for LSTM
# Define features (X) and target (y)
X = df_scaled.drop(columns=["close"]).values
y = df_scaled["close"].values

# Reshape X for LSTM input (samples, timesteps, features)
timesteps = 1  # You can adjust the timesteps if needed
X = X.reshape(X.shape[0], timesteps, X.shape[1])

# Split data into training and testing sets (no shuffling)
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 8: Build and Train the LSTM Neural Network
# Define the LSTM model
# Modify LSTM layers with regularization
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, activation='tanh', return_sequences=True,
         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dropout(0.4),
    LSTM(32, activation='tanh',
         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dropout(0.4),
    Dense(32, activation='relu',
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dense(1)
])

# Compile the model with a custom optimizer
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Step 9: Evaluate the Model
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred_actual = target_scaler.inverse_transform(y_pred)
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
mse = mean_squared_error(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)
mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual) * 100  # Percentage

# Step 10: Forecast Future Prices
# Prepare the last input data point
current_input = X_test[-1].reshape(1, timesteps, X_test.shape[2])

n_future = 120  # Number of days to forecast
future_predictions = []

for _ in range(n_future):
    # Predict the next value
    next_pred = model.predict(current_input)
    future_predictions.append(next_pred[0, 0])

    # Update the current input
    # Remove the first feature and append the new prediction
    new_features = current_input[0, 0, 1:]  # Exclude the first feature
    new_features = np.append(new_features, next_pred[0, 0])

    # Reshape and update current_input
    current_input = new_features.reshape(1, timesteps, -1)

# Inverse transform future predictions
future_predictions_actual = target_scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

# Create dates for future predictions
last_date = df.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_future)

# Step 11: Save Outputs to PDF
# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
loss_plot = 'loss_plot.png'
plt.savefig(loss_plot)
plt.close()

# Prediction Plot with Moving Averages
# Calculate moving averages and store in a dictionary
ma_data = {
    'ma50': df['close'].rolling(window=50).mean(),
    'ma200': df['close'].rolling(window=200).mean()
}

# Convert dictionary to DataFrame
ma_df = pd.DataFrame(ma_data, index=df.index)

# Concatenate moving averages DataFrame to the original df
df = pd.concat([df, ma_df], axis=1)

plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size:], y_test_actual, label='Actual Close Price')
plt.plot(df.index[train_size:], y_pred_actual, label='Predicted Close Price')
plt.plot(df.index[-len(y_test_actual):], df["ma50"][-len(y_test_actual):], label='50-Day MA', linestyle='--')
plt.plot(df.index[-len(y_test_actual):], df["ma200"][-len(y_test_actual):], label='200-Day MA', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{SYMBOL} Stock Price Prediction')
plt.legend()
prediction_plot = 'prediction_plot.png'
plt.savefig(prediction_plot)
plt.close()

# Forecast Plot with Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size:], y_test_actual, label='Actual Close Price')
plt.plot(df.index[train_size:], y_pred_actual, label='Predicted Close Price')
plt.plot(df.index[-len(y_test_actual):], df["ma50"][-len(y_test_actual):], label='50-Day MA', linestyle='--')
plt.plot(df.index[-len(y_test_actual):], df["ma200"][-len(y_test_actual):], label='200-Day MA', linestyle='--')
plt.plot(future_dates, future_predictions_actual, label='Future Predictions', marker='o')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{SYMBOL} Stock Price Forecast')
plt.legend()
forecast_plot = 'forecast_plot.png'
plt.savefig(forecast_plot)
plt.close()

# Prepare future predictions DataFrame
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Close Price': future_predictions_actual.flatten()
})

# Generate PDF Report
pdf_file = (f'{SYMBOL} Stock Report.pdf')
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
Story = []

# Add Title
title = f"{SYMBOL} Stock Price Prediction Report"
Story.append(Paragraph(title, styles['Title']))
Story.append(Spacer(1, 12))

# Add Hyperparameters and Training Summary
hyperparameters_text = f"""
<b>Model Hyperparameters:</b><br/>
Bidirectional LSTM Units: 128<br/>
LSTM Units: 64<br/>
Dense Units: 64 and 32 with LeakyReLU activation<br/>
Dropout Rates: 30% and 20%<br/>
Optimizer: Adam (Learning Rate: 0.0001)<br/>
Batch Size: 64<br/>
Epochs: {len(history.history['loss'])}<br/>
"""
Story.append(Paragraph(hyperparameters_text, styles['Normal']))
Story.append(Spacer(1, 12))

# Add Evaluation Metrics
metrics_text = f"""
<b>Evaluation Metrics:</b><br/>
Mean Squared Error (MSE): {mse:.2f}<br/>
Mean Absolute Error (MAE): {mae:.2f}<br/>
Mean Absolute Percentage Error (MAPE): {mape:.2f}%<br/>
R-squared Score: {r2:.2f}
"""
Story.append(Paragraph(metrics_text, styles['Normal']))
Story.append(Spacer(1, 12))

# Add Training and Validation Loss Plot
Story.append(Paragraph("<b>Training and Validation Loss:</b>", styles['Heading2']))
Story.append(RLImage(loss_plot, width=500, height=250))
Story.append(Spacer(1, 12))

# Add Prediction Plot
Story.append(Paragraph("<b>Prediction Plot:</b>", styles['Heading2']))
Story.append(RLImage(prediction_plot, width=500, height=250))
Story.append(Spacer(1, 12))

# Add Forecast Plot
Story.append(Paragraph("<b>Forecast Plot:</b>", styles['Heading2']))
Story.append(RLImage(forecast_plot, width=500, height=250))
Story.append(Spacer(1, 12))

# Add Future Predictions Table
Story.append(Paragraph("<b>Future Predictions:</b>", styles['Heading2']))

# Ensure the Date column is of type datetime
future_df['Date'] = pd.to_datetime(future_df['Date'])

# Format the Date column to 'YYYY-MM-DD'
future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')

# Convert future_df to table data for PDF
table_data = [future_df.columns.tolist()] + future_df.values.tolist()

# Chunk the table data to prevent overflow
chunk_size = 15  # Adjust based on page size
table_chunks = [table_data[i:i+chunk_size] for i in range(0, len(table_data), chunk_size)]

# Add each chunk to the PDF
for chunk in table_chunks:
    t = Table(chunk)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    Story.append(t)
    Story.append(Spacer(1, 12))

# Build the PDF
doc.build(Story)

print(f"Report has been generated and saved as {pdf_file}.")

# Clean up image files
os.remove(loss_plot)
os.remove(prediction_plot)
os.remove(forecast_plot)

# Print a message indicating that the process is complete
print("Model training and forecasting complete. Outputs have been saved.")