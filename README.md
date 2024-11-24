Overview

This Python project implements a pipeline for predicting stock prices using historical stock data, economic indicators, and technical indicators. It utilizes machine learning (specifically an LSTM-based neural network) to forecast stock prices and generates a comprehensive PDF report that includes:
	
 1.	Training and validation loss plots.
 2.	Actual vs. predicted stock prices.
 3.	Future stock price predictions.

Features

•	Fetches daily stock price data from the Alpha Vantage API.
•	Collects economic indicators from the FRED API.
•	Adds technical indicators (e.g., RSI, MACD, Bollinger Bands) using pandas_ta.
•	Applies feature engineering, including lagged features and normalization.
•	Builds and trains an LSTM neural network for time series forecasting.
•	Generates a PDF report with training metrics, plots, and future predictions.

Setup Instructions

Prerequisites

Make sure you have the following installed:
	•	Python 3.7 or higher
	•	Required libraries (install via pip):

    pip install numpy pandas matplotlib scikit-learn tensorflow pandas_ta reportlab 
    

Required API Keys

  1.	Alpha Vantage API Key:
  •	Obtain from Alpha Vantage.
  •	Replace the placeholder value for API_KEY_ALPHA in the code.

  3.	FRED API Key:
  •	Obtain from Federal Reserve Economic Data (FRED).
  •	Replace the placeholder value for API_KEY_FRED in the code.

Running the Code

Step 1: Fetch Stock and Economic Data

The script fetches:
	•	Daily stock prices for a specific stock (e.g., AAPL) from Alpha Vantage.
	•	Economic indicators (e.g., CPI, unemployment rate) from FRED.

Update these variables as needed:
	•	SYMBOL: The stock symbol to analyze (default: “AAPL”).
	•	indicators: A dictionary of economic indicator series from FRED.

Step 2: Feature Engineering

The pipeline includes:
	•	Date-based features (month, quarter).
	•	Technical indicators (RSI, MACD, Bollinger Bands).
	•	Lag features for both stock prices and economic indicators.

Step 3: Train the LSTM Model

  •	The model trains on 80% of the data and validates on the remaining 20%.
	•	Hyperparameters such as learning rate, dropout, and optimizer can be adjusted as needed.

Step 4: Forecast and Evaluate

  •	Generates future predictions for n_future days (default: 120).
	•	Evaluates the model using:
	•	Mean Squared Error (MSE)
	•	Mean Absolute Error (MAE)
	•	Mean Absolute Percentage Error (MAPE)
	•	R-squared (R²)

Step 5: Generate PDF Report

The script generates a PDF report ({SYMBOL} Stock Report.pdf) containing:
	•	Training and validation loss plots.
	•	Actual vs. predicted price plots.
	•	Forecasted future prices.

File Outputs

  •	PDF Report: {SYMBOL} Stock Report.pdf
	•	Future Predictions: A table with forecasted dates and predicted close prices.

Key Libraries

•	TensorFlow/Keras: For building the LSTM model.
•	pandas_ta: For calculating technical indicators.
•	scikit-learn: For normalization and evaluation metrics.
•	reportlab: For generating the PDF report.
•	Alpha Vantage API: For stock data.
•	FRED API: For economic data.

Customization

Changing the Stock Symbol

Modify the SYMBOL variable to analyze a different stock:

SYMBOL = "AAPL"  # Example: Apple stock

Adjusting Forecast Duration

Update n_future to change the forecasted period:

n_future = 180  # Forecast 180 days

Hyperparameter Tuning

Modify the LSTM layers, dropout rates, and optimizer settings in the model definition.

Example Output

  1.	Training and Validation Loss Plot
	•	Shows how the model’s loss evolves during training.

  2.	Actual vs. Predicted Prices
	•	Visual comparison of actual and predicted stock prices.

  3.	Future Predictions
	•	Table and plot of predicted stock prices for the next 120 days.

Notes

  •	Use valid API keys for both Alpha Vantage and FRED APIs.


Using requirements.txt and Virtual Environment

To ensure a smooth setup of your Python environment for this project, follow the steps below:

1. Create a Virtual Environment

A virtual environment helps isolate your project’s dependencies from the system’s Python environment.
	1.	Open a terminal or command prompt.
	2.	Navigate to the project directory:

cd /path/to/project

  3.	Create a virtual environment:

    python -m venv venv

This creates a directory named venv containing the virtual environment.

  4.	Activate the virtual environment:
	•	Windows: venv\Scripts\activate

  •	macOS/Linux: source venv/bin/activate

After activation, your terminal will display the virtual environment’s name, like (venv).

2. Install Dependencies

Use the requirements.txt file to install the project’s required libraries.
	1.	Ensure the virtual environment is activated.
	2.	Run the following command:

    pip install -r requirements.txt

This will install all the necessary libraries listed in the requirements.txt file.

3. Verify the Installation

To confirm all dependencies were installed correctly, run:

    pip list

This command will display a list of installed libraries and their versions.

4. Deactivate the Virtual Environment

When you’re done working, deactivate the virtual environment to return to the system’s Python environment:

    deactivate


5. Reuse the Environment

Next time you work on the project:
	1.	Navigate to the project directory.
	2.	Activate the virtual environment:

source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows

License

This project is open-source and free to use. For issues or contributions, please create a pull request or open an issue.

Happy coding and forecasting! 🚀
