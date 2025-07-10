
# Fetch news using yfinance only
def fetch_news_yfinance(ticker):
    """
    Fetch news using yfinance's .news attribute. Returns a list of dicts with keys: title, link, publisher, providerPublishTime
    """
    try:
        stock = yf.Ticker(ticker)
        news = getattr(stock, 'news', [])
        mapped_news = []
        for item in news:
            mapped_news.append({
                'title': item.get('title', ''),
                'link': item.get('link', '#'),
                'publisher': item.get('publisher', ''),
                'providerPublishTime': item.get('providerPublishTime', '')
            })
        return mapped_news
    except Exception as e:
        print(f"Exception in fetch_news_yfinance: {e}")
        return []
import yfinance as yf  # Library for fetching stock data
import pandas as pd  # Library for handling data in tables (like Excel)
import dash  # Library for creating a web-based dashboard
from dash import dcc, html  # Components for the Dash dashboard
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from textblob import TextBlob
import plotly.graph_objects as go  # For creating interactive graphs
# import torch  # PyTorch for building and training the prediction model
# import torch.nn as nn  # For creating neural network layers
from sklearn.preprocessing import MinMaxScaler  # For scaling data (e.g., making numbers smaller)
from rich.console import Console  # For printing colorful and formatted outputs in the terminal
from rich.table import Table  # For displaying data in a table format in the terminal
import numpy as np  # For working with numerical data and arrays
from dash.dependencies import Input, Output, State
import requests
from textblob import TextBlob

# Setting up global variables for later use
console = Console()  # A console object to print colorful messages
scaler = MinMaxScaler()  # A scaler object to normalize data (e.g., make values between 0 and 1)

# Defining a neural network for predicting stock prices
# class StockPredictor(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         """Set up the layers of the neural network."""
#         super(StockPredictor, self).__init__()
#         self.hidden_layer = nn.Linear(input_size, hidden_size)  # Hidden layer
#         self.output_layer = nn.Linear(hidden_size, output_size)  # Output layer
#
#     def forward(self, x):
#         """Defines how the input data flows through the network."""
#         x = torch.relu(self.hidden_layer(x))  # Apply activation function to hidden layer
#         x = self.output_layer(x)  # Compute the final output
#         return x


def fetch_stock_data(ticker, start_date=None, end_date=None):
    """
    Fetch stock data using yfinance.
    Ticker is the stock symbol (e.g., AAPL for Apple).
    Default: fetches last 5 years if no dates provided.
    """
    console.print(f"\n[bold green]Fetching stock data for {ticker}...[/bold green]")
    try:
        stock = yf.Ticker(ticker)  # Initialize the ticker
        # Get historical data for the stock
        if start_date and end_date:
            data = stock.history(start=start_date, end=end_date)
        else:
            data = stock.history(period="5y")  # Default: Last 5 years
        if data.empty:
            # If no data, show a message
            console.print(f"[bold red]No data found for ticker '{ticker}'. Please check the ticker or date range.[/bold red]")
        return data
    except Exception as e:
        # Handle any errors while fetching the data
        console.print(f"[bold red]Error fetching data: {e}[/bold red]")
        return pd.DataFrame()


def calculate_indicators(data):
    """
    Calculate indicators like EMA, Bollinger Bands, RSI, MACD, Stochastic Oscillator, ATR.
    """
    console.print("\n[bold blue]Calculating technical indicators...[/bold blue]")
    try:
        # Exponential Moving Averages (EMA)
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['Upper_Band'] = rolling_mean + (rolling_std * 2)
        data['Lower_Band'] = rolling_mean - (rolling_std * 2)

        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Stochastic Oscillator
        low14 = data['Low'].rolling(window=14).min()
        high14 = data['High'].rolling(window=14).max()
        data['%K'] = 100 * (data['Close'] - low14) / (high14 - low14)
        data['%D'] = data['%K'].rolling(window=3).mean()

        # ATR (Average True Range)
        data['H-L'] = data['High'] - data['Low']
        data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
        data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
        tr = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        data['ATR'] = tr.rolling(window=14).mean()

        console.print("[bold green]Indicators successfully calculated![/bold green]\n")
        return data
    except Exception as e:
        console.print(f"[bold red]Error calculating indicators: {e}[/bold red]")
        return data


def provide_insights(data):
    """
    Print a summary of the stock data in the terminal.
    Includes metrics like average price and RSI.
    """
    console.print("\n[bold magenta]--- Stock Insights ---[/bold magenta]")
    try:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric")  # Metric name
        table.add_column("Value")  # Metric value

        # Add rows with calculated values
        table.add_row("Average Closing Price", f"${data['Close'].mean():.2f}")
        table.add_row("Highest Closing Price", f"${data['Close'].max():.2f}")
        table.add_row("Lowest Closing Price", f"${data['Close'].min():.2f}")
        table.add_row("Volatility (Std Dev)", f"{data['Close'].std():.2f}")

        # Add RSI insights
        if 'RSI' in data.columns:
            last_rsi = data['RSI'].iloc[-1]
            if last_rsi > 70:
                table.add_row("RSI", f"{last_rsi:.2f} (Overbought)")  # RSI > 70 means overbought
            elif last_rsi < 30:
                table.add_row("RSI", f"{last_rsi:.2f} (Oversold)")  # RSI < 30 means oversold
            else:
                table.add_row("RSI", f"{last_rsi:.2f} (Neutral)")
        else:
            table.add_row("RSI", "Not enough data")

        # Print the table
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error providing insights: {e}[/bold red]")


def export_data(data, ticker, export_format):
    """
    Export stock data to a file (CSV, Excel, or JSON).
    export_format: 'csv', 'excel', or 'json' (case-insensitive)
    """
    try:
        # Ensure the data has no timezone info
        if pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = data.index.tz_localize(None)

        # Export based on argument
        export_format = export_format.strip().lower()
        if export_format == "csv":
            data.to_csv(f"{ticker}_analysis.csv")
            console.print(f"[bold green]Data successfully saved to {ticker}_analysis.csv.[/bold green]")
        elif export_format == "excel":
            try:
                data.to_excel(f"{ticker}_analysis.xlsx")
                console.print(f"[bold green]Data successfully saved to {ticker}_analysis.xlsx.[/bold green]")
            except ModuleNotFoundError:
                console.print("[bold red]Error: 'openpyxl' library is required for Excel export.[/bold red]")
        elif export_format == "json":
            data.to_json(f"{ticker}_analysis.json")
            console.print(f"[bold green]Data successfully saved to {ticker}_analysis.json.[/bold green]")
        else:
            console.print("[bold red]Invalid format. Export skipped.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error exporting data: {e}[/bold red]")


def predict_stock_prices(data):
    """
    Use a PyTorch neural network to predict future stock prices.
    Includes additional features and improved model architecture.
    Returns predictions and error metrics (RMSE, MAE).
    """
    console.print("\n[bold blue]--- Predicting Stock Prices ---[/bold blue]")
    try:
        # Add additional features for training
        data['Days'] = (data.index - data.index[0]).days
        data['Pct_Change'] = data['Close'].pct_change(fill_method=None)
        data['Volume_Change'] = data['Volume'].pct_change(fill_method=None)
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        data['Prev_Close_1'] = data['Close'].shift(1)
        data['Prev_Close_3'] = data['Close'].shift(3)
        data['Prev_Close_5'] = data['Close'].shift(5)
        data['Prev_Close_10'] = data['Close'].shift(10)
        data['DayOfWeek'] = data.index.dayofweek
        # Clean data: replace inf/-inf with NaN, then drop all NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        # Features for model
        features = [
            'Days', 'Close', 'EMA_20', 'EMA_50', 'RSI', 'Upper_Band', 'Lower_Band',
            'MACD', 'MACD_Signal', '%K', '%D', 'ATR',
            'Pct_Change', 'Volume_Change', 'Momentum',
            'Prev_Close_1', 'Prev_Close_3', 'Prev_Close_5', 'Prev_Close_10', 'DayOfWeek'
        ]
        scaled_features = scaler.fit_transform(data[features].values)
        X = scaled_features[:, [i for i in range(len(features)) if features[i] != 'Close']]  # Exclude 'Close' as feature
        y = scaled_features[:, features.index('Close')]    # Target (scaled Close price)
        # Convert to PyTorch tensors
        # X_tensor = torch.tensor(X, dtype=torch.float32)
        # y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        # Train/Test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        # Improved model with Dropout and LeakyReLU
        # class ImprovedStockPredictor(nn.Module):
        #     def __init__(self, input_size, hidden_size, output_size):
        #         super().__init__()
        #         self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        #         self.dropout = nn.Dropout(0.2)
        #         self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        #         self.output_layer = nn.Linear(hidden_size, output_size)
        #         self.leaky_relu = nn.LeakyReLU()
        #     def forward(self, x):
        #         x = self.leaky_relu(self.hidden_layer1(x))
        #         x = self.dropout(x)
        #         x = self.leaky_relu(self.hidden_layer2(x))
        #         x = self.output_layer(x)
        #         return x
        # model = ImprovedStockPredictor(input_size=X_train.shape[1], hidden_size=128, output_size=1)
        # criterion = nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # # Train the model
        # console.print("[cyan]Training the neural network...[/cyan]")
        # best_val_loss = float('inf')
        # patience = 50
        # patience_counter = 0
        # for epoch in range(1000):
        #     model.train()
        #     optimizer.zero_grad()
        #     predictions = model(X_train)
        #     loss = criterion(predictions, y_train)
        #     loss.backward()
        #     optimizer.step()
        #     # Early stopping on validation loss
        #     model.eval()
        #     with torch.no_grad():
        #         val_predictions = model(X_test)
        #         val_loss = torch.sqrt(criterion(val_predictions, y_test))
        #         if val_loss.item() < best_val_loss:
        #             best_val_loss = val_loss.item()
        #             patience_counter = 0
        #         else:
        #             patience_counter += 1
        #         if patience_counter > patience:
        #             console.print(f"Early stopping at epoch {epoch+1}")
        #             break
        #     if (epoch + 1) % 50 == 0:
        #         console.print(f"Epoch {epoch + 1}/1000 - Loss: {loss.item():.4f} - Val RMSE: {val_loss.item():.4f}")
        # # Calculate error metrics
        # model.eval()
        # with torch.no_grad():
        #     test_preds = model(X_test).squeeze().numpy()
        #     test_true = y_test.squeeze().numpy()
        #     rmse = np.sqrt(np.mean((test_preds - test_true) ** 2))
        #     mae = np.mean(np.abs(test_preds - test_true))
        # # Predict future prices
        # # --- Recursive walk-forward prediction for next 10 days ---
        # future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
        # walk_data = data.copy()
        # preds = []
        # last_valid_close = walk_data['Close'].iloc[-1]
        # for i, date in enumerate(future_dates):
        #     # Build feature vector using latest indicators
        #     days = (date - walk_data.index[0]).days
        #     prev_close_1 = walk_data['Close'].iloc[-1]
        #     prev_close_3 = walk_data['Close'].iloc[-3] if len(walk_data) >= 3 else walk_data['Close'].iloc[-1]
        #     prev_close_5 = walk_data['Close'].iloc[-5] if len(walk_data) >= 5 else walk_data['Close'].iloc[-1]
        #     prev_close_10 = walk_data['Close'].iloc[-10] if len(walk_data) >= 10 else walk_data['Close'].iloc[-1]
        #     pct_change = walk_data['Close'].pct_change(fill_method=None).iloc[-1]
        #     if np.isnan(pct_change) or np.isinf(pct_change):
        #         pct_change = 0.0
        #     vol_change = walk_data['Volume'].pct_change(fill_method=None).iloc[-1] if 'Volume' in walk_data.columns else 0
        #     if np.isnan(vol_change) or np.isinf(vol_change):
        #         vol_change = 0.0
        #     momentum = walk_data['Close'].iloc[-1] - walk_data['Close'].iloc[-11] if len(walk_data) >= 11 else 0
        #     day_of_week = date.dayofweek
        #     feature_vector = [
        #         days,
        #         0,
        #         walk_data['EMA_20'].iloc[-1],
        #         walk_data['EMA_50'].iloc[-1],
        #         walk_data['RSI'].iloc[-1],
        #         walk_data['Upper_Band'].iloc[-1],
        #         walk_data['Lower_Band'].iloc[-1],
        #         walk_data['MACD'].iloc[-1],
        #         walk_data['MACD_Signal'].iloc[-1],
        #         walk_data['%K'].iloc[-1],
        #         walk_data['%D'].iloc[-1],
        #         walk_data['ATR'].iloc[-1],
        #         pct_change,
        #         vol_change,
        #         momentum,
        #         prev_close_1,
        #         prev_close_3,
        #         prev_close_5,
        #         prev_close_10,
        #         day_of_week
        #     ]
        #     fv_check = feature_vector[:]
        #     fv_check[1] = walk_data['Close'].iloc[-1]
        #     if any([np.isnan(x) or np.isinf(x) for x in fv_check]):
        #         pred_close = last_valid_close
        #     else:
        #         try:
        #             scaled = scaler.transform([feature_vector])
        #             X_pred = np.delete(scaled, features.index('Close'), axis=1)
        #             X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32)
        #             pred_scaled = model(X_pred_tensor).detach().numpy().flatten()[0]
        #             scaled[0, features.index('Close')] = pred_scaled
        #             inv = scaler.inverse_transform(scaled)
        #             pred_close = inv[0, features.index('Close')]
        #             if np.isnan(pred_close) or np.isinf(pred_close):
        #                 pred_close = last_valid_close
        #         except Exception as e:
        #             pred_close = last_valid_close
        #     preds.append(pred_close)
        #     last_valid_close = pred_close
        #     # Prepare new row for walk_data
        #     new_row = {
        #         'Days': days,
        #         'Close': pred_close,
        #         'EMA_20': walk_data['EMA_20'].iloc[-1],
        #         'EMA_50': walk_data['EMA_50'].iloc[-1],
        #         'RSI': walk_data['RSI'].iloc[-1],
        #         'Upper_Band': walk_data['Upper_Band'].iloc[-1],
        #         'Lower_Band': walk_data['Lower_Band'].iloc[-1],
        #         'MACD': walk_data['MACD'].iloc[-1],
        #         'MACD_Signal': walk_data['MACD_Signal'].iloc[-1],
        #         '%K': walk_data['%K'].iloc[-1],
        #         '%D': walk_data['%D'].iloc[-1],
        #         'ATR': walk_data['ATR'].iloc[-1],
        #         'Pct_Change': pct_change,
        #         'Volume_Change': vol_change,
        #         'Momentum': momentum,
        #         'Prev_Close_1': prev_close_1,
        #         'Prev_Close_3': prev_close_3,
        #         'Prev_Close_5': prev_close_5,
        #         'Prev_Close_10': prev_close_10,
        #         'DayOfWeek': day_of_week,
        #         'Volume': walk_data['Volume'].iloc[-1] if 'Volume' in walk_data.columns else 0
        #     }
        #     walk_data = pd.concat([walk_data, pd.DataFrame([new_row], index=[date])], sort=False)
        #     # Recalculate indicators after appending new row
        #     walk_data = calculate_indicators(walk_data)
        # preds = [x if not (np.isnan(x) or np.isinf(x)) else last_valid_close for x in preds]
        # # If all predictions are the same, warn but do not alter predictions
        # if len(set(np.round(preds, 6))) == 1:
        #     console.print("[bold yellow]Warning: All predicted prices are the same. Model may not be learning or features are not updating.\nTry using a different ticker or check your model/data.[/bold yellow]")
        # predictions = pd.DataFrame({'Date': future_dates, 'Predicted Price': preds})
        # error_metrics = {'RMSE': float(rmse), 'MAE': float(mae)}
        # print("Predictions head:", predictions.head())
        predictions = pd.DataFrame()
        error_metrics = {'RMSE': None, 'MAE': None}
        return predictions, error_metrics
    except Exception as e:
        console.print(f"[bold red]Error in prediction: {e}[/bold red]")
        return pd.DataFrame(), {'RMSE': None, 'MAE': None}
    


def create_dashboard(data, ticker, predictions, error_metrics=None):
    """
    Build and launch a web dashboard for stock analysis using Dash.
    """
    app = dash.Dash(__name__)
    server = app.server  # Expose server for Azure

    # Responsive meta tag for mobile
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                /* Responsive styles for mobile */
                body { margin: 0; padding: 0; }
                .dash-table-container { overflow-x: auto; }
                /* Logo styles */
                .stocklens-logo {
                    height: 150px;
                    margin-right: 25px;
                    vertical-align: middle;
                    display: inline-block;
                    position: absolute;
                    top: 10px;
                    left: 20px;
                    z-index: 10;
                    transition: all 0.3s;
                }
                /* Button styles */
                button, .dash-button, .dash-spreadsheet-menu button {
                    background: linear-gradient(90deg, #2980b9 0%, #6dd5fa 100%);
                    color: #fff;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 20px;
                    font-size: 1rem;
                    font-family: inherit;
                    cursor: pointer;
                    box-shadow: 0 2px 8px rgba(41,128,185,0.08);
                    transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
                }
                button:hover, .dash-button:hover, .dash-spreadsheet-menu button:hover {
                    background: linear-gradient(90deg, #6dd5fa 0%, #2980b9 100%);
                    box-shadow: 0 4px 16px rgba(41,128,185,0.18);
                    transform: translateY(-2px) scale(1.04);
                }
                /* Feedback button override */
                .feedback-btn {
                    background: linear-gradient(90deg, #4285F4 0%, #34a853 100%);
                    color: #fff;
                }
                .feedback-btn:hover {
                    background: linear-gradient(90deg, #34a853 0%, #4285F4 100%);
                }
                @media (max-width: 600px) {
                    .dash-table-container, .dash-spreadsheet-container, .dash-table { font-size: 13px !important; }
                    .dash-table-container table { width: 100% !important; }
                    .dash-table-container th, .dash-table-container td { padding: 6px 4px !important; }
                    .stocklens-logo {
                        position: static;
                        display: block;
                        margin: 0 auto 10px auto;
                        height: 70px;
                        max-width: 90vw;
                    }
                    .main-content-mobile-margin {
                        margin-top: 10px !important;
                    }
                    .modern-input {
                        width: 95vw !important;
                        max-width: 400px !important;
                        font-size: 1.1rem !important;
                        padding: 14px 18px !important;
                    }
                }
                /* Modern input style */
                .modern-input {
                    width: 100%;
                    max-width: 400px;
                    font-size: 1.15rem;
                    padding: 14px 20px;
                    border: 1.5px solid #b2bec3;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(41,128,185,0.07);
                    outline: none;
                    transition: border 0.2s, box-shadow 0.2s;
                    background: #fafdff;
                    color: #222;
                    margin-bottom: 10px;
                }
                .modern-input:focus {
                    border: 2px solid #2980b9;
                    box-shadow: 0 4px 16px rgba(41,128,185,0.13);
                    background: #f0f8ff;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    # Add buy/sell signals based on EMA, only if data is not empty and has required columns
    if not data.empty and 'Close' in data.columns and 'EMA_20' in data.columns:
        data['Buy_Signal'] = (data['Close'] > data['EMA_20']) & (data['Close'].shift(1) <= data['EMA_20'])
        data['Sell_Signal'] = (data['Close'] < data['EMA_20']) & (data['Close'].shift(1) >= data['EMA_20'])
    else:
        data['Buy_Signal'] = False
        data['Sell_Signal'] = False

    # Create a graph with Plotly only if data is not empty and has required columns
    fig = go.Figure()
    required_cols = ['Close', 'EMA_20', 'Upper_Band', 'Lower_Band']
    if not data.empty and all(col in data.columns for col in required_cols):
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name="EMA 20", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], name="Upper Bollinger Band", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], name="Lower Bollinger Band", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(
            x=data[data['Buy_Signal']].index if 'Buy_Signal' in data.columns else [],
            y=data[data['Buy_Signal']]['Close'] if 'Buy_Signal' in data.columns else [],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=data[data['Sell_Signal']].index if 'Sell_Signal' in data.columns else [],
            y=data[data['Sell_Signal']]['Close'] if 'Sell_Signal' in data.columns else [],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ))
        fig.update_layout(
            title=f"{ticker} Stock Analysis with Indicators and Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Indicators"
        )
    else:
        fig.add_annotation(text="No valid data to display chart.",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=18, color="red"))
        fig.update_layout(title="No Data", xaxis_title="Date", yaxis_title="Price")
    # Removed stray code that caused SyntaxError

    # Add predictions to the dashboard
    if predictions is not None and not predictions.empty and 'Date' in predictions.columns and 'Predicted Price' in predictions.columns:
        prediction_component = dcc.Graph(figure=go.Figure(data=[
            go.Scatter(x=predictions['Date'], y=predictions['Predicted Price'], name="Predicted Price")
        ]))
    else:
        prediction_component = html.Div("No predictions available to display.")
    # Error metrics display
    if error_metrics and error_metrics['RMSE'] is not None:
        metrics_component = html.Div([
            html.H4("Prediction Error Metrics", style={"textAlign": "center", "color": "#8e44ad"}),
            html.Div([
                html.Span(f"RMSE: {error_metrics['RMSE']:.4f}", style={"marginRight": 20}),
                html.Span(f"MAE: {error_metrics['MAE']:.4f}")
            ], style={"textAlign": "center", "color": "#555", "fontSize": 16, "marginBottom": 20})
        ])
    else:
        metrics_component = html.Div()

    # Education section: collapsible info about indicators
    indicator_info = [
        {
            "title": "Exponential Moving Average (EMA)",
            "desc": "EMA is a type of moving average that places a greater weight and significance on the most recent data points. Used to identify trend direction and potential buy/sell signals."
        },
        {
            "title": "Bollinger Bands",
            "desc": "Bollinger Bands consist of a middle band (SMA) and two outer bands (standard deviations away). They help visualize price volatility and identify overbought/oversold conditions."
        },
        {
            "title": "Relative Strength Index (RSI)",
            "desc": "RSI is a momentum oscillator that measures the speed and change of price movements. Values above 70 indicate overbought, below 30 indicate oversold."
        },
        {
            "title": "MACD (Moving Average Convergence Divergence)",
            "desc": "MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. It helps identify potential buy/sell signals and trend reversals."
        },
        {
            "title": "Stochastic Oscillator",
            "desc": "The Stochastic Oscillator compares a particular closing price of a security to a range of its prices over a certain period of time. It is used to generate overbought and oversold trading signals."
        },
        {
            "title": "ATR (Average True Range)",
            "desc": "ATR measures market volatility by decomposing the entire range of an asset price for that period. Higher ATR values indicate higher volatility."
        },
        {
            "title": "Momentum",
            "desc": "Momentum measures the rate of the rise or fall in stock prices. It is calculated as the difference between the current price and the price from a set number of periods ago."
        },
        {
            "title": "Percentage Change",
            "desc": "Shows the percent change in price from one period to the next. Useful for measuring volatility and trend strength."
        },
        {
            "title": "Volume Change",
            "desc": "Shows the percent change in trading volume from one period to the next. Can indicate increased interest or activity in a stock."
        },
        {
            "title": "Prediction Error Metrics (RMSE & MAE)",
            "desc": "RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) are metrics that measure how close the model's predicted prices are to the actual prices. Lower values indicate better prediction accuracy. RMSE penalizes larger errors more, while MAE gives a straightforward average of absolute errors. These metrics help you evaluate the reliability of the model's predictions."
        }
    ]

    education_section = html.Div([
        html.H2("About Technical Indicators", style={"textAlign": "center", "color": "#2980b9", "marginTop": 40}),
        html.Div([
            html.Details([
                html.Summary(ind["title"], style={"fontWeight": "bold", "fontSize": 17, "cursor": "pointer"}),
                html.Div(ind["desc"], style={"marginLeft": 20, "marginBottom": 10, "color": "#555", "fontSize": 15})
            ], style={"marginBottom": 10, "background": "#f0f4f8", "borderRadius": 6, "padding": 8, "boxShadow": "0 1px 3px #ddd"})
            for ind in indicator_info
        ], style={"maxWidth": 700, "margin": "0 auto"})
    ])

    disclaimer_section = html.Div([
        html.Hr(),
        html.Div([
            html.Strong("Disclaimer:", style={"color": "#c0392b"}),
            html.Span(
                " This tool is for educational and informational purposes only. It does not constitute investment advice, recommendation, or solicitation to buy or sell any securities. The creators and contributors of this tool are not SEBI-registered investment advisors. Users are solely responsible for their own investment decisions and should consult a SEBI-registered financial advisor before making any investment decisions. The creators and contributors assume no liability for any losses or damages arising from the use of this tool. Past performance is not indicative of future results. All investments are subject to market risks, including the possible loss of principal. By using this tool, you acknowledge and accept these risks.",
                style={"color": "#555", "fontSize": 14}
            )
        ], style={"maxWidth": 900, "margin": "20px auto 0 auto", "padding": 12, "background": "#fff3cd", "border": "1px solid #ffeeba", "borderRadius": 6, "boxShadow": "0 1px 3px #eee"})
    ])

    app.layout = html.Div([
        # Floating Disclaimer Modal Overlay
        html.Div(
            id="disclaimer-modal",
            children=[
                html.Div([
                    html.H2("Disclaimer", style={"color": "#1565c0", "marginBottom": 12, "textAlign": "center"}),
                    html.P(
                        "This tool is for educational and informational purposes only. It does not constitute investment advice, recommendation, or solicitation to buy or sell any securities. The creator and contributors of this tool are not SEBI-registered investment advisors. Users are solely responsible for their own investment decisions and should consult a SEBI-registered financial advisor before making any investment decisions. The creators and contributors assume no liability for any losses or damages arising from the use of this tool. Past performance is not indicative of future results. All investments are subject to market risks, including the possible loss of principal. By using this tool, you acknowledge and accept these risks.",
                        style={"color": "#333", "fontSize": 16, "marginBottom": 24, "textAlign": "justify"}
                    ),
                    html.Button("I Agree", id="agree-btn", n_clicks=0, style={
                        "background": "#1565c0",
                        "color": "white",
                        "border": "none",
                        "padding": "12px 32px",
                        "borderRadius": "6px",
                        "fontSize": "1.1rem",
                        "fontWeight": "bold",
                        "boxShadow": "0 2px 8px rgba(21,101,192,0.13)",
                        "cursor": "pointer",
                        "margin": "0 auto",
                        "display": "block"
                    })
                ], style={
                    "background": "#fff",
                    "padding": "32px 24px 24px 24px",
                    "borderRadius": "12px",
                    "boxShadow": "0 4px 32px rgba(44,62,80,0.18)",
                    "maxWidth": "480px",
                    "width": "90vw",
                    "margin": "10vh auto 0 auto",
                    "textAlign": "center"
                })
            ],
            style={
                "position": "fixed",
                "top": 0,
                "left": 0,
                "width": "100vw",
                "height": "100vh",
                "background": "rgba(44,62,80,0.55)",
                "zIndex": 9999,
                "display": "flex",
                "alignItems": "flex-start",
                "justifyContent": "center"
            }
        ),
        # No persistent store, always prompt on page load
        html.Div([
            html.Img(
                src="/assets/logocode.jpg",
                className="stocklens-logo",
                alt="Logo"
            )
        ]),
        html.H1(
            "StockLens Hub",
            style={
                "textAlign": "center",
                "color": "#2c3e50",
                "marginTop": 20,
                "fontSize": "clamp(1.5rem, 5vw, 2.5rem)",
                "marginBottom": 0,
            }
        ),
        html.Div([
            dcc.Input(
                id="ticker-input",
                type="text",
                placeholder="Enter stock symbol(s) (e.g., AAPL, TCS.NS)",
                value=ticker,
                className="modern-input"
            ),
            html.Div([
                dcc.DatePickerSingle(id="start-date", placeholder="Start Date", style={"marginRight": 10, "marginBottom": 10}),
                dcc.DatePickerSingle(id="end-date", placeholder="End Date", style={"marginRight": 10, "marginBottom": 10}),
                html.Span(
                    "(Default: last 5 years. Set dates for custom analysis)",
                    style={"marginLeft": 12, "color": "#888", "fontSize": "1rem", "verticalAlign": "middle", "whiteSpace": "nowrap"}
                )
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": 4, "flexWrap": "wrap", "marginBottom": 10}),
            html.Button("Submit", id="submit-btn", n_clicks=0, style={"background": "#2980b9", "color": "white", "border": "none", "padding": "8px 18px", "borderRadius": 5, "fontSize": "1rem", "width": "90%", "maxWidth": 200})
        ], style={"textAlign": "center", "marginBottom": 30, "display": "flex", "flexDirection": "column", "alignItems": "center", "gap": 8}),
        # Removed export controls and download prompt
        dcc.Tabs(id="main-tabs", value="tab-analysis", children=[
            dcc.Tab(label="Analysis", value="tab-analysis"),
            dcc.Tab(label="News", value="tab-news"),
            dcc.Tab(label="Fundamentals", value="tab-fundamentals"),
            dcc.Tab(label="Financials", value="tab-financials"),
            dcc.Tab(label="Corporate Actions", value="tab-corpactions"),
            dcc.Tab(label="Market Sentiment", value="tab-sentiment"),
        ],
        style={"fontSize": "clamp(1rem, 3vw, 1.2rem)", "overflowX": "auto"}),
        dcc.Loading(
            id="dashboard-loading",
            type="circle",
            color="#1565c0",
            fullscreen=False,
            children=html.Div(id="dashboard-content")
        ),
        # dcc.Download(id="download-data") removed
        education_section,
        disclaimer_section,
        # Feedback button directly below disclaimer
        html.Div([
            html.A(
                html.Button("Submit Feedback", className="feedback-btn", style={
                    "margin": "18px auto 0 auto",
                    "display": "block"
                }),
                href="https://forms.gle/kdLnidUgrumkz9uS6",
                target="_blank",
                style={"textDecoration": "none", "display": "block", "width": "fit-content", "margin": "0 auto"}
            )
        ]),
        html.Hr(),
        html.Footer([
            html.Div([
                html.Span("Made with ", style={"color": "#888"}),
                html.Span("❤", style={"color": "#e74c3c", "fontSize": 18, "fontWeight": "bold"}),
                html.Span(" by Nikunj Maru", style={"color": "#888"})
            ], style={"textAlign": "center", "marginTop": 30, "marginBottom": 10, "fontSize": 16})
        ])
    ], style={"fontFamily": "Segoe UI, Arial, sans-serif", "backgroundColor": "#f8f9fa", "padding": 0, "minHeight": "100vh", "width": "100vw", "boxSizing": "border-box"})

    # Hide/show disclaimer modal for every visit (no persistence)
    @app.callback(
        Output("disclaimer-modal", "style"),
        [Input("agree-btn", "n_clicks")],
        [State("disclaimer-modal", "style")]
    )
    def toggle_modal(n_clicks, style):
        # If user clicks agree, set display none
        if n_clicks and n_clicks > 0:
            s = dict(style) if style else {}
            s["display"] = "none"
            return s
        # Otherwise, show modal
        s = dict(style) if style else {}
        s["display"] = "flex"
        return s

    @app.callback(
        Output("dashboard-content", "children"),
        [Input("submit-btn", "n_clicks"),
         Input("main-tabs", "value")],
        [State("ticker-input", "value"),
         State("start-date", "date"),
         State("end-date", "date")]
    )
    def update_dashboard(n_clicks, tab, ticker_val, start_date_val, end_date_val):
        # Support comma-separated tickers for peer comparison
        tickers = [t.strip() for t in ticker_val.split(",") if t.strip()] if ticker_val else []
        def normalize_ticker(t):
            t = t.upper()
            if "." in t:
                return t
            us_tickers = {"AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "BRK.B", "JPM", "V", "UNH", "HD", "PG", "MA", "DIS", "BAC", "XOM", "PFE", "WMT", "INTC", "CSCO", "T", "VZ", "KO", "PEP", "MRK", "CVX", "ABBV", "MCD", "COST", "ADBE", "CRM", "ABT", "CMCSA", "TMO", "ACN", "AVGO", "QCOM", "TXN", "LIN", "NEE", "DHR", "NKE", "MDT", "HON", "UNP", "AMGN", "LOW", "MS", "GS", "BLK", "AXP", "BA", "CAT", "GE", "IBM", "ORCL", "SBUX", "LMT", "MMM", "SPGI", "BKNG", "ISRG", "NOW", "PLD", "ZTS", "GILD", "SYK", "MDLZ", "MO", "TGT", "DE", "DUK", "SO", "USB", "C", "FDX", "GM", "F", "DAL", "UAL", "AAL", "UAL", "UAL"}
            if t in us_tickers:
                return t
            return t + ".NS"
        norm_tickers = [normalize_ticker(t) for t in tickers]
        main_ticker = norm_tickers[0] if norm_tickers else "AAPL"
        # Corporate Actions Tab (yfinance)
        if tab == "tab-corpactions":
            try:
                stock = yf.Ticker(main_ticker)
                dividends = stock.dividends
                splits = stock.splits
                # Format Dividends: last 6 entries, only date (no time), no timestamp
                def format_dividends_table(dividends):
                    if dividends is None or dividends.empty:
                        return html.Div("No Dividends data available.", style={"color": "#888", "fontSize": 15, "marginBottom": 18})
                    df = dividends.copy()
                    df = df.tail(10)
                    df = df.reset_index()
                    df = df.sort_values('Date', ascending=False).reset_index(drop=True)
                    df['Date'] = df['Date'].apply(lambda d: d.strftime('%d-%m-%Y'))
                    table_header = [html.Th("Date", style={"border": "1px solid #ccc", "padding": "8px", "background": "#e9ecef"}), html.Th("Dividend", style={"border": "1px solid #ccc", "padding": "8px", "background": "#e9ecef"})]
                    table_rows = [html.Tr([
                        html.Td(row['Date'], style={"border": "1px solid #ccc", "padding": "8px"}),
                        html.Td(f"₹{row[dividends.name]:.2f}", style={"border": "1px solid #ccc", "padding": "8px"})
                    ]) for _, row in df.iterrows()]
                    return html.Div([
                        html.H4("Dividends (Last 10 )", style={"color": "#1565c0", "marginTop": 18, "marginBottom": 8}),
                        html.Table([
                            html.Thead(html.Tr(table_header)),
                            html.Tbody(table_rows)
                        ], style={
                            "margin": "0 auto",
                            "fontSize": 15,
                            "background": "#f8f9fa",
                            "borderRadius": 6,
                            "boxShadow": "0 1px 3px #eee",
                            "padding": 10,
                            "width": "100%",
                            "maxWidth": 700,
                            "borderCollapse": "collapse",
                            "border": "1px solid #ccc"
                        })
                    ])
                # Format Splits: all entries, show as ratio, only date
                def format_splits_table(splits):
                    if splits is None or splits.empty:
                        return html.Div("No Stock Splits data available.", style={"color": "#888", "fontSize": 15, "marginBottom": 18})
                    df = splits.copy()
                    df = df[df != 1]  # Only show actual splits
                    if df.empty:
                        return html.Div("No Stock Splits data available.", style={"color": "#888", "fontSize": 15, "marginBottom": 18})
                    df = df.reset_index()
                    df = df.sort_values('Date', ascending=False).reset_index(drop=True)
                    df['Date'] = df['Date'].apply(lambda d: d.strftime('%d-%m-%Y'))
                    table_header = [html.Th("Date", style={"border": "1px solid #ccc", "padding": "8px", "background": "#e9ecef"}), html.Th("Split Ratio", style={"border": "1px solid #ccc", "padding": "8px", "background": "#e9ecef"})]
                    table_rows = []
                    for _, row in df.iterrows():
                        ratio = row[splits.name]
                        if ratio > 1:
                            split_str = f"{int(ratio)}:1"
                        else:
                            split_str = f"1:{int(1/ratio)}"
                        table_rows.append(html.Tr([
                            html.Td(row['Date'], style={"border": "1px solid #ccc", "padding": "8px"}),
                            html.Td(split_str, style={"border": "1px solid #ccc", "padding": "8px"})
                        ]))
                    return html.Div([
                        html.H4("Stock Splits ", style={"color": "#1565c0", "marginTop": 18, "marginBottom": 8}),
                        html.Table([
                            html.Thead(html.Tr(table_header)),
                            html.Tbody(table_rows)
                        ], style={
                            "margin": "0 auto",
                            "fontSize": 15,
                            "background": "#f8f9fa",
                            "borderRadius": 6,
                            "boxShadow": "0 1px 3px #eee",
                            "padding": 10,
                            "width": "100%",
                            "maxWidth": 700,
                            "borderCollapse": "collapse",
                            "border": "1px solid #ccc"
                        })
                    ])
                return html.Div([
                    html.H3(f"Corporate Actions for {main_ticker}", style={"textAlign": "center", "color": "#2c3e50", "marginTop": 20}),
                    format_dividends_table(dividends),
                    format_splits_table(splits)
                ], style={"maxWidth": 900, "margin": "0 auto", "padding": "2vw"})
            except Exception as e:
                return html.Div(f"Error fetching corporate actions: {e}", style={"color": "red", "textAlign": "center"})
        if n_clicks == 0 or not ticker_val:
            return html.Div("Enter a ticker and click Submit to view analysis.", style={"textAlign": "center", "color": "#888", "marginTop": 40})
        # Support comma-separated tickers for peer comparison
        tickers = [t.strip() for t in ticker_val.split(",") if t.strip()]
        # Helper: add .NS if not present and not a known US ticker
        def normalize_ticker(t):
            t = t.upper()
            # If already has a suffix, return as is
            if "." in t:
                return t
            # List of common US tickers (expand as needed)
            us_tickers = {"AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "BRK.B", "JPM", "V", "UNH", "HD", "PG", "MA", "DIS", "BAC", "XOM", "PFE", "WMT", "INTC", "CSCO", "T", "VZ", "KO", "PEP", "MRK", "CVX", "ABBV", "MCD", "COST", "ADBE", "CRM", "ABT", "CMCSA", "TMO", "ACN", "AVGO", "QCOM", "TXN", "LIN", "NEE", "DHR", "NKE", "MDT", "HON", "UNP", "AMGN", "LOW", "MS", "GS", "BLK", "AXP", "BA", "CAT", "GE", "IBM", "ORCL", "SBUX", "LMT", "MMM", "SPGI", "BKNG", "ISRG", "NOW", "PLD", "ZTS", "GILD", "SYK", "MDLZ", "MO", "TGT", "DE", "DUK", "SO", "USB", "C", "FDX", "GM", "F", "DAL", "UAL", "AAL", "UAL", "UAL"}
            if t in us_tickers:
                return t
            # Otherwise, assume Indian stock and add .NS
            return t + ".NS"
        norm_tickers = [normalize_ticker(t) for t in tickers]
        main_ticker = norm_tickers[0] if norm_tickers else "AAPL"
        data = fetch_stock_data(main_ticker, start_date_val, end_date_val)
        if data.empty:
            # Try fallback: if user entered without .NS and not in US list, try as-is (for BSE or other)
            fallback_ticker = tickers[0] if tickers else "AAPL"
            if fallback_ticker != main_ticker:
                data = fetch_stock_data(fallback_ticker, start_date_val, end_date_val)
        if data.empty:
            return html.Div("No data found for the given ticker or date range.", style={"color": "red", "textAlign": "center", "marginTop": 40})
        data = calculate_indicators(data)
        provide_insights(data)
        predictions, error_metrics = predict_stock_prices(data)
        # Buy/Sell signals
        data['Buy_Signal'] = (data['Close'] > data['EMA_20']) & (data['Close'].shift(1) <= data['EMA_20'])
        data['Sell_Signal'] = (data['Close'] < data['EMA_20']) & (data['Close'].shift(1) >= data['EMA_20'])
        # Main chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name="EMA 20", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], name="Upper Bollinger Band", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], name="Lower Bollinger Band", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(
            x=data[data['Buy_Signal']].index,
            y=data[data['Buy_Signal']]['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=data[data['Sell_Signal']].index,
            y=data[data['Sell_Signal']]['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ))
        fig.update_layout(
            title=f"{main_ticker} Stock Analysis with Indicators and Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Indicators",
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10),
            height=400
        )
        # Responsive graph style
        graph_style = {"marginBottom": 40, "width": "100%", "maxWidth": 900, "margin": "0 auto"}
        # Prediction chart
        if predictions is not None and not predictions.empty and 'Date' in predictions.columns and 'Predicted Price' in predictions.columns:
            prediction_component = dcc.Graph(figure=go.Figure(data=[
                go.Scatter(x=predictions['Date'], y=predictions['Predicted Price'], name="Predicted Price")
            ]), style=graph_style)
        else:
            prediction_component = html.Div("No predictions available to display.")
        # Error metrics display
        if error_metrics and error_metrics['RMSE'] is not None:
            metrics_component = html.Div([
                html.H4("Prediction Error Metrics", style={"textAlign": "center", "color": "#8e44ad"}),
                html.Div([
                    html.Span(f"RMSE: {error_metrics['RMSE']:.4f}", style={"marginRight": 20}),
                    html.Span(f"MAE: {error_metrics['MAE']:.4f}")
                ], style={"textAlign": "center", "color": "#555", "fontSize": 16, "marginBottom": 20})
            ])
        else:
            metrics_component = html.Div()
        # Peer Comparison Tab
        if tab == "tab-analysis" and len(tickers) > 1:
            # Fetch and plot all tickers (normalize each)
            traces = []
            for orig, norm in zip(tickers, norm_tickers):
                d = fetch_stock_data(norm, start_date_val, end_date_val)
                if d.empty and orig != norm:
                    d = fetch_stock_data(orig, start_date_val, end_date_val)
                if not d.empty:
                    traces.append(go.Scatter(x=d.index, y=d['Close'], name=orig))
            if traces:
                fig = go.Figure(traces)
                fig.update_layout(title="Peer Comparison: Closing Prices", xaxis_title="Date", yaxis_title="Price", autosize=True, margin=dict(l=10, r=10, t=40, b=10), height=400)
                return html.Div([
                    dcc.Graph(figure=fig, style=graph_style),
                    html.Div("Peer comparison of selected tickers.", style={"textAlign": "center", "color": "#888", "marginTop": 20})
                ])
            else:
                return html.Div("No data found for the selected tickers.", style={"color": "red", "textAlign": "center", "marginTop": 40})
        # Financial Statements Tab (yfinance)
        if tab == "tab-financials":
            try:
                stock = yf.Ticker(main_ticker)
                income = stock.financials
                balance = stock.balance_sheet
                cashflow = stock.cashflow
                def format_sensible_unit(val):
                    abs_val = abs(val)
                    if abs_val >= 1e12:
                        return f"{val/1e12:.2f}T"
                    elif abs_val >= 1e9:
                        return f"{val/1e9:.2f}B"
                    elif abs_val >= 1e7:
                        return f"{val/1e7:.2f}Cr"
                    elif abs_val >= 1e5:
                        return f"{val/1e5:.2f}L"
                    elif abs_val >= 1e6:
                        return f"{val/1e6:.2f}M"
                    else:
                        return f"{val:,.2f}"
                def make_table(df, title):
                    if df is None or df.empty:
                        return html.Div(f"No {title} data available.", style={"color": "#888", "marginBottom": 20})
                    df = df.fillna(0)
                    df = df.astype(float)
                    columns = [str(c) for c in df.columns]
                    # Transpose for better mobile view: Dates as rows, Metrics as columns
                    df_t = df.T
                    df_t.index = [str(i) for i in df_t.index]
                    table_header = [html.Th("Date", style={"textAlign": "center", "padding": "8px", "background": "#e9ecef", "fontWeight": "bold"})]
                    for metric in df_t.columns:
                        table_header.append(html.Th(metric, style={"textAlign": "center", "padding": "8px", "background": "#e9ecef"}))
                    table_rows = []
                    for date in df_t.index:
                        row = [html.Td(date, style={"fontWeight": "bold", "textAlign": "center", "padding": "8px", "background": "#f8f9fa"})]
                        for metric in df_t.columns:
                            val = df_t.at[date, metric]
                            cell = format_sensible_unit(val)
                            row.append(html.Td(cell, style={"textAlign": "right", "padding": "8px"}))
                        table_rows.append(html.Tr(row))
                    return html.Div([
                        html.H4(title, style={"marginTop": 20, "color": "#2c3e50", "textAlign": "center"}),
                        html.Div([
                            html.Table([
                                html.Thead(html.Tr(table_header)),
                                html.Tbody(table_rows)
                            ], style={
                                "width": "100%",
                                "borderCollapse": "collapse",
                                "fontSize": 14,
                                "background": "#fff",
                                "borderRadius": 6,
                                "boxShadow": "0 1px 3px #eee",
                                "overflowX": "auto"
                            })
                        ], style={"overflowX": "auto", "width": "100%"})
                    ], style={"marginBottom": 30, "overflowX": "auto", "background": "#f8f9fa", "borderRadius": 6, "padding": 10})
                # --- Units legend at the top ---
                units_legend = html.Div([
                    html.Strong("Units Legend: ", style={"color": "#2c3e50", "fontSize": 15}),
                    html.Span("Cr - Crore (10 Million), ", style={"color": "#555", "fontSize": 14}),
                    html.Span("L - Lakh (100 Thousand), ", style={"color": "#555", "fontSize": 14}),
                    html.Span("M - Million, ", style={"color": "#555", "fontSize": 14}),
                    html.Span("B - Billion, ", style={"color": "#555", "fontSize": 14}),
                    html.Span("T - Trillion", style={"color": "#555", "fontSize": 14})
                ], style={"marginBottom": 18, "marginTop": 8, "textAlign": "center", "background": "#e9ecef", "padding": "8px 10px", "borderRadius": 6, "maxWidth": 700, "marginLeft": "auto", "marginRight": "auto", "fontSize": 14})
                return html.Div([
                    units_legend,
                    make_table(income, "Income Statement"),
                    make_table(balance, "Balance Sheet"),
                    make_table(cashflow, "Cash Flow Statement"),
                    html.Div("Data from yfinance. All values in reporting currency. For more details, visit Yahoo Finance.", style={"color": "#888", "fontSize": 13, "textAlign": "center", "marginTop": 20})
                ], style={"maxWidth": 900, "margin": "0 auto", "padding": "2vw"})
            except Exception as e:
                return html.Div(f"Error fetching financial statements: {e}", style={"color": "red", "textAlign": "center"})
        # News tab: Show only news analytics, not the main chart
        if tab == "tab-news":
            try:
                main_ticker = norm_tickers[0] if norm_tickers else "AAPL"
                stock_name = tickers[0] if tickers else main_ticker
                headlines = []
                # Moneycontrol
                try:
                    mc_url = f"https://www.moneycontrol.com/news/tags/{stock_name.lower()}.html"
                    mc_resp = requests.get(mc_url, headers={"User-Agent": "Mozilla/5.0"})
                    mc_soup = BeautifulSoup(mc_resp.text, 'html.parser')
                    mc_count = 0
                    for item in mc_soup.select('li.clearfix'):
                        if mc_count >= 4:
                            break
                        title_tag = item.find('h2')
                        if not title_tag:
                            continue
                        title = title_tag.get_text(strip=True)
                        link_tag = title_tag.find('a')
                        link = link_tag['href'] if link_tag and link_tag.has_attr('href') else ''
                        if link and not link.startswith('http'):
                            link = 'https://www.moneycontrol.com' + link
                        snippet_tag = item.find('p')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                        date_tag = item.find('span', class_='dateline')
                        pub_date = date_tag.get_text(strip=True) if date_tag else None
                        headlines.append({'title': title, 'link': link, 'date': pub_date, 'snippet': snippet, 'source': 'Moneycontrol'})
                        mc_count += 1
                except Exception:
                    pass
                # Mint
                try:
                    mint_url = f"https://www.livemint.com/search-news?query={stock_name}"
                    mint_resp = requests.get(mint_url, headers={"User-Agent": "Mozilla/5.0"})
                    mint_soup = BeautifulSoup(mint_resp.text, 'html.parser')
                    mint_count = 0
                    for item in mint_soup.select('div.listingNewBox'):
                        if mint_count >= 4:
                            break
                        title_tag = item.find('a', class_='headline')
                        if not title_tag:
                            # fallback: try h2
                            title_tag = item.find('h2')
                        if not title_tag:
                            continue
                        title = title_tag.get_text(strip=True)
                        link = title_tag['href'] if title_tag.has_attr('href') else ''
                        if link and not link.startswith('http'):
                            link = 'https://www.livemint.com' + link
                        snippet_tag = item.find('p')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                        date_tag = item.find('span', class_='date')
                        if not date_tag:
                            date_tag = item.find('span', class_='lm-date')
                        pub_date = date_tag.get_text(strip=True) if date_tag else None
                        headlines.append({'title': title, 'link': link, 'date': pub_date, 'snippet': snippet, 'source': 'Mint'})
                        mint_count += 1
                except Exception:
                    pass
                # Economic Times
                try:
                    et_url = f"https://economictimes.indiatimes.com/topic/{stock_name}/news"
                    et_resp = requests.get(et_url, headers={"User-Agent": "Mozilla/5.0"})
                    et_soup = BeautifulSoup(et_resp.text, 'html.parser')
                    et_count = 0
                    for item in et_soup.select('div.eachStory'):
                        if et_count >= 4:
                            break
                        title_tag = item.find('a', class_='title')
                        if not title_tag:
                            continue
                        title = title_tag.get_text(strip=True)
                        link = 'https://economictimes.indiatimes.com' + title_tag['href']
                        snippet_tag = item.find('p')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                        date_tag = item.find('time')
                        if date_tag and date_tag.has_attr('data-time'):
                            pub_date = date_tag['data-time']
                        elif date_tag:
                            pub_date = date_tag.get_text(strip=True)
                        else:
                            pub_date = None
                        headlines.append({'title': title, 'link': link, 'date': pub_date, 'snippet': snippet, 'source': 'Economic Times'})
                        et_count += 1
                except Exception:
                    pass
                # Times of India
                try:
                    toi_url = f"https://timesofindia.indiatimes.com/topic/{stock_name}/news"
                    toi_resp = requests.get(toi_url, headers={"User-Agent": "Mozilla/5.0"})
                    toi_soup = BeautifulSoup(toi_resp.text, 'html.parser')
                    toi_count = 0
                    for item in toi_soup.select('div.story'):
                        if toi_count >= 4:
                            break
                        title_tag = item.find('a')
                        if not title_tag:
                            continue
                        title = title_tag.get_text(strip=True)
                        link = 'https://timesofindia.indiatimes.com' + title_tag['href']
                        snippet_tag = item.find('p')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                        date_tag = item.find('span', class_='time')
                        pub_date = date_tag.get_text(strip=True) if date_tag else None
                        headlines.append({'title': title, 'link': link, 'date': pub_date, 'snippet': snippet, 'source': 'Times of India'})
                        toi_count += 1
                except Exception:
                    pass
                # Upstox
                try:
                    upstox_url = f"https://upstox.com/news/search/?q={stock_name}"
                    upstox_resp = requests.get(upstox_url, headers={"User-Agent": "Mozilla/5.0"})
                    upstox_soup = BeautifulSoup(upstox_resp.text, 'html.parser')
                    upstox_count = 0
                    for item in upstox_soup.select('div.news-card, div.news-card-item'):
                        if upstox_count >= 4:
                            break
                        title_tag = item.find('a', class_='news-title')
                        if not title_tag:
                            continue
                        title = title_tag.get_text(strip=True)
                        link = title_tag['href'] if title_tag.has_attr('href') else ''
                        if link and not link.startswith('http'):
                            link = 'https://upstox.com' + link
                        snippet_tag = item.find('div', class_='news-description')
                        if not snippet_tag:
                            snippet_tag = item.find('p')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                        date_tag = item.find('span', class_='news-date')
                        pub_date = date_tag.get_text(strip=True) if date_tag else None
                        headlines.append({'title': title, 'link': link, 'date': pub_date, 'snippet': snippet, 'source': 'Upstox'})
                        upstox_count += 1
                except Exception:
                    pass
                # CNBC TV18
                try:
                    cnbc_url = f"https://www.cnbctv18.com/search/?query={stock_name}"
                    cnbc_resp = requests.get(cnbc_url, headers={"User-Agent": "Mozilla/5.0"})
                    cnbc_soup = BeautifulSoup(cnbc_resp.text, 'html.parser')
                    cnbc_count = 0
                    for item in cnbc_soup.select('div.search-story, div.search-result-story'):
                        if cnbc_count >= 4:
                            break
                        title_tag = item.find('a', class_='search-title')
                        if not title_tag:
                            continue
                        title = title_tag.get_text(strip=True)
                        link = title_tag['href'] if title_tag.has_attr('href') else ''
                        if link and not link.startswith('http'):
                            link = 'https://www.cnbctv18.com' + link
                        snippet_tag = item.find('div', class_='search-desc')
                        if not snippet_tag:
                            snippet_tag = item.find('p')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                        date_tag = item.find('span', class_='search-date')
                        pub_date = date_tag.get_text(strip=True) if date_tag else None
                        headlines.append({'title': title, 'link': link, 'date': pub_date, 'snippet': snippet, 'source': 'CNBC TV18'})
                        cnbc_count += 1
                except Exception:
                    pass
                # Business Today
                try:
                    bt_url = f"https://www.businesstoday.in/search.jsp?searchword={stock_name}"
                    bt_resp = requests.get(bt_url, headers={"User-Agent": "Mozilla/5.0"})
                    bt_soup = BeautifulSoup(bt_resp.text, 'html.parser')
                    bt_count = 0
                    for item in bt_soup.select('div.widget-listing, div.search-result-card'):
                        if bt_count >= 4:
                            break
                        title_tag = item.find('a', class_='widget-listing-title')
                        if not title_tag:
                            title_tag = item.find('a', class_='search-result-title')
                        if not title_tag:
                            continue
                        title = title_tag.get_text(strip=True)
                        link = title_tag['href'] if title_tag.has_attr('href') else ''
                        if link and not link.startswith('http'):
                            link = 'https://www.businesstoday.in' + link
                        snippet_tag = item.find('div', class_='widget-listing-description')
                        if not snippet_tag:
                            snippet_tag = item.find('p')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                        date_tag = item.find('span', class_='widget-listing-date')
                        if not date_tag:
                            date_tag = item.find('span', class_='search-result-date')
                        pub_date = date_tag.get_text(strip=True) if date_tag else None
                        headlines.append({'title': title, 'link': link, 'date': pub_date, 'snippet': snippet, 'source': 'Business Today'})
                        bt_count += 1
                except Exception:
                    pass
                # --- Enhanced News Analysis ---
                volatility_keywords = ["crash", "surge", "volatile", "downgrade", "upgrade"]
                sentiment_tags = ["bullish", "bearish", "selloff", "breakout", "buy rating", "sell rating", "downgrade", "upgrade"]
                stock_name_lower = stock_name.lower()
                total_mentions = 0
                for news in headlines:
                    text = (news['title'] + " " + news['snippet']).lower().strip()
                    blob = TextBlob(text)
                    news['sentiment'] = blob.sentiment.polarity
                    # Relevance Score: 1 = direct mention, 0.5 = partial, 0 = not mentioned
                    stock_name_lc = stock_name_lower.strip().lower()
                    if stock_name_lc and stock_name_lc in text:
                        news['relevance'] = 1.0
                        total_mentions += 1
                    elif stock_name_lc and any(word for word in stock_name_lc.split() if word in text):
                        news['relevance'] = 0.5
                        total_mentions += 1
                    else:
                        news['relevance'] = None
                    # Volatility keyword detection (case-insensitive, robust)
                    news['volatility_keywords'] = [kw for kw in volatility_keywords if kw.lower() in text]
                    # Market sentiment tags (case-insensitive, robust)
                    news['sentiment_tags'] = [tag for tag in sentiment_tags if tag.lower() in text]
                # Aggregate sentiment
                if headlines:
                    avg_score = sum(n['sentiment'] for n in headlines) / len(headlines)
                else:
                    avg_score = 0
                verdict = "Positive" if avg_score > 0.1 else "Negative" if avg_score < -0.1 else "Neutral"
                color = "green" if avg_score > 0.1 else "red" if avg_score < -0.1 else "gray"
                # Frequency of mentions (spike = high impact)
                mention_freq = total_mentions
                # Display
                sentiment_explanation = html.Details([
                    html.Summary("How to Interpret Sentiment & Relevance Scores", style={"fontWeight": "bold", "fontSize": 17, "color": "#1565c0", "cursor": "pointer", "marginTop": 30, "textAlign": "center"}),
                    html.Div([
                        html.P("The sentiment score ranges from -1 (very negative) to +1 (very positive). It is calculated using natural language processing on news headlines and snippets.", style={"fontSize": 15, "color": "#444", "marginBottom": 8}),
                        html.P("Relevance Score: 1 = news is about your stock, 0.5 = partial/possible mention, 0 = not about your stock.", style={"fontSize": 15, "color": "#444", "marginBottom": 8}),
                        html.P("Volatility Keywords: If present, news may indicate high volatility or major events.", style={"fontSize": 15, "color": "#444", "marginBottom": 8}),
                        html.P("Market Sentiment Tags: Tags like 'bullish', 'bearish', 'breakout', etc. detected in news.", style={"fontSize": 15, "color": "#444", "marginBottom": 8}),
                        html.P("Frequency of Mentions: More articles mentioning your stock in a short time = higher impact.", style={"fontSize": 15, "color": "#444", "marginBottom": 8}),
                        html.Ul([
                            html.Li("Scores above +0.1: Positive sentiment (good news dominates)", style={"color": "green", "fontSize": 15}),
                            html.Li("Scores below -0.1: Negative sentiment (bad news dominates)", style={"color": "#c0392b", "fontSize": 15}),
                            html.Li("Scores between -0.1 and +0.1: Neutral sentiment (mixed or balanced news)", style={"color": "#888", "fontSize": 15}),
                        ], style={"marginLeft": 20, "marginBottom": 8}),
                        html.P("An 'ideal' or 'best' sentiment score depends on your investment strategy:", style={"fontSize": 15, "color": "#444", "marginBottom": 4}),
                        html.Ul([
                            html.Li("For bullish investors: Higher positive scores (closer to +1) are preferred, indicating optimism in the news.", style={"color": "green", "fontSize": 15}),
                            html.Li("For bearish or contrarian strategies: Negative scores (closer to -1) may signal opportunities if you expect a reversal.", style={"color": "#c0392b", "fontSize": 15}),
                            html.Li("Neutral scores suggest a lack of strong news-driven momentum.", style={"color": "#888", "fontSize": 15}),
                        ], style={"marginLeft": 20}),
                        html.P("Always use sentiment and relevance as inputs among many. Sudden shifts in sentiment or a spike in mentions can precede price moves, but are not guarantees.", style={"fontSize": 14, "color": "#888", "marginTop": 8})
                    ], style={"maxWidth": 700, "margin": "0 auto"})
                ], open=False)
                return html.Div([
                    html.H3(f"News for {stock_name} (Moneycontrol, Mint, ET, TOI, Upstox, CNBC TV18, Business Today)", style={"textAlign": "center", "marginTop": 20}),
                    html.Div([
                        html.Span("Overall Sentiment Score: ", style={"fontWeight": "bold", "fontSize": 18}),
                        html.Span(f"{avg_score:.2f} ", style={"color": color, "fontWeight": "bold", "fontSize": 18}),
                        html.Span(f"({verdict})", style={"color": color, "fontWeight": "bold", "fontSize": 18}),
                        html.Span(f" | Mentions: {mention_freq}", style={"color": "#888", "fontSize": 16, "marginLeft": 10})
                    ], style={'textAlign': 'center', 'marginBottom': 10, 'marginTop': 10}),
                    sentiment_explanation,
                    html.Ul([
                        html.Li([
                            html.A(f"[{news['source']}] {news['title']}", href=news['link'], target="_blank"),
                            html.Br(),
                            html.Span(news['snippet'], style={'color': '#555', 'fontSize': 14}),
                            html.Br(),
                            html.Span(f"Sentiment: {news['sentiment']:.2f}", style={'color': 'green' if news['sentiment'] > 0 else 'red' if news['sentiment'] < 0 else 'gray', 'fontWeight': 'bold'}),
                            (html.Span(f" | Relevance: {news['relevance']}", style={'color': '#2980b9', 'fontSize': 13, 'marginLeft': 8}) if news.get('relevance') is not None else None),
                            (html.Span(f" | Volatility: {', '.join(news['volatility_keywords'])}", style={'color': '#e67e22', 'fontSize': 13, 'marginLeft': 8}) if news.get('volatility_keywords') else None),
                            (html.Span(f" | Tags: {', '.join(news['sentiment_tags'])}", style={'color': '#8e44ad', 'fontSize': 13, 'marginLeft': 8}) if news.get('sentiment_tags') else None),
                            html.Span(f" | Posted: {news['date'] if news['date'] else 'Unknown'}", style={'color': '#888', 'fontSize': 12})
                        ], style={"marginBottom": 18, "background": "#f8f9fa", "padding": 10, "borderRadius": 6, "boxShadow": "0 1px 3px #eee"})
                        for news in headlines if news.get('title') and news.get('source')
                    ], style={"maxWidth": 800, "margin": "0 auto", "marginTop": 20})
                ])
            except Exception as e:
                return html.Div(f"Error fetching news: {e}", style={"color": "red", "textAlign": "center"})
        # Fundamentals tab
        if tab == "tab-fundamentals":
            try:
                stock = yf.Ticker(main_ticker)
                info = stock.info
                keys = [
                    ("Symbol", "symbol"),
                    ("Name", "shortName"),
                    ("Sector", "sector"),
                    ("Industry", "industry"),
                    ("Market Cap", "marketCap"),
                    ("P/E Ratio", "trailingPE"),
                    ("EPS", "trailingEps"),
                    ("Dividend Yield", "dividendYield"),
                    ("52 Week High", "fiftyTwoWeekHigh"),
                    ("52 Week Low", "fiftyTwoWeekLow"),
                    ("Beta", "beta"),
                ]
                rows = [html.Tr([html.Td(label), html.Td(info.get(key, "-"))]) for label, key in keys]
                return html.Div([
                    html.H3(f"Fundamentals for {main_ticker}", style={"textAlign": "center", "color": "#2c3e50"}),
                    html.Table([
                        html.Tbody(rows)
                    ], style={"margin": "0 auto", "fontSize": 16, "background": "#f8f9fa", "borderRadius": 6, "boxShadow": "0 1px 3px #eee", "padding": 10})
                ])
            except Exception as e:
                return html.Div(f"Error fetching fundamentals: {e}", style={"color": "red", "textAlign": "center"})
        # Market Sentiment tab: summary and chart
        if tab == "tab-sentiment":
            try:
                data = fetch_stock_data(main_ticker, start_date_val, end_date_val)
                if data.empty:
                    return html.Div("No data found for the given ticker or date range.", style={"color": "red", "textAlign": "center", "marginTop": 40})
                data = calculate_indicators(data)
                # --- Sentiment logic ---
                verdicts = []
                details = []
                # RSI
                rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else None
                if rsi is not None:
                    if rsi > 70:
                        verdicts.append("Bearish (Overbought RSI)")
                        details.append(f"RSI: {rsi:.2f} (Overbought)")
                    elif rsi < 30:
                        verdicts.append("Bullish (Oversold RSI)")
                        details.append(f"RSI: {rsi:.2f} (Oversold)")
                    else:
                        details.append(f"RSI: {rsi:.2f} (Neutral)")
                # MACD
                macd = data['MACD'].iloc[-1] if 'MACD' in data.columns else None
                macd_signal = data['MACD_Signal'].iloc[-1] if 'MACD_Signal' in data.columns else None
                if macd is not None and macd_signal is not None:
                    if macd > macd_signal:
                        verdicts.append("Bullish (MACD > Signal)")
                        details.append(f"MACD: {macd:.2f} > Signal: {macd_signal:.2f} (Bullish)")
                    elif macd < macd_signal:
                        verdicts.append("Bearish (MACD < Signal)")
                        details.append(f"MACD: {macd:.2f} < Signal: {macd_signal:.2f} (Bearish)")
                    else:
                        details.append(f"MACD: {macd:.2f} ≈ Signal: {macd_signal:.2f} (Neutral)")
                # EMA Crossover
                close = data['Close'].iloc[-1]
                ema20 = data['EMA_20'].iloc[-1] if 'EMA_20' in data.columns else None
                ema50 = data['EMA_50'].iloc[-1] if 'EMA_50' in data.columns else None
                if ema20 is not None and ema50 is not None:
                    if ema20 > ema50:
                        verdicts.append("Bullish (EMA20 > EMA50)")
                        details.append(f"EMA20: {ema20:.2f} > EMA50: {ema50:.2f} (Bullish)")
                    elif ema20 < ema50:
                        verdicts.append("Bearish (EMA20 < EMA50)")
                        details.append(f"EMA20: {ema20:.2f} < EMA50: {ema50:.2f} (Bearish)")
                    else:
                        details.append(f"EMA20: {ema20:.2f} ≈ EMA50: {ema50:.2f} (Neutral)")
                # Momentum
                if 'Momentum' in data.columns:
                    momentum = data['Momentum'].iloc[-1]
                    if momentum > 0:
                        verdicts.append("Bullish (Positive Momentum)")
                        details.append(f"Momentum: {momentum:.2f} (Bullish)")
                    elif momentum < 0:
                        verdicts.append("Bearish (Negative Momentum)")
                        details.append(f"Momentum: {momentum:.2f} (Bearish)")
                    else:
                        details.append(f"Momentum: {momentum:.2f} (Neutral)")
                # Stochastic Oscillator
                if '%K' in data.columns and '%D' in data.columns:
                    k = data['%K'].iloc[-1]
                    d = data['%D'].iloc[-1]
                    if k > 80:
                        verdicts.append("Bearish (Stochastic Overbought)")
                        details.append(f"%K: {k:.2f} (Overbought)")
                    elif k < 20:
                        verdicts.append("Bullish (Stochastic Oversold)")
                        details.append(f"%K: {k:.2f} (Oversold)")
                    else:
                        details.append(f"%K: {k:.2f} (Neutral)")
                # Volatility (ATR)
                if 'ATR' in data.columns:
                    atr = data['ATR'].iloc[-1]
                    details.append(f"ATR (Volatility): {atr:.2f}")
                # Final verdict
                bullish = sum('Bullish' in v for v in verdicts)
                bearish = sum('Bearish' in v for v in verdicts)
                if bullish > bearish:
                    final_verdict = "Bullish"
                    color = "#27ae60"
                elif bearish > bullish:
                    final_verdict = "Bearish"
                    color = "#c0392b"
                else:
                    final_verdict = "Neutral"
                    color = "#f39c12"
                # Educational info for each indicator
                indicator_ranges = [
                    ("RSI", "30-70 is considered neutral. Above 70: Overbought (possible reversal down). Below 30: Oversold (possible reversal up)."),
                    ("MACD", "Bullish when MACD > Signal line. Bearish when MACD < Signal line. Crossovers are key signals."),
                    ("EMA20/EMA50", "Bullish when EMA20 > EMA50 (short-term trend up). Bearish when EMA20 < EMA50 (trend down). Crossovers signal trend changes."),
                    ("Momentum", "Positive: Upward momentum. Negative: Downward momentum. Near zero: Sideways/neutral."),
                    ("Stochastic %K", "20-80 is neutral. Above 80: Overbought. Below 20: Oversold. Crosses with %D can signal turns."),
                    ("ATR", "Higher ATR = higher volatility. No strict 'ideal', but compare to historical ATR for context.")
                ]
                return html.Div([
                    html.H3(f"Market Sentiment for {main_ticker}", style={"textAlign": "center", "color": color}),
                    html.Div(f"Final Verdict: {final_verdict}", style={"textAlign": "center", "fontSize": 22, "color": color, "marginTop": 20, "fontWeight": "bold"}),
                    html.Ul([
                        html.Li(detail, style={"fontSize": 16, "color": "#555", "marginBottom": 6}) for detail in details
                    ], style={"maxWidth": 600, "margin": "30px auto 0 auto"}),
                    html.Hr(),
                    html.H4("What are the ideal/best ranges for these indicators?", style={"textAlign": "center", "color": "#2980b9", "marginTop": 30}),
                    html.Ul([
                        html.Li([
                            html.Strong(f"{name}: ", style={"color": "#222"}),
                            html.Span(desc, style={"color": "#555", "fontSize": 15})
                        ], style={"marginBottom": 8}) for name, desc in indicator_ranges
                    ], style={"maxWidth": 700, "margin": "0 auto 20px auto"}),
                    html.Div("Sentiment verdict is based on a combination of RSI, MACD, EMA crossovers, Momentum, Stochastic Oscillator, and Volatility (ATR).", style={"color": "#888", "fontSize": 14, "textAlign": "center", "marginTop": 30})
                ])
            except Exception as e:
                return html.Div(f"Error calculating sentiment: {e}", style={"color": "red", "textAlign": "center"})
        # --- Chart Analysis Summary and Accuracy Note ---
        # Generate a short summary of the analyzed chart
        summary_lines = []
        # Trend direction
        if not data.empty and 'EMA_20' in data.columns:
            last_close = data['Close'].iloc[-1]
            last_ema = data['EMA_20'].iloc[-1]
            if last_close > last_ema:
                summary_lines.append("Current trend: Uptrend (price above EMA 20)")
            elif last_close < last_ema:
                summary_lines.append("Current trend: Downtrend (price below EMA 20)")
            else:
                summary_lines.append("Current trend: Sideways (price near EMA 20)")
        # Volatility
        if 'ATR' in data.columns:
            atr = data['ATR'].iloc[-1]
            summary_lines.append(f"Volatility (ATR): {atr:.2f}")
        # Recent buy/sell signals
        recent_signals = []
        if 'Buy_Signal' in data.columns and data['Buy_Signal'].iloc[-1]:
            recent_signals.append("Buy signal detected (price crossed above EMA 20)")
        if 'Sell_Signal' in data.columns and data['Sell_Signal'].iloc[-1]:
            recent_signals.append("Sell signal detected (price crossed below EMA 20)")
        if recent_signals:
            summary_lines.extend(recent_signals)
        # Momentum
        if 'Momentum' in data.columns:
            momentum = data['Momentum'].iloc[-1]
            if momentum > 0:
                summary_lines.append("Momentum: Positive")
            elif momentum < 0:
                summary_lines.append("Momentum: Negative")
            else:
                summary_lines.append("Momentum: Neutral")
        # Compose summary (only the summary of the chart analyzed)
        chart_summary = html.Div([
            html.Hr(),
            html.Div([
                html.Strong("Chart Summary: ", style={"color": "#1565c0", "fontSize": 16}),
                html.Span(" ".join(summary_lines) if summary_lines else "No significant signals detected.", style={"color": "#555", "fontSize": 15})
            ], style={"margin": "10px auto 0 auto", "textAlign": "center", "maxWidth": 700, "background": "#f0f4f8", "borderRadius": 6, "padding": 10, "boxShadow": "0 1px 3px #eee"})
        ])
        # Default: Analysis tab
        return html.Div([
            dcc.Graph(figure=fig, style=graph_style),
            metrics_component,
            chart_summary,
            html.H2("Predicted Prices for the Next 30 Days", style={"textAlign": "center", "color": "#34495e", "fontSize": "clamp(1.1rem, 4vw, 1.5rem)"}),
            prediction_component
        ])
    # Removed export data callback

    return app

# Expose server for Azure/gunicorn
# Default values for initial dashboard load
server = None
if __name__ != "__main__":
    default_ticker = ""
    default_data = fetch_stock_data(default_ticker)
    default_data = calculate_indicators(default_data)
    default_predictions, default_error_metrics = predict_stock_prices(default_data)
    app = create_dashboard(default_data, default_ticker, default_predictions, default_error_metrics)
    server = app.server

if __name__ == "__main__":
    default_ticker = ""
    default_data = fetch_stock_data(default_ticker)
    default_data = calculate_indicators(default_data)
    default_predictions, default_error_metrics = predict_stock_prices(default_data)
    app = create_dashboard(default_data, default_ticker, default_predictions, default_error_metrics)
    app.run(debug=True, use_reloader=False, port=8000)
