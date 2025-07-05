import yfinance as yf  # Library for fetching stock data
import pandas as pd  # Library for handling data in tables (like Excel)
import dash  # Library for creating a web-based dashboard
from dash import dcc, html  # Components for the Dash dashboard
import plotly.graph_objects as go  # For creating interactive graphs
import torch  # PyTorch for building and training the prediction model
import torch.nn as nn  # For creating neural network layers
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
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Set up the layers of the neural network."""
        super(StockPredictor, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)  # Hidden layer
        self.output_layer = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        """Defines how the input data flows through the network."""
        x = torch.relu(self.hidden_layer(x))  # Apply activation function to hidden layer
        x = self.output_layer(x)  # Compute the final output
        return x


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
        data['Pct_Change'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
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
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        # Train/Test split
        train_size = int(len(X_tensor) * 0.8)
        X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
        # Improved model with Dropout and LeakyReLU
        class ImprovedStockPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.hidden_layer1 = nn.Linear(input_size, hidden_size)
                self.dropout = nn.Dropout(0.2)
                self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
                self.output_layer = nn.Linear(hidden_size, output_size)
                self.leaky_relu = nn.LeakyReLU()
            def forward(self, x):
                x = self.leaky_relu(self.hidden_layer1(x))
                x = self.dropout(x)
                x = self.leaky_relu(self.hidden_layer2(x))
                x = self.output_layer(x)
                return x
        model = ImprovedStockPredictor(input_size=X_train.shape[1], hidden_size=128, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Train the model
        console.print("[cyan]Training the neural network...[/cyan]")
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            # Early stopping on validation loss
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_test)
                val_loss = torch.sqrt(criterion(val_predictions, y_test))
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter > patience:
                    console.print(f"Early stopping at epoch {epoch+1}")
                    break
            if (epoch + 1) % 50 == 0:
                console.print(f"Epoch {epoch + 1}/1000 - Loss: {loss.item():.4f} - Val RMSE: {val_loss.item():.4f}")
        # Calculate error metrics
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).squeeze().numpy()
            test_true = y_test.squeeze().numpy()
            rmse = np.sqrt(np.mean((test_preds - test_true) ** 2))
            mae = np.mean(np.abs(test_preds - test_true))
        # Predict future prices
        last_row = data.iloc[-1]
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        future_data = []
        for i, date in enumerate(future_dates):
            day_of_week = date.dayofweek
            prev_close_1 = last_row['Close'] if i == 0 else future_data[-1][features.index('Prev_Close_1')]
            prev_close_3 = last_row['Close'] if i < 3 else future_data[-3][features.index('Prev_Close_1')]
            prev_close_5 = last_row['Close'] if i < 5 else future_data[-5][features.index('Prev_Close_1')]
            prev_close_10 = last_row['Close'] if i < 10 else future_data[-10][features.index('Prev_Close_1')]
            row = [
                (data['Days'].max() + i + 1),
                prev_close_1,  # Close
                last_row['EMA_20'],
                last_row['EMA_50'],
                last_row['RSI'],
                last_row['Upper_Band'],
                last_row['Lower_Band'],
                last_row['MACD'],
                last_row['MACD_Signal'],
                last_row['%K'],
                last_row['%D'],
                last_row['ATR'],
                last_row['Pct_Change'],
                last_row['Volume_Change'],
                last_row['Momentum'],
                prev_close_1,
                prev_close_3,
                prev_close_5,
                prev_close_10,
                day_of_week
            ]
            future_data.append(row)
        future_scaled = scaler.transform(future_data)
        future_tensor = torch.tensor(future_scaled[:, [i for i in range(len(features)) if features[i] != 'Close']], dtype=torch.float32)
        future_prices_scaled = model(future_tensor).detach().numpy()
        # Inverse transform: set the predicted close in the right column
        future_full = np.copy(future_scaled)
        future_full[:, features.index('Close')] = future_prices_scaled.flatten()
        future_prices = scaler.inverse_transform(future_full)[:, features.index('Close')]
        predictions = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices.flatten()})
        error_metrics = {'RMSE': float(rmse), 'MAE': float(mae)}
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

    # Add buy/sell signals based on EMA
    data['Buy_Signal'] = (data['Close'] > data['EMA_20']) & (data['Close'].shift(1) <= data['EMA_20'])
    data['Sell_Signal'] = (data['Close'] < data['EMA_20']) & (data['Close'].shift(1) >= data['EMA_20'])

    # Create a graph with Plotly
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
        title=f"{ticker} Stock Analysis with Indicators and Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Indicators"
    )

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
        html.H1("Stock Market Analysis Tool", style={"textAlign": "center", "color": "#2c3e50", "marginTop": 20}),
        html.Div([
            dcc.Input(id="ticker-input", type="text", placeholder="Enter stock ticker (e.g., AAPL)", value=ticker, style={"marginRight": 10, "width": 200}),
            dcc.DatePickerSingle(id="start-date", placeholder="Start Date", style={"marginRight": 10}),
            dcc.DatePickerSingle(id="end-date", placeholder="End Date", style={"marginRight": 10}),
            html.Button("Submit", id="submit-btn", n_clicks=0, style={"background": "#2980b9", "color": "white", "border": "none", "padding": "8px 18px", "borderRadius": 5})
        ], style={"textAlign": "center", "marginBottom": 30}),
        html.Div(id="export-controls-container"),
        dcc.Tabs(id="main-tabs", value="tab-analysis", children=[
            dcc.Tab(label="Analysis", value="tab-analysis"),
            dcc.Tab(label="News", value="tab-news"),
            dcc.Tab(label="Fundamentals", value="tab-fundamentals"),
            dcc.Tab(label="Market Sentiment", value="tab-sentiment"),
        ]),
        html.Div(id="dashboard-content"),
        dcc.Download(id="download-data"),
        education_section,
        disclaimer_section,
        html.Hr(),
        html.Footer([
            html.Div([
                html.Span("Made with ", style={"color": "#888"}),
                html.Span("❤", style={"color": "#e74c3c", "fontSize": 18, "fontWeight": "bold"}),
                html.Span(" by Nikunj Maru", style={"color": "#888"})
            ], style={"textAlign": "center", "marginTop": 30, "marginBottom": 10, "fontSize": 16})
        ])
    ], style={"fontFamily": "Segoe UI, Arial, sans-serif", "backgroundColor": "#f8f9fa", "padding": 0, "minHeight": "100vh"})

    @app.callback(
        Output("export-controls-container", "children"),
        [Input("submit-btn", "n_clicks")],
        [State("ticker-input", "value"),
         State("start-date", "date"),
         State("end-date", "date")]
    )
    def show_export_controls(n_clicks, ticker_val, start_date_val, end_date_val):
        if n_clicks == 0 or not ticker_val:
            return None
        return html.Div([
            html.Div([
                dcc.Dropdown(
                    id="export-format",
                    options=[
                        {"label": "CSV", "value": "csv"},
                        {"label": "Excel", "value": "excel"},
                        {"label": "JSON", "value": "json"}
                    ],
                    value="csv",
                    style={"width": 120, "display": "inline-block", "marginRight": 10}
                ),
                html.Button("Export Data", id="export-btn", n_clicks=0, style={"background": "#27ae60", "color": "white", "border": "none", "padding": "8px 18px", "borderRadius": 5})
            ], style={"textAlign": "center", "marginBottom": 20, "marginTop": 0})
        ])

    @app.callback(
        Output("dashboard-content", "children"),
        [Input("submit-btn", "n_clicks"),
         Input("main-tabs", "value")],
        [State("ticker-input", "value"),
         State("start-date", "date"),
         State("end-date", "date")]
    )
    def update_dashboard(n_clicks, tab, ticker_val, start_date_val, end_date_val):
        if n_clicks == 0 or not ticker_val:
            return html.Div("Enter a ticker and click Submit to view analysis.", style={"textAlign": "center", "color": "#888", "marginTop": 40})
        # Fetch and process data
        data = fetch_stock_data(ticker_val, start_date_val, end_date_val)
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
            title=f"{ticker_val} Stock Analysis with Indicators and Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Indicators"
        )
        # Prediction chart
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
        # News tab
        if tab == "tab-news":
            try:
                stock = yf.Ticker(ticker_val)
                news = getattr(stock, 'news', [])
                if news:
                    news_items = [
                        html.Li([
                            html.A(item.get('title', 'No Title'), href=item.get('link', '#'), target="_blank"),
                            html.Br(),
                            html.Span(item.get('publisher', ''), style={"color": "#888", "fontSize": 13}),
                            html.Br(),
                            html.Span(item.get('providerPublishTime', ''), style={"color": "#aaa", "fontSize": 12})
                        ], style={"marginBottom": 15}) for item in news[:10]
                    ]
                    return html.Div([
                        html.H3(f"Latest News for {ticker_val}", style={"textAlign": "center", "color": "#2c3e50"}),
                        html.Ul(news_items)
                    ])
                else:
                    return html.Div("No news found for this ticker.", style={"textAlign": "center", "color": "#888", "marginTop": 40})
            except Exception as e:
                return html.Div(f"Error fetching news: {e}", style={"color": "red", "textAlign": "center"})
        # Fundamentals tab
        if tab == "tab-fundamentals":
            try:
                stock = yf.Ticker(ticker_val)
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
                    html.H3(f"Fundamentals for {ticker_val}", style={"textAlign": "center", "color": "#2c3e50"}),
                    html.Table([
                        html.Tbody(rows)
                    ], style={"margin": "0 auto", "fontSize": 16, "background": "#f8f9fa", "borderRadius": 6, "boxShadow": "0 1px 3px #eee", "padding": 10})
                ])
            except Exception as e:
                return html.Div(f"Error fetching fundamentals: {e}", style={"color": "red", "textAlign": "center"})
        # Market Sentiment tab (simple: based on news headlines)
        if tab == "tab-sentiment":
            try:
                stock = yf.Ticker(ticker_val)
                news = getattr(stock, 'news', [])
                if not news:
                    return html.Div("No news found for this ticker.", style={"textAlign": "center", "color": "#888", "marginTop": 40})
                # Simple sentiment: count positive/negative/neutral words in headlines
                sentiments = []
                for item in news[:10]:
                    title = item.get('title', '')
                    if title:
                        blob = TextBlob(title)
                        sentiments.append(blob.sentiment.polarity)
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    if avg_sentiment > 0.1:
                        sentiment_label = "Positive"
                        color = "#27ae60"
                    elif avg_sentiment < -0.1:
                        sentiment_label = "Negative"
                        color = "#c0392b"
                    else:
                        sentiment_label = "Neutral"
                        color = "#f39c12"
                    return html.Div([
                        html.H3(f"Market Sentiment for {ticker_val}", style={"textAlign": "center", "color": color}),
                        html.Div(f"Average Sentiment Score: {avg_sentiment:.2f} ({sentiment_label})", style={"textAlign": "center", "fontSize": 18, "color": color, "marginTop": 20}),
                        html.Ul([
                            html.Li(item.get('title', '')) for item in news[:10]
                        ], style={"marginTop": 20, "color": "#555", "fontSize": 15})
                    ])
                else:
                    return html.Div("No sentiment data available.", style={"textAlign": "center", "color": "#888", "marginTop": 40})
            except Exception as e:
                return html.Div(f"Error fetching sentiment: {e}", style={"color": "red", "textAlign": "center"})
        # Default: Analysis tab
        return html.Div([
            dcc.Graph(figure=fig, style={"marginBottom": 40}),
            metrics_component,
            html.H2("Predicted Prices for the Next 30 Days", style={"textAlign": "center", "color": "#34495e"}),
            prediction_component
        ])
    @app.callback(
        Output("download-data", "data"),
        Input("export-btn", "n_clicks"),
        State("ticker-input", "value"),
        State("start-date", "date"),
        State("end-date", "date"),
        State("export-format", "value"),
        prevent_initial_call=True
    )
    def export_data_callback(n_clicks, ticker_val, start_date_val, end_date_val, export_format):
        if not ticker_val:
            return None
        data = fetch_stock_data(ticker_val, start_date_val, end_date_val)
        data = calculate_indicators(data)
        if data.empty:
            return None
        if export_format == "csv":
            return dict(content=data.to_csv(), filename=f"{ticker_val}_analysis.csv")
        elif export_format == "excel":
            import io
            output = io.BytesIO()
            data.to_excel(output)
            output.seek(0)
            return dict(content=output.getvalue(), filename=f"{ticker_val}_analysis.xlsx", type="binary")
        elif export_format == "json":
            return dict(content=data.to_json(), filename=f"{ticker_val}_analysis.json")
        return None

    # Only run the Dash server if running locally (not under gunicorn/Azure)
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or __name__ == "__main__":
        app.run(debug=True, use_reloader=False, port=8000)
# Only run the dashboard if this script is executed directly (not imported by gunicorn)
if __name__ == "__main__":
    # Default values for initial dashboard load
    default_ticker = "AAPL"
    default_data = fetch_stock_data(default_ticker)
    default_data = calculate_indicators(default_data)
    default_predictions, default_error_metrics = predict_stock_prices(default_data)
    create_dashboard(default_data, default_ticker, default_predictions, default_error_metrics)










