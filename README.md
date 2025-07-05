# Stock Market Analysis and Prediction Tool

## Project Overview

This project is a **Stock Market Analysis and Prediction Tool** that allows users to:
1. Analyze historical stock market data.
2. View technical indicators such as **Exponential Moving Average (EMA)**, **Relative Strength Index (RSI)**, and **Bollinger Bands**.
3. Predict future stock prices using a **PyTorch Neural Network**.
4. Visualize data interactively with a **Dash dashboard**.
5. Export analyzed data in multiple formats (**CSV, Excel, JSON**).

This project was designed as a hands-on exploration of **stock market data analysis**, **machine learning**, and **data visualization**.

---

## Features

### 1. **Stock Data Fetching**
- Fetches historical stock data using the **Yahoo Finance API** (`yfinance`).
- Users can specify custom date ranges or analyze the last year of data by default.

### 2. **Technical Indicators**
- Calculates key indicators like:
  - **EMA (20-day and 50-day)**: Tracks average price trends.
  - **Bollinger Bands**: Visualizes price volatility.
  - **RSI**: Helps identify overbought or oversold stocks.
  - **Momentum**: Measures the speed of price changes.

### 3. **Data Insights**
- Provides a summary of stock data, including:
  - Average, highest, and lowest closing prices.
  - Price volatility (standard deviation).
  - RSI with insights on whether the stock is overbought, oversold, or neutral.

### 4. **Price Prediction**
- Predicts stock prices for the next 30 days using a **PyTorch Neural Network**.
- Features used for predictions include:
  - Historical prices
  - EMA
  - RSI
  - Bollinger Bands
  - Momentum indicators

### 5. **Interactive Dash Dashboard**
- Visualizes historical and predicted data interactively.
- Displays **buy/sell signals** using technical indicators.
- Easy-to-navigate interface with dynamic graphs.

### 6. **Export Options**
- Export analyzed data into CSV, Excel, or JSON formats for further use.

---

## How to Run the Project

### **Step 1: Install Requirements**
Ensure you have Python 3.10 or later installed. Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

### **Step 2: Run the Program**
Run the project in your terminal:
```bash
python fetch.py
```

### **Step 3: Follow the Instructions**
- Input the stock ticker (e.g., `AAPL` for Apple, `GME` for GameStop).
- Specify a date range for analysis or press Enter for the default last year of data.
- Select from the following options:
  1. Predict Stock Prices
  2. View Interactive Dashboard
  3. Export Data

---

## File Structure

```
.
├── fetch.py             # Main Python script
├── requirements.txt     # Required libraries and dependencies
└── README.md            # This documentation
```

---



