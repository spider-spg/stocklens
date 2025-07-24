from dash import html

def get_how_it_works_layout():
    return html.Div([
        html.H2("How It Works"),
        html.P("Learn how StockLens analyzes stock data, generates signals, and provides insights for investors.")
    ], style={"maxWidth": 700, "margin": "40px auto", "padding": 24, "background": "#f8f9fa", "borderRadius": 8})
