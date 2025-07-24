from dash import html

def get_explore_layout():
    return html.Div([
        html.H2("Explore StockLens"),
        html.P("Discover features, tools, and resources to analyze stocks and markets with StockLens.")
    ], style={"maxWidth": 700, "margin": "40px auto", "padding": 24, "background": "#f8f9fa", "borderRadius": 8})
