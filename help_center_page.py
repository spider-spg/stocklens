from dash import html

def get_help_center_layout():
    return html.Div([
        html.H2("Help Center"),
        html.P("Browse common questions and troubleshooting guides for StockLens users.")
    ], style={"maxWidth": 700, "margin": "40px auto", "padding": 24, "background": "#f8f9fa", "borderRadius": 8})
