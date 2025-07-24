from dash import html

def get_support_layout():
    return html.Div([
        html.H2("Support"),
        html.P("Find help articles, FAQs, and resources to get the most out of StockLens.")
    ], style={"maxWidth": 700, "margin": "40px auto", "padding": 24, "background": "#f8f9fa", "borderRadius": 8})
