from dash import html

def get_terms_of_service_layout():
    return html.Div([
        html.H2("Terms of Service"),
        html.P("Read the terms and conditions for using StockLens. By using this tool, you agree to abide by these terms.")
    ], style={"maxWidth": 700, "margin": "40px auto", "padding": 24, "background": "#f8f9fa", "borderRadius": 8})
